from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_from_disk
import torch
import os, random
import numpy as np

cache_dir = "/workspace/Sys_team/yuxuan_workspace/dsc180a/.cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Model name and paths for cached datasets
model_name = "NousResearch/Meta-Llama-3-8B"
test_model_name = 'TinyLlama/TinyLlama-1.1B-step-50K-105b'
tokenized_data_cache_dir = "./datasets/tokenized_data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_len = 2048


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Running on CPU.")


def print_dataset_shape(dataset, dataset_name="Dataset"):
    print(f"\n{dataset_name} Details:")
    print(f"  Total number of samples: {len(dataset)}")
    
    # Assuming all samples have the same structure, use the first sample for structure details
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, list):
            # Print list length and shape if it's a nested list
            print(f"  {key}: List of length {len(value)} with inner element length {len(value[0]) if len(value) > 0 else 'Unknown'}")
        elif isinstance(value, torch.Tensor):
            # Print tensor shape
            print(f"  {key}: Tensor of shape {value.shape}")
        else:
            # Print type for any other type
            print(f"  {key}: {type(value)}")


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(test_model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")

    # Load tokenized datasets from disk
    print("Loading pre-tokenized datasets from disk...")
    train_tokenized_datasets = load_from_disk(os.path.join(tokenized_data_cache_dir, "train"))
    validation_tokenized_dataset = load_from_disk(os.path.join(tokenized_data_cache_dir, "validation"))
    print("Pre-tokenized datasets loaded successfully.")

    # Print the shape and structure details of the datasets
    print_dataset_shape(train_tokenized_datasets, "Training Dataset")
    print_dataset_shape(validation_tokenized_dataset, "Validation Dataset")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(test_model_name, cache_dir=cache_dir, ignore_mismatched_sizes=True)
    model.config.use_cache = False
    model.to(device)
    print("Model loaded and moved to device.")

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        run_name="dsc180a_sys_exp",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
        deepspeed="/workspace/Sys_team/yuxuan_workspace/dsc180a/config/test_ds_zero2_config.json",
        fp16=True,
        gradient_checkpointing=True,
        report_to='wandb'
    )


    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized_datasets,
        eval_dataset=validation_tokenized_dataset
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Save final model and tokenizer
    model.save_pretrained('./final_model')
    tokenizer.save_pretrained('./final_model')
    print("Model and tokenizer saved successfully.")


if __name__ == "__main__":
    main()
