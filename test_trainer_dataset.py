from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_from_disk
import torch
import os, random
import numpy as np
import wandb
wandb.init()

cache_dir = "/workspace/Sys_team/yuxuan_workspace/dsc180a/.cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Model name and paths for cached datasets
model_name = "mlabonne/Meta-Llama-3-8B"
test_model_name = 'TinyLlama/TinyLlama-1.1B-step-50K-105b'
tokenized_data_cache_dir = "./datasets/tokenized_data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_len = 4096

# DeepSpeed configuration files
target_config = 'zero3'
ds_configs = {
    "zero1": "/workspace/Sys_team/yuxuan_workspace/dsc180a/config/test_ds_zero1_config.json",
    "zero2": "/workspace/Sys_team/yuxuan_workspace/dsc180a/config/test_ds_zero2_config.json",
    "zero3": "/workspace/Sys_team/yuxuan_workspace/dsc180a/config/test_ds_zero3_config.json"
}


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
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list) and isinstance(value[0], list):
            print(f"  {key}: {[len(value[0])]}")  # Approximate shape if it's a nested list
        else:
            print(f"  {key}: {type(value)}")


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
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
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, ignore_mismatched_sizes=True)
    model = AutoModelForCausalLM.from_config(model.config)
    model.config.use_cache = False
    model.to(device)
    print("Model loaded and moved to device.")

    # # Training arguments
    # training_args = TrainingArguments(
    #     output_dir='./model_checkpoints',
    #     run_name="dsc180a_sys_exp",
    #     overwrite_output_dir=True,
    #     num_train_epochs=3,
    #     per_device_train_batch_size=1,
    #     save_steps=10_000,
    #     save_total_limit=2,
    #     deepspeed="/workspace/Sys_team/yuxuan_workspace/dsc180a/config/test_ds_zero3_config.json",
    #     fp16=True,
    #     gradient_checkpointing=True,
    #     report_to='wandb'
    # )

    # Training arguments without gradient checkpointing
    training_args = TrainingArguments(
        output_dir=f'./model_checkpoints/{target_config}',
        run_name=f"dsc180a_sys_exp_{target_config}",
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        num_train_epochs=2,
        save_steps=10_000,
        save_total_limit=2,
        deepspeed=ds_configs[target_config],
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
    model.save_pretrained(f'./final_model/{target_config}')
    tokenizer.save_pretrained(f'./final_model/{target_config}')
    print("Model and tokenizer saved successfully.")


if __name__ == "__main__":
    main()
