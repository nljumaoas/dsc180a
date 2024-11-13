from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainerCallback
import torch
import transformers
import os, random
import tempfile
import numpy as np

cache_dir = "/workspace/Sys_team/yuxuan_workspace/dsc180a/.cache"
os.makedirs(cache_dir, exist_ok=True) 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TMPDIR"] = "/workspace/Sys_team/yuxuan_workspace/dsc180a/.cache/tmp"
# tempfile.tempdir = "/workspace/Sys_team/yuxuan_workspace/dsc180a/.cache/tmp"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
model_name = "NousResearch/Meta-Llama-3-8B"
test_model_name = 'TinyLlama/TinyLlama-1.1B-step-50K-105b'
# Paths to load processed datasets
train_dataset_path = "./datasets/train_tokenized_dataset.pt"
validation_dataset_path = "./datasets/validation_tokenized_dataset.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
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
    # Print device information
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Running on CPU.")



def print_sample_structure(sample, sample_name="Sample"):
    print(f"{sample_name} structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list) and isinstance(value[0], list):
            print(f"  {key}: {[len(value[0])]}")  # Approximate shape if it's a nested list
        else:
            print(f"  {key}: {type(value)}")



def main():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(test_model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Load dataset from the Hugging Face datasets library
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]", cache_dir = cache_dir, download_mode="reuse_cache_if_exists", ignore_verifications=True)

    # Load 10% of the train and validation splits
    train_subset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]", cache_dir=cache_dir, download_mode="force_redownload")
    validation_subset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:1%]", cache_dir=cache_dir, download_mode="force_redownload")


    # Tokenize the texts
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_len)


    train_tokenized_datasets = train_subset.map(tokenize_function, batched=True, remove_columns = train_subset.column_names, batch_size = 16)
    validation_tokenized_dataset = validation_subset.map(tokenize_function, batched=True, remove_columns = validation_subset.column_names, batch_size = 16)

    train_tokenized_datasets = torch.load(train_dataset_path)
    validation_tokenized_dataset = torch.load(validation_dataset_path)
    # Check tokenized sample structure
    print_sample_structure(train_tokenized_datasets[0], "Train tokenized data")
    print_sample_structure(validation_tokenized_dataset[0], "Validation tokenized data")

    # Load the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # TinyLlama uses a causal (not masked) language model, similar to GPT-2
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(test_model_name, cache_dir = cache_dir, ignore_mismatched_sizes=True)
    model = AutoModelForCausalLM.from_config(model.config)
    model.config.use_cache = False
    model.to(device)


    # Define the training arguments
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

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized_datasets,
        eval_dataset=validation_tokenized_dataset
    )


    # Start the training
    trainer.train()


    # Save the final model and tokenizer
    model.save_pretrained('./final_model')
    tokenizer.save_pretrained('./final_model')


if __name__ == "__main__":
    main()
