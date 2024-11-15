from transformers import AutoTokenizer
from datasets import load_dataset
import os, random
import numpy as np
import torch


cache_dir = "/workspace/Sys_team/yuxuan_workspace/dsc180a/.cache"
model_name = "mlabonne/Meta-Llama-3-8B"
max_seq_len = 512

# Define paths for Hugging Face dataset caching
tokenized_data_cache_dir = "./datasets/tokenized_data"


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)



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


def prepare_and_cache_dataset():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset (you can use the Hugging Face library to download and load it directly)
    train_subset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]", cache_dir=cache_dir)
    validation_subset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:1%]", cache_dir=cache_dir)

    # Tokenization function with max length truncation
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_len)

    # Apply tokenization and save to cache
    train_tokenized_datasets = train_subset.map(tokenize_function, batched=True, remove_columns=["text"])
    validation_tokenized_dataset = validation_subset.map(tokenize_function, batched=True, remove_columns=["text"])
    print_dataset_shape(train_tokenized_datasets, "Training Dataset")
    print_dataset_shape(validation_tokenized_dataset, "Validation Dataset")


    # Save the tokenized datasets to Hugging Face's cache
    train_tokenized_datasets.save_to_disk(os.path.join(tokenized_data_cache_dir, "train"))
    validation_tokenized_dataset.save_to_disk(os.path.join(tokenized_data_cache_dir, "validation"))

    print("Tokenized datasets saved to disk.")

if __name__ == "__main__":
    prepare_and_cache_dataset()
