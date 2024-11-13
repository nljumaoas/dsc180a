from transformers import AutoTokenizer
from datasets import load_dataset
import os

cache_dir = "/workspace/Sys_team/yuxuan_workspace/dsc180a/.cache"
model_name = "NousResearch/Meta-Llama-3-8B"
test_model_name = 'TinyLlama/TinyLlama-1.1B-step-50K-105b'
max_seq_len = 2048

# Define paths for Hugging Face dataset caching
tokenized_data_cache_dir = "./datasets/tokenized_data"

def prepare_and_cache_dataset():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(test_model_name, cache_dir=cache_dir)
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

    # Save the tokenized datasets to Hugging Face's cache
    train_tokenized_datasets.save_to_disk(os.path.join(tokenized_data_cache_dir, "train"))
    validation_tokenized_dataset.save_to_disk(os.path.join(tokenized_data_cache_dir, "validation"))

    print("Tokenized datasets saved to disk.")

if __name__ == "__main__":
    prepare_and_cache_dataset()
