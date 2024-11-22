from transformers import AutoTokenizer
from datasets import load_dataset
import os
from utilities import seed_everything, print_dataset_shape

# Define constants for caching paths and sequence length
CACHE_DIR = "./hf_cache"
TOKENIZER_DIR = './configs/llama_tokenizer_configs'
TOKENIZED_DATA_CACHE_DIR = "./datasets/SlimPajama-6B_tokenized_data_baseline"
MAX_SEQ_LEN = 2048
DATASET_NAME = 'DKYoon/SlimPajama-6B'
NUM_PROC = 50

# Set random seed
seed_everything(42)

def prepare_and_cache_dataset():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    print(f"Start Loading Dataset: {DATASET_NAME}...")
    train_subset = load_dataset(DATASET_NAME, split="train[:10%]", cache_dir=CACHE_DIR, download_mode='reuse_cache_if_exists')
    validation_subset = load_dataset(DATASET_NAME, split="validation[:10%]", cache_dir=CACHE_DIR, download_mode='reuse_cache_if_exists')

    # Tokenization function with max length truncation
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)

    print(f"Tokenizing Dataset...")
    # Apply tokenization and save to cache
    train_tokenized_datasets = train_subset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=NUM_PROC)
    validation_tokenized_dataset = validation_subset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=NUM_PROC)

    # Display dataset shapes
    print_dataset_shape(train_tokenized_datasets, "Training Dataset")
    print_dataset_shape(validation_tokenized_dataset, "Validation Dataset")

    # Save the tokenized datasets to disk
    train_tokenized_datasets.save_to_disk(os.path.join(TOKENIZED_DATA_CACHE_DIR, "train"))
    validation_tokenized_dataset.save_to_disk(os.path.join(TOKENIZED_DATA_CACHE_DIR, "validation"))

    print(f"Tokenized datasets saved to disk: {TOKENIZED_DATA_CACHE_DIR}.")



if __name__ == "__main__":
    prepare_and_cache_dataset()
