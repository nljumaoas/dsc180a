from transformers import AutoTokenizer
from datasets import load_dataset
import os, random
import numpy as np
import torch


cache_dir = "./hf_cache"
max_seq_len = 2048

# Define paths for Hugging Face dataset caching
tokenized_data_cache_dir = "./datasets/SlimPajama-6B_tokenized_data_packing"


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
    tokenizer = AutoTokenizer.from_pretrained('/workspace/ML_team/llama_tokenizer_1b', cache_dir=cache_dir)
    #tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset (you can use the Hugging Face library to download and load it directly)
    train_subset = load_dataset('DKYoon/SlimPajama-6B', split="train", cache_dir=cache_dir, download_mode='reuse_cache_if_exists')
    validation_subset = load_dataset('DKYoon/SlimPajama-6B', split="validation", cache_dir=cache_dir, download_mode='reuse_cache_if_exists')

    # Tokenization function with max length truncation
    def pack_sequences_truncate(batch, max_length=1024):
        input_ids, attention_masks = [], []
        current_chunk = {"input_ids": [], "attention_mask": []}
        current_length = 0
        
        for tokens in batch['text']:
            # Check if adding the sequence exceeds max length
            tokenized = tokenizer(tokens, truncation=True, max_length=max_length, padding=False)
            if current_length + len(tokenized['input_ids'])+1 <= max_length:
                #print("current: ", current_length, " added: ", len(tokenized['input_ids'])+1)
                current_chunk["input_ids"].extend(tokenized['input_ids'] + [tokenizer.eos_token_id])
                current_length += len(tokenized['input_ids'])+1
            else:
                # Save current chunk and start a new one
                if current_length < max_length:
                    current_chunk["input_ids"].extend(tokenized['input_ids'][:max_length-current_length-1] + [tokenizer.eos_token_id])
                    input_ids.append(current_chunk["input_ids"])
                    attention_masks.append([1]*max_length)
                    current_chunk = {"input_ids": []}
                    current_length = 0
                else:
                    input_ids.append(current_chunk["input_ids"])
                    attention_masks.append([1]*max_length)
                    current_chunk = {"input_ids": []}
                    if len(tokenized['input_ids']) == max_length:
                        current_chunk["input_ids"].extend(tokenized['input_ids'][:max_length-1] + [tokenizer.eos_token_id])
                        current_length = max_length
                    else:
                        current_chunk["input_ids"].extend(tokenized['input_ids'] + [tokenizer.eos_token_id])
                        current_length = len(tokenized['input_ids'])+1

        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_masks), "labels": torch.tensor(input_ids)}
    
    train_tokenized_datasets = train_subset.map(pack_sequences_truncate, batched=True, remove_columns=['text', 'meta', '__index_level_0__'], num_proc=50)
    validation_tokenized_dataset = validation_subset.map(pack_sequences_truncate, batched=True, remove_columns=['text', 'meta', '__index_level_0__'], num_proc=50)

    # Apply tokenization and save to cache
    #train_tokenized_datasets = train_subset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=50)
    #validation_tokenized_dataset = validation_subset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=50)
    print_dataset_shape(train_tokenized_datasets, "Training Dataset")
    print_dataset_shape(validation_tokenized_dataset, "Validation Dataset")


    # Save the tokenized datasets to Hugging Face's cache
    train_tokenized_datasets.save_to_disk(os.path.join(tokenized_data_cache_dir, "train"))
    validation_tokenized_dataset.save_to_disk(os.path.join(tokenized_data_cache_dir, "validation"))

    #print("Tokenized datasets saved to disk.")

if __name__ == "__main__":
    prepare_and_cache_dataset()
