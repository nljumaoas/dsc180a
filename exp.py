from datasets import load_dataset
from transformers import AutoTokenizer
import os

# Define cache directory
CACHE_DIR = "./hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Dataset source and base model
dataset_source = "timdettmers/openassistant-guanaco"
base_model = "Qwen/Qwen1.5-0.5B"

# Load dataset with caching
print("Loading dataset...")
dataset = load_dataset(dataset_source, cache_dir=CACHE_DIR)
print("Dataset loaded successfully.")

# Load tokenizer with caching
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded successfully.")

# Define tokenization function
def tokenize_function(examples):
    MAX_LENGTH = 1024  # Reduce max length for better visualization
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    tokenized["labels"] = tokenized["input_ids"][:]  # Copy input_ids to labels
    return tokenized

# Tokenize dataset with caching
print("Tokenizing dataset (this may take some time)...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    cache_file_name=os.path.join(CACHE_DIR, "tokenized_dataset.arrow"),
)
print("Tokenization completed.")

# Save the first sample to a text file
output_file = os.path.join(CACHE_DIR, "tokenized_sample.txt")
first_sample = tokenized_dataset["train"][0]  # Get the first sample
print(f"Saving the first tokenized sample to {output_file}...")

try:
    with open(output_file, "w") as f:
        f.write("First Tokenized Sample:\n\n")
        for key, value in first_sample.items():
            f.write(f"{key}: {value}\n")
    print(f"First tokenized sample saved successfully to {output_file}")
except Exception as e:
    print(f"Error saving sample to file: {e}")

# Print first sample to console
print("\nFirst tokenized sample:")
for key, value in first_sample.items():
    print(f"{key}: {value}")
