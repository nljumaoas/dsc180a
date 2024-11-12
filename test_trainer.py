from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainerCallback
import torch
import transformers
import os


cache_dir = "/workspace/Sys_team/yuxuan_workspace/dsc180a/.cache"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Device count:", torch.cuda.device_count())
    # Print device information
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Running on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-step-50K-105b', cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Load dataset from the Hugging Face datasets library
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]", cache_dir = cache_dir, download_mode="reuse_cache_if_exists", ignore_verifications=True)

    # Load 10% of the train and validation splits
    train_subset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]", cache_dir=cache_dir, download_mode="force_redownload")
    validation_subset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:1%]", cache_dir=cache_dir, download_mode="force_redownload")


    # Tokenize the texts
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)


    train_tokenized_datasets = train_subset.map(tokenize_function, batched=True, remove_columns = train_subset.column_names, batch_size = 16)
    validation_tokenized_dataset = validation_subset.map(tokenize_function, batched=True, remove_columns = validation_subset.column_names, batch_size = 16)

    # Check tokenized sample structure
    print_sample_structure(train_tokenized_datasets[0], "Train tokenized data")
    print_sample_structure(validation_tokenized_dataset[0], "Validation tokenized data")

    # Load the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # TinyLlama uses a causal (not masked) language model, similar to GPT-2
    )


    # Load the model
    model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-step-50K-105b', cache_dir = cache_dir, ignore_mismatched_sizes=True)
    model.to(device)


    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
        deepspeed="/workspace/Sys_team/yuxuan_workspace/dsc180a/config/test_ds_zero1_config.json",
        fp16=True,
        gradient_checkpointing = True,
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
