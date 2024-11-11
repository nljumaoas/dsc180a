from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import os
from trl import SFTTrainer
from accelerate import init_empty_weights
from datasets import load_dataset, DatasetDict
import tensorboard


# Load Data and Model
cache_dir = "/workspace/Sys_team/yuxuan_workspace/dsc180a/.cache"
model_name = "Weyaxi/Einstein-v8-Llama3.2-1B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
tokenizer.pad_token = tokenizer.eos_token

print(f"Start loading the original model and reinitializing its weights")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir ).to(device)
model = AutoModelForCausalLM.from_config(model.config)
model.to(device)
model_size = sum(t.numel() for t in model.parameters())
print(f"TinyLlama size: {model_size/1000**2:.1f}M parameters")

# - Load Small Dataset
ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train[:1%]", cache_dir=cache_dir)
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation[:1%]", cache_dir=cache_dir)

raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)


context_length = 512

# - Batch Tokenized Dataset
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


print(f"Start batch tokenizing datasets")
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names, batch_size=64, num_proc=64
)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# Set up training arguments
torch.cuda.empty_cache()
args = TrainingArguments(
    output_dir="model_checkpoints",
    per_device_train_batch_size=18,
    per_device_eval_batch_size=18,
    evaluation_strategy="steps",
    eval_steps=30,
    logging_steps=50,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_0,
    fp16=True,
    report_to=["tensorboard"],
    logging_dir = "tensorboard_logs",
    gradient_checkpointing=True
)


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"]
)

print(f"Start to train the model: {model_name}")
trainer.train()