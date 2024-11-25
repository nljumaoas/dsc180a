import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_from_disk
import wandb
import os
import random
import numpy as np
import deepspeed
from torch.cuda.amp import autocast
from utilities import seed_everything, check_cuda_availability, determine_compute_dtype_and_attention, TrainerMemoryMonitor


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["MASTER_ADDR"] = "localhost" 
os.environ["MASTER_PORT"] = "9994"
# os.environ["NCCL_P2P_DISABLE"] = '1'
    
seed_everything(42)
check_cuda_availability()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize training hyperparameters

## I/O Paths
data_path = "./datasets/SlimPajama-6B_tokenized_data"
# data_path = "/workspace/ML_team/datasets_70_1024/tokenized_data"
dataset_train = load_from_disk(os.path.join("/workspace/ML_team/datasets_pack_full/tokenized_data", "train"))
dataset_eval = load_from_disk(os.path.join("/workspace/ML_team/datasets_pack_full/tokenized_data", "validation"))
model_path = './configs/model_configs/llama_1b_config.json'
checkpoint_output_dir = './model_checkpoints'
deepspeed_config = './configs/deepspeed_configs/test_ds_zero3_plus_config.json'
tokenizer_config = './configs/llama_tokenizer_configs'
logging_dir = './logs'

## Training args
num_proc = 50
attn_implementation = determine_compute_dtype_and_attention()
eval_strategy = "steps"
vis_app = 'wandb'
save_strategy = 'no'
logging_steps = 100
eval_steps = 50
num_epoch = 3
gradient_checkpointing_status = False
batch_size = 4
gradient_checkpointing = True
fp16 = not torch.cuda.is_bf16_supported()
bf16 = torch.cuda.is_bf16_supported()
learning_rate = 3e-4
gradient_accumulation = 16
weight_decay = 0.1 * learning_rate

# Wandb variables
wandb_key = '2b4c37f67aa8d460e76224d4348ddb16fbb843e5'
project_name = 'memory_test'
entity_name = 'fjiang7-ucsd'


def initialize_model(config_path='./configs/model_configs/llama_8b_config.json'):
    # the default config path is for llama 3.1 8b model
    with deepspeed.zero.Init():
        config = AutoConfig.from_pretrained(config_path)
        model = AutoModelForCausalLM.from_config(config, 
                                                 attn_implementation=attn_implementation["attn_implementation"], 
                                                 torch_dtype=attn_implementation["compute_dtype"])
    return model

def main():
    
    wandb.login(key = wandb_key)  # Log in directly without setting env variable
    wandb.init(project=project_name, entity=entity_name)


    # load tokenizer and model
    print(f"Computing type: {attn_implementation['compute_dtype']}")
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config)
    tokenizer.pad_token = tokenizer.eos_token
    model = initialize_model(model_path)

    # model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.config.use_cache=False

    print(f"Embed tokens weight shape: {model.model.embed_tokens.weight.shape}")

    dataset_train = load_from_disk(os.path.join(data_path, "train"))
    dataset_eval = load_from_disk(os.path.join(data_path, "validation"))


    torch.cuda.empty_cache()  # Clear any residual GPU memory usage
    print(f"fp16 status: {fp16}; bf16 status: {bf16}")

    training_args = TrainingArguments(
        output_dir = checkpoint_output_dir,
        evaluation_strategy = eval_strategy,
        eval_steps = eval_steps,
        learning_rate = learning_rate,
        per_device_train_batch_size = batch_size, 
        per_device_eval_batch_size = batch_size,
        num_train_epochs = num_epoch,
        weight_decay = weight_decay,
        gradient_accumulation_steps = gradient_accumulation,
        report_to = "wandb",
        logging_dir = logging_dir,
        logging_steps = logging_steps,
        lr_scheduler_type="cosine",
        save_steps = 500,
        deepspeed = deepspeed_config,
        fp16 = fp16,
        bf16 = bf16,  
        warmup_steps=500,
        gradient_checkpointing = gradient_checkpointing_status,
        save_strategy = save_strategy,
        save_total_limit=2,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = TrainerMemoryMonitor(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=data_collator
    )

    print("Start training...")
    trainer.train()

    model.save_pretrained("./final_model/target_model_config")
    tokenizer.save_pretrained("./final_model/target_tokenizer_config")
    print("Saved final model and tokenizer.")

    wandb.finish()


if __name__ == "__main__":
    main()