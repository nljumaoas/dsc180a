import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_from_disk
import wandb
import os
import random
import numpy as np
import deepspeed
from torch.cuda.amp import autocast
from utilities import seed_everything, check_cuda_availability, determine_compute_dtype_and_attention, count_parameters, inspect_model_params


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4"
os.environ["MASTER_ADDR"] = "localhost" 
os.environ["MASTER_PORT"] = "9994"
# os.environ["NCCL_P2P_DISABLE"] = '1'
    
seed_everything(42)
check_cuda_availability()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize training hyperparameters
## I/O Paths
# data_path = "./datasets/SlimPajama-6B_tokenized_data"
data_path = "/workspace/ML_team/datasets_pack_full/tokenized_data"
model_path = './configs/model_configs/llama_750M_config_modify.json'
checkpoint_output_dir = './model_checkpoints'
deepspeed_config = './configs/deepspeed_configs/test_ds_zero2_config.json'
tokenizer_config = '/workspace/ML_team/llama_tokenizer_1b'
logging_dir = './logs'

## Training args
num_proc = 50
attn_implementation = determine_compute_dtype_and_attention()
eval_strategy = "steps"
vis_app = 'wandb'
save_strategy = 'steps'
save_steps = 10000
logging_steps = 20
eval_steps = 10000
num_epoch = 1
batch_size = 12
gradient_checkpointing = False
fp16 = not torch.cuda.is_bf16_supported()
bf16 = torch.cuda.is_bf16_supported()
learning_rate = 5e-4
gradient_accumulation = 5
weight_decay = 0.1 * learning_rate
save_total_limit=3

# Wandb variables
wandb_key = 'ae05f44c8d5afe19940ef81e6f5cf69063392241'
project_name = 'llama-training'
entity_name = 'fjiang7-ucsd'


def initialize_model(config_path='./configs/model_configs/llama_8b_config.json'):
    # with deepspeed.zero.Init():
    #     config = AutoConfig.from_pretrained(config_path)
    #     model = AutoModelForCausalLM.from_config(config, 
    #                                              attn_implementation=attn_implementation["attn_implementation"], 
    #                                              torch_dtype=attn_implementation["compute_dtype"])
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_config(config, 
                                             attn_implementation=attn_implementation["attn_implementation"], 
                                             torch_dtype=attn_implementation["compute_dtype"])
    print(f"Total number of trainable parameters: {count_parameters(model):,}")
    inspect_model_params(model)
    

    # inspect_model_params(model)
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
    model.to(device)
    model.config.use_cache=False



    dataset_train = load_from_disk(os.path.join(data_path, "train/chunk1"))
    dataset_eval = load_from_disk(os.path.join(data_path, "validation"))

    # Select the first 76% of the training data
    print(f"the full traiing shape: {len(dataset_train)}")
    train_size = int(0.3179 * len(dataset_train))  # Calculate 40% of the dataset size
    print(f"the training shape: {train_size / 10**9}")
    dataset_train = dataset_train.select(range(train_size))

    # Select the first 76% of the evaluation data
    eval_size = int(0.3179 * len(dataset_eval))  # Calculate 40% of the dataset size
    dataset_eval = dataset_eval.select(range(eval_size))


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
        save_steps = save_steps,
        deepspeed = deepspeed_config,
        fp16 = fp16,
        bf16 = bf16,  
        warmup_steps=500,
        gradient_checkpointing = gradient_checkpointing,
        save_strategy = save_strategy,
        save_total_limit=save_total_limit,
    )

    def custom_collator(batch):
        # Assuming the batch contains 'input_ids', 'attention_mask', and 'labels'
        input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
        labels = input_ids
        attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=custom_collator
    )

    print("Start training...")
    trainer.train()

    model.save_pretrained("./final_model/target_model_config")
    tokenizer.save_pretrained("./final_model/target_tokenizer_config")
    print("Saved final model and tokenizer.")

    wandb.finish()


if __name__ == "__main__":
    main()