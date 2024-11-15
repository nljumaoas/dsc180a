import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_from_disk
import wandb
import os
import random
import numpy as np
import deepspeed

# Set up environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

# Check cuda support
if torch.cuda.is_available():
    print("CUDA is available. Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Running on CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize wandb visualization app - replace these with your wandb info
# wandb_key = '5c18e0e1920e548d7cd21774c89c6e9a28facc65'
# project_name = 'llama3-8b-training'
# entity_name = 'yuxuan_zhang13'
# wandb.login(key = wandb_key)
# wandb.init(project=project_name, entity=entity_name)


# Initialize training hyperparameters
## I/O Paths
data_path = "/workspace/ML_team/datasets/tokenized_data"
model_path = './configs/config.json'
checkpoint_output_dir = './model_checkpoints'
deepspeed_config = './config/test_ds_zero3_plus_config.json'
logging_dir = './logs'

## Training args
attn_implementation = 'flash_attention_2'
eval_strategy = "steps"
vis_app = 'wandb'
save_strategy = 'no'
logging_steps = 100
eval_steps = 100
num_epoch = 3
batch_size = 1
gradient_checkpointing = True
fp16 = True
learning_rate = 2e-5

def initialize_model(config_path='./configs/config.json'):
    # the default config path is for llama 3.1 8b model
    with deepspeed.zero.Init():
        config = LlamaConfig.from_pretrained(config_path, attn_implementation = attn_implementation)
        model = LlamaForCausalLM(config=config)
    return model

def print_memory_usage(step, stage):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"Step {step} ({stage}): Allocated: {allocated / (1024 ** 3):.2f} GB, Reserved: {reserved / (1024 ** 3):.2f} GB")

class TrainerMemoryMonitor(Trainer):
    def training_step(self, model, inputs):
        step = self.state.global_step
        print_memory_usage(step, "training_step> before")
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            print_memory_usage(step, "forward pass: -before")
            loss = self.compute_loss(model, inputs)
            print_memory_usage(step, "forward pass: -after")

        # On multi-GPU parallel training use mean() to average 
        if self.args.n_gpu > 1:
            loss = loss.mean()  
        
        # Clean cache
        torch.cuda.empty_cache()
        print_memory_usage(step, "backward pass> before")

        # Use `self.accelerator.backward()` for mixed-precision backward pass
        self.accelerator.backward(loss)

        print_memory_usage(step, "backward pass> after")
        print_memory_usage(step, "training_step> after")
        
        # Clean cache again
        torch.cuda.empty_cache()
        
        return loss.detach() / self.args.gradient_accumulation_steps
def main():

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("/workspace/ML_team/train/llama_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    model = initialize_model()
    model.config.use_cache=False
    model.to(device)

    # Load data and data collator
    print("Loading dataset...")
    dataset_train = load_from_disk(os.path.join("/workspace/ML_team/datasets/tokenized_data", "train"))
    dataset_eval = load_from_disk(os.path.join("/workspace/ML_team/datasets/tokenized_data", "validation"))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up trainer
    print("Setting up trainer...")
    training_args = TrainingArguments(
        output_dir = checkpoint_output_dir,
        evaluation_strategy = eval_strategy,
        eval_steps = 100,
        num_train_epochs = num_epoch,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        report_to = vis_app,
        logging_dir = logging_dir,
        logging_steps = logging_steps,
        learning_rate = learning_rate,
        deepspeed = deepspeed_config,
        fp16 = fp16,
        gradient_checkpointing = gradient_checkpointing,
        save_strategy = save_strategy,
    )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     tokenizer=tokenizer,
    #     train_dataset=dataset_train,
    #     eval_dataset=dataset_eval,
    #     data_collator=data_collator
    # )
    trainer = TrainerMemoryMonitor(
        model = model, 
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
    )

    # Start training and visualizing
    print("Start training...")
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()