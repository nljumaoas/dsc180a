from transformers import LlamaConfig, LlamaForCausalLM
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
import os
import random
import numpy as np
import torch
import json

CACHE_DIR = "./hf_cache"
SAMPLE_OUTPUT_FILE = os.path.join(CACHE_DIR, "tokenized_sample_output.txt")


def determine_compute_dtype_and_attention():
    """
    Determines the appropriate compute dtype and attention implementation
    based on the system's bfloat16 support.

    Returns:
        dict: A dictionary containing the compute dtype and attention implementation.
    """
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'
    
    print("-------------------------")
    print(f"**compute_dtype={compute_dtype}")
    print(f"**attn_implementation={attn_implementation}")
    print("-------------------------")
    
    return {"compute_dtype": compute_dtype, "attn_implementation": attn_implementation}

def check_cuda_availability():
    """
    Checks if CUDA is available and prints the available CUDA devices.

    Returns:
        dict: A dictionary containing the number of CUDA devices and their names.
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available. Device count: {device_count}")
        device_names = {}
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device_name}")
            device_names[i] = device_name
        return {"available": True, "device_count": device_count, "devices": device_names}
    else:
        print("CUDA is not available. Running on CPU.")
        return {"available": False, "device_count": 0, "devices": {}}


def estimate_memory_usage(config_path='./configs/model_configs/llama_8b_config.json', num_gpus=4, num_nodes=1):
    """
    Estimates memory usage for DeepSpeed ZeRO Stage 2 and 3.

    Args:
        config_path (str): Path to the model configuration file.
        num_gpus (int): Number of GPUs per node.
        num_nodes (int): Number of nodes.

    Returns:
        dict: A dictionary containing memory estimates for ZeRO Stage 2 and ZeRO Stage 3.
    """
    try:
        # Load model configuration

        config = LlamaConfig.from_pretrained(config_path)
        model = LlamaForCausalLM(config=config)

        if hasattr(config, "_name_or_path"):
            print(f"Estimating memory usage for model: {config._name_or_path}")
        else:
            print("Model name not available. Proceeding with memory estimation.")
        
        # Estimate memory for ZeRO Stage 3
        zero3_mem_estimate = estimate_zero3_model_states_mem_needs_all_live(
            model, num_gpus_per_node=num_gpus, num_nodes=num_nodes
        )
        
        # Estimate memory for ZeRO Stage 2
        zero2_mem_estimate = estimate_zero2_model_states_mem_needs_all_live(
            model, num_gpus_per_node=num_gpus, num_nodes=num_nodes
        )
        
        return {
            "ZeRO Stage 3": zero3_mem_estimate,
            "ZeRO Stage 2": zero2_mem_estimate,
        }
    except Exception as e:
        raise RuntimeError(f"Error estimating memory usage: {e}")

def seed_everything(seed: int):
    """
    Sets random seed for reproducibility across different libraries.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_samples_to_json(samples, output_file):
    """
    Saves the given samples as a JSON file.

    Args:
        samples (list): A list of dataset samples to save.
        output_file (str): Path to the file where samples will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(output_file, "w") as f:
            json.dump(samples, f, indent=4)  # Save the dataset slice as JSON with indentation
        print(f"Samples saved successfully to: {output_file}")
    except Exception as e:
        print(f"Error saving samples to file: {e}")


def print_dataset_shape(dataset, dataset_name="Dataset", output_file="./hf_cache/tokenized_sample_output.json"):
    """
    Prints the shape and structure of a dataset and saves the first three samples
    to a file using the `save_samples_to_file` function.

    Args:
        dataset: The dataset to analyze.
        dataset_name (str): Name of the dataset for display.
        output_file (str): Path to the file where samples will be saved.
    """
    print(f"\n{dataset_name} Details:")
    print(f"  Total number of samples: {len(dataset)}")
    
    # Get the first three samples
    samples = dataset[:3]
    
    save_samples_to_json(samples, output_file)