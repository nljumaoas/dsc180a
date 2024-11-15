from transformers import AutoModel, LlamaConfig, LlamaForCausalLM
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
num_gpus = 4
config_path='./configs/config.json'
config = LlamaConfig.from_pretrained(config_path)
model = LlamaForCausalLM(config=config)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=num_gpus, num_nodes=1)
estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=num_gpus, num_nodes=1)
