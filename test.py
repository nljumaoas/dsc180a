from utilities import estimate_memory_usage, infer_from_checkpoint, count_parameters
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

NUM_GPUS=2


if __name__ == "__main__":
    
    # estimate_memory_usage(num_gpus=NUM_GPUS, num_nodes=1)
    # llm_reply = infer_from_checkpoint()
    # print(f'Test LLM replies: {llm_reply}')

    # a_n = 0.076
    # a_d = 0.095
    # a_c = 0.057
    # # a_c_min = 0.05
    # a_b = 0.21
    # a_s = 0.76
    # a_c_min = 1/(1/a_s + 1/a_b + 1/a_n)
    # N_c = 8.8 * 10**13
    # D_c = 5.4 * 10**13
    # C_c = 1.6 * 10**7
    # C_c_min = 3.1 * 10**8
    # B_c = 2.1*10**8
    # S_c = 2.1 * 10**3
    # computing_budget = 25.85
    # print(f"a_c_min: {1/(1/a_s + 1/a_b + 1/a_n)}")
    # print(f"the computing budget: {computing_budget}")
    # N = computing_budget**(a_c_min/a_n)
    # print(f"the number of parameters: {computing_budget**(a_c_min/a_n)}")
    # print(f"the number of parameters(exta): {computing_budget**0.73}")
    # print(f"the batch size: {computing_budget**(a_c_min/a_b)}")
    # print(f"the batch size(extra): {computing_budget**0.24}")
    # S = computing_budget**(a_c_min/a_s)
    # print(f"the training steps: {computing_budget**(a_c_min/a_s)}")
    # print(f"the training steps: {computing_budget**0.03}")
    # print(f"the optimal loss: {(3.1*10**8 /computing_budget)**0.05}")
    # print(f"Corresponding compute loss: {(N_c /N)**a_n + (S_c/S)**a_s }")
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_config(config, 
                                             attn_implementation=attn_implementation["attn_implementation"], 
                                             torch_dtype=attn_implementation["compute_dtype"])
    print(f"Total number of trainable parameters: {count_parameters(model):,}")
    def inspect_model_params(model):
        for name, param in model.named_parameters():
            print(f"{name}: {param.numel()}")

    inspect_model_params(model)


