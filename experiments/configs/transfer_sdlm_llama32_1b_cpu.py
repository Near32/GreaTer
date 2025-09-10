import os

os.sys.path.append("..")
from configs.template import get_config as default_config


def get_config():
    config = default_config()
    
    # Override the attack method to use SDLM
    config.attack = 'sdlm_opt'
    config.transfer = False
    config.progressive_goals = False
    config.stop_on_success = True
    
    # Tokenizer configuration matching transfer_llama3.py
    config.tokenizer_paths = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct"
    ]
    config.tokenizer_kwargs = [
        {"use_fast": False, "add_bos_token": False, "pad_token": "<|end_of_text|>"}, 
        {"use_fast": False, "add_bos_token": False, "pad_token": "<|end_of_text|>"}
    ]
    
    # Model configuration for Llama-3-8B-Instruct
    #config.model_paths = [
    #    'HuggingFaceTB/SmolLM2-135M-Instruct', 
    #    'HuggingFaceTB/SmolLM2-135M-Instruct',
    #]
    config.model_paths = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
    ]
    config.conversation_templates = [
        'llama-3', 
        'llama-3',
    ]
    config.devices = [
        'cpu', #'cuda:0', 
        'cpu', #'cuda:0',
    ]
    config.torch_dtype = 'bfloat16'
    
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]
    # Enable 4-bit quantization to fit the model in GPU memory
    '''
    config.quantization_config = {
        'load_in_4bit': True,
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_use_double_quant': True,
        'bnb_4bit_compute_dtype': 'bfloat16'
    }
    '''

    # SDLM-specific parameters (optimized for Llama-3-8B-Instruct)
    config.sdlm_variable_kwargs = {
        'learning_rate': 0.001,     # Lower learning rate for stability
        'logit_scaler': 5.0,       # Adjusted for better gradient flow
        'temperature': 0.1,        # Lower temperature for more focused sampling
        'learnable_temperature': True,
        'init_strategy': 'fluency',
        'hard': False,
    }
    
    config.sdlm_model_stgs_logits_generation = True
    config.sdlm_model_kwargs = {
        "hard": False,
        "temperature": 0.7,        # Slightly higher temperature for model sampling
        "learnable_temperature": False,
        "hidden_state_conditioning": True,  # Enabled for better performance
    }
    
    # SDLM configuration
    config.acc_grad_n_examples = -1
    config.update_solution_max_new_tokens = 512
    config.gradient_comp_batch_size = 1
        
    # Optimization parameters (optimized for CPU and small model)
    config.n_steps = 50          # Fewer steps for faster experimentation
    config.batch_size = 16        # Larger batch size possible with smaller model
    config.topk = 10              # More focused sampling
    config.topq = 5             # More focused sampling
    config.temp = 0.1             # Lower temperature for more focused sampling at the time of new control sampling !!!!
    config.target_weight = 1.0
    config.control_weight = 0.4   # Higher control weight for better guidance
    config.test_steps = 5         # Check more frequently
    
    # Early stopping
    config.stop_on_success = True
    config.early_stopping = True
    config.early_stopping_steps = 3
    
    # Logging
    config.logfile = 'results/sdlm_llama3_1b_cpu_gsm8k.json'
    config.verbose = True
    
    return config
