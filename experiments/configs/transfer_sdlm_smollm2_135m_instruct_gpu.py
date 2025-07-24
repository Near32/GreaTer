import os

os.sys.path.append("..")
from configs.template import get_config as default_config


def get_config():
    config = default_config()
    
    # Override the attack method to use SDLM
    config.attack = 'sdlm_opt'

    # SDLM:
    config.acc_grad_n_examples = -1
    config.update_solution_max_new_tokens = 128
    config.gradient_comp_batch_size = 1

    # Use DistilGPT2 model (much smaller and faster)
    # With 2 :
    config.model_kwargs *= 2
    config.model_paths = ['HuggingFaceTB/SmolLM2-135M-Instruct', 'HuggingFaceTB/SmolLM2-135M-Instruct']
    config.tokenizer_kwargs *= 2
    config.tokenizer_paths = ['HuggingFaceTB/SmolLM2-135M-Instruct', 'HuggingFaceTB/SmolLM2-135M-Instruct']
    #config.conversation_templates = ['gpt-2', 'gpt-2']
    #config.devices = ['cpu', 'cpu']
    # Or with only one:
    #config.model_paths = ['HuggingFaceTB/SmolLM2-135M-Instruct']
    #config.tokenizer_paths = ['HuggingFaceTB/SmolLM2-135M-Instruct']
    #config.model_paths = ['distil]
    #config.tokenizer_paths = ['HuggingFaceTB/SmolLM2-135M-Instruct']
    #config.conversation_templates = ['gemma-2'] #['gpt-2']
    config.conversation_templates = ['smollm-2', 'smollm-2'] #['gpt-2']
    #config.devices = ['cuda:0']
    config.devices = ['cuda:0', 'cuda:0']
    config.torch_dtype = 'float32'  # More stable on CPU
    
    # No quantization needed for DistilGPT2 as it's already small
    
    # SDLM-specific parameters (optimized for CPU and small model)
    config.sdlm_params = { #TODO: model & variable params
        'learning_rate': 0.1,    # Higher learning rate for faster convergence
        'logit_scaler': 10.0,
        'temperature': 0.1,      # Lower temperature for more focused sampling
        'learnable_temperature': True,  
        'hard': False,
    }
    
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
    config.logfile = 'results/sdlm_smollm2_135m_instruct_gpu_gsm8k.json'
    config.verbose = True
    
    return config
