import os

os.sys.path.append("..")
from configs.template import get_config as default_config


def get_config():
    config = default_config()
    
    # Override the attack method to use SDLM
    config.attack = 'sdlm_opt'
    
    # Use DistilGPT2 model (much smaller and faster)
    config.model_paths = ['distilbert/distilgpt2', 'distilbert/distilgpt2']
    config.tokenizer_paths = ['distilbert/distilgpt2', 'distilbert/distilgpt2']
    config.conversation_templates = ['gpt-2', 'gpt-2']
    config.devices = ['cpu', 'cpu']
    
    # Force CPU usage and optimize for CPU
    config.torch_dtype = 'float32'  # More stable on CPU
    
    # No quantization needed for DistilGPT2 as it's already small
    
    # SDLM-specific parameters (optimized for CPU and small model)
    config.sdlm_params = {
        'learning_rate': 0.1,    # Higher learning rate for faster convergence
        'temperature': 0.7,      # Lower temperature for more focused sampling
        'learnable_temperature': True,  
        'hidden_state_conditioning': False,  # Disable for CPU efficiency
        'stgs_hard': False,
    }
    
    # Optimization parameters (optimized for CPU and small model)
    config.n_steps = 50          # Fewer steps for faster experimentation
    config.batch_size = 16        # Larger batch size possible with smaller model
    config.topk = 10              # More focused sampling
    config.topq = 5             # More focused sampling
    config.temp = 1             # Lower temperature for more focused sampling
    config.target_weight = 1.0
    config.control_weight = 0.4   # Higher control weight for better guidance
    config.test_steps = 5         # Check more frequently
    
    # Early stopping
    config.stop_on_success = True
    config.early_stopping = True
    config.early_stopping_steps = 3
    
    # Logging
    config.logfile = 'results/sdlm_distilgpt2_cpu_gsm8k.json'
    config.verbose = True
    
    return config
