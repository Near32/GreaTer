from experiments.configs.llama3_8b_instruct import get_config as get_base_config

def get_config():
    config = get_base_config()
    
    # Override the attack method to use SDLM
    config.attack = 'sdlm_opt'
    
    # SDLM-specific parameters
    config.sdlm_params = {
        'learning_rate': 0.1,  # Learning rate for SDLM updates
        'temperature': 1.0,    # Sampling temperature
        'learnable_temperature': True,  # Whether to make temperature learnable
        'hidden_state_conditioning': True,  # Use hidden states for conditioning
        'stgs_hard': False,  # Whether to use hard sampling
    }
    
    # Optimization parameters
    config.n_steps = 100  # Number of optimization steps
    config.batch_size = 16  # Batch size for optimization
    config.topk = 50  # Top-k sampling
    config.topq = 0.9  # Top-p sampling (nucleus sampling)
    config.temp = 1.0  # Temperature for sampling
    config.target_weight = 1.0  # Weight for target loss
    config.control_weight = 0.2  # Weight for control loss
    config.test_steps = 10  # Test every N steps
    
    # Early stopping
    config.stop_on_success = True  # Stop early if attack succeeds
    config.early_stopping = True  # Enable early stopping
    config.early_stopping_steps = 10  # Stop if no improvement for N steps
    
    # Logging
    config.logfile = 'results/sdlm_optimization.json'  # Log file path
    config.verbose = True  # Print progress
    
    return config
