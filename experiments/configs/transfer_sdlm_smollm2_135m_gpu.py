import os

os.sys.path.append("..")
from configs.template import get_config as default_config
from ml_collections import config_dict


def get_config():
    config = default_config()

    # SDLM attack configuration
    config.attack = 'sdlm_opt'
    config.transfer = False
    config.progressive_goals = False
    config.stop_on_success = True
    config.gradient_clip_stategy = None #'whole-50'

    # Tokenizer / model configuration for SmolLM2-135M-Instruct
    config.tokenizer_paths = [
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    ]
    config.tokenizer_kwargs = [
        {"use_fast": False, "add_bos_token": False, "pad_token": "<|im_end|>"},
        {"use_fast": False, "add_bos_token": False, "pad_token": "<|im_end|>"}
    ]

    config.model_paths = [
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    ]
    config.conversation_templates = [
        'smollm-2',
        'smollm-2',
    ]
    config.devices = [
        'cuda:0',
        'cuda:0',
    ]
    config.dtypes = [
        'float16',
        'float16',
    ]
    config.torch_dtype = 'float16'

    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False, "do_sample": True},
        {"low_cpu_mem_usage": True, "use_cache": False, "do_sample": True}
    ]

    # SDLM-specific parameters
    sdlm_variable_kwargs = config_dict.ConfigDict()
    sdlm_variable_kwargs.learning_rate = 0.1
    sdlm_variable_kwargs.logit_scaler = 5.0
    sdlm_variable_kwargs.temperature = 0.7
    sdlm_variable_kwargs.learnable_temperature = True
    sdlm_variable_kwargs.decouple_learnable_temperature = False
    sdlm_variable_kwargs.init_strategy = 'fluency'
    sdlm_variable_kwargs.hard = False
    sdlm_variable_kwargs.lr_scheduler_type = config_dict.FieldReference(
        None, field_type=str
    )
    sdlm_variable_kwargs.lr_scheduler_total_steps = config_dict.FieldReference(
        None, field_type=int
    )
    sdlm_variable_kwargs.lr_scheduler_final_lr = config_dict.FieldReference(
        None, field_type=float
    )
    sdlm_variable_kwargs.lr_scheduler_gamma = config_dict.FieldReference(
        None, field_type=float
    )
    config.sdlm_variable_kwargs = sdlm_variable_kwargs

    config.sdlm_model_stgs_logits_generation = True
    config.sdlm_fluency_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    config.sdlm_model_kwargs = {
        "hard": False,
        "temperature": 0.7,
        "learnable_temperature": False,
        "hidden_state_conditioning": False,
        "use_bpttoken": False,
        "dropout": 0.0,
    }

    # SDLM configuration
    config.acc_grad_n_examples = -1
    config.update_solution_max_new_tokens = 512
    config.gradient_comp_batch_size = 1

    # Optimization parameters
    config.n_steps = 50
    config.batch_size = 2
    config.topk = 10
    config.topq = 5
    config.temp = 0.5
    config.target_weight = 0.1
    config.control_weight = 0.0
    config.test_steps = 5

    # Early stopping
    config.early_stopping = True
    config.early_stopping_steps = 3

    # Logging
    config.logfile = 'results/sdlm_smollm2_135m_gpu_gsm8k.json'
    config.verbose = True

    return config
