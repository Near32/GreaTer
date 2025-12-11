from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    # Experiment type
    config.log_first = False
    config.test_only = False
    config.transfer = False
    config.use_wandb = False
    config.project = 'GreaTer-SDLM'

    # SDLM:
    config.acc_grad_n_examples = -1
    config.max_new_tokens_answer = 8 
    config.update_solution_max_new_tokens = 256
    config.gradient_comp_batch_size = 1
    config.do_sample = True 
    config.loss_type = 'offline'
    config.gradient_clip_strategy = 'none'
    config.online_teacher_forcing = False
    config.use_differentiable_cache = False
    config.gradient_health = config_dict.ConfigDict()
    config.gradient_health.value_clip = config_dict.FieldReference(
        None, field_type=float
    )
    config.gradient_health.max_norm = config_dict.FieldReference(
        None, field_type=float
    )
    config.sdlm_grad_variance_samples = 0
    config.sdlm_grad_variance_period = 1
    config.sdlm_grad_bias_samples = 0
    config.sdlm_grad_bias_period = 1
    config.sdlm_grad_bias_reference_samples = 0
    config.sdlm_grad_bias_reference_batch_size = 2
    config.sdlm_grad_bias_reference_use_baseline = True
    config.sdlm_grad_bias_reference_reward_scale = 1.0
    config.sdlm_grad_bias_reference_baseline_beta = 0.9

    # Evaluation-only generation overrides
    config.eval_generate_kwargs = config_dict.ConfigDict()
    config.eval_generate_kwargs.temperature = config_dict.FieldReference(
        None, field_type=float
    )
    config.eval_generate_kwargs.top_p = config_dict.FieldReference(
        None, field_type=float
    )
    config.eval_generate_kwargs.do_sample = config_dict.FieldReference(
        None, field_type=bool
    )

    # Prompt evaluation speed controls
    config.use_test_optimized = False

    # General parameters 
    config.target_weight=0.85
    config.control_weight=0.15
    config.progressive_goals=False
    config.progressive_models=False
    config.anneal=True
    config.incr_control=False
    config.stop_on_success=False
    config.verbose=True
    config.allow_non_ascii=False
    config.num_train_models=1

    # Results
    config.result_prefix = 'results/individual_vicuna7b'

    # tokenizers
    config.tokenizer_paths=['/data/vicuna/vicuna-7b-v1.3']
    config.tokenizer_kwargs=[{"use_fast": False}]
    
    config.model_paths=['/data/vicuna/vicuna-7b-v1.3']
    config.model_kwargs=[{"low_cpu_mem_usage": True, "use_cache": False}]
    config.conversation_templates=['vicuna']
    config.devices=['auto']
    config.dtypes=['auto']

    # data
    config.seed = 0
    config.data_seed = -1
    config.torch_deterministic = False
    config.torch_allow_tf32 = False
    config.train_data = ''
    config.valid_data = ''
    config.test_data = ''
    config.n_train_data = 50
    config.n_valid_data = 0
    config.n_test_data = 0
    config.data_offset = 0
    config.validate_on = 'loss'

    # attack-related parameters
    config.attack = 'gcg'
    config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    config.n_steps = 105
    config.test_steps = 100
    config.batch_size = 100
    config.lr = 0.01
    config.topk = 10
    config.topq = 5
    config.temp = 1.0
    config.filter_cand = True


    # Extraction parameter
    config.em_from_gen_str =True
    config.extractor_text = "Therefore, the final answer option is  $ "

    config.gbda_deterministic = True

    return config
