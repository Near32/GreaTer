from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    # Experiment type
    config.transfer = False
    config.use_wandb = False
    config.project = 'GreaTer-SDLM'

    # SDLM:
    config.acc_grad_n_examples = -1
    config.max_new_tokens_answer = 8 
    config.update_solution_max_new_tokens = 256
    config.gradient_comp_batch_size = 1

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

    # data
    config.train_data = ''
    config.test_data = ''
    config.n_train_data = 50
    config.n_test_data = 0
    config.data_offset = 0

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
    config.extractor_text = "Therefore, the final answer option is  $ "

    config.gbda_deterministic = True

    return config
