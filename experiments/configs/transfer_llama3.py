import os

os.sys.path.append("..")
from configs.template import get_config as default_config


def get_config():
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = True
    config.stop_on_success = False
    # config.tokenizer_paths = [
    #     "/scratch1/sfd5525/model_files/l-2-7b-chat-hf.main.c1b0db933684edbfe29a06fa47eb19cc48025e93/",
    #     "/scratch1/sfd5525/model_files/l-2-7b-chat-hf.main.c1b0db933684edbfe29a06fa47eb19cc48025e93/"
    # ]
    config.tokenizer_paths = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ]

    # config.tokenizer_paths = [
    #     "google/gemma-7b-it",
    #     "meta-llama/Llama-2-7b-chat-hf"
    # ]
    config.tokenizer_kwargs = [{"use_fast": False, "add_bos_token": False, "pad_token": "<|end_of_text|>"}, {"use_fast": False, "add_bos_token": False, "pad_token": "<|end_of_text|>"}]

    # config.model_paths = [
    #     "meta-llama/Llama-2-7b-chat-hf",
    #     "meta-llama/Llama-2-7b-chat-hf"
    # ]

    config.model_paths = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ]

    # config.model_paths = [
    #     "google/gemma-7b-it",
    #     "meta-llama/Llama-2-7b-chat-hf"
    # ]

    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]
    config.conversation_templates = ["llama-3", "llama-3"]
    # config.conversation_templates = ["gemma", "llama-2"]
    config.devices = ["cuda:0", "cuda:1"]

    # config.devices = ["0,1", "2,3"]

    return config