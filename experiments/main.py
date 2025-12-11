'''A main script to run attack for LLMs.'''
import time
import importlib
import numpy as np
import random
import torch
import torch.multiprocessing as mp

# AMD:
# Disable optimized attention backends
#torch.backends.cuda.enable_flash_sdp(False)
#torch.backends.cuda.enable_mem_efficient_sdp(False)
#torch.backends.cuda.enable_math_sdp(True)

from absl import app
from ml_collections import config_flags
import dill
import wandb

import os
import time
from time import sleep
import sys
sys.path.append('..')
from llm_opt.base.attack_manager import get_goals_and_targets, get_workers
import logging


_CONFIG = config_flags.DEFINE_config_file('config')
logging.getLogger('root').setLevel(logging.ERROR)

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

def main(_):

    mp.set_start_method('spawn')

    params = _CONFIG.value
    
    attack_lib = dynamic_import(f'llm_opt.{params.attack}')

    if params.use_wandb:
        wandb.init(
            project=params.project,
            #name=params.result_prefix,
            config=params,
        )

    data_dict = get_goals_and_targets(params)
    train_goals, train_targets, test_goals, test_targets, train_final_target, test_final_target = \
        data_dict['train_goals'], \
        data_dict['train_targets'], \
        data_dict['test_goals'], \
        data_dict['test_targets'], \
        data_dict['train_final_targets'], \
        data_dict['test_final_targets'] 

    valid_goals = data_dict['valid_goals']
    valid_targets = data_dict['valid_targets']
    valid_final_target = data_dict['valid_final_targets']

    seed = params.seed

    torch.manual_seed(seed)
    if hasattr(torch.backends, "cudnn") and params.get('torch_deterministic', False):
        print("ALLOWING DETERMINISTIC & BENCHMARK")
        sleep(5)
        #torch.backends.cudnn.deterministic = False 
        #TODO: after debug : True
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True) 
        #torch.use_deterministic_algorithms(True) 
        torch.backends.cudnn.benchmark = True 
        #TODO: after debug: False 
        #torch.backends.cudnn.benchmark = False 
    
    if hasattr(torch.backends, "cudnn") and params.get('torch_allow_tf32', False):
        print("ALLOWING TF32")
        sleep(5)
        # Reduce memory allocation overhead
        #torch.backends.cuda.matmul.allow_tf32 = True
        #torch.backends.cudnn.allow_tf32 = True
        # UserWarning: 
        # Please use the new API settings to control TF32 behavior, 
        # such as torch.backends.cudnn.conv.fp32_precision = 'tf32' 
        # or torch.backends.cuda.matmul.fp32_precision = 'ieee'. 
        # Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, 
        # torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() 
        # and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. 
        # Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices 
        # (Triggered internally at /__w/TheRock/TheRock/external-builds/pytorch/pytorch/aten/src/ATen/Context.cpp:45.)
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        torch.backends.cuda.matmul.fp32_precision = 'ieee'


    np.random.seed(seed)
    random.seed(seed)

    workers, test_workers = get_workers(params, eval=(params.attack == 'sdlm_opt'))

    managers = {
        "AP": attack_lib.Prompter,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPrompter,
    }

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    #if params.transfer:
    prompt_optimizer = attack_lib.ProgressiveMultiPrompter(
        train_goals,
        train_targets,
        workers,
        progressive_models=params.progressive_models,
        progressive_goals=params.progressive_goals,
        control_init=params.control_init,
        extractor_text=params.extractor_text,
        logfile=f"{params.result_prefix}_{timestamp}.json",
        managers=managers,
        valid_goals=valid_goals,
        valid_targets=valid_targets,
        test_goals=test_goals,
        test_targets=test_targets,
        test_workers=test_workers,
        mpa_deterministic=params.gbda_deterministic,
        mpa_lr=params.lr,
        mpa_batch_size=params.batch_size,
        mpa_n_steps=params.n_steps,
        train_final_target=train_final_target,
        valid_final_target=valid_final_target,
        test_final_target =  test_final_target,
        params=params,
    )
    
    print(f"Starting optimisation over {len(train_goals)} examples...")
    prompt_optimizer.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size, 
        topk=params.topk,
        temp=params.temp,
        topq=params.topq,
        target_weight=params.target_weight,
        control_weight=params.control_weight,
        test_steps=getattr(params, 'test_steps', 1),
        anneal=params.anneal,
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
        allow_non_ascii=params.allow_non_ascii,
        params=params,
    )

    for worker in workers + test_workers:
        worker.stop()

if __name__ == '__main__':
    __spec__ = None
    app.run(main)
