'''A main script to run attack for LLMs.'''
import time
import importlib
import numpy as np
import torch.multiprocessing as mp
from absl import app
from ml_collections import config_flags
import dill

import os
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

    print(params)

    train_goals, train_targets, test_goals, test_targets, train_final_target, test_final_target = get_goals_and_targets(params)

    workers, test_workers = get_workers(params)

    managers = {
        "AP": attack_lib.Prompter,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPrompter,
    }

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    if params.transfer:
        prompt_optimizer = attack_lib.ProgressiveMultiPrompter(
            train_goals,
            train_targets,
            workers,
            progressive_models=params.progressive_models,
            progressive_goals=params.progressive_goals,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets,
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
            train_final_target=train_final_target,
            test_final_target =  test_final_target
        )

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
    )

    for worker in workers + test_workers:
        worker.stop()

if __name__ == '__main__':
    app.run(main)