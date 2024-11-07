import time
import importlib
from absl import app
from ml_collections import config_flags
import os
import sys
sys.path.append('..')
from llm_opt.base.attack_manager import get_goals_and_targets, get_workers
from prompt_optimizer.utils.prompt_utils import get_workers

_CONFIG = config_flags.DEFINE_config_file('config')

def dynamic_import(module):
    return importlib.import_module(module)

def main():

    params = _CONFIG.VALUE
    attack_lib = dynamic_import(f'llm_attacks.{params.attack}')
    print(params)

    train_goals, train_target, test_goals, test_targets = get_goals_and_targets(params)
    workers, test_workers = get_workers(params)


