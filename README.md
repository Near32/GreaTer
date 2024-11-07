# 

- Clone the Repository
- Change directory to ```prompt_optimization/experiments/```
- Install requirements ```pip install transformers==4.38.0```
- Install latest transformers specific version: ```pip install transformers==4.38.0```. You may face some errors, but still you should see 4.38.0 version installed. If otherwise let me know.
- Get access from huggingface-hub
```
# only required for model access from huggingface. If you already got it figured out, don't worry about it.
from huggingface_hub import login

# Define your Hugging Face token
hf_token = "YOUR_HUGGING_FACE_TOKEN"  # I can provide my personal token, if required

# Log in to Hugging Face
login(token=hf_token)
```
- There are a lot of moving parts right now in BBH. For example, for each types of the task, we need to got to ``attack_manager.py`` and change the variable named ``addition2``, so that optimization extraction uses correct extraction format. otherwise the loss function will not work.
- In ``gcg_attack.py``, ``intersection_across_examples=10`` is given which works well so far. This might need to be changed for different tasks.
- Also in ``gcg_attack,py``, ``top_k=15``, This might need to be changed, if Intersection size during optimization is too low.
- Right now I am using the addition as ``addition = ". Use "`` in gcg_attack.py. I started to use it after some experiments. This was previously ``To solve this type of problem, use``. I still might need to do more experiment to validate this choice.
- I have the evaluation code in my hand, will share it later. This is basically a modified version from OPRO that I am using.
- In ``experiments/configs``, configs for ``transfer_llama3`` and ``transfer_llama2`` exists. The gpus need to be changed to assign which gpu you want to use. Yes we are using two gpus for each run. Realistically we dont even need one full gpu, but this is to slighly do the experiments faster.
- Run Llama3-8b with disambiguation_qa task from BBH:
- 
```python run.py --config="./configs/transfer_llama3.py" --config.train_data="../data/BBH/disambiguation_qa.json" --config.test_data="../data/BBH/disambiguation_qa.json" --config.result_prefix="results/transfer_llama3_gcg_bbh_disambiguation_qa.json" --config.progressive_goals=True --config.stop_on_success=False --config.allow_non_ascii=False --config.num_train_models=1 --config.n_train_data=25 --config.n_test_data=25 --config.n_steps=800 --config.test_steps=500 --config.anneal=True --config.batch_size=64 --config.topk=40 --config.control_init="logical reasoning and think step by step. Finally give the actual correct answer." --config.control_weight=0.20 --config.target_weight=1.0```
