# GreaTer - Gradient Over Reasoning makes Smaller Language Models Strong Prompt Optimizers

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
*Note:
This is an early unoptimized version of the code. In the next version, we will incorporate KV-cache to ensure the whole forward pipeline can be effectively done in one single call, which will make this substantially faster.*
- Follow ``run_llama3_all.sh`` or ``run_gemma2_mcq.sh`` to find out the running command for different BBH tasks.
- You can change ``top_k`` and ``top_q`` as required, which may impact performance.
- In ``experiments/configs``, configs for ``transfer_llama3`` and ``transfer_gemma2`` exists. The gpus need to be changed to assign which gpu you want to use. Yes we are using two gpus for each run. Realistically we dont even need one full gpu, but this is to slighly do the experiments faster.
- Example Run gemma2_2b with tracking_shuffled_objects_five_objects task from BBH:
- 
```python run.py --config="./configs/transfer_gemma2_2b.py" --config.train_data="../data/BBH/tracking_shuffled_objects_five_objects.json" --config.test_data="../data/BBH/tracking_shuffled_objects_five_objects.json" --config.result_prefix="results/transfer_llama32_1b_tracking_shuffled_objects_five_objects.json" --config.progressive_goals=True --config.stop_on_success=False --config.allow_non_ascii=False --config.num_train_models=1 --config.n_train_data=50 --config.n_test_data=50 --config.n_steps=100 --config.test_steps=500 --config.anneal=True --config.batch_size=64 --config.topk=11 --config.topq=6 --config.control_init=" proper logical reasoning and think step by step. Finally give the actual correct answer." --config.extractor_text="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $" --config.control_weight=0.20 --config.target_weight=1.0```
