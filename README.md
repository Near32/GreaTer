# GreaTer - Gradient Over Reasoning makes Smaller Language Models Strong Prompt Optimizers

Source code and scripts for *GreaTer - Gradient Over Reasoning makes Smaller Language Models Strong Prompt Optimizers*

## Running GreaTer

- Clone the Repository
- Change directory to ```prompt_optimization/experiments/```
- Install requirements ```pip install requirements.txt```. Ignore the errors due to mismatch with fschat version.
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
This implementation is built on top of [llm attacks](https://github.com/llm-attacks/llm-attacks) codebase. We will further update by incorporating KV-cache to ensure the whole forward pipeline can be effectively done in one single call, which will make this faster.*
- To Run gemma2_2b with tracking_shuffled_objects_five_objects task from BBH:

```
python main.py --config="./configs/transfer_gemma2_2b.py" --config.train_data="../data/BBH/tracking_shuffled_objects_five_objects.json" --config.test_data="../data/BBH/tracking_shuffled_objects_five_objects.json" --config.result_prefix="results/transfer_llama32_1b_tracking_shuffled_objects_five_objects.json" --config.progressive_goals=True --config.stop_on_success=False --config.allow_non_ascii=False --config.num_train_models=1 --config.n_train_data=50 --config.n_test_data=50 --config.n_steps=100 --config.test_steps=500 --config.anneal=True --config.topk=11 --config.topq=6 --config.control_init=" proper logical reasoning and think step by step. Finally give the actual correct answer." --config.extractor_text="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $" --config.control_weight=0.20 --config.target_weight=1.0
```
- Depending on the task to optimize, you need to change the ```extractor_text``` so that you can extract the correct answer properly, which is fundamental in calculation of the loss.