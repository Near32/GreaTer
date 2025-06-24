#!/bin/bash

# Create the logs directory if it doesn't exist
mkdir -p sdlm_gsm8k_logs

# Set the task name and extractor text for GSM8K
task_name="gsm8k"
extractor_text="Therefore, the final answer (use exactly this format: \$NUMBER\$, where NUMBER is a positive or negative integer) is $"

# Set the number of GPUs to use (adjust based on your system)
NUM_GPUS=2
GPUS="0,1"  # Using first two GPUs, modify as needed

# Set paths and parameters
TRAIN_DATA="../data/GSM8K/train.jsonl"
TEST_DATA="../data/GSM8K/test.jsonl"
LOG_FILE="sdlm_gsm8k_logs/gsm8k_optimization.log"
RESULT_PREFIX="results/sdlm_llama3_gsm8k"

# Print configuration
echo "Starting SDLM optimization for GSM8K"
echo "Using GPUs: $GPUS"
echo "Train data: $TRAIN_DATA"
echo "Test data: $TEST_DATA"
echo "Log file: $LOG_FILE"

# Run the optimization
CUDA_VISIBLE_DEVICES=$GPUS python main.py \
    --config="./configs/transfer_sdlm.py" \
    --config.train_data="$TRAIN_DATA" \
    --config.test_data="$TEST_DATA" \
    --config.result_prefix="$RESULT_PREFIX" \
    --config.progressive_goals=True \
    --config.stop_on_success=False \
    --config.allow_non_ascii=False \
    --config.num_train_models=1 \
    --config.n_train_data=100 \
    --config.n_test_data=100 \
    --config.n_steps=200 \
    --config.test_steps=20 \
    --config.anneal=True \
    --config.batch_size=16 \
    --config.topk=50 \
    --config.topq=0.9 \
    --config.control_init="Let's solve this math problem step by step. First, I will understand the problem, then break it down into smaller, manageable parts, and finally arrive at the correct answer." \
    --config.extractor_text="$extractor_text" \
    --config.control_weight=0.25 \
    --config.target_weight=1.0 \
    > "$LOG_FILE" 2>&1

echo "SDLM optimization for GSM8K completed!"
echo "Results saved to: $RESULT_PREFIX*"
