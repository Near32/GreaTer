#!/bin/bash

# Create directories
mkdir -p sdlm_smollm2_135m_logs
mkdir -p results

# Set paths - using a small subset of data for quick testing
#TRAIN_DATA="../data/grade_school_math/train.csv"
#TRAIN_DATA="../data/grade_school_math/processed/train.csv"
#TRAIN_DATA="../data/grade_school_math/train.tsv"
TRAIN_DATA="../data/grade_school_math/train.jsonl"
#TEST_DATA="../data/grade_school_math/test.csv"
#TEST_DATA="../data/grade_school_math/processed/test.csv"
#TEST_DATA="../data/grade_school_math/test.tsv"
TEST_DATA="../data/grade_school_math/test.jsonl"
LOG_FILE="sdlm_smollm2_135m_logs/gsm8k_optimization.log"
RESULT_PREFIX="results/sdlm_smollm2_135m_cpu_gsm8k"

# Print configuration
echo "========================================"
echo "SDLM Optimization with smollm2_135m (CPU)"
echo "========================================"
echo "Model: smollm2_135m"
echo "Device: CPU"
echo "Train data: $TRAIN_DATA"
echo "Test data: $TEST_DATA"
echo "Log file: $LOG_FILE"
echo "Results prefix: $RESULT_PREFIX"
echo "========================================"

# Set extractor text for GSM8K format
extractor_text="Therefore, the final answer (use exactly this format: \$NUMBER\$, where NUMBER is a positive or negative integer) is $"

# Set environment variables for CPU optimization
export OMP_NUM_THREADS=$(nproc)  # Use all available CPU cores
export TOKENIZERS_PARALLELISM=false

# Run the optimization
echo "Starting optimization on CPU with SmolLM2-135m_instruct..."
python -m ipdb -c c main.py \
    --config="./configs/transfer_sdlm_smollm2_135m_instruct_cpu.py" \
    --config.use_wandb=True \
    --config.project="GreaTer-SDLM" \
    --config.train_data="$TRAIN_DATA" \
    --config.test_data="$TEST_DATA" \
    --config.result_prefix="$RESULT_PREFIX" \
    --config.stop_on_success=True \
    --config.allow_non_ascii=False \
    --config.num_train_models=1 \
    --config.n_train_data=3 \
    --config.n_test_data=3 \
    --config.sdlm_variable_kwargs.learning_rate=0.01 \
    --config.sdlm_variable_kwargs.init_strategy='fluency' \
    --config.sdlm_variable_kwargs.temperature=1.0 \
    --config.sdlm_variable_kwargs.learnable_temperature=False \
    --config.acc_grad_n_examples=-1 \
    --config.update_solution_max_new_tokens=16 \
    --config.n_steps=50 \
    --config.test_steps=1 \
    --config.anneal=True \
    --config.batch_size=10 \
    --config.temp=0.1 \
    --config.topk=10 \
    --config.topq=5 \
    --config.control_init="Let's solve this math problem step by step. First, I will understand the problem, then break it down into smaller, manageable parts, and finally arrive at the correct answer." \
    --config.extractor_text="$extractor_text" \
    --config.control_weight=0.4 \
    --config.target_weight=1.0 #\
    #> "$LOG_FILE" 2>&1

# Print completion message
echo "========================================"
echo "CPU Optimization with SmolLM2-135m_instruct completed!"
echo "Results saved to: $RESULT_PREFIX*"
echo "Log file: $LOG_FILE"
echo "========================================"

# Print the last few lines of the log file
echo -e "\n=== Tail of the log file ==="
tail -n 20 "$LOG_FILE"
echo "========================================"
