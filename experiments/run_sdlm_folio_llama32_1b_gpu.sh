#!/bin/bash

# Create directories
mkdir -p sdlm_llama32_1b_logs
mkdir -p results

# Set paths - using a small subset of data for quick testing
TRAIN_DATA="../data/FOLIO/train.csv"
VALID_DATA="../data/FOLIO/dev.csv"
TEST_DATA="../data/FOLIO/test.csv"
LOG_FILE="sdlm_llama32_1b_logs/folio_optimization.log"
RESULT_PREFIX="results/sdlm_llama32_1b_gpu_folio"

# Print configuration
echo "========================================"
echo "SDLM Optimization with Llama-3.2-1B-Instruct (GPU)"
echo "========================================"
echo "Model: meta-llama/Llama-3.2-1B-Instruct"
echo "Device: GPU"
echo "Train data: $TRAIN_DATA"
echo "Valid data: $VALID_DATA"
echo "Test data: $TEST_DATA"
echo "Log file: $LOG_FILE"
echo "Results prefix: $RESULT_PREFIX"
echo "========================================"

# Set extractor text for FOLIO format
extractor_text="Therefore, the final answer (use exactly this format: \$(LETTER)\$, where LETTER is an uppercase letter, like A or B) is \$("
# "Therefore, the final answer (use exactly this format: \$NUMBER\$, where NUMBER is a positive or negative integer) is $"
control_init="Use proper logical reasoning and think step by step. Finally, give the actuacl correct_answer."
#"Let's solve this math problem step by step. First, I will understand the problem, then break it down into smaller, manageable parts, and finally arrive at the correct answer." \

# Set environment variables for CPU optimization
export OMP_NUM_THREADS=$(nproc)  # Use all available CPU cores
export TOKENIZERS_PARALLELISM=false

# Run the optimization
echo "Starting optimization on GPU with Llama-3.2-1B-Instruct..."
python -m ipdb -c c main.py \
    --config="./configs/transfer_sdlm_llama32_1b_gpu.py" \
    --config.use_wandb=True \
    --config.project="GreaTer-SDLM-FOLIO" \
    --config.train_data="$TRAIN_DATA" \
    --config.valid_data="$VALID_DATA" \
    --config.test_data="$TEST_DATA" \
    --config.result_prefix="$RESULT_PREFIX" \
    --config.stop_on_success=True \
    --config.allow_non_ascii=False \
    --config.num_train_models=1 \
    --config.do_sample=False \
    --config.torch_deterministic=True \
    --config.seed=10 \
    --config.n_train_data=50 \
    --config.n_valid_data=100 \
    --config.n_test_data=203 \
    --config.sdlm_variable_kwargs.learning_rate=0.1 \
    --config.sdlm_variable_kwargs.init_strategy='fluency' \
    --config.sdlm_variable_kwargs.temperature=0.5 \
    --config.sdlm_variable_kwargs.learnable_temperature=True \
    --config.sdlm_model_kwargs.learnable_temperature=False \
    --config.sdlm_model_kwargs.temperature=0.7 \
    --config.sdlm_model_kwargs.hidden_state_conditioning=False \
    --config.acc_grad_n_examples=1 \
    --config.gradient_comp_batch_size=1 \
    --config.update_solution_max_new_tokens=256 \
    --config.max_new_tokens_answer=8 \
    --config.n_steps=2 \
    --config.test_steps=1 \
    --config.anneal=True \
    --config.batch_size=8 \
    --config.temp=0.6 \
    --config.topk=10 \
    --config.topq=5 \
    --config.control_init="$control_init" \
    --config.extractor_text="$extractor_text" \
    --config.control_weight=0.3 \
    --config.target_weight=1.0 #> "$LOG_FILE" 2>&1

# Print completion message
echo "========================================"
echo "GPU Optimization with Llama-3.2-1B-Instruct completed!"
echo "Results saved to: $RESULT_PREFIX*"
echo "Log file: $LOG_FILE"
echo "========================================"

# Print the last few lines of the log file
echo -e "\n=== Tail of the log file ==="
tail -n 20 "$LOG_FILE"
echo "========================================"

