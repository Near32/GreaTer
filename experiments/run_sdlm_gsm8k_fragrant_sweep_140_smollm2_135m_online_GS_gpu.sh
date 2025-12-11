#!/bin/bash

# Create directories
mkdir -p sdlm_smollm2_135m_logs
mkdir -p results

# Set paths - using a small subset of data for quick testing
TRAIN_DATA="../data/grade_school_math/train.jsonl"
TEST_DATA="../data/grade_school_math/test.jsonl"
LOG_FILE="sdlm_smollm2_135m_logs/gsm8k_optimization.log"
RESULT_PREFIX="results_test/online_sdlm_smollm2_135m_gpu_gsm8k"

# Print configuration
echo "========================================"
echo "SDLM Optimization with SmolLM2-135M-Instruct (GPU)"
echo "========================================"
echo "Model: HuggingFaceTB/SmolLM2-135M-Instruct"
echo "Device: GPU"
echo "Train data: $TRAIN_DATA"
echo "Test data: $TEST_DATA"
echo "Log file: $LOG_FILE"
echo "Results prefix: $RESULT_PREFIX"
echo "========================================"

# Set extractor text for GSM8K format
#extractor_text="Therefore, the final answer (use exactly this format: \$NUMBER\$, where NUMBER is a positive or negative integer) is $"
#extractor_text="Therefore, the final answer (with format: $ANSWER$) is $"
extractor_text="Therefore, the final answer (with format: \$NUMBER\$, where NUMBER is a positive or negative integer) is $"
control_init="Let's solve this math problem step by step. First, I will understand the problem, then break it down into smaller, manageable parts, and finally arrive at the correct answer."


# Set environment variables for CPU optimization
export OMP_NUM_THREADS=$(nproc)  # Use all available CPU cores
export TOKENIZERS_PARALLELISM=false

# Run the optimization
echo "Starting optimization on GPU with SmolLM2-135M-Instruct..."
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:4096 \
HIPBLAS_PREFER_FP16=1 \
WANDB_CACHE_DIR=./wandb_cache/ \
WANDB_DIR=./wandb_dir/ \
WANDB_DATA_DIR=./wandb_data_dir/ \
python -m ipdb -c c main.py \
    --config="./configs/transfer_sdlm_smollm2_135m_gpu.py" \
    --config.use_wandb=True \
    --config.project="GreaTer-SDLM" \
    --config.train_data="$TRAIN_DATA" \
    --config.test_data="$TEST_DATA" \
    --config.result_prefix="$RESULT_PREFIX" \
    --config.stop_on_success=True \
    --config.allow_non_ascii=False \
    --config.num_train_models=1 \
    --config.torch_deterministic=False \
    --config.torch_allow_tf32=True \
    --config.torch_dtype='float16' \
    --config.do_sample=False \
    --config.loss_type='online' \
    --config.use_differentiable_cache=False \
    --config.online_teacher_forcing=True \
    --config.gradient_clip_strategy='None' \
    --config.gradient_health.value_clip=0.0 \
    --config.gradient_health.max_norm=0.0 \
    --config.seed=10 \
    --config.data_seed=10 \
    --config.validate_on="accuracy" \
    --config.n_train_data=100 \
    --config.n_valid_data=100 \
    --config.n_test_data=10000 \
    --config.sdlm_variable_kwargs.learning_rate=0.1 \
    --config.sdlm_variable_kwargs.init_strategy='fluency' \
    --config.sdlm_variable_kwargs.temperature=10.0 \
    --config.sdlm_variable_kwargs.logit_scaler=1.0 \
    --config.sdlm_variable_kwargs.learnable_temperature=True \
    --config.sdlm_variable_kwargs.decouple_learnable_temperature=True \
    --config.sdlm_variable_kwargs.lr_scheduler_type='None' \
    --config.sdlm_variable_kwargs.lr_scheduler_total_steps=24 \
    --config.sdlm_variable_kwargs.lr_scheduler_final_lr=0.01 \
    --config.sdlm_model_stgs_logits_generation=True \
    --config.sdlm_model_kwargs.hard=False \
    --config.sdlm_model_kwargs.learnable_temperature=False \
    --config.sdlm_model_kwargs.temperature=10.0 \
    --config.sdlm_model_kwargs.hidden_state_conditioning=False \
    --config.sdlm_model_kwargs.use_bpttoken=False \
    --config.sdlm_model_kwargs.dropout=0.0 \
    --config.acc_grad_n_examples=1 \
    --config.gradient_comp_batch_size=1 \
    --config.update_solution_max_new_tokens=512 \
    --config.max_new_tokens_answer=8 \
    --config.n_steps=30 \
    --config.test_steps=10 \
    --config.log_first=False \
    --config.anneal=True \
    --config.batch_size=2 \
    --config.temp=0.5 \
    --config.topk=10 \
    --config.topq=5 \
    --config.eval_generate_kwargs.temperature=0.2 \
    --config.eval_generate_kwargs.top_p=0.9 \
    --config.eval_generate_kwargs.do_sample=True \
    --config.control_init="$control_text" \
    --config.extractor_text="$extractor_text" \
    --config.em_from_gen_str=False \
    --config.control_weight=0.0 \
    --config.target_weight=0.1 #> "$LOG_FILE" 2>&1

# Scheduler notes:
#   * Linear decay (used above): specify lr_scheduler_type='linear', lr_scheduler_final_lr, and optionally lr_scheduler_total_steps (defaults to config.n_steps).
#   * Exponential decay: use lr_scheduler_type='exponential' and provide either lr_scheduler_gamma (per-step multiplier) or lr_scheduler_final_lr (gamma inferred).

# Print completion message
echo "========================================"
echo "GPU Optimization with SmolLM2-135M-Instruct completed!"
echo "Results saved to: $RESULT_PREFIX*"
echo "Log file: $LOG_FILE"
echo "========================================"

# Print the last few lines of the log file
echo -e "\n=== Tail of the log file ==="
tail -n 20 "$LOG_FILE"
echo "========================================"
