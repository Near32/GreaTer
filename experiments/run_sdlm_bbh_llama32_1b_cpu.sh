#!/bin/bash

# Create directories
mkdir -p sdlm_llama32_1b_logs
mkdir -p results

LOG_FILE="sdlm_llama32_1b_logs/bbh_optimization.log"

# Get the extractor texts
declare -A extractor_texts=(
    ["multistep_arithmetic_two"]="Therefore, the final answer (use exactly this format: \$NUMBER\$, where NUMBER is a positive or negative integer) is $"
    ["tracking_shuffled_objects_five_objects"]="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $"
    ["object_counting"]="Therefore, the final answer (use exactly this format: \$NUMBER\$, where NUMBER is a positive integer) is $"
    ["date_understanding"]="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $"
    ["disambiguation_qa"]="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C') is $"
    ["formal_fallacies"]="Therefore, the final answer (use exact format: '$ valid' or '$ invalid') is $ "
    ["geometric_shapes"]="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E' or '\$F' or '\$G' or '\$H' or '\$I' or '\$J') is $"
    ["salient_translation_error_detection"]="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E' or '\$F') is $"
    ["penguins_in_a_table"]="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $"
    ["causal_judgement"]="Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $ "
    ["logical_deduction_five_objects"]="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $"
    ["movie_recommendation"]="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D') is $"
    ["navigate"]="Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $ "
    ["web_of_lies"]="Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $ "
    ["sports_understanding"]="Therefore, the final answer (use exact format: '$ yes' or '$ no') is $ "
    ["reasoning_about_colored_objects"]="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E' or '\$F' or '\$G' or '\$H' or '\$I' or '\$J' or '\$K' or '\$L' or '\$M' or '\$N' or '\$O' or '\$P' or '\$Q' or '\$R') is $"
    ["hyperbaton"]="Therefore, the final answer (use exact format: '\$A' or '\$B') is $"
    ["ruin_names"]="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D') is $"
    ["snarks"]="Therefore, the final answer (use exact format: '\$A' or '\$B') is $"
    ["temporal_sequences"]="Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D') is $"
    ["boolean_expressions"]="Therefore, the final answer (use exact format: '$ True' or '$ False') is $ "
)

# Select tasks to run (you can modify this list as needed)
selected_tasks=(
    "tracking_shuffled_objects_five_objects"
    "date_understanding"
    "disambiguation_qa"
    "formal_fallacies"
    "causal_judgement"
    "logical_deduction_five_objects"
    "navigate"
    "sports_understanding"
)

task_name=${selected_tasks[0]}

# Set paths - using a small subset of data for quick testing
TRAIN_DATA="../data/BBH/${task_name}.json"
TEST_DATA="../data/BBH/${task_name}.json"
RESULT_PREFIX="results/sdlm_llama32_1b_cpu_bbh_${task_name}"

# Print configuration
echo "========================================"
echo "SDLM Optimization with Llama-3.2-1B-Instruct (CPU)"
echo "========================================"
echo "Model: meta-llama/Llama-3.2-1B-Instruct"
echo "Device: CPU"
echo "Train data: $TRAIN_DATA"
echo "Test data: $TEST_DATA"
echo "Log file: $LOG_FILE"
echo "Results prefix: $RESULT_PREFIX"
echo "========================================"

# Set environment variables for CPU optimization
export OMP_NUM_THREADS=$(nproc)  # Use all available CPU cores
export TOKENIZERS_PARALLELISM=false

control_init="Let's solve this math problem step by step. First, I will understand the problem, then break it down into smaller, manageable parts, and finally arrive at the correct answer." \
extractor_text=${extractor_texts[$task_name]}

# Run the optimization
echo "Starting optimization on CPU with Llama-3.2-1B-Instruct..."
python -m ipdb -c c main.py \
    --config="./configs/transfer_sdlm_llama32_1b_cpu.py" \
    --config.use_wandb=True \
    --config.project="GreaTer-SDLM" \
    --config.train_data="$TRAIN_DATA" \
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
    --config.n_test_data=100 \
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
    --config.em_from_gen_str=False \
    --config.extractor_text="$extractor_text" \
    --config.control_weight=0.3 \
    --config.target_weight=1.0 #> "$LOG_FILE" 2>&1

# Print completion message
echo "========================================"
echo "CPU Optimization with Llama-3.2-1B-Instruct completed!"
echo "Results saved to: $RESULT_PREFIX*"
echo "Log file: $LOG_FILE"
echo "========================================"

# Print the last few lines of the log file
echo -e "\n=== Tail of the log file ==="
tail -n 20 "$LOG_FILE"
echo "========================================"

