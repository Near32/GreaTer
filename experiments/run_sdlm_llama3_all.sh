#!/bin/bash

# Create the logs directory if it doesn't exist
mkdir -p sdlm_llama3_logs

# Number of GPUs needed per task
NUM_GPUS_PER_TASK=2

# Total number of GPUs
TOTAL_GPUS=8

# Array to keep track of active jobs and their associated GPUs
declare -A active_jobs

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

# Function to start a task with two GPUs
start_task() {
    local task_name=$1
    local gpu1=$2
    local gpu2=$3
    local gpu_string="$gpu1,$gpu2"
    echo "Starting SDLM optimization for task: $task_name on GPUs: $gpu_string"
    
    CUDA_VISIBLE_DEVICES=$gpu_string python main.py \
        --config="./configs/transfer_sdlm.py" \
        --config.train_data="../data/BBH/${task_name}.json" \
        --config.test_data="../data/BBH/${task_name}.json" \
        --config.result_prefix="results/sdlm_llama3_${task_name}.json" \
        --config.progressive_goals=True \
        --config.stop_on_success=False \
        --config.allow_non_ascii=False \
        --config.num_train_models=1 \
        --config.n_train_data=50 \
        --config.n_test_data=50 \
        --config.n_steps=100 \
        --config.test_steps=10 \
        --config.anneal=True \
        --config.batch_size=16 \
        --config.topk=50 \
        --config.topq=0.9 \
        --config.control_init="Think step by step and provide a clear, logical explanation. Finally, give the correct answer." \
        --config.extractor_text="${extractor_texts[$task_name]}" \
        --config.control_weight=0.20 \
        --config.target_weight=1.0 \
        > "sdlm_llama3_logs/${task_name}.log" 2>&1 &

    # Store the PID and associated GPUs
    active_jobs[$!]="$gpu1 $gpu2"
}

# Function to check and clean up completed jobs
cleanup_completed_jobs() {
    for pid in "${!active_jobs[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            # Process has completed, free up GPUs
            read -r gpu1 gpu2 <<< "${active_jobs[$pid]}"
            echo "Task with PID $pid completed, freeing GPUs: $gpu1,$gpu2"
            unset "active_jobs[$pid]"
        fi
    done
}

# Function to get available GPUs
get_available_gpus() {
    local used_gpus=()
    local available_gpus=()
    
    # Get all GPUs that are currently in use
    for gpu_used in "${active_jobs[@]}"; do
        used_gpus+=($gpu_used)
    done
    
    # Find available GPUs
    for ((i=0; i<TOTAL_GPUS; i++)); do
        if ! [[ " ${used_gpus[@]} " =~ " $i " ]]; then
            available_gpus+=($i)
        fi
    done
    
    echo "${available_gpus[@]}"
}

# Main loop to schedule tasks
for task_name in "${selected_tasks[@]}"; do
    while true; do
        # Clean up completed jobs
        cleanup_completed_jobs
        
        # Get available GPUs
        available_gpus=($(get_available_gpus))
        
        # Check if we have enough GPUs to start a new task
        if [ ${#available_gpus[@]} -ge $NUM_GPUS_PER_TASK ]; then
            # Take the first two available GPUs
            gpu1=${available_gpus[0]}
            gpu2=${available_gpus[1]}
            start_task "$task_name" "$gpu1" "$gpu2"
            break
        else
            # Wait a bit before checking again
            echo "Waiting for GPUs to become available..."
            sleep 60
        fi
    done
done

# Wait for all jobs to complete
echo "All tasks started. Waiting for completion..."
wait

echo "All SDLM optimization tasks completed!"
