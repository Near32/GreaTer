#!/bin/bash

# Create the logs directory if it doesn't exist
mkdir -p gemma2-9b-logs

# Number of GPUs needed per task
NUM_GPUS_PER_TASK=2

# Total number of GPUs
TOTAL_GPUS=6

# Array to keep track of active jobs and their associated GPUs
declare -A active_jobs

# Get the extractor texts as before
declare -A extractor_texts=(
    ["multistep_arithmetic_two"]="Therefore, the final answer (use exactly this format: **NUMBER**, where NUMBER is a positive or negative integer) is **"
    ["tracking_shuffled_objects_five_objects"]="Therefore, the final answer (use exact format: '$ A' or '$ B' or '$ C' or '$ D' or '$ E') is $"
    ["object_counting"]="Therefore, the final answer (use exactly this format: **NUMBER**, where NUMBER is a positive integer) is **"
    ["date_understanding"]="Therefore, the final answer (use exact format: '$ A' or '$ B' or '$ C' or '$ D' or '$ E') is $"
    ["disambiguation_qa"]="Therefore, the final answer (use exact format: '$ A' or '$ B' or '$ C') is $"
    ["formal_fallacies"]="Therefore, the final answer (use exact format: '$ valid' or '$ invalid') is $"
#    ["geometric_shapes"]="Therefore, the final answer (use exact format: '$ A' or '$ B' or '$ C' or '$ D' or '$ E' or '$ F' or '$ G' or '$ H' or '$ I' or '$ J') is $"
    ["salient_translation_error_detection"]="Therefore, the final answer (use exact format: '$ A' or '$ B' or '$ C' or '$ D' or '$ E' or '$ F') is $"
    ["penguins_in_a_table"]="Therefore, the final answer (use exact format: '$ A' or '$ B' or '$ C' or '$ D' or '$ E') is $"
    ["causal_judgement"]="Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $"
    ["logical_deduction_five_objects"]="Therefore, the final answer (use exact format: '$ A' or '$ B' or '$ C' or '$ D' or '$ E') is $"
    ["movie_recommendation"]="Therefore, the final answer (use exact format: '$ A' or '$ B' or '$ C' or '$ D') is $"
    ["navigate"]="Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $"
    ["web_of_lies"]="Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $"
    ["sports_understanding"]="Therefore, the final answer (use exact format: '$ yes' or '$ no') is $"
    ["reasoning_about_colored_objects"]="Therefore, the final answer (use exact format: '$ A' or '$ B' or '$ C' or '$ D' or '$ E' or '$ F' or '$ G' or '$ H' or '$ I' or '$ J' or '$ K' or '$ L' or '$ M' or '$ N' or '$ O' or '$ P' or '$ Q' or '$ R') is $"

    ["hyperbaton"]="Therefore, the final answer (use exact format: '$ A' or '$ B') is $"
    ["ruin_names"]="Therefore, the final answer (use exact format: '$ A' or '$ B' or '$ C' or '$ D') is $"

    ["snarks"]="Therefore, the final answer (use exact format: '$ A' or '$ B') is $"
    ["temporal_sequences"]="Therefore, the final answer (use exact format: '$ A' or '$ B' or '$ C' or '$ D') is $"
)

#selected_tasks=("salient_translation_error_detection" "formal_fallacies" "date_understanding" "snarks" "reasoning_about_colored_objects" "navigate")  # Add the tasks you want to run
selected_tasks=("object_counting" "tracking_shuffled_objects_five_objects" "hyperbaton" "causal_judgement" "movie_recommendation")  # Add the tasks you want to run


# Function to start a task with two GPUs
start_task() {
    local task_name=$1
    local gpu1=$2
    local gpu2=$3
    local gpu_string="$gpu1,$gpu2"
    echo "Starting task: $task_name on GPUs: $gpu_string"
    CUDA_VISIBLE_DEVICES=$gpu_string python main.py \
        --config="./configs/transfer_gemma2_2b.py" \
        --config.train_data="../data/BBH/${task_name}.json" \
        --config.test_data="../data/BBH/${task_name}.json" \
        --config.result_prefix="results/transfer_gemma2_2b_${task_name}_ffr.json" \
        --config.progressive_goals=True \
        --config.stop_on_success=False \
        --config.allow_non_ascii=False \
        --config.num_train_models=1 \
        --config.n_train_data=50 \
        --config.n_test_data=50 \
        --config.n_steps=125 \
        --config.test_steps=500 \
        --config.anneal=True \
        --config.batch_size=64 \
        --config.topk=40 \
        --config.topq=6 \
        --config.control_init="proper logical reasoning and think step by step. Finally give the actual correct answer." \
        --config.extractor_text="${extractor_texts[$task_name]}" \
        --config.control_weight=0.20 \
        --config.target_weight=1.0 \
        > "gemma2-2b-logs/${task_name}_rr.log" 2>&1 &

    # Store the PID and associated GPUs
    active_jobs[$!]="$gpu1 $gpu2"
}

# Main loop to schedule tasks
for task_name in "${selected_tasks[@]}"; do
    while true; do
        # Check how many GPUs are available
        available_gpus=()
        for i in $(seq 1 $((TOTAL_GPUS))); do
            gpu_in_use=false
            for pid in "${!active_jobs[@]}"; do
                if [[ ${active_jobs[$pid]} =~ (^|[[:space:]])$i($|[[:space:]]) ]]; then
                    gpu_in_use=true
                    break
                fi
            done
            if [ "$gpu_in_use" = false ]; then
                available_gpus+=($i)
            fi
        done

        # If exactly two GPUs are available, start the task
        if [ ${#available_gpus[@]} -ge $NUM_GPUS_PER_TASK ]; then
            start_task $task_name ${available_gpus[0]} ${available_gpus[1]}
            break
        fi

        # Check for any finished jobs
        for pid in "${!active_jobs[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                echo "Task with PID $pid completed. GPUs ${active_jobs[$pid]} are now available."
                unset active_jobs[$pid]
            fi
        done

        # Sleep for a short time before checking again
        sleep 5
    done
done

# Final wait to ensure all tasks are completed before exiting the script
wait
echo "All tasks completed."