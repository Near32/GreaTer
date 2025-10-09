#!/usr/bin/env bash
set -euo pipefail

# Basic demo workflow for next_token_distribution_analysis.py on Llama 3.2 1B Instruct.
# Configure WANDB_PROJECT and WANDB_ENTITY before running, and ensure the Hugging Face
# credentials for the model are available in the environment if required.

PROJECT_NAME=${WANDB_PROJECT:-NextTokenDistributionAnalysis}
ENTITY_NAME=${WANDB_ENTITY:-near32}
RUN_NAME=${RUN_NAME:-llama32_1b_instruct_demo}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}

${PYTHON_BIN} -m ipdb -c c "${SCRIPT_DIR}/next_token_distribution_analysis.py" \
  --wandb-project "${PROJECT_NAME}" \
  --wandb-entity "${ENTITY_NAME}" \
  --run-name "${RUN_NAME}" \
  --model-name "meta-llama/Llama-3.2-1B-Instruct" \
  --skip-random-init \
  --wandb-action both \
  --dtypes bfloat16 \
  --num-anchor-sets 32 \
  --anchor-cluster-count 32 \
  --anchor-cluster-blend-steps 5 \
  --candidate-source token\
  --num-sentences 8 \
  --seq-length 1 \
  --num-generations 1 \
  --k-values 8 \
  --ratios 0.0 0.25 0.5 0.75 1.0 \
  --distribution-types k_hot \
  --temperature 0.1 \
  --top-k-sampling None \
  --top-p-sampling None \
  --random-seed 1234 \
  --similarity-pool-size 256 \
  --heatmap-max-points 64 \
  --allow-pad-fallback #\
  #--wandb-source-run "${ENTITY_NAME}/${PROJECT_NAME}/uo1sv5nm" \

echo "Demo run completed. Check W&B for logged metrics and plots." >&2
