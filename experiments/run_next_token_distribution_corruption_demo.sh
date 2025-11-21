#!/usr/bin/env bash
set -euo pipefail

# Demo workflow for next_token_distribution_corruption_study.py
# Creates a compact sweep over corruption regimes (head / tail / blended) on a causal
# language model (defaults to Llama 3.2 1B Instruct) using a thin slice of the
# Wikitext-2 dataset. The experiment now reports cross entropy alongside entropy
# changes, plotting mean ± std bands for quick inspection. Adjust environment
# variables below to customise execution without editing this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}

MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-1B-Instruct"}
TOKENIZER_NAME=${TOKENIZER_NAME:-$MODEL_NAME}
DATASET_NAME=${DATASET_NAME:-wikitext}
DATASET_CONFIG=${DATASET_CONFIG:-wikitext-2-raw-v1}
DATASET_SPLIT=${DATASET_SPLIT:-train[:512]}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/artifacts/next_token_corruption/demo}
BLEND_RATIOS=${BLEND_RATIOS:-"0.01 0.1"}
CORRUPTION_STRENGTHS=${CORRUPTION_STRENGTHS:-"0.1 0.9"}
# Configure WandB if desired; use environment variables to override.
PROJECT_NAME=${WANDB_PROJECT:-NextTokenDistributionCorruption}
ENTITY_NAME=${WANDB_ENTITY:-near32}
RUN_NAME=${RUN_NAME:-corruption_demo}
TRANSFER_DELTA_MIN=${TRANSFER_DELTA_MIN:-0.05}
TRANSFER_DELTA_MAX=${TRANSFER_DELTA_MAX:-0.95}

mkdir -p "${OUTPUT_DIR}"

PYTORCH_DISABLE_FLASH_ATTENTION=1 ${PYTHON_BIN} -m ipdb -c c "${SCRIPT_DIR}/next_token_distribution_corruption_study.py" \
  --model-name "${MODEL_NAME}" \
  --tokenizer-name "${TOKENIZER_NAME}" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-config "${DATASET_CONFIG}" \
  --dataset-split "${DATASET_SPLIT}" \
  --corruption-modes transfer \
  --error-estimator quantile \
  --wandb-project "${PROJECT_NAME}" \
  --wandb-entity "${ENTITY_NAME}" \
  --wandb-run-name "${RUN_NAME}" \
  --wandb-mode online \
  --wandb-action both \
  --max-samples 128 \
  --prompt-length 64 \
  --batch-size 64 \
  --corruption-step 1 \
  --corruption-steps 50 \
  --corruption-targets high high_to_any low low_to_any random\
  --blend-ratios ${BLEND_RATIOS} \
  --corruption-strengths ${CORRUPTION_STRENGTHS} \
  --transfer-delta-min ${TRANSFER_DELTA_MIN} \
  --transfer-delta-max ${TRANSFER_DELTA_MAX} \
  --log-level INFO \
  --plot-log-log \
  --plot-log 
  #--output-dir "${OUTPUT_DIR}" \
  #--local-reload

echo "Corruption study demo completed. Outputs in ${OUTPUT_DIR}. Check W&B project ${PROJECT_NAME} for logs." >&2
