#!/usr/bin/env bash
set -euo pipefail

# Demo workflow for next_token_distribution_analysis_freq.py using dictionary span sampling.
# Configure WANDB_PROJECT and WANDB_ENTITY before running, and ensure the Hugging Face
# credentials for the model (and BERT encoder) are available in the environment if required.

PROJECT_NAME=${WANDB_PROJECT:-NextTokenDistributionAnalysis}
ENTITY_NAME=${WANDB_ENTITY:-near32}
RUN_NAME=${RUN_NAME:-llama32_1b_dictionary_freq_demo}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
ANCHOR_FREQUENCY_TEXT=${ANCHOR_FREQUENCY_TEXT:-}
# Set ANCHOR_FREQUENCY_TEXT to a corpus file if you want anchors sampled by observed frequency.

ANCHOR_FREQUENCY_ARGS=()
if [[ -n "${ANCHOR_FREQUENCY_TEXT}" ]]; then
  ANCHOR_FREQUENCY_ARGS+=(--anchor-frequency-text "${ANCHOR_FREQUENCY_TEXT}")
fi

${PYTHON_BIN} -m ipdb -c c "${SCRIPT_DIR}/next_token_distribution_analysis_freq.py" \
  --wandb-project "${PROJECT_NAME}" \
  --wandb-entity "${ENTITY_NAME}" \
  --run-name "${RUN_NAME}" \
  --model-name "meta-llama/Llama-3.2-1B-Instruct" \
  --wandb-action both \
  --dtypes bfloat16 \
  --skip-random-init \
  --num-sentences 32 \
  --seq-length 1 \
  --num-generations 1 \
  --num-anchor-sets 2 \
  --anchor-cluster-count 128 256 \
  --anchor-cluster-blend-steps 4 \
  --anchor-cluster-seed 1234 \
  --candidate-source dictionary \
  --embedding-mode bert \
  --dictionary-url "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt" \
  --max-dictionary-entries 0 \
  "${ANCHOR_FREQUENCY_ARGS[@]}" \
  --span-encoder-batch-size 64 \
  --span-encoder-max-length 256 \
  --bert-model-name "sentence-transformers/all-MiniLM-L6-v2" \
  --bert-layer-index -1 \
  --k-values 4 \
  --ratios 0.0 0.5 1.0 \
  --distribution-types k_hot \
  --temperature 0.1 \
  --top-k-sampling None \
  --top-p-sampling None \
  --support-top-p 1.0 \
  --support-similarity-threshold 0.5 0.9 \
  --facet-row-field support_similarity_threshold \
  --facet-col-field anchor_cluster_count \
  --facet-hue-field dtype \
  --random-seed 1234 \
  --similarity-pool-size 128000 \
  --heatmap-max-points 64 \
  --plot-symmetric-kl-violin \
  --allow-pad-fallback #\
  #--wandb-source-run "${ENTITY_NAME}/${PROJECT_NAME}/xwoi4lm2" \
  #vuehqslg" \
  #ix2zjc99" \

echo "Dictionary demo run completed. Check W&B for logged metrics and plots." >&2
