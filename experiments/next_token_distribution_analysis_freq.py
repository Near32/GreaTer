"""Experiment script to probe next-token distribution stability across input sampling strategies.

This script addresses the following research question:

    When passing as input N sentences of M tokens (initially M=1), sampled from
    a configurable token distribution, do the next-token output distributions of
    a language model remain the same across those inputs?

Key features
------------
* Supports both pre-trained and randomly initialised language models.
* Allows sweeping dtypes (bfloat16 through float64) and optional quantised modes.
* Samples inputs via one-hot / k-hot distributions with random, similarity-based,
  or blended token selection strategies.
* Measures pairwise cross-entropy (and derived KL divergence) between the N·P
  next-token distributions, comparing magnitudes against dtype machine epsilon.
* Logs detailed metrics, curves, and heatmaps to Weights & Biases for analysis.

The implementation is intentionally modular so different samplers, model families,
and metric aggregations can be plugged in with minimal changes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set
from urllib import request as urllib_request
from urllib.error import URLError

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError("matplotlib is required for plotting heatmaps") from exc

from matplotlib.legend import Legend

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None  # type: ignore

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb must be installed to run this experiment") from exc

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans

try:  # BitsAndBytes is optional – only needed when quantisation is requested.
    from transformers import BitsAndBytesConfig

    _BITSANDBYTES_AVAILABLE = True
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore
    _BITSANDBYTES_AVAILABLE = False

from tqdm.auto import tqdm


DTYPE_ALIASES: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
}

def _parse_optional_int(value: str) -> Optional[int]:
    """Parse string to int, handling 'None' as None."""
    if value.lower() == 'none':
        return None
    return int(value)


def _parse_optional_float(value: str) -> Optional[float]:
    """Parse string to float, handling 'None' as None."""
    if value.lower() == 'none':
        return None
    return float(value)
    
SUPPORTED_DISTRIBUTIONS = {"one_hot", "k_hot"}
SUPPORTED_QUANTISATION = {"none", "8bit", "4bit"}

SCALAR_METRIC_FIELDS = [
    "mean_cross_entropy",
    "mean_kl",
    "mean_symmetric_kl",
    "mean_wasserstein",
    "max_symmetric_kl",
    "max_wasserstein",
    "fraction_symmetric_kl_below_epsilon",
    "mean_support_similarity",
    "mean_entropy",
]


@dataclass
class ExperimentSettings:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str]
    wandb_project: str
    wandb_entity: Optional[str]
    run_name: Optional[str]
    wandb_mode: str
    wandb_action: str
    wandb_source_run: Optional[str]
    trust_remote_code: bool
    dtypes: List[str]
    quantisation: str
    include_pretrained: bool
    include_random_init: bool
    device: str
    num_sentences: int
    seq_length: int
    num_generations: int
    k_values: List[int]
    ratios: List[float]
    distribution_types: List[str]
    temperature: float
    top_k_sampling: Optional[int]
    top_p_sampling: Optional[float]
    generation_seed_offset: int
    random_seed: int
    num_anchor_sets: int
    anchor_seed_stride: int
    candidate_source: str
    dictionary_path: Optional[Path]
    max_dictionary_entries: Optional[int]
    dictionary_url: Optional[str]
    dictionary_cache_dir: Path
    anchor_frequency_text: Optional[Path]
    embedding_mode: str
    bert_model_name: Optional[str]
    bert_layer_index: int
    span_encoder_batch_size: int
    span_encoder_max_length: int
    similarity_pool_size: int
    max_similarity_candidates: int
    support_top_p: float
    support_similarity_threshold: Optional[float]
    support_similarity_threshold_options: List[Optional[float]]
    facet_row_field: Optional[str]
    facet_col_field: Optional[str]
    facet_hue_field: Optional[str]
    heatmap_max_points: Optional[int]
    plot_symmetric_kl_violin: bool
    offline: bool
    pad_token_fallback_to_eos: bool
    plot_output_path: str
    anchor_cluster_count: int
    anchor_cluster_count_options: List[int]
    anchor_cluster_blends: List[float]
    anchor_cluster_seed: int
    anchor_cluster_cache_dir: Path


@dataclass
class DTypeInfo:
    name: str
    torch_dtype: torch.dtype
    epsilon: float


@dataclass
class CandidatePool:
    keys: List[str]
    texts: List[str]
    input_id_lists: List[List[int]]
    embeddings: torch.Tensor
    cluster_assignments: Optional[np.ndarray] = None
    cluster_sizes: Optional[np.ndarray] = None
    cluster_to_indices: Optional[List[List[int]]] = None
    sampling_weights: Optional[np.ndarray] = None


@dataclass
class SequenceBlueprint:
    """Defines reproducible inputs for model evaluation."""

    sample_id: int
    anchor_key: str
    anchor_text: str
    anchor_input_ids: List[int]
    support_keys: List[str]
    support_texts: List[str]
    support_input_ids: List[List[int]]
    support_probabilities: List[float]
    support_similarities: List[float]
    chosen_support_index: int
    generation_seeds: List[int]


def parse_args(argv: Optional[Sequence[str]] = None) -> ExperimentSettings:
    parser = argparse.ArgumentParser(
        description="Analyse next-token distribution stability under varied inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-name",
        dest="model_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3.2-1B-Instruct",
        help="Pretrained model identifier or path.",
    )
    parser.add_argument(
        "--tokenizer-name",
        dest="tokenizer_name_or_path",
        type=str,
        default=None,
        help="Optional tokenizer identifier if different from model.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code.")

    parser.add_argument("--wandb-project", type=str, default="NextTokenDistributionAnalysis", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity.")
    parser.add_argument("--run-name", type=str, default=None, help="Override W&B run name.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices={"online", "offline", "disabled"},
        help="Controls wandb.init mode.",
    )
    parser.add_argument(
        "--wandb-action",
        type=str,
        default="log",
        choices={"log", "reload", "both"},
        help="Choose between running the experiment, reloading past data, or both.",
    )
    parser.add_argument(
        "--wandb-source-run",
        type=str,
        default=None,
        help="Run path (entity/project/run_id) to reload when --wandb-action includes 'reload'.",
    )

    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["bfloat16", "float32", "float64"],
        help="List of torch dtypes to evaluate (by name).",
    )
    parser.add_argument(
        "--quantisation",
        type=str,
        default="none",
        choices=sorted(SUPPORTED_QUANTISATION),
        help="Enable low-bit quantisation when supported.",
    )
    parser.add_argument("--skip-pretrained", action="store_true", help="Exclude pretrained weights.")
    parser.add_argument("--skip-random-init", action="store_true", help="Exclude randomly initialised model.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device string ('cuda', 'cpu', etc.). Use 'auto' to prefer CUDA when available.",
    )

    parser.add_argument("--num-sentences", type=int, default=8, help="Number of distinct first tokens (N).")
    parser.add_argument(
        "--seq-length",
        type=int,
        default=1,
        help="Total input length M. Values >1 trigger autoregressive generation of M-1 tokens.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Number of continuations P per initial token when M>1.",
    )
    parser.add_argument(
        "--num-anchor-sets",
        type=int,
        default=1,
        help="Number of independent anchor batches to sample (O repetitions).",
    )
    parser.add_argument(
        "--anchor-seed-stride",
        type=int,
        default=7919,
        help="Stride added to the random seed between anchor batches to decorrelate samples.",
    )
    parser.add_argument(
        "--anchor-cluster-count",
        type=int,
        nargs="+",
        default=[0],
        help="One or more k-means cluster counts over candidate embeddings (0 disables clustering).",
    )
    parser.add_argument(
        "--anchor-cluster-blends",
        type=float,
        nargs="+",
        default=None,
        help="Blend values between largest (0) and smallest (1) clusters when sampling anchors.",
    )
    parser.add_argument(
        "--anchor-cluster-blend-steps",
        type=int,
        default=3,
        help="If --anchor-cluster-blends not provided, generate this many evenly spaced blend values.",
    )
    parser.add_argument(
        "--anchor-cluster-seed",
        type=int,
        default=None,
        help="Seed used for k-means centroid initialization (defaults to --random-seed).",
    )
    parser.add_argument(
        "--anchor-cluster-cache-dir",
        type=str,
        default="~/.cache/ntd_anchor_clusters",
        help="Directory for cached k-means cluster assignments when using token candidates.",
    )
    parser.add_argument(
        "--candidate-source",
        type=str,
        default="token",
        choices={"token", "dictionary"},
        help="Source of initial candidates: LM tokens or spans from a dictionary file.",
    )
    parser.add_argument(
        "--dictionary-path",
        type=str,
        default=None,
        help="Optional path to a file containing candidate spans (one per line) when using dictionary source.",
    )
    parser.add_argument(
        "--dictionary-url",
        type=str,
        default="https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
        help="URL to download dictionary spans when using dictionary source.",
    )
    parser.add_argument(
        "--dictionary-cache-dir",
        type=str,
        default="~/.cache/ntd_dictionary",
        help="Cache directory for downloaded dictionary files.",
    )
    parser.add_argument(
        "--max-dictionary-entries",
        type=int,
        default=2000,
        help="Maximum number of dictionary spans to load (useful to cap runtime).",
    )
    parser.add_argument(
        "--anchor-frequency-text",
        type=str,
        default=None,
        help=(
            "Optional path to a text file whose token frequencies should weight dictionary anchor sampling. "
            "Only applies when --candidate-source=dictionary."
        ),
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="model",
        choices={"model", "bert"},
        help="Embedding backend for similarity computations.",
    )
    parser.add_argument(
        "--bert-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Transformer model used for span embeddings when --embedding-mode=bert.",
    )
    parser.add_argument(
        "--bert-layer-index",
        type=int,
        default=-1,
        help="Hidden layer index to pool from in the span encoder (default: final layer).",
    )
    parser.add_argument(
        "--span-encoder-batch-size",
        type=int,
        default=64,
        help="Batch size for computing span embeddings with the BERT encoder.",
    )
    parser.add_argument(
        "--span-encoder-max-length",
        type=int,
        default=64,
        help="Maximum token length for span encoder inputs (truncates longer spans).",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 2, 8, 32],
        help="Support sizes for k-hot sampling.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=None,
        help="Explicit blend ratios between random (0) and similarity-based (1) token selection.",
    )
    parser.add_argument(
        "--num-ratio-steps",
        type=int,
        default=5,
        help="If --ratios not provided, generate this many evenly spaced ratios in [0, 1].",
    )
    parser.add_argument(
        "--distribution-types",
        type=str,
        nargs="+",
        default=["one_hot", "k_hot"],
        choices=sorted(SUPPORTED_DISTRIBUTIONS),
        help="Distribution families to evaluate.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature for M>1.")
    parser.add_argument(
        "--top-k-sampling",
        type=_parse_optional_int,
        default=None,
        help="Apply top-k during generation when supplied (M>1 only). Use 'None' to disable.",
    )
    parser.add_argument(
        "--top-p-sampling",
        type=_parse_optional_float,
        default=None,
        help="Apply nucleus sampling during generation when supplied (M>1 only). Use 'None' to disable.",
    )
    parser.add_argument("--generation-seed-offset", type=int, default=13, help="Seed offset for generations.")
    parser.add_argument("--random-seed", type=int, default=1234, help="Global RNG seed.")

    parser.add_argument(
        "--similarity-pool-size",
        type=int,
        default=256,
        help="Candidate pool when choosing similar tokens (per anchor).",
    )
    parser.add_argument(
        "--max-similarity-candidates",
        type=int,
        default=4096,
        help="Limit on total candidates considered for similarity scoring.",
    )
    parser.add_argument(
        "--support-top-p",
        type=float,
        default=1.0,
        help="Restrict similarity-ranked support candidates to this cumulative probability mass.",
    )
    parser.add_argument(
        "--support-similarity-threshold",
        type=_parse_optional_float,
        nargs="+",
        default=[None],
        help="One or more cosine similarity thresholds for support candidates; use 'None' to disable.",
    )
    parser.add_argument(
        "--heatmap-max-points",
        type=int,
        default=256,
        help="Optional cap on heatmap resolution (number of distributions).",
    )
    parser.add_argument(
        "--plot-symmetric-kl-violin",
        action="store_true",
        help="Generate a violin plot (with quartile markers) for each symmetric KL heatmap column.",
    )
    parser.add_argument(
        "--no-heatmap-cap",
        action="store_true",
        help="Disable the --heatmap-max-points cap.",
    )
    parser.add_argument(
        "--allow-pad-fallback",
        action="store_true",
        help="When tokenizer lacks pad token, reuse EOS as padding.",
    )
    parser.add_argument(
        "--plot-output-path",
        type=str,
        default="wandb_ratio_cross_entropy.png",
        help="Where to save ratio vs cross-entropy plot when reloading without active wandb run.",
    )
    parser.add_argument(
        "--facet-row-field",
        type=str,
        default="top_p_sampling",
        help="History field used for facet rows when generating aggregated plots (use 'none' to disable).",
    )
    parser.add_argument(
        "--facet-col-field",
        type=str,
        default="anchor_cluster_count",
        help="History field used for facet columns when generating aggregated plots (use 'none' to disable).",
    )
    parser.add_argument(
        "--facet-hue-field",
        type=str,
        default="none",#"series_label",
        help="History field mapped to line hue in facet plots (use 'none' for a single line per facet).",
    )

    args = parser.parse_args(argv)

    if args.ratios is None:
        ratios = np.linspace(0.0, 1.0, num=max(2, args.num_ratio_steps), endpoint=True)
    else:
        ratios = np.array(args.ratios, dtype=float)

    ratios = np.clip(ratios, 0.0, 1.0)
    ratios = sorted(set(float(r) for r in ratios))

    if args.quantisation != "none" and not _BITSANDBYTES_AVAILABLE:
        raise ValueError("bitsandbytes is required for quantisation modes")

    if args.no_heatmap_cap:
        heatmap_cap = None
    else:
        heatmap_cap = max(1, args.heatmap_max_points)

    include_pretrained = not args.skip_pretrained
    include_random_init = not args.skip_random_init

    if not include_pretrained and not include_random_init:
        raise ValueError("At least one of pretrained or random_init modes must be enabled.")

    dictionary_path = Path(args.dictionary_path).expanduser() if args.dictionary_path else None
    dictionary_url = args.dictionary_url
    dictionary_cache_dir = Path(args.dictionary_cache_dir).expanduser()
    if args.candidate_source == "dictionary" and not (dictionary_path or dictionary_url):
        raise ValueError("Provide --dictionary-path or --dictionary-url when --candidate-source=dictionary")
    anchor_frequency_text = Path(args.anchor_frequency_text).expanduser() if args.anchor_frequency_text else None
    if anchor_frequency_text and args.candidate_source != "dictionary":
        raise ValueError("--anchor-frequency-text is only supported with --candidate-source=dictionary")

    if args.candidate_source == "dictionary" and args.embedding_mode != "bert":
        raise ValueError("Dictionary candidates require --embedding-mode=bert to compute span embeddings")

    offline = args.wandb_mode in {"offline", "disabled"}

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.anchor_cluster_blends is None:
        anchor_blends = np.linspace(
            0.0, 1.0, num=max(2, args.anchor_cluster_blend_steps), endpoint=True
        )
    else:
        anchor_blends = np.array(args.anchor_cluster_blends, dtype=float)
    anchor_blends = np.clip(anchor_blends, 0.0, 1.0)
    anchor_blends = sorted(set(float(b) for b in anchor_blends))

    anchor_cluster_seed = args.anchor_cluster_seed if args.anchor_cluster_seed is not None else args.random_seed
    anchor_cluster_cache_dir = Path(args.anchor_cluster_cache_dir).expanduser()

    raw_anchor_counts = args.anchor_cluster_count or [0]
    anchor_counts_ordered: List[int] = []
    seen_counts: Set[int] = set()
    for count in raw_anchor_counts:
        value = max(0, int(count))
        if value not in seen_counts:
            anchor_counts_ordered.append(value)
            seen_counts.add(value)
    if not anchor_counts_ordered:
        anchor_counts_ordered = [0]
    anchor_cluster_count_primary = anchor_counts_ordered[0]

    support_top_p = max(0.0, min(1.0, args.support_top_p))

    raw_thresholds = args.support_similarity_threshold or [None]
    threshold_ordered: List[Optional[float]] = []
    seen_thresholds: Set[Optional[float]] = set()
    for threshold in raw_thresholds:
        value = None if threshold is None else float(threshold)
        if value not in seen_thresholds:
            threshold_ordered.append(value)
            seen_thresholds.add(value)
    if not threshold_ordered:
        threshold_ordered = [None]
    support_similarity_threshold = threshold_ordered[0]

    def _normalise_facet_field(raw: Optional[str]) -> Optional[str]:
        if raw is None:
            return None
        cleaned = raw.strip()
        if not cleaned or cleaned.lower() == "none":
            return None
        return cleaned

    facet_row_field = _normalise_facet_field(args.facet_row_field)
    facet_col_field = _normalise_facet_field(args.facet_col_field)
    facet_hue_field = _normalise_facet_field(args.facet_hue_field)

    return ExperimentSettings(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
        wandb_mode=args.wandb_mode,
        wandb_action=args.wandb_action,
        wandb_source_run=args.wandb_source_run,
        trust_remote_code=args.trust_remote_code,
        dtypes=[dt.lower() for dt in args.dtypes],
        quantisation=args.quantisation,
        include_pretrained=include_pretrained,
        include_random_init=include_random_init,
        device=device,
        num_sentences=max(1, args.num_sentences),
        seq_length=max(1, args.seq_length),
        num_generations=max(1, args.num_generations),
        k_values=[max(1, k) for k in args.k_values],
        ratios=ratios,
        distribution_types=[dist.lower() for dist in args.distribution_types],
        temperature=max(0.0, args.temperature),
        top_k_sampling=args.top_k_sampling,
        top_p_sampling=args.top_p_sampling,
        generation_seed_offset=args.generation_seed_offset,
        random_seed=args.random_seed,
        num_anchor_sets=max(1, args.num_anchor_sets),
        anchor_seed_stride=max(1, args.anchor_seed_stride),
        candidate_source=args.candidate_source,
        dictionary_path=dictionary_path,
        max_dictionary_entries=(max(1, args.max_dictionary_entries) if args.max_dictionary_entries else None),
        dictionary_url=dictionary_url,
        dictionary_cache_dir=dictionary_cache_dir,
        anchor_frequency_text=anchor_frequency_text,
        embedding_mode=args.embedding_mode,
        bert_model_name=args.bert_model_name,
        bert_layer_index=args.bert_layer_index,
        span_encoder_batch_size=max(1, args.span_encoder_batch_size),
        span_encoder_max_length=max(1, args.span_encoder_max_length),
        similarity_pool_size=max(1, args.similarity_pool_size),
        max_similarity_candidates=max(1, args.max_similarity_candidates),
        support_top_p=support_top_p,
        support_similarity_threshold=support_similarity_threshold,
        support_similarity_threshold_options=threshold_ordered,
        facet_row_field=facet_row_field,
        facet_col_field=facet_col_field,
        facet_hue_field=facet_hue_field,
        heatmap_max_points=heatmap_cap,
        plot_symmetric_kl_violin=args.plot_symmetric_kl_violin,
        offline=offline,
        pad_token_fallback_to_eos=args.allow_pad_fallback,
        plot_output_path=args.plot_output_path,
        anchor_cluster_count=anchor_cluster_count_primary,
        anchor_cluster_count_options=anchor_counts_ordered,
        anchor_cluster_blends=anchor_blends,
        anchor_cluster_seed=anchor_cluster_seed,
        anchor_cluster_cache_dir=anchor_cluster_cache_dir,
    )


def resolve_dtype(dtype_name: str) -> DTypeInfo:
    key = dtype_name.lower()
    if key not in DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype alias: {dtype_name}")
    torch_dtype = DTYPE_ALIASES[key]
    epsilon = float(torch.finfo(torch_dtype).eps)
    return DTypeInfo(name=key, torch_dtype=torch_dtype, epsilon=epsilon)


def init_wandb(settings: ExperimentSettings) -> wandb.sdk.wandb_run.Run:
    mode = settings.wandb_mode
    #os.environ.setdefault("WANDB_SILENT", "true")

    def _format_sequence(values: Sequence) -> str:
        if not values:
            return "none"
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append(f"{value:.2f}")
            else:
                formatted.append(str(value))
        return "-".join(formatted)

    model_slug = settings.model_name_or_path.split("/")[-1]
    hp_summary = (
        f"model={model_slug}__N={settings.num_sentences}__M={settings.seq_length}__P={settings.num_generations}__"
        f"O={settings.num_anchor_sets}__k={_format_sequence(settings.k_values)}__ratio={_format_sequence(settings.ratios)}__"
        f"AC={settings.anchor_cluster_count}__ACBlend={_format_sequence(settings.anchor_cluster_blends)}"
    )
    base_name = settings.run_name.strip() if settings.run_name else ""
    if base_name:
        run_name = f"{base_name}__{hp_summary}"
    else:
        run_name = hp_summary

    run_name = run_name.replace("/", "-")

    run = wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        name=run_name,
        mode=mode,
        config={
            "model_name": settings.model_name_or_path,
            "tokenizer_name": settings.tokenizer_name_or_path,
            "device": settings.device,
            "dtypes": settings.dtypes,
            "quantisation": settings.quantisation,
            "num_sentences": settings.num_sentences,
            "seq_length": settings.seq_length,
            "num_generations": settings.num_generations,
            "num_anchor_sets": settings.num_anchor_sets,
            "k_values": settings.k_values,
            "ratios": settings.ratios,
            "distribution_types": settings.distribution_types,
            "temperature": settings.temperature,
            "top_k_sampling": settings.top_k_sampling,
            "top_p_sampling": settings.top_p_sampling,
            "random_seed": settings.random_seed,
            "similarity_pool_size": settings.similarity_pool_size,
            "max_similarity_candidates": settings.max_similarity_candidates,
            "support_top_p": settings.support_top_p,
            "support_similarity_threshold": settings.support_similarity_threshold,
            "support_similarity_threshold_options": settings.support_similarity_threshold_options,
            "heatmap_max_points": settings.heatmap_max_points,
            "include_pretrained": settings.include_pretrained,
            "include_random_init": settings.include_random_init,
            "candidate_source": settings.candidate_source,
            "embedding_mode": settings.embedding_mode,
            "dictionary_path": str(settings.dictionary_path) if settings.dictionary_path else None,
            "dictionary_url": settings.dictionary_url,
            "anchor_frequency_text": str(settings.anchor_frequency_text)
            if settings.anchor_frequency_text
            else None,
            "bert_model_name": settings.bert_model_name,
            "bert_layer_index": settings.bert_layer_index,
            "span_encoder_batch_size": settings.span_encoder_batch_size,
            "span_encoder_max_length": settings.span_encoder_max_length,
            "anchor_cluster_count": settings.anchor_cluster_count,
            "anchor_cluster_count_options": settings.anchor_cluster_count_options,
            "anchor_cluster_blends": settings.anchor_cluster_blends,
            "anchor_cluster_seed": settings.anchor_cluster_seed,
            "anchor_cluster_cache_dir": str(settings.anchor_cluster_cache_dir),
            "facet_row_field": settings.facet_row_field,
            "facet_col_field": settings.facet_col_field,
            "facet_hue_field": settings.facet_hue_field,
        },
    )
    return run


def load_tokenizer(settings: ExperimentSettings):
    tokenizer_name = settings.tokenizer_name_or_path or settings.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=settings.trust_remote_code)
    if tokenizer.pad_token_id is None and settings.pad_token_fallback_to_eos:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must provide pad_token_id; enable --allow-pad-fallback to reuse EOS.")
    tokenizer.padding_side = "left"
    return tokenizer


def load_pretrained_model(
    settings: ExperimentSettings,
    dtype_info: DTypeInfo,
    device: torch.device,
) -> AutoModelForCausalLM:
    quantisation = settings.quantisation
    kwargs: Dict[str, object] = {
        "trust_remote_code": settings.trust_remote_code,
    }

    if quantisation == "none":
        kwargs.update({
            "torch_dtype": dtype_info.torch_dtype,
        })
        model = AutoModelForCausalLM.from_pretrained(settings.model_name_or_path, **kwargs)
        model.to(device)
    else:
        if quantisation == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantisation == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype_info.torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        else:
            raise ValueError(f"Unsupported quantisation mode: {quantisation}")
        kwargs.update({
            "quantization_config": quant_config,
            "device_map": "auto",
        })
        model = AutoModelForCausalLM.from_pretrained(settings.model_name_or_path, **kwargs)

    model.eval()
    return model


def load_random_initialised_model(
    settings: ExperimentSettings,
    dtype_info: DTypeInfo,
    device: torch.device,
) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(settings.model_name_or_path, trust_remote_code=settings.trust_remote_code)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=settings.trust_remote_code)
    model.to(device)
    model.to(dtype=dtype_info.torch_dtype)
    model.eval()
    return model


def collect_valid_token_ids(tokenizer) -> List[int]:
    vocab_size = tokenizer.vocab_size
    specials = set(tokenizer.all_special_ids or [])
    return [token_id for token_id in range(vocab_size) if token_id not in specials]


def normalised_embeddings(model: AutoModelForCausalLM, valid_ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    embedding_layer = model.get_input_embeddings()
    weight = embedding_layer.weight
    if hasattr(weight, "dequantize"):
        weight_tensor = weight.dequantize()
    else:
        weight_tensor = weight.detach()
    weight_tensor = weight_tensor.float().cpu()

    valid_weight = weight_tensor[valid_ids]
    normalised = F.normalize(valid_weight, p=2, dim=1)
    return valid_weight, normalised


def load_span_encoder(settings: ExperimentSettings, device: torch.device):
    if settings.embedding_mode != "bert":
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(settings.bert_model_name)
    model = AutoModel.from_pretrained(settings.bert_model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def compute_span_embeddings(
    texts: Sequence[str],
    tokenizer,
    model: AutoModel,
    settings: ExperimentSettings,
    device: torch.device,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    for start in range(0, len(texts), settings.span_encoder_batch_size):
        batch_texts = texts[start : start + settings.span_encoder_batch_size]
        encoded = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=settings.span_encoder_max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)
        if settings.bert_layer_index >= 0:
            hidden = outputs.hidden_states[settings.bert_layer_index]
        else:
            hidden = outputs.last_hidden_state
        cls_embeddings = hidden[:, 0, :].detach().cpu()
        embeddings.append(cls_embeddings)

    stacked = torch.cat(embeddings, dim=0)
    return F.normalize(stacked.float(), p=2, dim=1)


def resolve_dictionary_path(settings: ExperimentSettings) -> Path:
    if settings.dictionary_path is not None:
        return settings.dictionary_path
    if not settings.dictionary_url:
        raise ValueError("Dictionary source requires a path or URL")
    cache_dir = settings.dictionary_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    hashed = hashlib.sha256(settings.dictionary_url.encode("utf-8")).hexdigest()[:16]
    extension = Path(settings.dictionary_url).suffix or ".txt"
    cached_file = cache_dir / f"dictionary_{hashed}{extension}"
    if not cached_file.exists():
        try:
            with urllib_request.urlopen(settings.dictionary_url) as response:
                data = response.read()
        except URLError as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to download dictionary from {settings.dictionary_url}: {exc}"
            ) from exc
        cached_file.write_bytes(data)
    return cached_file


def build_dictionary_candidate_pool(
    settings: ExperimentSettings,
    dictionary_file: Path,
    tokenizer: AutoTokenizer,
    span_tokenizer,
    span_model: AutoModel,
    device: torch.device,
) -> CandidatePool:
    texts: List[str] = []
    input_ids: List[List[int]] = []

    with dictionary_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if settings.max_dictionary_entries and len(texts) >= settings.max_dictionary_entries:
                break
            span = line.strip()
            if not span:
                continue
            tokenized = tokenizer.encode(span, add_special_tokens=False)
            if not tokenized:
                continue
            texts.append(span)
            input_ids.append([int(tok) for tok in tokenized])

    if not texts:
        raise ValueError("No valid dictionary entries found after tokenization")

    embeddings = compute_span_embeddings(texts, span_tokenizer, span_model, settings, device)
    keys = [f"span_{idx}" for idx in range(len(texts))]
    return CandidatePool(keys=keys, texts=texts, input_id_lists=input_ids, embeddings=embeddings)


def compute_anchor_frequency_weights_from_text(
    settings: ExperimentSettings,
    tokenizer: AutoTokenizer,
    pool: CandidatePool,
) -> Optional[np.ndarray]:
    text_path = settings.anchor_frequency_text
    if text_path is None:
        return None
    if not text_path.exists():
        raise FileNotFoundError(f"Anchor frequency text file not found: {text_path}")

    try:
        text_content = text_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to read anchor frequency text from {text_path}") from exc

    if not text_content.strip():
        return None

    token_ids = tokenizer.encode(text_content, add_special_tokens=False, truncation=False)
    if not token_ids:
        return None

    length_to_sequences: Dict[int, Dict[Tuple[int, ...], List[int]]] = {}
    for idx, seq in enumerate(pool.input_id_lists):
        if not seq:
            continue
        length = len(seq)
        if length <= 0:
            continue
        seq_tuple = tuple(seq)
        bucket = length_to_sequences.setdefault(length, {})
        bucket.setdefault(seq_tuple, []).append(idx)

    if not length_to_sequences:
        return None

    counts = np.zeros(len(pool.input_id_lists), dtype=np.float64)
    token_count = len(token_ids)
    for length, sequence_map in length_to_sequences.items():
        if length == 0 or token_count < length:
            continue
        if length == 1:
            for token in token_ids:
                indices = sequence_map.get((token,))
                if not indices:
                    continue
                for seq_idx in indices:
                    counts[seq_idx] += 1.0
            continue

        max_start = token_count - length + 1
        for start in range(max_start):
            window = tuple(token_ids[start : start + length])
            indices = sequence_map.get(window)
            if not indices:
                continue
            for seq_idx in indices:
                counts[seq_idx] += 1.0

    total = counts.sum()
    if total <= 0:
        return None
    return counts / total


def maybe_apply_anchor_frequency_weights(
    settings: ExperimentSettings,
    tokenizer: AutoTokenizer,
    pool: CandidatePool,
) -> None:
    weights = compute_anchor_frequency_weights_from_text(settings, tokenizer, pool)
    if weights is not None:
        pool.sampling_weights = weights


def build_token_candidate_pool(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    candidate_ids: Sequence[int],
) -> CandidatePool:
    _, similarity_embeddings = normalised_embeddings(model, candidate_ids)
    texts = []
    for token_id in candidate_ids:
        token_str = tokenizer.convert_ids_to_tokens(int(token_id))
        if isinstance(token_str, list):
            token_str = token_str[0] if token_str else ""
        label = token_str if token_str else str(int(token_id))
        texts.append(f"{int(token_id)}:{label}")
    input_ids = [[int(token_id)] for token_id in candidate_ids]
    keys = [str(int(token_id)) for token_id in candidate_ids]
    return CandidatePool(keys=keys, texts=texts, input_id_lists=input_ids, embeddings=similarity_embeddings)


def slugify_for_path(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value)


def build_cluster_cache_path(
    settings: ExperimentSettings,
    mode_name: str,
    dtype_name: str,
) -> Path:
    model_slug = slugify_for_path(settings.model_name_or_path.split("/")[-1])
    mode_slug = slugify_for_path(mode_name)
    dtype_slug = slugify_for_path(dtype_name)
    settings.anchor_cluster_cache_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{model_slug}_{mode_slug}_{dtype_slug}_clusters{settings.anchor_cluster_count}"
        f"_seed{settings.anchor_cluster_seed}.npz"
    )
    return settings.anchor_cluster_cache_dir / filename


def log_candidate_embedding_table(
    pool: CandidatePool,
    mode_name: str,
    dtype_name: str,
    anchor_cluster_count: int,
    settings: ExperimentSettings,
) -> None:
    columns = [
        "mode",
        "dtype",
        "anchor_cluster_count",
        "candidate_key",
        "candidate_text",
        "cluster_id",
        "cluster_size",
        "embedding",
    ]
    table = wandb.Table(columns=columns)
    assignments = pool.cluster_assignments
    cluster_sizes = pool.cluster_sizes
    for idx, key in enumerate(pool.keys):
        cluster_id = int(assignments[idx]) if assignments is not None else None
        cluster_size = (
            int(cluster_sizes[cluster_id])
            if cluster_sizes is not None and cluster_id is not None and cluster_id < len(cluster_sizes)
            else None
        )
        table.add_data(
            mode_name,
            dtype_name,
            anchor_cluster_count,
            key,
            pool.texts[idx],
            cluster_id,
            cluster_size,
            pool.embeddings[idx].tolist(),
        )

    wandb.log(
        {
            "candidate_embeddings": table,
            "candidate_source": settings.candidate_source,
            "anchor_cluster_count": anchor_cluster_count,
        }
    )

@torch.no_grad()
def initialize_candidate_clusters(
    pool: CandidatePool,
    num_clusters: int,
    random_seed: int,
    cache_path: Optional[Path] = None,
):
    if num_clusters <= 0 or pool.embeddings.shape[0] < max(2, num_clusters):
        pool.cluster_assignments = None
        pool.cluster_sizes = None
        pool.cluster_to_indices = None
        return

    assignments: Optional[np.ndarray] = None
    cluster_sizes: Optional[np.ndarray] = None

    if cache_path and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            assignments = cached["assignments"]
            cluster_sizes = cached["sizes"]
            if assignments.shape[0] != pool.embeddings.shape[0]:
                assignments = None
                cluster_sizes = None
        except Exception:  # pragma: no cover
            assignments = None
            cluster_sizes = None

    if assignments is None or cluster_sizes is None:
        embeddings_np = pool.embeddings.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init="auto")
        assignments = kmeans.fit_predict(embeddings_np)
        cluster_sizes = np.bincount(assignments, minlength=num_clusters).astype(np.int64)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, assignments=assignments, sizes=cluster_sizes)

    cluster_to_indices_map: Dict[int, List[int]] = defaultdict(list)
    for idx, cluster_id in enumerate(assignments):
        cluster_to_indices_map[int(cluster_id)].append(idx)

    pool.cluster_assignments = assignments
    pool.cluster_sizes = cluster_sizes
    pool.cluster_to_indices = [cluster_to_indices_map[c] for c in range(num_clusters)]


def select_cluster_index(
    rng: np.random.Generator,
    pool: CandidatePool,
    anchor_cluster_blend: float,
):
    assert pool.cluster_sizes is not None
    sizes = pool.cluster_sizes.astype(np.float64)
    num_clusters = len(sizes)
    if num_clusters == 0:
        raise ValueError("No clusters available for selection")

    large_basis = sizes.copy()
    if pool.sampling_weights is not None and pool.cluster_to_indices is not None:
        cluster_mass = np.zeros(num_clusters, dtype=np.float64)
        for cluster_id, member_indices in enumerate(pool.cluster_to_indices):
            if not member_indices:
                continue
            cluster_mass[cluster_id] = float(np.sum(pool.sampling_weights[member_indices]))
        if cluster_mass.sum() > 0:
            large_basis = cluster_mass

    total_large = large_basis.sum()
    if total_large <= 0:
        weights_large = np.full(num_clusters, 1.0 / num_clusters, dtype=np.float64)
    else:
        weights_large = large_basis / total_large

    small_basis = sizes
    if small_basis.sum() <= 0:
        weights_small = np.full(num_clusters, 1.0 / num_clusters, dtype=np.float64)
    else:
        inv = np.zeros_like(small_basis, dtype=np.float64)
        positive_mask = small_basis > 0
        inv[positive_mask] = 1.0 / (small_basis[positive_mask] + 1e-9)
        if inv.sum() <= 0:
            weights_small = np.full(num_clusters, 1.0 / num_clusters, dtype=np.float64)
        else:
            weights_small = inv / inv.sum()

    weights = (1.0 - anchor_cluster_blend) * weights_large + anchor_cluster_blend * weights_small
    total_weights = weights.sum()
    if total_weights <= 0:
        weights = np.full(num_clusters, 1.0 / num_clusters, dtype=np.float64)
    else:
        weights = weights / total_weights

    cluster_idx = int(rng.choice(num_clusters, p=weights))
    return cluster_idx


def sample_support_from_pool(
    rng: np.random.Generator,
    pool: CandidatePool,
    k: int,
    ratio: float,
    similarity_pool_size: int,
    anchor_cluster_blend: float,
    anchor_cluster_count: int,
    support_top_p: float,
    support_similarity_threshold: Optional[float],
) -> Tuple[int, List[int], List[float], List[float]]:
    num_candidates = len(pool.keys)
    if num_candidates == 0:
        raise ValueError("Candidate pool is empty")
    if k > num_candidates:
        raise ValueError("k exceeds number of available candidates")

    candidate_probs: Optional[np.ndarray] = None
    if pool.sampling_weights is not None:
        total_weight = float(pool.sampling_weights.sum())
        if total_weight > 0:
            candidate_probs = pool.sampling_weights / total_weight

    if (
        anchor_cluster_count > 0
        and pool.cluster_assignments is not None
        and pool.cluster_to_indices is not None
        and len(pool.cluster_to_indices) >= anchor_cluster_count
    ):
        cluster_idx = select_cluster_index(rng, pool, anchor_cluster_blend)
        cluster_candidates = pool.cluster_to_indices[cluster_idx]
        if candidate_probs is not None:
            cluster_probs = candidate_probs[cluster_candidates]
            cluster_total = float(cluster_probs.sum())
            if cluster_total > 0:
                anchor_index = int(
                    rng.choice(cluster_candidates, p=(cluster_probs / cluster_total))
                )
            else:
                anchor_index = int(rng.choice(cluster_candidates))
        else:
            anchor_index = int(rng.choice(cluster_candidates))
    else:
        if candidate_probs is not None:
            anchor_index = int(rng.choice(num_candidates, p=candidate_probs))
        else:
            anchor_index = int(rng.integers(0, num_candidates))
    support_indices: List[int] = [anchor_index]

    if k > 1:
        anchor_vector = pool.embeddings[anchor_index]
        sims = pool.embeddings @ anchor_vector
        sims[anchor_index] = -float("inf")

        pool_lim = min(similarity_pool_size, num_candidates - 1)
        if pool_lim <= 0:
            raise ValueError("Similarity pool size too small relative to number of candidates")

        values, indices = torch.topk(sims, k=pool_lim, largest=True)
        values = values.cpu()
        indices = indices.cpu()

        sorted_values, sorted_order = torch.sort(values, descending=True)
        sorted_indices = indices[sorted_order]

        if support_similarity_threshold is not None:
            sim_mask = sorted_values >= support_similarity_threshold
            if sim_mask.sum() == 0:
                sim_mask[0] = True
            sorted_values = sorted_values[sim_mask]
            sorted_indices = sorted_indices[sim_mask]

        if sorted_indices.numel() == 0:
            sorted_values = values
            sorted_indices = indices

        if support_top_p < 1.0:
            probs = torch.softmax(sorted_values, dim=0)
            cumulative = torch.cumsum(probs, dim=0)
            mask = cumulative <= support_top_p
            first_exceed = torch.nonzero(cumulative > support_top_p, as_tuple=False)
            if first_exceed.numel() > 0:
                mask[first_exceed[0].item()] = True
            if mask.sum() == 0:
                mask[0] = True
            filtered_indices = sorted_indices[mask]
        else:
            filtered_indices = sorted_indices

        top_similar = filtered_indices.numpy()
        rng.shuffle(top_similar)

        remaining_slots = k - 1
        similar_slots = min(remaining_slots, int(round(ratio * remaining_slots)))
        random_slots = remaining_slots - similar_slots

        chosen_similar = top_similar[:similar_slots].tolist()
        support_indices.extend(chosen_similar)

        if random_slots > 0:
            excluded = np.zeros(num_candidates, dtype=bool)
            excluded[support_indices] = True
            available = np.where(~excluded)[0]
            if available.size < random_slots:
                raise ValueError("Not enough random candidates to fill support set")
            random_choices = rng.choice(available, size=random_slots, replace=False).tolist()
            support_indices.extend(random_choices)

    probabilities = [1.0 / len(support_indices)] * len(support_indices)

    anchor_vec = pool.embeddings[anchor_index]
    similarities = [float((anchor_vec * pool.embeddings[idx]).sum().item()) for idx in support_indices]

    return anchor_index, support_indices, probabilities, similarities


def build_blueprints(
    settings: ExperimentSettings,
    rng: np.random.Generator,
    pool: CandidatePool,
    k: int,
    ratio: float,
    anchor_cluster_blend: float,
    support_top_p: float,
    support_similarity_threshold: Optional[float],
) -> List[SequenceBlueprint]:
    blueprints: List[SequenceBlueprint] = []
    for sample_idx in range(settings.num_sentences):
        anchor_index, support_indices, probabilities, similarities = sample_support_from_pool(
            rng,
            pool,
            k=k,
            ratio=ratio,
            similarity_pool_size=settings.similarity_pool_size,
            anchor_cluster_blend=anchor_cluster_blend,
            anchor_cluster_count=settings.anchor_cluster_count,
            support_top_p=support_top_p,
            support_similarity_threshold=support_similarity_threshold,
        )
        probs_array = np.array(probabilities, dtype=np.float64)
        probs_array /= probs_array.sum()
        chosen_support = int(rng.choice(len(support_indices), p=probs_array))
        generation_seeds = [
            int(rng.integers(0, 2**31) + settings.generation_seed_offset * (idx + 1))
            for idx in range(settings.num_generations)
        ]

        blueprints.append(
            SequenceBlueprint(
                sample_id=sample_idx,
                anchor_key=pool.keys[anchor_index],
                anchor_text=pool.texts[anchor_index],
                anchor_input_ids=pool.input_id_lists[anchor_index],
                support_keys=[pool.keys[idx] for idx in support_indices],
                support_texts=[pool.texts[idx] for idx in support_indices],
                support_input_ids=[pool.input_id_lists[idx] for idx in support_indices],
                support_probabilities=[float(p) for p in probs_array],
                support_similarities=[float(s) for s in similarities],
                chosen_support_index=chosen_support,
                generation_seeds=generation_seeds,
            )
        )
    return blueprints


def expand_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_ids: torch.Tensor,
    settings: ExperimentSettings,
    generator_seed: int,
    device: torch.device,
) -> torch.Tensor:
    if settings.seq_length == 1:
        return base_ids

    max_new_tokens = settings.seq_length - base_ids.shape[1]
    if max_new_tokens <= 0:
        return base_ids

    do_sample = settings.temperature > 0.0

    if tokenizer.pad_token_id is not None:
        attention_mask = (base_ids != tokenizer.pad_token_id).long()
        if torch.count_nonzero(attention_mask) == 0:
            attention_mask = torch.ones_like(base_ids, device=device)
    else:
        attention_mask = torch.ones_like(base_ids, device=device)

    generation_kwargs = {
        "input_ids": base_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": settings.temperature if do_sample else None,
        "top_k": settings.top_k_sampling,
        "top_p": settings.top_p_sampling,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "return_dict_in_generate": False,
        "output_scores": False,
    }

    sanitized_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    seed = int(generator_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generated = model.generate(**sanitized_kwargs)
    return generated


def collect_next_token_distributions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    settings: ExperimentSettings,
    blueprints: Sequence[SequenceBlueprint],
    device: torch.device,
    anchor_set_index: int,
) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    distributions: List[torch.Tensor] = []
    metadata: List[Dict[str, object]] = []

    for blueprint in blueprints:
        chosen_input_ids = blueprint.support_input_ids[blueprint.chosen_support_index]
        base_ids = torch.tensor([chosen_input_ids], device=device, dtype=torch.long)
        for gen_idx, seed in enumerate(blueprint.generation_seeds):
            if settings.seq_length > 1:
                expanded = expand_sequence(
                    model=model,
                    tokenizer=tokenizer,
                    base_ids=base_ids,
                    settings=settings,
                    generator_seed=seed,
                    device=device,
                )
            else:
                expanded = base_ids

            with torch.inference_mode():
                attention_mask = torch.ones_like(expanded)
                outputs = model(input_ids=expanded, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)

            distributions.append(torch.stack([probs.squeeze(0), log_probs.squeeze(0)], dim=0))
            metadata.append(
                {
                    "sample_id": blueprint.sample_id,
                    "generation_id": gen_idx,
                    "anchor_key": blueprint.anchor_key,
                    "anchor_text": blueprint.anchor_text,
                    "anchor_input_ids": blueprint.anchor_input_ids,
                    "chosen_support_index": blueprint.chosen_support_index,
                    "support_keys": blueprint.support_keys,
                    "support_texts": blueprint.support_texts,
                    "support_input_ids": blueprint.support_input_ids,
                    "support_probabilities": blueprint.support_probabilities,
                    "support_similarities": blueprint.support_similarities,
                    "anchor_set_index": anchor_set_index,
                }
            )

    stacked = torch.stack(distributions, dim=0)
    return stacked, metadata


def compute_cross_entropy_metrics(
    stacked_distributions: torch.Tensor,
    dtype_info: DTypeInfo,
) -> Dict[str, object]:
    probs = stacked_distributions[:, 0, :].to(torch.float64)
    log_probs = stacked_distributions[:, 1, :].to(torch.float64)

    eps = torch.finfo(torch.float64).eps
    probs = torch.clamp(probs, min=eps)

    cross_entropy_matrix = -(probs @ log_probs.T)
    entropy_vector = -torch.sum(probs * log_probs, dim=1, keepdim=True)
    kl_matrix = cross_entropy_matrix - entropy_vector
    sym_kl = 0.5 * (kl_matrix + kl_matrix.T)

    cdf = torch.cumsum(probs, dim=1)
    wasserstein = torch.sum(torch.abs(cdf.unsqueeze(1) - cdf.unsqueeze(0)), dim=2)

    dtype_eps = dtype_info.epsilon
    matches = (sym_kl.abs() <= dtype_eps).float()
    fraction_below_eps = matches.mean().item()

    metrics = {
        "cross_entropy_matrix": cross_entropy_matrix.cpu(),
        "kl_matrix": kl_matrix.cpu(),
        "symmetric_kl_matrix": sym_kl.cpu(),
        "wasserstein_matrix": wasserstein.cpu(),
        "mean_cross_entropy": cross_entropy_matrix.mean().item(),
        "mean_kl": kl_matrix.mean().item(),
        "mean_symmetric_kl": sym_kl.mean().item(),
        "mean_wasserstein": wasserstein.mean().item(),
        "max_symmetric_kl": sym_kl.abs().max().item(),
        "max_wasserstein": wasserstein.max().item(),
        "fraction_symmetric_kl_below_epsilon": fraction_below_eps,
        "dtype_epsilon": dtype_eps,
        "mean_entropy": entropy_vector.mean().item(),
    }
    return metrics


def maybe_downsample_matrix(matrix: torch.Tensor, max_points: Optional[int]) -> torch.Tensor:
    if max_points is None or matrix.shape[0] <= max_points:
        return matrix
    indices = torch.linspace(0, matrix.shape[0] - 1, steps=max_points)
    indices = indices.round().long()
    return matrix[indices][:, indices]


def plot_heatmap(matrix: torch.Tensor, title: str, subtitle: Optional[str] = None) -> plt.Figure:
    array = matrix.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(array, aspect="auto", origin="lower")
    full_title = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(full_title)
    ax.set_xlabel("Distribution index")
    ax.set_ylabel("Distribution index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_symmetric_kl_violin(
    matrix: torch.Tensor,
    title: str,
    subtitle: Optional[str] = None,
) -> plt.Figure:
    """Render per-column symmetric KL distributions as violins with quartile markers."""

    array = matrix.cpu().numpy()
    num_rows, num_cols = array.shape

    column_indices = np.repeat(np.arange(num_cols), num_rows)
    row_indices = np.tile(np.arange(num_rows), num_cols)
    values = array.reshape(-1)

    off_diagonal_mask = row_indices != column_indices
    if off_diagonal_mask.any():
        column_indices = column_indices[off_diagonal_mask]
        values = values[off_diagonal_mask]

    if values.size == 0:
        # Fallback for degenerate single-distribution matrices: include the diagonal values.
        column_indices = np.repeat(np.arange(num_cols), num_rows)
        values = array.reshape(-1)

    fig_width = max(6.0, 0.35 * max(1, num_cols) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    full_title = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(full_title)
    ax.set_xlabel("Distribution index")
    ax.set_ylabel("Symmetric KL divergence")

    if pd is not None and sns is not None:
        df = pd.DataFrame({"column": column_indices, "symmetric_kl": values})
        df["column_label"] = df["column"].astype(str)
        violin_order = sorted(df["column_label"].unique(), key=lambda label: int(label))
        sns.violinplot(
            data=df,
            x="column_label",
            y="symmetric_kl",
            order=violin_order,
            ax=ax,
            inner="quartile",
            cut=0,
            linewidth=1.0,
        )
        ax.set_xticklabels(violin_order)
    else:
        dataset: List[np.ndarray] = []
        for col in range(num_cols):
            column_values = array[:, col].astype(float)
            if num_rows == num_cols and num_rows > 1:
                column_values = np.delete(column_values, col)
            if column_values.size == 0:
                column_values = np.zeros(1, dtype=float)
            dataset.append(column_values)
        violin = ax.violinplot(
            dataset,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            quantiles=[0.25, 0.5, 0.75],
        )
        for body in violin["bodies"]:
            body.set_facecolor("#4C72B0")
            body.set_edgecolor("black")
            body.set_alpha(0.7)
        if "cquantiles" in violin:
            violin["cquantiles"].set_color("black")
            violin["cquantiles"].set_linewidth(1.0)
        ax.set_xticks(np.arange(1, num_cols + 1))
        ax.set_xticklabels([str(i) for i in range(num_cols)])

    fig.tight_layout()
    return fig


def plot_anchor_support_bar(
    blueprint: SequenceBlueprint,
    mode_name: str,
    dtype_name: str,
    k: int,
    ratio: float,
    distribution_type: str,
    anchor_set_index: int,
) -> plt.Figure:
    support = list(zip(blueprint.support_texts, blueprint.support_similarities))
    paired = sorted(support, key=lambda item: item[1], reverse=True)

    labels = []
    sorted_sims = []
    for text, sim in paired:
        safe_text = (text or "").replace("\n", "\\n")
        labels.append(safe_text)
        sorted_sims.append(sim)

    indices = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 4))
    ax.bar(indices, sorted_sims, color="tab:blue")
    ax.set_ylabel("Cosine similarity")
    lower = min(sorted_sims) if sorted_sims else 0.0
    upper = max(sorted_sims) if sorted_sims else 1.0
    padding = max(0.05, 0.05 * (upper - lower + 1e-6))
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(
        "Anchor Support Similarities\n"
        f"mode={mode_name}, dtype={dtype_name}, k={k}, ratio={ratio:.2f}, dist={distribution_type}, "
        f"anchor_set={anchor_set_index}, anchor_text={blueprint.anchor_text}"
    )
    fig.tight_layout()
    return fig


def _coerce_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def reload_and_plot(
    settings: ExperimentSettings,
    run_path: str,
) -> Tuple[
    plt.Figure,
    plt.Figure,
    plt.Figure,
    Optional[plt.Figure],
    plt.Figure,
    plt.Figure,
    Optional[plt.Figure],
    plt.Figure,
    Optional[plt.Figure],
    Optional[plt.Figure],
    Optional[plt.Figure],
    Optional["pd.DataFrame"],
]:
    api = wandb.Api()
    target_run = api.run(run_path)

    # Get ALL records first without any key filtering
    print(f"Scanning run {run_path} for all available data...")
    all_history = list(target_run.scan_history())

    if not all_history:
        raise ValueError(f"No data found in run {run_path}. The run may not have completed or may be empty.")

    print(f"Found {len(all_history)} total records in run")

    # Analyze what keys are available across all records
    all_available_keys = set()
    for record in all_history:
        all_available_keys.update(record.keys())

    print(f"Available keys across all records: {sorted(all_available_keys)}")

    # Group records by their key patterns to understand data structure
    record_types = {}
    for record in all_history:
        key_set = frozenset(record.keys())
        if key_set not in record_types:
            record_types[key_set] = []
        record_types[key_set].append(record)

    print(f"Found {len(record_types)} different record types:")
    for i, (key_set, records) in enumerate(record_types.items()):
        print(f"  Type {i+1}: {len(records)} records with keys: {sorted(key_set)}")

    wasserstein_available = "mean_wasserstein" in all_available_keys

    # For plotting, we need records with the core metrics
    core_metric_keys = {
        "mode",
        "dtype",
        "k",
        "ratio",
        "anchor_cluster_blend",
        "mean_cross_entropy",
        "mean_symmetric_kl",
        "fraction_symmetric_kl_below_epsilon",
    }
    if wasserstein_available:
        core_metric_keys.add("mean_wasserstein")

    # Find records that have the core metrics (for plotting)
    core_records = []
    for record in all_history:
        if core_metric_keys.issubset(set(record.keys())):
            core_records.append(record)

    print(f"Found {len(core_records)} records with core metrics for plotting")

    if not core_records:
        raise ValueError(f"No records found with core metrics in run {run_path}.")

    # Use core records for plotting (they have the essential data)
    history = core_records

    # Also collect additional statistical data from other record types if available
    aggregated_records = []
    for record in all_history:
        if "sample_count" in record or "num_anchor_sets" in record:
            aggregated_records.append(record)

    print(f"Found {len(aggregated_records)} records with statistical/aggregation data")

    if aggregated_records:
        print("Statistical data available - plots will include error bars")
    else:
        print("No statistical data found - plots will show means only")

    # Create a mapping of statistical data by configuration for enhanced plotting
    stats_by_config = {}
    for record in aggregated_records:
        # Create a key based on the configuration parameters that should match core records
        config_key = (
            record.get("mode"),
            record.get("dtype"),
            record.get("k"),
            record.get("ratio"),
            record.get("anchor_cluster_blend"),
            record.get("anchor_cluster_count"),
            record.get("support_similarity_threshold")
        )

        if config_key not in stats_by_config:
            stats_by_config[config_key] = {}

        # Extract statistical measures and store them by metric type
        for metric_base in ["mean_cross_entropy", "mean_symmetric_kl", "mean_wasserstein", "fraction_symmetric_kl_below_epsilon"]:
            for stat_type in ["std", "stderr"]:
                stat_key = f"{metric_base}_{stat_type}"
                if stat_key in record and record[stat_key] is not None:
                    if metric_base not in stats_by_config[config_key]:
                        stats_by_config[config_key][metric_base] = {}
                    stats_by_config[config_key][metric_base][stat_type] = record[stat_key]

        # Also capture sample count if available
        if "sample_count" in record:
            stats_by_config[config_key]["sample_count"] = record["sample_count"]

    print(f"Organized statistical data for {len(stats_by_config)} unique configurations")

    ce_series = defaultdict(list)
    frac_series = defaultdict(list)
    skl_series = defaultdict(list)
    wasserstein_series = defaultdict(list)
    ce_anchor_points = defaultdict(list)
    frac_anchor_points = defaultdict(list)
    skl_anchor_points = defaultdict(list)
    wasserstein_anchor_points = defaultdict(list)
    blend_ce = defaultdict(list)
    blend_skl = defaultdict(list)
    blend_sim = defaultdict(list)
    blend_wasserstein = defaultdict(list)
    history_records: List[Dict[str, object]] = []
    history_df: Optional["pd.DataFrame"] = None

    for row in history:
        mode_name = row.get("mode", "unknown")
        dtype_name = row.get("dtype", "unknown")
        k_value = row.get("k", "unknown")
        anchor_blend = float(row.get("anchor_cluster_blend", 0.0) or 0.0)
        anchor_identifier = row.get("anchor_set_index")
        ratio_value = row.get("ratio")
        numeric_ratio = _coerce_float(ratio_value)
        ratio_float = numeric_ratio if numeric_ratio is not None else float("nan")
        anchor_count_raw = row.get("anchor_cluster_count")
        try:
            anchor_count_value = int(anchor_count_raw) if anchor_count_raw is not None else None
        except (TypeError, ValueError):
            anchor_count_value = anchor_count_raw
        anchor_count_options = row.get("anchor_cluster_count_options")
        support_threshold_value_raw = row.get("support_similarity_threshold")
        support_threshold_value = _coerce_float(support_threshold_value_raw)
        support_threshold_options = row.get("support_similarity_threshold_options")

        mean_ce = row.get("mean_cross_entropy")
        ce_std = row.get("mean_cross_entropy_std", 0.0)  # Default to 0.0 if missing
        ce_stderr = row.get("mean_cross_entropy_stderr", 0.0)  # Default to 0.0 if missing
        frac_value = row.get("fraction_symmetric_kl_below_epsilon")
        frac_stderr = row.get("fraction_symmetric_kl_below_epsilon_stderr", 0.0)  # Default to 0.0 if missing
        mean_skl = row.get("mean_symmetric_kl")
        skl_std = row.get("mean_symmetric_kl_std", 0.0)  # Default to 0.0 if missing
        skl_stderr = row.get("mean_symmetric_kl_stderr", 0.0)  # Default to 0.0 if missing
        mean_wass = row.get("mean_wasserstein")
        wass_std = row.get("mean_wasserstein_std", 0.0)
        wass_stderr = row.get("mean_wasserstein_stderr", 0.0)
        max_wass = row.get("max_wasserstein")
        max_wass_std = row.get("max_wasserstein_std", 0.0)
        max_wass_stderr = row.get("max_wasserstein_stderr", 0.0)
        mean_sim = row.get("mean_support_similarity")
        sim_stderr = row.get("mean_support_similarity_stderr", 0.0)  # Default to 0.0 if missing
        mean_entropy = row.get("mean_entropy")
        entropy_stderr = row.get("mean_entropy_stderr", 0.0)  # Default to 0.0 if missing

        # Try to enhance with statistical data from aggregated records if available
        current_config_key = (mode_name, dtype_name, k_value, anchor_blend, anchor_count_value, support_threshold_value)
        if current_config_key in stats_by_config:
            config_stats = stats_by_config[current_config_key]

            # Use statistical data if available and more reliable than individual record data
            for metric_base in ["mean_cross_entropy", "mean_symmetric_kl", "mean_wasserstein", "fraction_symmetric_kl_below_epsilon"]:
                if metric_base in config_stats and "stderr" in config_stats[metric_base]:
                    stat_value = config_stats[metric_base]["stderr"]
                    if stat_value is not None and not (isinstance(stat_value, float) and math.isnan(stat_value)):
                        if metric_base == "mean_cross_entropy" and ce_stderr == 0.0:
                            ce_stderr = stat_value
                        elif metric_base == "mean_symmetric_kl" and skl_stderr == 0.0:
                            skl_stderr = stat_value
                        elif metric_base == "mean_wasserstein" and wass_stderr == 0.0:
                            wass_stderr = stat_value
                        elif metric_base == "fraction_symmetric_kl_below_epsilon" and frac_stderr == 0.0:
                            frac_stderr = stat_value

                # Also try to use std if stderr is not available
                if metric_base in config_stats and "std" in config_stats[metric_base]:
                    stat_value = config_stats[metric_base]["std"]
                    if stat_value is not None and not (isinstance(stat_value, float) and math.isnan(stat_value)):
                        if metric_base == "mean_cross_entropy" and ce_std == 0.0:
                            ce_std = stat_value
                        elif metric_base == "mean_symmetric_kl" and skl_std == 0.0:
                            skl_std = stat_value
                        elif metric_base == "mean_wasserstein" and wass_std == 0.0:
                            wass_std = stat_value

            # Use sample count if available
            if "sample_count" in config_stats and sample_size == 1:
                sample_size = int(config_stats["sample_count"])

        sample_size = int(row.get("sample_count") or row.get("num_anchor_sets") or 0)
        if sample_size == 0:
            sample_size = 1

        record = {
            "mode": mode_name,
            "dtype": dtype_name,
            "k": k_value,
            "ratio_raw": ratio_value,
            "ratio": ratio_float,
            "anchor_set_index": anchor_identifier,
            "anchor_cluster_blend": anchor_blend,
            "anchor_cluster_count": anchor_count_value,
            "anchor_cluster_count_options": anchor_count_options,
            "top_p_sampling": row.get("top_p_sampling"),
            "support_similarity_threshold": support_threshold_value,
            "support_similarity_threshold_raw": support_threshold_value_raw,
            "support_similarity_threshold_options": support_threshold_options,
            "mean_cross_entropy": float(mean_ce) if mean_ce is not None else float("nan"),
            "mean_cross_entropy_std": float(ce_std) if ce_std is not None else 0.0,
            "mean_cross_entropy_stderr": float(ce_stderr) if ce_stderr is not None else 0.0,
            "fraction_symmetric_kl_below_epsilon": float(frac_value) if frac_value is not None else float("nan"),
            "fraction_symmetric_kl_below_epsilon_stderr": float(frac_stderr) if frac_stderr is not None else 0.0,
            "mean_symmetric_kl": float(mean_skl) if mean_skl is not None else float("nan"),
            "mean_symmetric_kl_std": float(skl_std) if skl_std is not None else 0.0,
            "mean_symmetric_kl_stderr": float(skl_stderr) if skl_stderr is not None else 0.0,
            "mean_wasserstein": float(mean_wass) if mean_wass is not None else float("nan"),
            "mean_wasserstein_std": float(wass_std) if wass_std is not None else 0.0,
            "mean_wasserstein_stderr": float(wass_stderr) if wass_stderr is not None else 0.0,
            "max_wasserstein": float(max_wass) if max_wass is not None else float("nan"),
            "max_wasserstein_std": float(max_wass_std) if max_wass_std is not None else 0.0,
            "max_wasserstein_stderr": float(max_wass_stderr) if max_wass_stderr is not None else 0.0,
            "mean_support_similarity": float(mean_sim) if mean_sim is not None else float("nan"),
            "mean_support_similarity_stderr": float(sim_stderr) if sim_stderr is not None else 0.0,
            "mean_entropy": float(mean_entropy) if mean_entropy is not None else float("nan"),
            "mean_entropy_stderr": float(entropy_stderr) if entropy_stderr is not None else 0.0,
            "sample_count": sample_size,
        }
        if anchor_count_value is None:
            series_anchor = "None"
        elif isinstance(anchor_count_value, (int, np.integer)):
            series_anchor = str(int(anchor_count_value))
        else:
            series_anchor = str(anchor_count_value)
        if support_threshold_value is None and support_threshold_value_raw in ("None", None):
            support_label = "None"
        else:
            support_label = (
                "None"
                if support_threshold_value is None
                else f"{support_threshold_value:.4f}".rstrip("0").rstrip(".")
            )
        record["series_label"] = (
            f"{mode_name}|{dtype_name}|k={k_value}|blend={anchor_blend:.2f}|clusters={series_anchor}|support_thr={support_label}"
        )
        history_records.append(record)

        key_ratio = (mode_name, dtype_name, k_value, anchor_blend, anchor_count_value, support_label)
        key_blend = (mode_name, dtype_name, k_value, anchor_count_value, support_label)

        if isinstance(ratio_value, str) and ratio_value == "aggregate" and anchor_identifier == "aggregate_ratio":
            if mean_ce is not None:
                blend_ce[key_blend].append(
                    {
                        "blend": anchor_blend,
                        "mean": float(mean_ce),
                        "std": float(ce_std or 0.0),
                        "stderr": float(ce_stderr or 0.0),
                        "n": sample_size,
                    }
                )
            if mean_skl is not None:
                blend_skl[key_blend].append(
                    {
                        "blend": anchor_blend,
                        "mean": float(mean_skl),
                        "stderr": float(skl_stderr or 0.0),
                        "n": sample_size,
                    }
                )
            if mean_wass is not None:
                blend_wasserstein[key_blend].append(
                    {
                        "blend": anchor_blend,
                        "mean": float(mean_wass),
                        "stderr": float(wass_stderr or 0.0),
                        "n": sample_size,
                    }
                )
            if mean_sim is not None:
                blend_sim[key_blend].append(
                    {
                        "blend": anchor_blend,
                        "mean": float(mean_sim),
                        "stderr": float(sim_stderr or 0.0),
                        "n": sample_size,
                    }
                )
            continue

        if numeric_ratio is None:
            continue
        ratio_float = numeric_ratio

        if anchor_identifier in (None, "aggregate"):
            if mean_ce is not None:
                ce_series[key_ratio].append(
                    {
                        "ratio": ratio_float,
                        "mean": float(mean_ce),
                        "std": float(ce_std or 0.0),
                        "stderr": float(ce_stderr or 0.0),
                        "n": sample_size,
                        "entropy": float(mean_entropy) if mean_entropy is not None else None,
                    }
                )
            if frac_value is not None:
                frac_series[key_ratio].append(
                    {
                        "ratio": ratio_float,
                        "mean": float(frac_value),
                        "stderr": float(frac_stderr or 0.0),
                        "n": sample_size,
                    }
                )
            if mean_skl is not None:
                skl_series[key_ratio].append(
                    {
                        "ratio": ratio_float,
                        "mean": float(mean_skl),
                        "stderr": float(skl_stderr or 0.0),
                        "n": sample_size,
                    }
                )
            if mean_wass is not None:
                wasserstein_series[key_ratio].append(
                    {
                        "ratio": ratio_float,
                        "mean": float(mean_wass),
                        "stderr": float(wass_stderr or 0.0),
                        "n": sample_size,
                    }
                )
        else:
            if mean_ce is not None:
                ce_anchor_points[key_ratio].append((ratio_float, float(mean_ce)))
            if frac_value is not None:
                frac_anchor_points[key_ratio].append((ratio_float, float(frac_value)))
            if mean_skl is not None:
                skl_anchor_points[key_ratio].append((ratio_float, float(mean_skl)))
            if mean_wass is not None:
                wasserstein_anchor_points[key_ratio].append((ratio_float, float(mean_wass)))

    if not ce_series:
        raise ValueError(f"No ratio/mean_cross_entropy data found in run {run_path}.")

        message = f"No ratio/mean_cross_entropy data found in run {run_path}."

        def _empty_plot(title: str, xlabel: str, ylabel: str) -> plt.Figure:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
            fig.tight_layout()
            return fig

        placeholder = _empty_plot(
            "Blend Ratio vs Mean Cross-Entropy",
            "Blend ratio (random → similarity)",
            "Mean cross-entropy",
        )
        frac_placeholder = _empty_plot(
            "Blend Ratio vs Fraction |Symmetric KL| ≤ ε",
            "Blend ratio (random → similarity)",
            "Fraction <= ε",
        )
        skl_placeholder = _empty_plot(
            "Blend Ratio vs Mean Symmetric KL",
            "Blend ratio (random → similarity)",
            "Mean symmetric KL",
        )
        wasserstein_placeholder = _empty_plot(
            "Blend Ratio vs Mean Wasserstein Distance",
            "Blend ratio (random → similarity)",
            "Mean Wasserstein distance",
        )
        blend_ce_placeholder = _empty_plot(
            "Anchor Blend vs Mean Cross-Entropy",
            "Anchor blend (large clusters → small clusters)",
            "Mean cross-entropy",
        )
        blend_skl_placeholder = _empty_plot(
            "Anchor Blend vs Mean Symmetric KL",
            "Anchor blend (large clusters → small clusters)",
            "Mean symmetric KL",
        )
        blend_sim_placeholder = _empty_plot(
            "Anchor Blend vs Mean Support Similarity",
            "Anchor blend (large clusters → small clusters)",
            "Mean support similarity",
        )
        blend_wass_placeholder = _empty_plot(
            "Anchor Blend vs Mean Wasserstein Distance",
            "Anchor blend (large clusters → small clusters)",
            "Mean Wasserstein distance",
        )

        return (
            placeholder,
            frac_placeholder,
            skl_placeholder,
            wasserstein_placeholder,
            blend_ce_placeholder,
            blend_skl_placeholder,
            blend_wass_placeholder,
            blend_sim_placeholder,
            None,
            None,
            None,
            history_df,
        )

    if pd is not None:
        history_df = pd.DataFrame(history_records)
        if not history_df.empty:
            numeric_columns = [
                "ratio",
                "anchor_cluster_blend",
                "anchor_cluster_count",
                "top_p_sampling",
                "support_similarity_threshold",
                "mean_cross_entropy",
                "mean_cross_entropy_std",
                "mean_cross_entropy_stderr",
                "fraction_symmetric_kl_below_epsilon",
                "fraction_symmetric_kl_below_epsilon_stderr",
                "mean_symmetric_kl",
                "mean_symmetric_kl_std",
                "mean_symmetric_kl_stderr",
                "mean_wasserstein",
                "mean_wasserstein_std",
                "mean_wasserstein_stderr",
                "max_wasserstein",
                "max_wasserstein_std",
                "max_wasserstein_stderr",
                "mean_support_similarity",
                "mean_support_similarity_stderr",
                "mean_entropy",
                "mean_entropy_stderr",
                "sample_count",
            ]
            for column in numeric_columns:
                if column in history_df.columns:
                    history_df[column] = pd.to_numeric(history_df[column], errors="coerce")

    ce_fig = plot_ratio_series(
        ce_series,
        ce_anchor_points,
        "Blend Ratio vs Mean Cross-Entropy",
        "Mean cross-entropy",
        show_entropy=True,
        include_std=True,
    )
    frac_fig = plot_ratio_series(
        frac_series,
        frac_anchor_points,
        "Blend Ratio vs Fraction |Symmetric KL| ≤ ε",
        "Fraction <= ε",
    )
    frac_ax = frac_fig.axes[0]
    frac_ax.set_ylim(0.0, 1.0)
    skl_fig = plot_ratio_series(skl_series, skl_anchor_points, "Blend Ratio vs Mean Symmetric KL", "Mean symmetric KL")
    wasserstein_fig: Optional[plt.Figure] = None
    if wasserstein_series:
        wasserstein_fig = plot_ratio_series(
            wasserstein_series,
            wasserstein_anchor_points,
            "Blend Ratio vs Mean Wasserstein Distance",
            "Mean Wasserstein distance",
            include_std=True,
        )
    elif wasserstein_available:
        raise ValueError(f"No ratio/mean_wasserstein data found in run {run_path}.")

    blend_ce_fig = plot_blend_series(
        blend_ce,
        "Anchor Blend vs Mean Cross-Entropy",
        "Mean cross-entropy",
        include_std=True,
    )
    blend_skl_fig = plot_blend_series(
        blend_skl,
        "Anchor Blend vs Mean Symmetric KL",
        "Mean symmetric KL",
    )
    blend_wasserstein_fig: Optional[plt.Figure] = None
    if blend_wasserstein:
        blend_wasserstein_fig = plot_blend_series(
            blend_wasserstein,
            "Anchor Blend vs Mean Wasserstein Distance",
            "Mean Wasserstein distance",
            include_std=True,
        )
    elif wasserstein_available:
        raise ValueError(f"No anchor-blend Wasserstein data found in run {run_path}.")
    blend_sim_fig = plot_blend_series(
        blend_sim,
        "Anchor Blend vs Mean Support Similarity",
        "Mean support similarity",
    )
    facet_ce_fig: Optional[plt.Figure] = None
    facet_skl_fig: Optional[plt.Figure] = None
    facet_wasserstein_fig: Optional[plt.Figure] = None
    if history_df is not None:
        try:
            facet_ce_fig = plot_metric_facet(
                history_df,
                "mean_cross_entropy",
                "Mean cross-entropy",
                settings.facet_row_field,
                settings.facet_col_field,
                settings.facet_hue_field,
            )
        except ImportError:
            facet_ce_fig = None
        try:
            facet_skl_fig = plot_metric_facet(
                history_df,
                "mean_symmetric_kl",
                "Mean symmetric KL",
                settings.facet_row_field,
                settings.facet_col_field,
                settings.facet_hue_field,
            )
        except ImportError:
            facet_skl_fig = None
        if wasserstein_available:
            try:
                facet_wasserstein_fig = plot_metric_facet(
                    history_df,
                    "mean_wasserstein",
                    "Mean Wasserstein distance",
                    settings.facet_row_field,
                    settings.facet_col_field,
                    settings.facet_hue_field,
                )
            except ImportError:
                facet_wasserstein_fig = None

    return (
        ce_fig,
        frac_fig,
        skl_fig,
        wasserstein_fig,
        blend_ce_fig,
        blend_skl_fig,
        blend_wasserstein_fig,
        blend_sim_fig,
        facet_ce_fig,
        facet_skl_fig,
        facet_wasserstein_fig,
        history_df,
    )


def deprecated_plot_metric_facet(
    history_df: pd.DataFrame,
    metric_column: str,
    title: str,
    row_field: Optional[str],
    col_field: Optional[str],
    hue_field: Optional[str],
) -> plt.Figure:
    """
    DEPRECATED: Original facet plotting function with data filtering bug.
    Kept for comparison purposes.

    This function only shows ~2 points per facet instead of expected 9 points
    due to incorrect data grouping/filtering logic.
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("seaborn is required for facet plotting")

    if row_field is None and col_field is None:
        # Single plot case
        fig, ax = plt.subplots(figsize=(8, 6))
        # Basic scatter plot - this is likely where the bug is
        # Only showing 2 points instead of 9
        ax.scatter(history_df['ratio'], history_df[metric_column])
        ax.set_xlabel('Blend ratio')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig

    # Create facet grid
    g = sns.FacetGrid(
        history_df,
        row=row_field,
        col=col_field,
        hue=hue_field,
        height=4,
        aspect=1.2,
        sharex=True,
        sharey=True,
    )

    # Add scatter plots to each facet
    # This is likely where the data filtering bug occurs
    # Only showing 2 points instead of expected 9 (3 dtypes × 3 ratios)
    g.map_dataframe(
        sns.scatterplot,
        x='ratio',
        y=metric_column,
        alpha=0.7,
        s=60,
    )

    # Customize the plot
    g.set_axis_labels('Blend ratio (random → similarity)', title)
    g.set_titles(template='{row_name} | {col_name}')
    g.add_legend(title=hue_field)

    # Tight layout
    g.fig.tight_layout()

    return g.fig


def plot_metric_facet(
    history_df: pd.DataFrame,
    metric_column: str,
    title: str,
    row_field: Optional[str],
    col_field: Optional[str],
    hue_field: Optional[str],
) -> plt.Figure:
    """Build a facet grid of bar plots with standard error overlays for the metric."""
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("seaborn is required for facet plotting") from exc

    if history_df is None or history_df.empty:
        raise ValueError("No data available for facet plotting")
    if metric_column not in history_df.columns:
        raise ValueError(f"Metric column '{metric_column}' not found in data")

    if (
        "anchor_cluster_blend" in history_df.columns
        and not history_df["anchor_cluster_blend"].isna().all()
    ):
        x_col = "anchor_cluster_blend"
        x_label = "Anchor cluster blend (large → small)"
    elif "ratio" in history_df.columns and not history_df["ratio"].isna().all():
        x_col = "ratio"
        x_label = "Blend ratio (random → similarity)"
    else:
        raise ValueError(
            "Expected either 'anchor_cluster_blend' or 'ratio' column with non-NaN values for plotting"
        )

    required_cols = [metric_column, x_col]
    if any(column not in history_df.columns for column in required_cols):
        missing = [col for col in required_cols if col not in history_df.columns]
        raise ValueError(f"Required columns missing: {missing}")

    plot_df = history_df.dropna(subset=required_cols).copy()
    if plot_df.empty:
        raise ValueError("No data available for facet plotting after filtering")

    def _ordered_categories(series: pd.Series) -> List[object]:
        non_na = series.dropna()
        if non_na.empty:
            return []
        if pd.api.types.is_numeric_dtype(non_na):
            unique_vals = sorted(non_na.unique().tolist())
        else:
            unique_vals = []
            for value in series:
                if pd.isna(value) or value in unique_vals:
                    continue
                unique_vals.append(value)
        return unique_vals

    x_order = _ordered_categories(plot_df[x_col])
    if not x_order:
        raise ValueError("Unable to determine x-axis values for facet plotting")

    def _format_tick_label(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            magnitude = abs(float(value))
            precision = 3 if magnitude < 1.0 else 2
            formatted = f"{float(value):.{precision}f}"
            formatted = formatted.rstrip("0").rstrip(".")
            return formatted if formatted else "0"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    x_label_order = [_format_tick_label(val) for val in x_order]
    value_to_label = {val: label for val, label in zip(x_order, x_label_order)}

    def _map_value_to_label(value: object) -> str:
        if value in value_to_label:
            return value_to_label[value]
        if isinstance(value, (float, np.floating)):
            for candidate, label in value_to_label.items():
                if np.isclose(candidate, value):
                    return label
        return _format_tick_label(value)

    plot_df["_x_label"] = plot_df[x_col].apply(_map_value_to_label)

    hue_order = (
        _ordered_categories(plot_df[hue_field])
        if hue_field and hue_field in plot_df.columns
        else None
    )
    row_order = (
        _ordered_categories(plot_df[row_field])
        if row_field and row_field in plot_df.columns
        else None
    )
    col_order = (
        _ordered_categories(plot_df[col_field])
        if col_field and col_field in plot_df.columns
        else None
    )

    stderr_col = f"{metric_column}_stderr" if f"{metric_column}_stderr" in plot_df.columns else None
    sample_count_col = "sample_count" if "sample_count" in plot_df.columns else None

    catplot_kwargs = {
        "data": plot_df,
        "kind": "bar",
        "x": "_x_label",
        "y": metric_column,
        "row": row_field,
        "col": col_field,
        "hue": hue_field,
        "order": x_label_order,
        "height": 2.8,
        "aspect": 1.2,
        "sharex": True,
        "sharey": False,
        "errorbar": None,
        "legend": False,
    }
    if hue_order:
        catplot_kwargs["hue_order"] = hue_order
    if row_field and row_order:
        catplot_kwargs["row_order"] = row_order
    if col_field and col_order:
        catplot_kwargs["col_order"] = col_order

    try:
        facet = sns.catplot(**catplot_kwargs)
    except TypeError as exc:
        if "errorbar" not in str(exc):
            raise
        legacy_kwargs = dict(catplot_kwargs)
        legacy_kwargs.pop("errorbar", None)
        legacy_kwargs["ci"] = None
        facet = sns.catplot(**legacy_kwargs)

    grouping_fields: List[str] = ["_x_label"]
    if hue_field:
        grouping_fields.append(hue_field)
    if row_field:
        grouping_fields.append(row_field)
    if col_field:
        grouping_fields.append(col_field)

    agg_map = {metric_column: "mean"}
    if stderr_col:
        agg_map[stderr_col] = "mean"
    if sample_count_col:
        agg_map[sample_count_col] = "max"

    aggregated = (
        plot_df.groupby(grouping_fields, dropna=False, as_index=False).agg(agg_map)
        if grouping_fields
        else plot_df
    )

    zero_stderr_records: List[Dict[str, object]] = []

    if stderr_col:
        from matplotlib.container import BarContainer

        if hue_field:
            hue_levels = hue_order or _ordered_categories(plot_df[hue_field])
        else:
            hue_levels = [None]

        for (row_val, col_val), ax in facet.axes_dict.items():
            subset = aggregated
            if row_field:
                if pd.isna(row_val):
                    subset = subset[subset[row_field].isna()]
                else:
                    subset = subset[subset[row_field] == row_val]
            if col_field:
                if pd.isna(col_val):
                    subset = subset[subset[col_field].isna()]
                else:
                    subset = subset[subset[col_field] == col_val]
            if subset.empty:
                continue

            bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
            if not hue_field and bar_containers:
                bar_containers = bar_containers[:1]
            elif hue_field:
                bar_containers = bar_containers[: len(hue_levels)]

            for idx, hue_val in enumerate(hue_levels):
                if idx >= len(bar_containers):
                    break
                container = bar_containers[idx]
                for bar_idx, bar in enumerate(container.patches):
                    if bar_idx >= len(x_label_order):
                        continue
                    x_label_value = x_label_order[bar_idx]
                    mask = subset["_x_label"] == x_label_value
                    if hue_field:
                        mask &= subset[hue_field] == hue_val
                    row_data = subset[mask]
                    if row_data.empty:
                        continue
                    row_value = row_data[row_field].iloc[0] if row_field else None
                    col_value = row_data[col_field].iloc[0] if col_field else None
                    err = float(row_data[stderr_col].iloc[0])
                    if not np.isfinite(err) or err < 0.0:
                        continue
                    height = bar.get_height()
                    if not np.isfinite(height):
                        continue
                    center = bar.get_x() + bar.get_width() / 2
                    if abs(err) < 1e-12:
                        sample_count_val: Optional[float] = None
                        if sample_count_col and sample_count_col in row_data.columns:
                            sample_count_val = float(row_data[sample_count_col].iloc[0])
                        source_mask = plot_df["_x_label"] == x_label_value
                        if hue_field:
                            source_mask &= plot_df[hue_field] == hue_val
                        if row_field:
                            if pd.isna(row_value):
                                source_mask &= plot_df[row_field].isna()
                            else:
                                source_mask &= plot_df[row_field] == row_value
                        if col_field:
                            if pd.isna(col_value):
                                source_mask &= plot_df[col_field].isna()
                            else:
                                source_mask &= plot_df[col_field] == col_value
                        contributing_points = plot_df[source_mask]
                        n_points = int(contributing_points.shape[0])
                        non_zero_inputs = (
                            int(
                                np.sum(
                                    np.abs(
                                        contributing_points[stderr_col]
                                        .dropna()
                                        .to_numpy(dtype=float)
                                    )
                                    > 1e-12
                                )
                            )
                            if stderr_col in contributing_points.columns
                            else 0
                        )
                        zero_stderr_records.append(
                            {
                                "row": row_value,
                                "col": col_value,
                                "hue": hue_val,
                                "x": x_label_value,
                                "sample_count": sample_count_val,
                                "points": n_points,
                                "non_zero_inputs": non_zero_inputs,
                            }
                        )
                        ax.scatter(
                            center,
                            height,
                            marker="x",
                            color=bar.get_facecolor(),
                            s=40,
                            linewidths=1.4,
                            zorder=5,
                        )
                        continue
                    ax.errorbar(
                        center,
                        height,
                        yerr=err,
                        color=bar.get_facecolor(),
                        capsize=4,
                        linewidth=1.0,
                    )

    facet.set_axis_labels(x_label, title)
    if row_field or col_field:
        facet.set_titles(template="{row_name} | {col_name}")
    else:
        facet.axes.flat[0].set_title(title)

    legend: Optional[Legend] = None
    if hue_field:
        facet.add_legend(title=hue_field)
        legend = getattr(facet, "_legend", None)
        if legend is not None:
            legend.set_bbox_to_anchor(
                (1.02, 0.5),
                transform=facet.fig.transFigure,
                borderaxespad=0.0,
            )
            legend.set_loc("center left")

    descriptor_parts: List[str] = []
    if row_field:
        descriptor_parts.append(f"rows={row_field}")
    if col_field:
        descriptor_parts.append(f"cols={col_field}")
    if hue_field:
        descriptor_parts.append(f"hue={hue_field}")
    title_suffix = f" ({', '.join(descriptor_parts)})" if descriptor_parts else ""

    for ax in facet.axes.flat:
        ax.tick_params(axis="x", labelsize=8)
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")

    if zero_stderr_records:
        debug_mode = bool(os.environ.get("GREATER_FACET_DEBUG"))
        print(
            f"WARNING: {metric_column} facet – zero stderr for {len(zero_stderr_records)} bar(s); marked with 'x' on the plot."
        )
        if debug_mode:
            print("   GREATER_FACET_DEBUG is set – printing full list of zero-stderr bars.")
        preview_limit = len(zero_stderr_records) if debug_mode else 12
        for record in itertools.islice(zero_stderr_records, preview_limit):
            parts: List[str] = []
            def _format_meta(value: object) -> str:
                if value is None:
                    return "None"
                if isinstance(value, (float, np.floating, int, np.integer)):
                    return _format_tick_label(value)
                return str(value)
            if row_field and record["row"] is not None:
                parts.append(f"{row_field}={_format_meta(record['row'])}")
            if col_field and record["col"] is not None:
                parts.append(f"{col_field}={_format_meta(record['col'])}")
            if hue_field and record["hue"] is not None:
                parts.append(f"{hue_field}={_format_meta(record['hue'])}")
            parts.append(f"{x_col}={record['x']}")
            sample_count_val = record.get("sample_count")
            if sample_count_val is not None and np.isfinite(sample_count_val):
                sample_note = f"sample_count≈{sample_count_val:.0f}"
                if sample_count_val <= 1.0:
                    sample_note += " (only one sample)"
            else:
                sample_note = "sample_count unavailable"
            points_note = "inputs=unknown"
            points_raw = record.get("points")
            if isinstance(points_raw, int):
                non_zero = record.get("non_zero_inputs")
                if isinstance(non_zero, int):
                    points_note = (
                        f"inputs={points_raw} (non-zero stderr entries={non_zero})"
                    )
                else:
                    points_note = f"inputs={points_raw}"
            print(f" - {' | '.join(parts)} → {sample_note}; {points_note}")
        if len(zero_stderr_records) > preview_limit:
            remaining = len(zero_stderr_records) - preview_limit
            row_fields = {record.get("row") for record in zero_stderr_records}
            col_fields = {record.get("col") for record in zero_stderr_records}
            hue_fields = {record.get("hue") for record in zero_stderr_records if record.get("hue") is not None}
            print(
                f"   … {remaining} additional zero-stderr bar(s) suppressed"
                f" (rows≈{len(row_fields)}, cols≈{len(col_fields)}, hues≈{len(hue_fields) or 0})."
                " Re-run with more seeds or set GREATER_FACET_DEBUG=1 to inspect all entries."
            )
        print(
            "   Zero stderr typically means only one run contributed or upstream aggregation skipped stderr; "
            "consider aggregating across multiple seeds/samples."
        )

    layout_rect = [0.0, 0.0, 0.82 if legend is not None else 1.0, 0.92]
    facet.fig.tight_layout(rect=layout_rect)
    facet.fig.suptitle(f"{title}{title_suffix}", y=0.98)

    return facet.fig


def plot_ratio_series(
    series_dict,
    anchor_dict,
    title,
    ylabel,
    show_entropy: bool = False,
    include_std: bool = False,
):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    entropy_label_added = False
    if not series_dict:
        raise ValueError("No data available for ratio series plot")
        ax.set_title(title)
        ax.set_xlabel("Blend ratio (random → similarity)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    for (mode_name, dtype_name, k_value, anchor_blend, anchor_count, support_label), points in series_dict.items():
        points.sort(key=lambda item: item["ratio"])
        ratios = [p["ratio"] for p in points]
        means = [p["mean"] for p in points]
        errs = [p["stderr"] for p in points]
        stds = [abs(p.get("std", 0.0)) if p.get("std") is not None else 0.0 for p in points]
        n_value = points[0].get("n") if points else None
        label = (
            f"{mode_name}|{dtype_name}|k={k_value}|blend={anchor_blend:.2f}|"
            f"clusters={anchor_count}|support_thr={support_label}|n={n_value}"
        )
        container = ax.errorbar(ratios, means, yerr=errs, marker="o", capsize=4, label=label)
        if include_std:
            lower = [m - s if not math.isnan(s) else m for m, s in zip(means, stds)]
            upper = [m + s if not math.isnan(s) else m for m, s in zip(means, stds)]
            color = container.lines[0].get_color() if container.lines else None
            ax.fill_between(ratios, lower, upper, color=color, alpha=0.15)
        anchor_vals = anchor_dict.get((mode_name, dtype_name, k_value, anchor_blend, anchor_count, support_label))
        if anchor_vals:
            color = container.lines[0].get_color() if container.lines else None
            ax.scatter(
                [item[0] for item in anchor_vals],
                [item[1] for item in anchor_vals],
                alpha=0.3,
                s=15,
                color=color,
            )
        if show_entropy:
            entropy_values = [p.get("entropy") for p in points if p.get("entropy") is not None and not math.isnan(p.get("entropy"))]
            if entropy_values:
                ax.plot(
                    ratios,
                    entropy_values,
                    linestyle="--",
                    color="red",
                    alpha=0.5,
                    label="Mean entropy" if not entropy_label_added else None,
                )
                entropy_label_added = True
    ax.set_title(title)
    ax.set_xlabel("Blend ratio (random → similarity)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize="small", ncol=1, frameon=False)
    fig.tight_layout()
    return fig


def plot_blend_series(
    series_dict,
    title: str,
    ylabel: str,
    include_std: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not series_dict:
        raise ValueError("No data available for blend series plot")
        ax.set_title(title)
        ax.set_xlabel("Anchor blend (large clusters → small clusters)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    for (mode_name, dtype_name, k_value, anchor_count, support_label), points in series_dict.items():
        points.sort(key=lambda item: item["blend"])
        blends = [p["blend"] for p in points]
        means = [p["mean"] for p in points]
        errs = [p["stderr"] for p in points]
        stds = [abs(p.get("std", 0.0)) if p.get("std") is not None else 0.0 for p in points]
        n_value = points[0].get("n") if points else None
        label = (
            f"{mode_name}|{dtype_name}|k={k_value}|clusters={anchor_count}|"
            f"support_thr={support_label}|n={n_value}"
        )
        line = ax.errorbar(blends, means, yerr=errs, marker="o", capsize=4, label=label)
        if include_std:
            color = line.lines[0].get_color() if line.lines else None
            lower = [m - s if not math.isnan(s) else m for m, s in zip(means, stds)]
            upper = [m + s if not math.isnan(s) else m for m, s in zip(means, stds)]
            ax.fill_between(blends, lower, upper, color=color, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Anchor blend (large clusters → small clusters)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize="small", ncol=1, frameon=False)
    fig.tight_layout()
    return fig

    
def run_experiment(settings: ExperimentSettings) -> Optional[wandb.sdk.wandb_run.Run]:
    run = init_wandb(settings)
    tokenizer = load_tokenizer(settings)

    device = torch.device(settings.device)
    valid_token_ids = collect_valid_token_ids(tokenizer)
    if settings.max_similarity_candidates and settings.max_similarity_candidates < len(valid_token_ids):
        candidate_ids = valid_token_ids[: settings.max_similarity_candidates]
    else:
        candidate_ids = valid_token_ids

    dictionary_pool: Optional[CandidatePool] = None
    dictionary_file_path: Optional[Path] = None
    if settings.candidate_source == "dictionary":
        span_tokenizer, span_model = load_span_encoder(settings, device)
        if span_tokenizer is None or span_model is None:
            raise ValueError("Failed to load span encoder for dictionary candidates")
        dictionary_file_path = resolve_dictionary_path(settings)
        settings.dictionary_path = dictionary_file_path
        dictionary_pool = build_dictionary_candidate_pool(
            settings,
            dictionary_file_path,
            tokenizer,
            span_tokenizer,
            span_model,
            device,
        )
        maybe_apply_anchor_frequency_weights(settings, tokenizer, dictionary_pool)

    logged_candidate_tables: Set[Tuple[str, str, int]] = set()

    def compute_stats(values: Sequence[float]) -> Tuple[float, float, float]:
        filtered = [v for v in values if v is not None and not math.isnan(v)]
        if not filtered:
            return float("nan"), 0.0, 0.0
        arr = np.array(filtered, dtype=np.float64)
        mean_value = float(arr.mean())
        if arr.size > 1:
            std_value = float(arr.std(ddof=1))
        else:
            std_value = 0.0
        stderr_value = float(std_value / math.sqrt(arr.size)) if arr.size else 0.0
        return mean_value, std_value, stderr_value

    for dtype_name in tqdm(settings.dtypes, desc="dtypes", dynamic_ncols=True):
        dtype_info = resolve_dtype(dtype_name)
        dtype_device = torch.device("cpu") if dtype_info.torch_dtype == torch.float64 and device.type == "cuda" else device

        models_to_run: List[Tuple[str, AutoModelForCausalLM]] = []
        if settings.include_pretrained:
            models_to_run.append(("pretrained", load_pretrained_model(settings, dtype_info, dtype_device)))
        if settings.include_random_init:
            models_to_run.append(("random_init", load_random_initialised_model(settings, dtype_info, dtype_device)))

        for mode_name, model in tqdm(models_to_run, desc=f"{dtype_name} modes", dynamic_ncols=True, leave=False):
            if settings.candidate_source == "token":
                base_candidate_pool = build_token_candidate_pool(model, tokenizer, candidate_ids)
            else:
                if dictionary_pool is None:
                    raise ValueError("Dictionary candidate pool not initialised")
                base_candidate_pool = dictionary_pool

            for support_threshold in settings.support_similarity_threshold_options:
                settings.support_similarity_threshold = support_threshold

                for anchor_count in settings.anchor_cluster_count_options:
                    settings.anchor_cluster_count = anchor_count

                    if settings.candidate_source == "token":
                        candidate_pool = base_candidate_pool
                        cache_path = None
                        if anchor_count > 0:
                            cache_path = build_cluster_cache_path(settings, mode_name, dtype_name)
                        initialize_candidate_clusters(
                            candidate_pool,
                            anchor_count,
                            settings.anchor_cluster_seed,
                            cache_path=cache_path,
                        )
                    else:
                        candidate_pool = base_candidate_pool
                        initialize_candidate_clusters(
                            candidate_pool,
                            anchor_count,
                            settings.anchor_cluster_seed,
                        )

                    table_key = (mode_name, dtype_name, anchor_count)
                    if table_key not in logged_candidate_tables:
                        log_candidate_embedding_table(
                            candidate_pool,
                            mode_name,
                            dtype_name,
                            anchor_count,
                            settings,
                        )
                        logged_candidate_tables.add(table_key)

                    for distribution_type in tqdm(
                        settings.distribution_types,
                        desc=f"{mode_name} distributions",
                        dynamic_ncols=True,
                        leave=False,
                    ):
                        k_list = [1] if distribution_type == "one_hot" else settings.k_values

                        for k in tqdm(k_list, desc=f"{distribution_type} k", dynamic_ncols=True, leave=False):
                            for anchor_blend in settings.anchor_cluster_blends:
                                ratio_aggregated_logs: List[Dict[str, object]] = []
                                for ratio in tqdm(
                                    settings.ratios,
                                    desc=f"ratio sweep (k={k})",
                                    dynamic_ncols=True,
                                    leave=False,
                                ):
                                    scalar_records: List[Dict[str, float]] = []
                                    last_num_distributions: Optional[int] = None
                                    anchor_indices: Iterable[int] = range(settings.num_anchor_sets)
                                    if settings.num_anchor_sets > 1:
                                        anchor_indices = tqdm(
                                            anchor_indices,
                                            desc=(
                                                f"anchor sets (k={k}, ratio={ratio:.2f}, blend={anchor_blend:.2f})"
                                            ),
                                            dynamic_ncols=True,
                                            leave=False,
                                        )

                                    for anchor_set_index in anchor_indices:
                                        seed_offset = settings.anchor_seed_stride * int(anchor_set_index)
                                        rng_seed = settings.random_seed + seed_offset
                                        rng = np.random.default_rng(rng_seed)
                                        blueprints = build_blueprints(
                                            settings=settings,
                                            rng=rng,
                                            pool=candidate_pool,
                                            k=k,
                                            ratio=ratio,
                                            anchor_cluster_blend=anchor_blend,
                                            support_top_p=settings.support_top_p,
                                            support_similarity_threshold=settings.support_similarity_threshold,
                                        )

                                stacked, metadata = collect_next_token_distributions(
                                    model=model,
                                    tokenizer=tokenizer,
                                    settings=settings,
                                    blueprints=blueprints,
                                    device=dtype_device,
                                    anchor_set_index=int(anchor_set_index),
                                )

                                support_similarity_values: List[float] = []
                                for blueprint in blueprints:
                                    sims = blueprint.support_similarities
                                    if len(sims) > 1:
                                        support_similarity_values.append(float(np.mean(sims[1:])))
                                    elif sims:
                                        support_similarity_values.append(float(sims[0]))
                                mean_support_similarity = (
                                    float(np.mean(support_similarity_values))
                                    if support_similarity_values
                                    else float("nan")
                                )

                                metrics = compute_cross_entropy_metrics(stacked, dtype_info)
                                metrics["mean_support_similarity"] = mean_support_similarity
                                scalar_records.append(
                                    {field: float(metrics[field]) for field in SCALAR_METRIC_FIELDS}
                                )
                                last_num_distributions = stacked.shape[0]

                                heatmap_ce = maybe_downsample_matrix(
                                    metrics["cross_entropy_matrix"], settings.heatmap_max_points
                                )
                                heatmap_sym_kl = maybe_downsample_matrix(
                                    metrics["symmetric_kl_matrix"], settings.heatmap_max_points
                                )
                                heatmap_wasserstein = maybe_downsample_matrix(
                                    metrics["wasserstein_matrix"], settings.heatmap_max_points
                                )

                                heatmap_context = (
                                    "mode={mode}, dtype={dtype}, k={k}, ratio={ratio:.2f}, dist={dist}, "
                                    "N={num_sentences}, M={seq_len}, P={num_generations}, anchorset={anchor_set}, "
                                    "candidate_source={candidate_source}, blend={blend:.2f}, clusters={clusters}, "
                                    "support_thr={support_thr}"
                                ).format(
                                    mode=mode_name,
                                    dtype=dtype_info.name,
                                    k=k,
                                    ratio=ratio,
                                    dist=distribution_type,
                                    num_sentences=settings.num_sentences,
                                    seq_len=settings.seq_length,
                                    num_generations=settings.num_generations,
                                    anchor_set=anchor_set_index,
                                    candidate_source=settings.candidate_source,
                                    blend=anchor_blend,
                                    clusters=settings.anchor_cluster_count,
                                    support_thr=(
                                        "None"
                                        if settings.support_similarity_threshold is None
                                        else f"{settings.support_similarity_threshold:.4f}".rstrip("0").rstrip(".")
                                    ),
                                )

                                ce_fig = plot_heatmap(
                                    heatmap_ce,
                                    title="Cross-Entropy Heatmap",
                                    subtitle=heatmap_context,
                                )
                                skl_fig = plot_heatmap(
                                    heatmap_sym_kl,
                                    title="Symmetric KL Heatmap",
                                    subtitle=heatmap_context,
                                )
                                skl_violin_fig: Optional[plt.Figure] = None
                                if settings.plot_symmetric_kl_violin:
                                    skl_violin_fig = plot_symmetric_kl_violin(
                                        heatmap_sym_kl,
                                        title="Symmetric KL Violin Plot",
                                        subtitle=heatmap_context,
                                    )
                                wasserstein_fig = plot_heatmap(
                                    heatmap_wasserstein,
                                    title="Wasserstein Distance Heatmap",
                                    subtitle=heatmap_context,
                                )

                                log_payload = {
                                    "mode": mode_name,
                                    "dtype": dtype_info.name,
                                    "k": k,
                                    "ratio": ratio,
                                    "distribution_type": distribution_type,
                                    "candidate_source": settings.candidate_source,
                                    "embedding_mode": settings.embedding_mode,
                                    "anchor_cluster_count": settings.anchor_cluster_count,
                                    "anchor_cluster_count_options": settings.anchor_cluster_count_options,
                                    "anchor_cluster_blend": anchor_blend,
                                    "top_p_sampling": settings.top_p_sampling,
                                    "support_top_p": settings.support_top_p,
                                    "support_similarity_threshold": settings.support_similarity_threshold,
                                    "support_similarity_threshold_options": settings.support_similarity_threshold_options,
                                    "anchor_set_index": int(anchor_set_index),
                                    "mean_cross_entropy": metrics["mean_cross_entropy"],
                                    "mean_kl": metrics["mean_kl"],
                                    "mean_symmetric_kl": metrics["mean_symmetric_kl"],
                                    "mean_wasserstein": metrics["mean_wasserstein"],
                                    "max_symmetric_kl": metrics["max_symmetric_kl"],
                                    "max_wasserstein": metrics["max_wasserstein"],
                                    "fraction_symmetric_kl_below_epsilon": metrics[
                                        "fraction_symmetric_kl_below_epsilon"
                                    ],
                                    "mean_support_similarity": metrics["mean_support_similarity"],
                                    "mean_entropy": metrics["mean_entropy"],
                                    "dtype_epsilon": metrics["dtype_epsilon"],
                                    "num_distributions": stacked.shape[0],
                                    "cross_entropy_heatmap": wandb.Image(
                                        ce_fig,
                                        caption=f"Cross-Entropy | {heatmap_context}"
                                    ),
                                    "symmetric_kl_heatmap": wandb.Image(
                                        skl_fig,
                                        caption=f"Symmetric KL | {heatmap_context}"
                                    ),
                                    **(
                                        {
                                            "symmetric_kl_violin": wandb.Image(
                                                skl_violin_fig,
                                                caption=f"Symmetric KL Violin | {heatmap_context}",
                                            )
                                        }
                                        if skl_violin_fig is not None
                                        else {}
                                    ),
                                    "wasserstein_heatmap": wandb.Image(
                                        wasserstein_fig,
                                        caption=f"Wasserstein Distance | {heatmap_context}"
                                    ),
                                }
                                if dictionary_file_path is not None:
                                    log_payload["dictionary_path"] = str(dictionary_file_path)
                                if settings.anchor_frequency_text is not None:
                                    log_payload["anchor_frequency_text"] = str(settings.anchor_frequency_text)
                                wandb.log(log_payload)

                                ce_fig.clf()
                                skl_fig.clf()
                                wasserstein_fig.clf()
                                plt.close(ce_fig)
                                plt.close(skl_fig)
                                if skl_violin_fig is not None:
                                    skl_violin_fig.clf()
                                    plt.close(skl_violin_fig)
                                plt.close(wasserstein_fig)

                                metadata_table = wandb.Table(
                                    columns=[
                                        "mode",
                                        "dtype",
                                        "k",
                                        "ratio",
                                        "anchor_cluster_blend",
                                        "anchor_set_index",
                                        "sample_id",
                                        "generation_id",
                                        "anchor_key",
                                        "anchor_text",
                                        "anchor_input_ids",
                                        "support_keys",
                                        "support_texts",
                                        "support_input_ids",
                                        "support_probabilities",
                                        "support_similarities",
                                        "chosen_support_index",
                                    ]
                                )
                                for row in metadata:
                                    metadata_table.add_data(
                                        mode_name,
                                        dtype_info.name,
                                        k,
                                        ratio,
                                        anchor_blend,
                                        anchor_set_index,
                                        row["sample_id"],
                                        row["generation_id"],
                                        row["anchor_key"],
                                        row["anchor_text"],
                                        row["anchor_input_ids"],
                                        row["support_keys"],
                                        row["support_texts"],
                                        row["support_input_ids"],
                                        row["support_probabilities"],
                                        row["support_similarities"],
                                        row["chosen_support_index"],
                                    )

                                metadata_log = {
                                    "distribution_metadata": metadata_table,
                                    "anchor_cluster_count": settings.anchor_cluster_count,
                                    "anchor_cluster_count_options": settings.anchor_cluster_count_options,
                                    "top_p_sampling": settings.top_p_sampling,
                                    "support_top_p": settings.support_top_p,
                                    "support_similarity_threshold": settings.support_similarity_threshold,
                                    "support_similarity_threshold_options": settings.support_similarity_threshold_options,
                                }
                                if dictionary_file_path is not None:
                                    metadata_log["dictionary_path"] = str(dictionary_file_path)
                                if settings.anchor_frequency_text is not None:
                                    metadata_log["anchor_frequency_text"] = str(settings.anchor_frequency_text)
                                wandb.log(metadata_log)

                                for blueprint in blueprints:
                                    support_fig = plot_anchor_support_bar(
                                        blueprint=blueprint,
                                        mode_name=mode_name,
                                        dtype_name=dtype_info.name,
                                        k=k,
                                        ratio=ratio,
                                        distribution_type=distribution_type,
                                        anchor_set_index=int(anchor_set_index),
                                    )

                                    wandb.log(
                                        {
                                            "anchor_support_bar": wandb.Image(
                                                support_fig,
                                                caption=(
                                                    f"Anchor support | mode={mode_name}, dtype={dtype_info.name}, "
                                                    f"k={k}, ratio={ratio:.2f}, dist={distribution_type}, "
                                                    f"anchor_set={anchor_set_index}, blend={anchor_blend:.2f}, "
                                                    f"sample={blueprint.sample_id}"
                                                ),
                                            ),
                                            "mode": mode_name,
                                            "dtype": dtype_info.name,
                                            "k": k,
                                            "ratio": ratio,
                                            "distribution_type": distribution_type,
                                            "anchor_cluster_count": settings.anchor_cluster_count,
                                            "anchor_cluster_count_options": settings.anchor_cluster_count_options,
                                            "anchor_cluster_blend": anchor_blend,
                                            "top_p_sampling": settings.top_p_sampling,
                                            "support_top_p": settings.support_top_p,
                                            "support_similarity_threshold": settings.support_similarity_threshold,
                                            "support_similarity_threshold_options": settings.support_similarity_threshold_options,
                                            "anchor_set_index": int(anchor_set_index),
                                            "sample_id": blueprint.sample_id,
                                            "anchor_key": blueprint.anchor_key,
                                            "anchor_text": blueprint.anchor_text,
                                            "candidate_source": settings.candidate_source,
                                            "dictionary_path": str(dictionary_file_path)
                                            if dictionary_file_path is not None
                                            else None,
                                            "anchor_frequency_text": str(settings.anchor_frequency_text)
                                            if settings.anchor_frequency_text is not None
                                            else None,
                                        }
                                    )

                                    support_fig.clf()
                                    plt.close(support_fig)

                        if scalar_records:
                            sample_count = len(scalar_records)
                            aggregated_log = {
                                "mode": mode_name,
                                "dtype": dtype_info.name,
                                "k": k,
                                "ratio": ratio,
                                "distribution_type": distribution_type,
                                "anchor_set_index": "aggregate",
                                "num_anchor_sets": settings.num_anchor_sets,
                                "dtype_epsilon": dtype_info.epsilon,
                                "num_distributions": last_num_distributions,
                                "candidate_source": settings.candidate_source,
                                "embedding_mode": settings.embedding_mode,
                                "anchor_cluster_count": settings.anchor_cluster_count,
                                "anchor_cluster_count_options": settings.anchor_cluster_count_options,
                                "anchor_cluster_blend": anchor_blend,
                                "top_p_sampling": settings.top_p_sampling,
                                "support_top_p": settings.support_top_p,
                                "support_similarity_threshold": settings.support_similarity_threshold,
                                "support_similarity_threshold_options": settings.support_similarity_threshold_options,
                                "sample_count": sample_count,
                            }
                            for field in SCALAR_METRIC_FIELDS:
                                values = [record[field] for record in scalar_records]
                                mean_value, std_value, stderr_value = compute_stats(values)
                                aggregated_log[field] = mean_value
                                aggregated_log[f"{field}_mean"] = mean_value
                                aggregated_log[f"{field}_std"] = std_value
                                aggregated_log[f"{field}_stderr"] = stderr_value
                            if dictionary_file_path is not None:
                                aggregated_log["dictionary_path"] = str(dictionary_file_path)
                            if settings.anchor_frequency_text is not None:
                                aggregated_log["anchor_frequency_text"] = str(settings.anchor_frequency_text)
                            wandb.log(aggregated_log)
                            ratio_aggregated_logs.append(dict(aggregated_log))

                        if ratio_aggregated_logs:
                            blend_log = {
                                "mode": mode_name,
                                "dtype": dtype_info.name,
                                "k": k,
                                "ratio": "aggregate",
                                "distribution_type": distribution_type,
                                "anchor_set_index": "aggregate_ratio",
                                "anchor_cluster_blend": anchor_blend,
                                "candidate_source": settings.candidate_source,
                                "embedding_mode": settings.embedding_mode,
                                "anchor_cluster_count": settings.anchor_cluster_count,
                                "anchor_cluster_count_options": settings.anchor_cluster_count_options,
                                "top_p_sampling": settings.top_p_sampling,
                                "support_top_p": settings.support_top_p,
                                "support_similarity_threshold": settings.support_similarity_threshold,
                                "support_similarity_threshold_options": settings.support_similarity_threshold_options,
                                "sample_count": len(ratio_aggregated_logs),
                                "num_anchor_sets": settings.num_anchor_sets,
                            }
                            for field in SCALAR_METRIC_FIELDS:
                                values = [rec.get(field) for rec in ratio_aggregated_logs]
                                mean_value, std_value, stderr_value = compute_stats(values)
                                blend_log[field] = mean_value
                                blend_log[f"{field}_mean"] = mean_value
                                blend_log[f"{field}_std"] = std_value
                                blend_log[f"{field}_stderr"] = stderr_value
                            if dictionary_file_path is not None:
                                blend_log["dictionary_path"] = str(dictionary_file_path)
                            if settings.anchor_frequency_text is not None:
                                blend_log["anchor_frequency_text"] = str(settings.anchor_frequency_text)
                            wandb.log(blend_log)

    return run

def main(argv: Optional[Sequence[str]] = None) -> None:
    settings = parse_args(argv)
    torch.manual_seed(settings.random_seed)
    np.random.seed(settings.random_seed)

    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
    run_path: Optional[str] = settings.wandb_source_run

    if settings.wandb_action in {"log", "both"}:
        wandb_run = run_experiment(settings)
        if wandb_run is not None and getattr(wandb_run, "path", None):
            run_path = wandb_run.path

    if settings.wandb_action in {"reload", "both"}:
        if not run_path:
            raise ValueError(
                "--wandb-source-run must be provided when reloading without a freshly logged run."
            )
        (
            ce_fig,
            frac_fig,
            skl_fig,
            wasserstein_fig,
            blend_ce_fig,
            blend_skl_fig,
            blend_wasserstein_fig,
            blend_sim_fig,
            facet_ce_fig,
            facet_skl_fig,
            facet_wasserstein_fig,
            history_df,
        ) = reload_and_plot(settings, run_path)
        plot_infos = [
            ("cross_entropy", ce_fig, "ratio_cross_entropy_curve"),
            ("fraction_within_epsilon", frac_fig, "ratio_fraction_within_epsilon"),
            ("mean_symmetric_kl", skl_fig, "ratio_mean_symmetric_kl"),
            ("anchor_blend_cross_entropy", blend_ce_fig, "anchor_blend_cross_entropy"),
            ("anchor_blend_mean_symmetric_kl", blend_skl_fig, "anchor_blend_mean_symmetric_kl"),
            ("anchor_blend_support_similarity", blend_sim_fig, "anchor_blend_support_similarity"),
        ]
        if wasserstein_fig is not None:
            plot_infos.append(("mean_wasserstein", wasserstein_fig, "ratio_mean_wasserstein"))
        if blend_wasserstein_fig is not None:
            plot_infos.append(
                ("anchor_blend_mean_wasserstein", blend_wasserstein_fig, "anchor_blend_mean_wasserstein")
            )
        if facet_ce_fig is not None:
            plot_infos.append(("facet_cross_entropy", facet_ce_fig, "facet_cross_entropy"))
        if facet_skl_fig is not None:
            plot_infos.append(("facet_mean_symmetric_kl", facet_skl_fig, "facet_mean_symmetric_kl"))
        if facet_wasserstein_fig is not None:
            plot_infos.append(("facet_mean_wasserstein", facet_wasserstein_fig, "facet_mean_wasserstein"))

        base_path = Path(settings.plot_output_path)
        base_dir = base_path.parent
        base_dir.mkdir(parents=True, exist_ok=True)

        for suffix, fig, _ in plot_infos:
            png_path = base_dir / f"{base_path.stem}_{suffix}.png"
            pdf_path = base_dir / f"{base_path.stem}_{suffix}.pdf"
            fig.savefig(png_path, dpi=200)
            fig.savefig(pdf_path, format="pdf")

        csv_path: Optional[Path] = None
        if history_df is not None and not history_df.empty:
            csv_path = base_dir / f"{base_path.stem}_aggregated_metrics.csv"
            history_df.to_csv(csv_path, index=False)

        if wandb_run is not None and hasattr(wandb_run, "log"):
            wandb_payload = {key: wandb.Image(fig) for _, fig, key in plot_infos}
            wandb_run.log(wandb_payload)
            if history_df is not None and not history_df.empty:
                table_df = history_df.copy()
                if pd is not None and "anchor_set_index" in table_df.columns:
                    table_df["anchor_set_index"] = pd.to_numeric(
                        table_df["anchor_set_index"], errors="coerce"
                    )
                wandb_run.log({"aggregated_metrics_table": wandb.Table(dataframe=table_df)})
        else:
            for suffix, _, _ in plot_infos:
                print(
                    f"Saved plot to {base_path.stem}_{suffix}.png and .pdf"
                )
            if csv_path is not None:
                print(f"Saved aggregated metrics to {csv_path}")

        for _, fig, _ in plot_infos:
            plt.close(fig)

    if wandb_run is not None and hasattr(wandb_run, "finish"):
        wandb_run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
