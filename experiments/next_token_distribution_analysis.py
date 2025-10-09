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
    "max_symmetric_kl",
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
    embedding_mode: str
    bert_model_name: Optional[str]
    bert_layer_index: int
    span_encoder_batch_size: int
    span_encoder_max_length: int
    similarity_pool_size: int
    max_similarity_candidates: int
    support_top_p: float
    support_similarity_threshold: Optional[float]
    heatmap_max_points: Optional[int]
    offline: bool
    pad_token_fallback_to_eos: bool
    plot_output_path: str
    anchor_cluster_count: int
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
        default=0,
        help="Number of k-means clusters over candidate embeddings (0 disables clustering).",
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
        default=None,
        help="Minimum cosine similarity required for support candidates; use 'None' to disable.",
    )
    parser.add_argument(
        "--heatmap-max-points",
        type=int,
        default=256,
        help="Optional cap on heatmap resolution (number of distributions).",
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
    support_top_p = max(0.0, min(1.0, args.support_top_p))
    support_similarity_threshold = (
        float(args.support_similarity_threshold)
        if args.support_similarity_threshold is not None
        else None
    )

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
        embedding_mode=args.embedding_mode,
        bert_model_name=args.bert_model_name,
        bert_layer_index=args.bert_layer_index,
        span_encoder_batch_size=max(1, args.span_encoder_batch_size),
        span_encoder_max_length=max(1, args.span_encoder_max_length),
        similarity_pool_size=max(1, args.similarity_pool_size),
        max_similarity_candidates=max(1, args.max_similarity_candidates),
        support_top_p=support_top_p,
        support_similarity_threshold=support_similarity_threshold,
        heatmap_max_points=heatmap_cap,
        offline=offline,
        pad_token_fallback_to_eos=args.allow_pad_fallback,
        plot_output_path=args.plot_output_path,
        anchor_cluster_count=max(0, args.anchor_cluster_count),
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
            "heatmap_max_points": settings.heatmap_max_points,
            "include_pretrained": settings.include_pretrained,
            "include_random_init": settings.include_random_init,
            "candidate_source": settings.candidate_source,
            "embedding_mode": settings.embedding_mode,
            "dictionary_path": str(settings.dictionary_path) if settings.dictionary_path else None,
            "dictionary_url": settings.dictionary_url,
            "bert_model_name": settings.bert_model_name,
            "bert_layer_index": settings.bert_layer_index,
            "span_encoder_batch_size": settings.span_encoder_batch_size,
            "span_encoder_max_length": settings.span_encoder_max_length,
            "anchor_cluster_count": settings.anchor_cluster_count,
            "anchor_cluster_blends": settings.anchor_cluster_blends,
            "anchor_cluster_seed": settings.anchor_cluster_seed,
            "anchor_cluster_cache_dir": str(settings.anchor_cluster_cache_dir),
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
    settings: ExperimentSettings,
) -> None:
    columns = [
        "mode",
        "dtype",
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
    if sizes.sum() == 0:
        return int(rng.integers(0, len(sizes)))
    weights_large = sizes / sizes.sum()
    inv = 1.0 / (sizes + 1e-9)
    weights_small = inv / inv.sum()
    weights = (1.0 - anchor_cluster_blend) * weights_large + anchor_cluster_blend * weights_small
    weights = weights / weights.sum()
    cluster_idx = int(rng.choice(len(sizes), p=weights))
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

    if (
        anchor_cluster_count > 0
        and pool.cluster_assignments is not None
        and pool.cluster_to_indices is not None
        and len(pool.cluster_to_indices) >= anchor_cluster_count
    ):
        cluster_idx = select_cluster_index(rng, pool, anchor_cluster_blend)
        cluster_candidates = pool.cluster_to_indices[cluster_idx]
        anchor_index = int(rng.choice(cluster_candidates))
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

    dtype_eps = dtype_info.epsilon
    matches = (sym_kl.abs() <= dtype_eps).float()
    fraction_below_eps = matches.mean().item()

    metrics = {
        "cross_entropy_matrix": cross_entropy_matrix.cpu(),
        "kl_matrix": kl_matrix.cpu(),
        "symmetric_kl_matrix": sym_kl.cpu(),
        "mean_cross_entropy": cross_entropy_matrix.mean().item(),
        "mean_kl": kl_matrix.mean().item(),
        "mean_symmetric_kl": sym_kl.mean().item(),
        "max_symmetric_kl": sym_kl.abs().max().item(),
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


def reload_and_plot(
    settings: ExperimentSettings,
    run_path: str,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure, plt.Figure, plt.Figure]:
    api = wandb.Api()
    target_run = api.run(run_path)
    history = target_run.scan_history(
        keys=[
            "mode",
            "dtype",
            "k",
            "ratio",
            "anchor_cluster_blend",
            "anchor_set_index",
            "mean_cross_entropy",
            "mean_cross_entropy_stderr",
            "fraction_symmetric_kl_below_epsilon",
            "fraction_symmetric_kl_below_epsilon_stderr",
            "mean_symmetric_kl",
            "mean_symmetric_kl_stderr",
            "mean_support_similarity",
            "mean_support_similarity_stderr",
        ]
    )

    ce_series = defaultdict(list)
    frac_series = defaultdict(list)
    skl_series = defaultdict(list)
    ce_anchor_points = defaultdict(list)
    frac_anchor_points = defaultdict(list)
    skl_anchor_points = defaultdict(list)
    blend_ce = defaultdict(list)
    blend_skl = defaultdict(list)
    blend_sim = defaultdict(list)

    for row in history:
        mode_name = row.get("mode", "unknown")
        dtype_name = row.get("dtype", "unknown")
        k_value = row.get("k", "unknown")
        anchor_blend = float(row.get("anchor_cluster_blend", 0.0) or 0.0)
        anchor_identifier = row.get("anchor_set_index")
        ratio_value = row.get("ratio")

        mean_ce = row.get("mean_cross_entropy")
        ce_stderr = row.get("mean_cross_entropy_stderr")
        frac_value = row.get("fraction_symmetric_kl_below_epsilon")
        frac_stderr = row.get("fraction_symmetric_kl_below_epsilon_stderr")
        mean_skl = row.get("mean_symmetric_kl")
        skl_stderr = row.get("mean_symmetric_kl_stderr")
        mean_sim = row.get("mean_support_similarity")
        sim_stderr = row.get("mean_support_similarity_stderr")
        mean_entropy = row.get("mean_entropy")
        entropy_stderr = row.get("mean_entropy_stderr")
        sample_size = int(row.get("sample_count") or row.get("num_anchor_sets") or 0)
        if sample_size == 0:
            sample_size = 1

        key_ratio = (mode_name, dtype_name, k_value, anchor_blend)
        key_blend = (mode_name, dtype_name, k_value)

        if isinstance(ratio_value, str) and ratio_value == "aggregate" and anchor_identifier == "aggregate_ratio":
            if mean_ce is not None:
                blend_ce[key_blend].append(
                    {
                        "blend": anchor_blend,
                        "mean": float(mean_ce),
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

        if not isinstance(ratio_value, (int, float)):
            continue
        ratio_float = float(ratio_value)

        if anchor_identifier in (None, "aggregate"):
            if mean_ce is not None:
                ce_series[key_ratio].append(
                    {
                        "ratio": ratio_float,
                        "mean": float(mean_ce),
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
        else:
            if mean_ce is not None:
                ce_anchor_points[key_ratio].append((ratio_float, float(mean_ce)))
            if frac_value is not None:
                frac_anchor_points[key_ratio].append((ratio_float, float(frac_value)))
            if mean_skl is not None:
                skl_anchor_points[key_ratio].append((ratio_float, float(mean_skl)))

    if not ce_series:
        raise ValueError(f"No ratio/mean_cross_entropy data found in run {run_path}")

    def plot_ratio_series(series_dict, anchor_dict, title, ylabel, show_entropy: bool = False):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        entropy_label_added = False
        if not series_dict:
            ax.set_title(title)
            ax.set_xlabel("Blend ratio (random → similarity)")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            fig.tight_layout()
            return fig

        for (mode_name, dtype_name, k_value, anchor_blend), points in series_dict.items():
            points.sort(key=lambda item: item["ratio"])
            ratios = [p["ratio"] for p in points]
            means = [p["mean"] for p in points]
            errs = [p["stderr"] for p in points]
            n_value = points[0].get("n") if points else None
            label = f"{mode_name}|{dtype_name}|k={k_value}|blend={anchor_blend:.2f}|n={n_value}"
            container = ax.errorbar(ratios, means, yerr=errs, marker="o", capsize=4, label=label)
            anchor_vals = anchor_dict.get((mode_name, dtype_name, k_value, anchor_blend))
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

    ce_fig = plot_ratio_series(
        ce_series,
        ce_anchor_points,
        "Blend Ratio vs Mean Cross-Entropy",
        "Mean cross-entropy",
        show_entropy=True,
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

    def plot_blend_series(series_dict, title: str, ylabel: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if not series_dict:
            ax.set_title(title)
            ax.set_xlabel("Anchor blend (large clusters → small clusters)")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            fig.tight_layout()
            return fig

        for (mode_name, dtype_name, k_value), points in series_dict.items():
            points.sort(key=lambda item: item["blend"])
            blends = [p["blend"] for p in points]
            means = [p["mean"] for p in points]
            errs = [p["stderr"] for p in points]
            n_value = points[0].get("n") if points else None
            label = f"{mode_name}|{dtype_name}|k={k_value}|n={n_value}"
            ax.errorbar(blends, means, yerr=errs, marker="o", capsize=4, label=label)
        ax.set_title(title)
        ax.set_xlabel("Anchor blend (large clusters → small clusters)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize="small", ncol=1, frameon=False)
        fig.tight_layout()
        return fig

    blend_ce_fig = plot_blend_series(
        blend_ce,
        "Anchor Blend vs Mean Cross-Entropy",
        "Mean cross-entropy",
    )
    blend_skl_fig = plot_blend_series(
        blend_skl,
        "Anchor Blend vs Mean Symmetric KL",
        "Mean symmetric KL",
    )
    blend_sim_fig = plot_blend_series(
        blend_sim,
        "Anchor Blend vs Mean Support Similarity",
        "Mean support similarity",
    )

    return ce_fig, frac_fig, skl_fig, blend_ce_fig, blend_skl_fig, blend_sim_fig


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
        if settings.anchor_cluster_count > 0:
            initialize_candidate_clusters(
                dictionary_pool,
                settings.anchor_cluster_count,
                settings.anchor_cluster_seed,
            )

    logged_candidate_tables: Set[Tuple[str, str]] = set()

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
                candidate_pool = build_token_candidate_pool(model, tokenizer, candidate_ids)
                cache_path = None
                if settings.anchor_cluster_count > 0:
                    cache_path = build_cluster_cache_path(settings, mode_name, dtype_name)
                    initialize_candidate_clusters(
                        candidate_pool,
                        settings.anchor_cluster_count,
                        settings.anchor_cluster_seed,
                        cache_path=cache_path,
                    )
                table_key = (mode_name, dtype_name)
                if table_key not in logged_candidate_tables:
                    log_candidate_embedding_table(candidate_pool, mode_name, dtype_name, settings)
                    logged_candidate_tables.add(table_key)
            else:
                if dictionary_pool is None:
                    raise ValueError("Dictionary candidate pool not initialised")
                candidate_pool = dictionary_pool

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

                                heatmap_context = (
                                    "mode={mode}, dtype={dtype}, k={k}, ratio={ratio:.2f}, dist={dist}, "
                                    "N={num_sentences}, M={seq_len}, P={num_generations}, anchorset={anchor_set}, "
                                    "candidate_source={candidate_source}, blend={blend:.2f}"
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

                                log_payload = {
                                    "mode": mode_name,
                                    "dtype": dtype_info.name,
                                    "k": k,
                                    "ratio": ratio,
                                    "distribution_type": distribution_type,
                                    "candidate_source": settings.candidate_source,
                                    "embedding_mode": settings.embedding_mode,
                                "anchor_cluster_blend": anchor_blend,
                                "support_top_p": settings.support_top_p,
                                "anchor_set_index": int(anchor_set_index),
                                    "mean_cross_entropy": metrics["mean_cross_entropy"],
                                    "mean_kl": metrics["mean_kl"],
                                    "mean_symmetric_kl": metrics["mean_symmetric_kl"],
                                    "max_symmetric_kl": metrics["max_symmetric_kl"],
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
                                }
                                if dictionary_file_path is not None:
                                    log_payload["dictionary_path"] = str(dictionary_file_path)
                                wandb.log(log_payload)

                                ce_fig.clf()
                                skl_fig.clf()
                                plt.close(ce_fig)
                                plt.close(skl_fig)

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

                                metadata_log = {"distribution_metadata": metadata_table}
                                if dictionary_file_path is not None:
                                    metadata_log["dictionary_path"] = str(dictionary_file_path)
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
                                            "anchor_cluster_blend": anchor_blend,
                                            "anchor_set_index": int(anchor_set_index),
                                            "sample_id": blueprint.sample_id,
                                            "anchor_key": blueprint.anchor_key,
                                            "anchor_text": blueprint.anchor_text,
                                            "candidate_source": settings.candidate_source,
                                            "dictionary_path": str(dictionary_file_path)
                                            if dictionary_file_path is not None
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
                                "anchor_cluster_blend": anchor_blend,
                                "support_top_p": settings.support_top_p,
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
                                "support_top_p": settings.support_top_p,
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
        ce_fig, frac_fig, skl_fig, blend_ce_fig, blend_skl_fig, blend_sim_fig = reload_and_plot(
            settings, run_path
        )
        plot_infos = [
            ("cross_entropy", ce_fig, "ratio_cross_entropy_curve"),
            ("fraction_within_epsilon", frac_fig, "ratio_fraction_within_epsilon"),
            ("mean_symmetric_kl", skl_fig, "ratio_mean_symmetric_kl"),
            ("anchor_blend_cross_entropy", blend_ce_fig, "anchor_blend_cross_entropy"),
            ("anchor_blend_mean_symmetric_kl", blend_skl_fig, "anchor_blend_mean_symmetric_kl"),
            ("anchor_blend_support_similarity", blend_sim_fig, "anchor_blend_support_similarity"),
        ]

        base_path = Path(settings.plot_output_path)
        base_dir = base_path.parent
        base_dir.mkdir(parents=True, exist_ok=True)

        for suffix, fig, _ in plot_infos:
            png_path = base_dir / f"{base_path.stem}_{suffix}.png"
            pdf_path = base_dir / f"{base_path.stem}_{suffix}.pdf"
            fig.savefig(png_path, dpi=200)
            fig.savefig(pdf_path, format="pdf")

        if wandb_run is not None and hasattr(wandb_run, "log"):
            wandb_payload = {key: wandb.Image(fig) for _, fig, key in plot_infos}
            wandb_run.log(wandb_payload)
        else:
            for suffix, _, _ in plot_infos:
                print(
                    f"Saved plot to {base_path.stem}_{suffix}.png and .pdf"
                )

        for _, fig, _ in plot_infos:
            plt.close(fig)

    if wandb_run is not None and hasattr(wandb_run, "finish"):
        wandb_run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
