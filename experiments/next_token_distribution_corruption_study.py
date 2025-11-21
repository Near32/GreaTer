#!/usr/bin/env python
"""Explore how perturbing next-token distributions affects symmetric KL divergence.

This script loads a causal language model, samples prompts from a Wikipedia split,
collects the model's next-token distributions, progressively corrupts them, and
measures the resulting cross entropy, symmetric KL divergence, total variation,
entropy shifts, and multiple mass-shift diagnostics (per-selection, symmetric,
and clamped) within the perturbed support.
Corruption regimes target either high-probability (head) tokens,
low-probability (tail) tokens, or configurable mixtures of the two. Both the
number of tokens affected and the corruption strength can be swept to produce
grids suited to seaborn facet plots.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
from string import ascii_uppercase

import numpy as np

import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # Optional dependencies for analysis artefacts.
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-7
TRANSFER_ONLY_TARGETS = {"high_to_any", "low_to_any"}


@dataclass
class PromptDistribution:
    """Container for a prompt, its encoded length, and next-token probabilities."""

    sample_id: int
    prompt_text: str
    input_length: int
    probs: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sym-KL impact of corrupting LM next-token distributions")
    parser.add_argument("--model-name", default="distilgpt2", help="Model identifier for AutoModelForCausalLM")
    parser.add_argument("--tokenizer-name", default=None, help="Tokenizer identifier (defaults to model)")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow loading models with custom code")
    parser.add_argument("--dataset-name", default="wikipedia", help="Hugging Face dataset name")
    parser.add_argument("--dataset-config", default="20220301.en", help="Dataset configuration/subset")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to sample prompts from")
    parser.add_argument("--text-column", default="text", help="Column containing raw text")
    parser.add_argument("--max-samples", type=int, default=16, help="How many prompts to evaluate")
    parser.add_argument("--prompt-length", type=int, default=64, help="Maximum number of tokens per prompt")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility")
    parser.add_argument("--device", default=None, help="Computation device override (cpu / cuda / cuda:0 / mps)")
    parser.add_argument("--corruption-step", type=int, default=32, help="Stride for corruption counts")
    parser.add_argument("--corruption-steps", type=int, default=10, help="How many increments to evaluate")
    parser.add_argument(
        "--corruption-counts",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of corruption counts (overrides step settings)",
    )
    parser.add_argument(
        "--corruption-targets",
        nargs="*",
        default=("high", "low", "random"),
        choices=("high", "low", "random", "high_to_any", "low_to_any"),
        help=(
            "Distribution regions to perturb. "
            "Use *_to_any to let head (or tail) tokens donate probability mass to any recipient "
            "(transfer mode only)."
        ),
    )
    parser.add_argument(
        "--corruption-modes",
        nargs="*",
        choices=("scale", "transfer"),
        default=("scale",),
        help="Corruption mechanisms to apply; specify multiple to compare across modes.",
    )
    parser.add_argument(
        "--corruption-strength",
        type=float,
        default=0.6,
        help="Maximum multiplicative deviation applied to targeted tokens",
    )
    parser.add_argument(
        "--corruption-strengths",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional list of corruption strengths. When provided, overrides --corruption-strength "
            "and sweeps multiple magnitudes (used for facet columns in the default plot)."
        ),
    )
    parser.add_argument(
        "--blend-ratios",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional fractions in [0, 1] capturing the share of high-probability tokens to corrupt "
            "when mixing head and tail perturbations. The remainder targets low-probability tokens."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for tables and plots. Defaults to an auto-named folder derived "
            "from the model, dataset, and corruption settings."
        ),
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating seaborn plots even if available")
    parser.add_argument(
        "--plot-log",
        action="store_true",
        help="Also render plots with a logarithmic x-axis.",
    )
    parser.add_argument(
        "--plot-log-log",
        action="store_true",
        help="Render generated plots on logarithmic scales for both axes.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for forward passes")
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Persist prompt texts alongside metrics (may be large)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity",
    )
    parser.add_argument(
        "--mass-shift-threshold",
        type=float,
        default=1e-4,
        help="Absolute probability delta threshold used when computing symmetric mass shift statistics.",
    )
    parser.add_argument(
        "--mass-clamp-value",
        type=float,
        default=0.1,
        help="Clamping value applied to |Δp| when computing clamped mass shift metrics.",
    )
    parser.add_argument(
        "--error-estimator",
        type=str,
        choices=("quantile", "std"),
        default="std",
        help=(
            "Statistic used for asymmetric error bars. "
            "'quantile' uses the 16th/84th percentiles, 'std' reuses the sample standard deviation."
        ),
    )
    parser.add_argument(
        "--transfer-delta-min",
        type=float,
        default=0.05,
        help="Minimum absolute probability mass moved per transfer when using transfer corruption.",
    )
    parser.add_argument(
        "--transfer-delta-max",
        type=float,
        default=0.15,
        help="Maximum absolute probability mass moved per transfer when using transfer corruption.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="NextTokenDistributionCorruption",
        help="Weights & Biases project name (ignored when --wandb-action=none).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity / team name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name override.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default="disabled",
        help="W&B initialisation mode (disabled skips logging even when action requests it).",
    )
    parser.add_argument(
        "--wandb-action",
        type=str,
        choices=("none", "log", "reload", "both"),
        default="none",
        help="Whether to log a new W&B run, reload a previous run, or both.",
    )
    parser.add_argument(
        "--wandb-source-run",
        type=str,
        default=None,
        help="Run path (entity/project/run_id) to reload when --wandb-action includes 'reload'.",
    )
    parser.add_argument(
        "--wandb-artifact-name",
        type=str,
        default="corruption-study",
        help="Artifact name used when logging results to W&B (versioned automatically).",
    )
    parser.add_argument(
        "--local-reload",
        action="store_true",
        help="Reuse artifacts already present in --output-dir instead of recomputing the experiment.",
    )
    return parser.parse_args()


def init_wandb(args: argparse.Namespace, metadata: dict | None = None):
    if args.wandb_action not in {"log", "both"}:
        return None
    if args.wandb_mode == "disabled":
        LOGGER.info("Skipping W&B logging because --wandb-mode=disabled")
        return None
    if wandb is None:  # pragma: no cover - optional dependency
        raise ImportError("wandb must be installed for logging; pip install wandb")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=metadata or {},
    )
    LOGGER.info("Initialised W&B run: %s", run.name if run is not None else "<None>")
    return run


def finish_wandb(run) -> None:
    if run is None:
        return
    if hasattr(run, "finish"):
        run.finish()


def log_to_wandb(
    run,
    results_df,
    summary_df,
    plot_paths: List[Path],
    artifact_files: List[Path],
    metadata: dict,
) -> None:
    if run is None:
        return
    log_payload = {}
    if pd is not None and results_df is not None:
        log_payload["corruption_results_table"] = wandb.Table(dataframe=results_df)
    if pd is not None and summary_df is not None:
        log_payload["corruption_summary_table"] = wandb.Table(dataframe=summary_df)
        aggregate_means = {
            key: float(summary_df[key].mean())
            for key in summary_df.columns
            if key.startswith("mean_")
        }
        if aggregate_means:
            run.summary.update(aggregate_means)
    for path in plot_paths:
        if path.exists():
            log_payload[f"plots/{path.name}"] = wandb.Image(str(path))
    if log_payload:
        run.log(log_payload)

    artifact = wandb.Artifact(
        name=metadata.get("artifact_name", "corruption-study"),
        type="corruption-study",
        metadata={k: v for k, v in metadata.items() if k != "artifact_name"},
    )
    for file_path in artifact_files:
        if file_path.exists():
            artifact.add_file(str(file_path), name=file_path.name)
    if artifact.manifest.entries:
        run.log_artifact(artifact, aliases=["latest"])


def reload_from_wandb(args: argparse.Namespace, output_dir: Path, run_path: str | None = None):
    if wandb is None:  # pragma: no cover - optional dependency
        raise ImportError("wandb must be installed to reload runs")
    target_run = run_path or args.wandb_source_run
    if not target_run:
        raise ValueError("Run path required to reload; provide --wandb-source-run or log a run first")

    api = wandb.Api()
    run = api.run(target_run)
    artifacts = [art for art in run.logged_artifacts() if art.type == "corruption-study"]
    if not artifacts:
        raise ValueError(f"No 'corruption-study' artifacts found for run {target_run}")
    def _artifact_version(artifact):
        try:
            return int(str(artifact.version).lstrip("v"))
        except (TypeError, ValueError):
            return -1

    artifacts.sort(key=_artifact_version)
    artifact = artifacts[-1]
    reload_root = output_dir / "wandb_reload"
    reload_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading artifact %s to %s", artifact.name, reload_root)
    artifact_dir = Path(artifact.download(root=str(reload_root)))

    copied_files: List[Path] = []
    for entry in artifact.manifest.entries.keys():
        src = artifact_dir / entry
        dest = output_dir / Path(entry).name
        try:
            shutil.copy2(src, dest)
            copied_files.append(dest)
        except FileNotFoundError:
            LOGGER.debug("Artifact entry %s missing locally", entry)

    results_df = None
    if pd is not None:
        estimator_suffix = "" if args.error_estimator == "std" else f"_{args.error_estimator}"
        parquet_path = output_dir / f"corruption_results{estimator_suffix}.parquet"
        csv_path = output_dir / f"corruption_results{estimator_suffix}.csv"
        fallback_parquet = output_dir / "corruption_results.parquet"
        fallback_csv = output_dir / "corruption_results.csv"
        if parquet_path.exists():
            results_df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            results_df = pd.read_csv(csv_path)
        elif estimator_suffix:
            if fallback_parquet.exists():
                results_df = pd.read_parquet(fallback_parquet)
            elif fallback_csv.exists():
                results_df = pd.read_csv(fallback_csv)

    if copied_files:
        LOGGER.info("Reloaded files: %s", ", ".join(path.name for path in copied_files))
    return results_df


def reload_from_local(
    output_dir: Path,
    no_plots: bool,
    mass_shift_threshold: float,
    mass_clamp_value: float,
    plot_log: bool,
    plot_log_log: bool,
    error_estimator: str,
):
    results_df = None
    summary_df = None
    plot_paths: List[Path] = []
    artifact_files: List[Path] = []

    estimator_suffix = "" if error_estimator == "std" else f"_{error_estimator}"

    if pd is not None:
        parquet_path = output_dir / f"corruption_results{estimator_suffix}.parquet"
        csv_path = output_dir / f"corruption_results{estimator_suffix}.csv"
        fallback_parquet = output_dir / "corruption_results.parquet"
        fallback_csv = output_dir / "corruption_results.csv"
        if parquet_path.exists():
            results_df = pd.read_parquet(parquet_path)
            artifact_files.append(parquet_path)
        elif csv_path.exists():
            results_df = pd.read_csv(csv_path)
            artifact_files.append(csv_path)
        elif estimator_suffix:
            if fallback_parquet.exists():
                results_df = pd.read_parquet(fallback_parquet)
                artifact_files.append(fallback_parquet)
            elif fallback_csv.exists():
                results_df = pd.read_csv(fallback_csv)
                artifact_files.append(fallback_csv)

        summary_csv = output_dir / f"corruption_summary{estimator_suffix}.csv"
        summary_txt = output_dir / f"corruption_summary{estimator_suffix}.txt"
        fallback_summary_csv = output_dir / "corruption_summary.csv"
        fallback_summary_txt = output_dir / "corruption_summary.txt"
        if summary_csv.exists():
            summary_df = pd.read_csv(summary_csv)
            artifact_files.append(summary_csv)
        elif summary_txt.exists():
            artifact_files.append(summary_txt)
        elif estimator_suffix:
            if fallback_summary_csv.exists():
                summary_df = pd.read_csv(fallback_summary_csv)
                artifact_files.append(fallback_summary_csv)
            elif fallback_summary_txt.exists():
                artifact_files.append(fallback_summary_txt)
    else:
        summary_txt = output_dir / f"corruption_summary{estimator_suffix}.txt"
        fallback_summary_txt = output_dir / "corruption_summary.txt"
        if summary_txt.exists():
            artifact_files.append(summary_txt)
        elif estimator_suffix and fallback_summary_txt.exists():
            artifact_files.append(fallback_summary_txt)

    if results_df is not None and not no_plots:
        plot_paths = maybe_create_plots(
            results_df,
            output_dir,
            mass_shift_threshold,
            mass_clamp_value,
            log_x=False,
            log_y=False,
            error_estimator=error_estimator,
            estimator_tag=estimator_suffix,
        )
        if plot_log:
            plot_paths.extend(
                maybe_create_plots(
                    results_df,
                    output_dir,
                    mass_shift_threshold,
                    mass_clamp_value,
                    log_x=True,
                    log_y=False,
                    error_estimator=error_estimator,
                    estimator_tag=estimator_suffix,
                    suffix="_logx",
                )
            )
        if plot_log_log:
            plot_paths.extend(
                maybe_create_plots(
                    results_df,
                    output_dir,
                    mass_shift_threshold,
                    mass_clamp_value,
                    log_x=True,
                    log_y=True,
                    error_estimator=error_estimator,
                    estimator_tag=estimator_suffix,
                    suffix="_log",
                )
            )
        artifact_files.extend(plot_paths)

    return results_df, summary_df, plot_paths, artifact_files
def resolve_counts(args: argparse.Namespace) -> List[int]:
    if args.corruption_counts:
        counts = sorted({max(0, int(value)) for value in args.corruption_counts})
    else:
        max_count = args.corruption_step * args.corruption_steps
        counts = list(range(0, max_count + 1, args.corruption_step))
    return counts


def resolve_blend_ratios(args: argparse.Namespace) -> List[float]:
    if args.blend_ratios is None:
        return []
    ratios = np.asarray(args.blend_ratios, dtype=float)
    if np.isnan(ratios).any():  # pragma: no cover - guardrail for invalid inputs
        raise ValueError("blend ratios must be numeric")
    ratios = np.clip(ratios, 0.0, 1.0)
    return sorted(set(float(r) for r in ratios))


def resolve_strengths(args: argparse.Namespace) -> List[float]:
    if args.corruption_strengths:
        strengths = np.asarray(args.corruption_strengths, dtype=float)
    else:
        strengths = np.asarray([args.corruption_strength], dtype=float)
    if np.isnan(strengths).any():  # pragma: no cover
        raise ValueError("corruption strengths must be numeric")
    strengths = np.clip(strengths, 0.0, None)
    return [float(s) for s in sorted(set(strengths.tolist()))]


def load_prompts(args: argparse.Namespace) -> List[str]:
    LOGGER.info("Loading dataset %s/%s (%s)", args.dataset_name, args.dataset_config, args.dataset_split)
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    if args.max_samples > len(dataset):
        LOGGER.warning("max_samples=%d exceeds dataset length %d; clipping", args.max_samples, len(dataset))
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.max_samples, len(dataset)), replace=False)
    texts: List[str] = []
    for idx in indices:
        entry = dataset[int(idx)]
        text = entry.get(args.text_column, "")
        if not isinstance(text, str) or not text.strip():
            continue
        texts.append(" ".join(text.split()))
    if not texts:
        raise ValueError("No valid prompts extracted from dataset; check text column or preprocessing")
    LOGGER.info("Collected %d prompts", len(texts))
    return texts


def load_model_and_tokenizer(args: argparse.Namespace):
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name or model_name
    LOGGER.info("Loading tokenizer %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    LOGGER.info("Loading causal LM %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device %s", device)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def compute_next_token_distributions(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    max_length: int,
    batch_size: int,
) -> List[PromptDistribution]:
    results: List[PromptDistribution] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encodings = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        last_token_indices = attention_mask.sum(dim=1) - 1
        next_token_logits = logits[torch.arange(logits.size(0), device=device), last_token_indices]
        probs = torch.softmax(next_token_logits, dim=-1).cpu().numpy()
        lengths = attention_mask.sum(dim=1).cpu().tolist()
        for idx, (prompt, length, prob_vector) in enumerate(zip(batch_prompts, lengths, probs)):
            results.append(
                PromptDistribution(
                    sample_id=start + idx,
                    prompt_text=prompt,
                    input_length=int(length),
                    probs=prob_vector.astype(np.float64, copy=True),
                )
            )
    return results


def compute_symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    p_norm = np.clip(p, EPSILON, None)
    q_norm = np.clip(q, EPSILON, None)
    p_norm /= p_norm.sum()
    q_norm /= q_norm.sum()
    kl_pq = float(np.sum(p_norm * (np.log(p_norm) - np.log(q_norm))))
    kl_qp = float(np.sum(q_norm * (np.log(q_norm) - np.log(p_norm))))
    return kl_pq + kl_qp


def compute_cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Cross entropy H(p, q) measured in nats."""
    p_norm = np.clip(p, EPSILON, None)
    q_norm = np.clip(q, EPSILON, None)
    p_norm /= p_norm.sum()
    q_norm /= q_norm.sum()
    return float(-np.sum(p_norm * np.log(q_norm)))


def compute_entropy(p: np.ndarray) -> float:
    """Shannon entropy (nats) of a probability distribution."""
    p_norm = np.clip(p, EPSILON, None)
    p_norm /= p_norm.sum()
    return float(-np.sum(p_norm * np.log(p_norm)))


def compute_wasserstein_distance(p: np.ndarray, q: np.ndarray) -> float:
    """First Wasserstein distance (Earth mover's) on the discrete token index line."""
    p_norm = np.clip(p, EPSILON, None)
    q_norm = np.clip(q, EPSILON, None)
    p_norm /= p_norm.sum()
    q_norm /= q_norm.sum()
    cdf_diff = np.cumsum(p_norm - q_norm)
    return float(np.sum(np.abs(cdf_diff)))


def compute_total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two discrete distributions."""
    p_norm = np.clip(p, EPSILON, None)
    q_norm = np.clip(q, EPSILON, None)
    p_norm /= p_norm.sum()
    q_norm /= q_norm.sum()
    return float(0.5 * np.sum(np.abs(p_norm - q_norm)))


def _scale_corrupt_distribution(
    base_probs: np.ndarray,
    num_corruptions: int,
    target: str,
    strength: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, dict]:
    adjusted = base_probs.astype(np.float64, copy=True)
    vocab_size = adjusted.shape[0]
    count = int(min(max(num_corruptions, 0), vocab_size))
    if count == 0:
        return adjusted, {
            "selection_size": 0,
            "selected_mass_before": 0.0,
            "selected_mass_after": 0.0,
            "mass_shift_selected": 0.0,
        }
    order = np.argsort(base_probs)
    if target == "high":
        selected = order[::-1][:count]
        lower = max(1.0 - strength, 1e-3)
        factors = rng.uniform(lower, 1.0, size=count)
    elif target == "low":
        selected = order[:count]
        factors = rng.uniform(1.0, 1.0 + strength, size=count)
    elif target == "random":
        selected = rng.choice(vocab_size, size=count, replace=False)
        lower = max(1.0 - strength, 1e-3)
        upper = 1.0 + strength
        factors = rng.uniform(lower, upper, size=count)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported corruption target: {target}")
    adjusted[selected] *= factors
    adjusted = np.clip(adjusted, EPSILON, None)
    adjusted /= adjusted.sum()
    mass_shift_selected = float(np.sum(np.abs(base_probs[selected] - adjusted[selected]))) if count else 0.0

    metadata = {
        "selection_size": int(count),
        "selected_mass_before": float(base_probs[selected].sum()),
        "selected_mass_after": float(adjusted[selected].sum()),
        "mass_shift_selected": mass_shift_selected,
    }
    return adjusted, metadata


def _scale_corrupt_distribution_blend(
    base_probs: np.ndarray,
    num_corruptions: int,
    high_fraction: float,
    strength: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, dict]:
    adjusted = base_probs.astype(np.float64, copy=True)
    vocab_size = adjusted.shape[0]
    count = int(min(max(num_corruptions, 0), vocab_size))
    if count == 0:
        metadata = {
            "selection_size": 0,
            "selected_mass_before": 0.0,
            "selected_mass_after": 0.0,
            "high_count": 0,
            "low_count": 0,
            "high_mass_before": 0.0,
            "high_mass_after": 0.0,
            "low_mass_before": 0.0,
            "low_mass_after": 0.0,
            "requested_high_fraction": float(high_fraction),
            "effective_high_fraction": 0.0,
            "mass_shift_selected": 0.0,
        }
        return adjusted, metadata

    clipped_fraction = float(np.clip(high_fraction, 0.0, 1.0))
    # Sample how many tokens to draw from the high-probability region.
    high_count = int(rng.binomial(count, clipped_fraction))
    high_count = min(high_count, count)
    low_count = count - high_count

    order = np.argsort(base_probs)
    order_desc = order[::-1]
    used_mask = np.zeros(vocab_size, dtype=bool)

    high_selected: np.ndarray
    if high_count > 0:
        high_pool_size = min(vocab_size, max(high_count * 2, 4))
        high_pool = order_desc[:high_pool_size]
        if high_pool.size < high_count:
            high_pool = order_desc  # Fallback to full sorted list.
        high_selected = np.array(rng.choice(high_pool, size=high_count, replace=False), dtype=int)
        used_mask[high_selected] = True
    else:
        high_selected = np.empty(0, dtype=int)

    if low_count > 0:
        low_candidates = order[~used_mask[order]]
        low_pool_size = min(low_candidates.size, max(low_count * 2, 4))
        low_pool = low_candidates[:low_pool_size]
        if low_pool.size < low_count:
            low_pool = low_candidates  # Fallback if pool is too small.
        low_selected = np.array(rng.choice(low_pool, size=low_count, replace=False), dtype=int)
    else:
        low_selected = np.empty(0, dtype=int)

    selected = np.concatenate([high_selected, low_selected])
    if selected.size == 0:
        metadata = {
            "selection_size": 0,
            "selected_mass_before": 0.0,
            "selected_mass_after": 0.0,
            "high_count": 0,
            "low_count": 0,
            "high_mass_before": 0.0,
            "high_mass_after": 0.0,
            "low_mass_before": 0.0,
            "low_mass_after": 0.0,
            "requested_high_fraction": clipped_fraction,
            "effective_high_fraction": 0.0,
            "mass_shift_selected": 0.0,
        }
        return adjusted, metadata

    strength = float(max(strength, 0.0))
    lower = max(1.0 - strength, 1e-3)
    upper = 1.0 + strength
    if high_selected.size:
        high_factors = rng.uniform(lower, 1.0, size=high_selected.size)
        adjusted[high_selected] *= high_factors
    if low_selected.size:
        low_factors = rng.uniform(1.0, upper, size=low_selected.size)
        adjusted[low_selected] *= low_factors

    adjusted = np.clip(adjusted, EPSILON, None)
    adjusted /= adjusted.sum()

    selected_mass_before = float(base_probs[selected].sum())
    selected_mass_after = float(adjusted[selected].sum())
    mass_shift_selected = float(np.sum(np.abs(base_probs[selected] - adjusted[selected])))

    high_mass_before = float(base_probs[high_selected].sum()) if high_selected.size else 0.0
    high_mass_after = float(adjusted[high_selected].sum()) if high_selected.size else 0.0
    low_mass_before = float(base_probs[low_selected].sum()) if low_selected.size else 0.0
    low_mass_after = float(adjusted[low_selected].sum()) if low_selected.size else 0.0

    effective_high_fraction = float(high_selected.size / selected.size) if selected.size else 0.0

    metadata = {
        "selection_size": int(selected.size),
        "selected_mass_before": selected_mass_before,
        "selected_mass_after": selected_mass_after,
        "high_count": int(high_selected.size),
        "low_count": int(low_selected.size),
        "high_mass_before": high_mass_before,
        "high_mass_after": high_mass_after,
        "low_mass_before": low_mass_before,
        "low_mass_after": low_mass_after,
        "requested_high_fraction": clipped_fraction,
        "effective_high_fraction": effective_high_fraction,
        "mass_shift_selected": mass_shift_selected,
    }
    return adjusted, metadata


def _prepare_transfer_pools(base_probs: np.ndarray, pool_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(base_probs)
    high_pool = order[::-1][:pool_size]
    low_pool = order[:pool_size]
    all_indices = np.arange(base_probs.shape[0])
    return high_pool, low_pool, all_indices


def _sample_pair(
    rng: np.random.Generator,
    donor_pool: np.ndarray,
    recipient_pool: np.ndarray,
    all_indices: np.ndarray,
    allow_same_pool: bool = False,
) -> Tuple[int, int]:
    donor_pool = donor_pool if donor_pool.size else all_indices
    recipient_pool = recipient_pool if recipient_pool.size else all_indices
    if allow_same_pool and donor_pool.size >= 2 and np.array_equal(donor_pool, recipient_pool):
        chosen = rng.choice(donor_pool, size=2, replace=False)
        return int(chosen[0]), int(chosen[1])
    donor_idx = int(rng.choice(donor_pool))
    recipient_idx = int(rng.choice(recipient_pool))
    if donor_idx == recipient_idx:
        for _ in range(5):
            candidate = int(rng.choice(recipient_pool))
            if candidate != donor_idx:
                recipient_idx = candidate
                break
        else:
            candidates = all_indices[all_indices != donor_idx]
            if candidates.size:
                recipient_idx = int(rng.choice(candidates))
    return donor_idx, recipient_idx


def _transfer_corrupt_distribution(
    base_probs: np.ndarray,
    num_transfers: int,
    target: str,
    rng: np.random.Generator,
    delta_min: float,
    delta_max: float,
) -> Tuple[np.ndarray, dict]:
    adjusted = base_probs.astype(np.float64, copy=True)
    vocab_size = adjusted.shape[0]
    count = int(max(0, num_transfers))
    if count == 0:
        return adjusted, {
            "selection_size": 0,
            "selected_mass_before": 0.0,
            "selected_mass_after": 0.0,
            "mass_shift_selected": 0.0,
        }
    pool_size = max(2, min(vocab_size, max(4, count * 2)))
    high_pool, low_pool, all_indices = _prepare_transfer_pools(base_probs, pool_size)
    touched: set[int] = set()

    delta_min = max(0.0, float(delta_min))
    delta_max = max(delta_min, float(delta_max))

    for _ in range(count):
        if target == "high":
            donor_pool = high_pool
            recipient_pool = high_pool
            allow_same_pool = True
        elif target == "low":
            donor_pool = low_pool
            recipient_pool = low_pool
            allow_same_pool = True
        elif target == "random":
            donor_pool = all_indices
            recipient_pool = all_indices
            allow_same_pool = True
        elif target == "high_to_any":
            donor_pool = high_pool
            recipient_pool = all_indices
            allow_same_pool = False
        elif target == "low_to_any":
            donor_pool = low_pool
            recipient_pool = all_indices
            allow_same_pool = False
        else:
            raise ValueError(f"Unsupported corruption target: {target}")
        donor_idx, recipient_idx = _sample_pair(
            rng,
            donor_pool,
            recipient_pool,
            all_indices,
            allow_same_pool=allow_same_pool,
        )
        if donor_idx == recipient_idx:
            continue
        donor_mass = adjusted[donor_idx]
        if donor_mass <= 0.0:
            continue
        delta_sample = rng.uniform(delta_min, delta_max)
        delta = min(delta_sample, donor_mass)
        if delta <= 0.0:
            continue
        adjusted[donor_idx] -= delta
        adjusted[recipient_idx] += delta
        touched.add(int(donor_idx))
        touched.add(int(recipient_idx))

    if touched:
        touched_array = np.fromiter(touched, dtype=int)
        selected_mass_before = float(base_probs[touched_array].sum())
        selected_mass_after = float(adjusted[touched_array].sum())
        mass_shift_selected = float(np.sum(np.abs(base_probs[touched_array] - adjusted[touched_array])))
    else:
        touched_array = np.array([], dtype=int)
        selected_mass_before = 0.0
        selected_mass_after = 0.0
        mass_shift_selected = 0.0

    metadata = {
        "selection_size": int(len(touched)),
        "selected_mass_before": selected_mass_before,
        "selected_mass_after": selected_mass_after,
        "mass_shift_selected": mass_shift_selected,
    }
    return adjusted, metadata


def _transfer_corrupt_distribution_blend(
    base_probs: np.ndarray,
    num_transfers: int,
    high_fraction: float,
    rng: np.random.Generator,
    delta_min: float,
    delta_max: float,
) -> Tuple[np.ndarray, dict]:
    adjusted = base_probs.astype(np.float64, copy=True)
    vocab_size = adjusted.shape[0]
    count = int(max(0, num_transfers))
    if count == 0:
        metadata = {
            "selection_size": 0,
            "selected_mass_before": 0.0,
            "selected_mass_after": 0.0,
            "high_count": 0,
            "low_count": 0,
            "high_mass_before": 0.0,
            "high_mass_after": 0.0,
            "low_mass_before": 0.0,
            "low_mass_after": 0.0,
            "requested_high_fraction": float(high_fraction),
            "effective_high_fraction": 0.0,
            "mass_shift_selected": 0.0,
        }
        return adjusted, metadata

    clipped_fraction = float(np.clip(high_fraction, 0.0, 1.0))
    pool_size = max(2, min(vocab_size, max(4, count * 2)))
    high_pool, low_pool, all_indices = _prepare_transfer_pools(base_probs, pool_size)
    high_set = set(int(i) for i in high_pool.tolist())
    low_set = set(int(i) for i in low_pool.tolist())

    delta_min = max(0.0, float(delta_min))
    delta_max = max(delta_min, float(delta_max))

    touched: set[int] = set()
    high_transfers = 0
    total_transfers = 0

    for _ in range(count):
        donor_from_high = rng.random() < clipped_fraction
        donor_pool = high_pool if donor_from_high else low_pool
        recipient_pool = low_pool if donor_from_high else high_pool
        donor_idx, recipient_idx = _sample_pair(rng, donor_pool, recipient_pool, all_indices, allow_same_pool=False)
        if donor_idx == recipient_idx:
            continue
        donor_mass = adjusted[donor_idx]
        if donor_mass <= 0.0:
            continue
        delta_sample = rng.uniform(delta_min, delta_max)
        delta = min(delta_sample, donor_mass)
        if delta <= 0.0:
            continue
        adjusted[donor_idx] -= delta
        adjusted[recipient_idx] += delta
        touched.add(int(donor_idx))
        touched.add(int(recipient_idx))
        if donor_from_high:
            high_transfers += 1
        total_transfers += 1

    if touched:
        touched_array = np.fromiter(touched, dtype=int)
        selected_mass_before = float(base_probs[touched_array].sum())
        selected_mass_after = float(adjusted[touched_array].sum())
        mass_shift_selected = float(np.sum(np.abs(base_probs[touched_array] - adjusted[touched_array])))
    else:
        touched_array = np.array([], dtype=int)
        selected_mass_before = 0.0
        selected_mass_after = 0.0
        mass_shift_selected = 0.0

    high_touched = [idx for idx in touched if idx in high_set]
    low_touched = [idx for idx in touched if idx in low_set]

    metadata = {
        "selection_size": int(len(touched)),
        "selected_mass_before": selected_mass_before,
        "selected_mass_after": selected_mass_after,
        "high_count": len(high_touched),
        "low_count": len(low_touched),
        "high_mass_before": float(base_probs[high_touched].sum()) if high_touched else 0.0,
        "high_mass_after": float(adjusted[high_touched].sum()) if high_touched else 0.0,
        "low_mass_before": float(base_probs[low_touched].sum()) if low_touched else 0.0,
        "low_mass_after": float(adjusted[low_touched].sum()) if low_touched else 0.0,
        "requested_high_fraction": clipped_fraction,
        "effective_high_fraction": float(high_transfers / total_transfers) if total_transfers else 0.0,
        "mass_shift_selected": mass_shift_selected,
    }
    return adjusted, metadata


def corrupt_distribution(
    base_probs: np.ndarray,
    num_corruptions: int,
    target: str,
    strength: float,
    rng: np.random.Generator,
    mode: str,
    transfer_settings: Tuple[float, float],
) -> Tuple[np.ndarray, dict]:
    if mode == "transfer":
        delta_min, delta_max = transfer_settings
        return _transfer_corrupt_distribution(
            base_probs,
            num_corruptions,
            target,
            rng,
            delta_min,
            delta_max,
        )
    if mode == "scale":
        return _scale_corrupt_distribution(
            base_probs,
            num_corruptions,
            target,
            strength,
            rng,
        )
    raise ValueError(f"Unsupported corruption mode: {mode}")


def corrupt_distribution_blend(
    base_probs: np.ndarray,
    num_corruptions: int,
    high_fraction: float,
    strength: float,
    rng: np.random.Generator,
    mode: str,
    transfer_settings: Tuple[float, float],
) -> Tuple[np.ndarray, dict]:
    if mode == "transfer":
        delta_min, delta_max = transfer_settings
        return _transfer_corrupt_distribution_blend(
            base_probs,
            num_corruptions,
            high_fraction,
            rng,
            delta_min,
            delta_max,
        )
    if mode == "scale":
        return _scale_corrupt_distribution_blend(
            base_probs,
            num_corruptions,
            high_fraction,
            strength,
            rng,
        )
    raise ValueError(f"Unsupported corruption mode: {mode}")


def _quantile_errors(arr: np.ndarray) -> Tuple[float, float]:
    """Return non-negative deviations below/above the mean based on the central 68% interval."""
    if arr.size <= 1:
        return 0.0, 0.0
    mean = float(arr.mean())
    if arr.size >= 3:
        lower_percentile = float(np.percentile(arr, 16))
        upper_percentile = float(np.percentile(arr, 84))
        err_low = max(mean - lower_percentile, 0.0)
        err_high = max(upper_percentile - mean, 0.0)
    else:
        err_low = max(mean - float(arr.min()), 0.0)
        err_high = max(float(arr.max()) - mean, 0.0)
    return err_low, err_high


def _metric_stats(values: np.ndarray, estimator: str) -> Tuple[float, float, float, float]:
    """Return mean, sample std, and asymmetric lower/upper errors for an array."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if estimator == "quantile":
        err_low, err_high = _quantile_errors(arr)
    elif estimator == "std":
        err_low = err_high = max(std, 0.0)
    else:  # pragma: no cover - validated in CLI
        raise ValueError(f"Unsupported error estimator: {estimator}")
    return mean, std, err_low, err_high


def summarize_results(records: List[dict], error_estimator: str) -> List[dict]:
    grouped = {}
    for row in records:
        key = (
            row["corruption_target"],
            row["num_corruptions"],
            row.get("blend_ratio"),
            row.get("corruption_strength"),
            row.get("corruption_mode", "scale"),
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for (target, count, ratio, strength, mode), rows in grouped.items():
        values = np.array([r["symmetric_kl"] for r in rows], dtype=np.float64)
        cross_values = np.array([r["cross_entropy"] for r in rows], dtype=np.float64)
        entropy_orig_values = np.array([r["entropy_original"] for r in rows], dtype=np.float64)
        entropy_corr_values = np.array([r["entropy_corrupted"] for r in rows], dtype=np.float64)
        wasser_values = np.array([r["wasserstein_distance"] for r in rows], dtype=np.float64)
        tv_values = np.array([r["total_variation"] for r in rows], dtype=np.float64)
        mass_shift_values = np.array([r.get("mass_shift_selected", 0.0) for r in rows], dtype=np.float64)
        sym_mass_shift_values = np.array([r.get("symmetric_mass_shift", 0.0) for r in rows], dtype=np.float64)
        sym_mass_count_values = np.array([r.get("symmetric_mass_count", 0) for r in rows], dtype=np.float64)
        clamp_mass_values = np.array([r.get("clamped_mass_shift", 0.0) for r in rows], dtype=np.float64)
        clamp_mass_count_values = np.array([r.get("clamped_mass_count", 0) for r in rows], dtype=np.float64)
        if ratio is not None:
            ratio_float = float(ratio)
            ratio_value = None if np.isnan(ratio_float) else ratio_float
        else:
            ratio_value = None

        (
            mean_skl,
            std_skl,
            err_low_skl,
            err_high_skl,
        ) = _metric_stats(values, error_estimator)
        (
            mean_cross,
            std_cross,
            err_low_cross,
            err_high_cross,
        ) = _metric_stats(cross_values, error_estimator)
        (
            mean_wasser,
            std_wasser,
            err_low_wasser,
            err_high_wasser,
        ) = _metric_stats(wasser_values, error_estimator)
        (
            mean_tv,
            std_tv,
            err_low_tv,
            err_high_tv,
        ) = _metric_stats(tv_values, error_estimator)
        (
            mean_mass_shift,
            std_mass_shift,
            err_low_mass_shift,
            err_high_mass_shift,
        ) = _metric_stats(mass_shift_values, error_estimator)
        (
            mean_sym_mass_shift,
            std_sym_mass_shift,
            err_low_sym_mass_shift,
            err_high_sym_mass_shift,
        ) = _metric_stats(sym_mass_shift_values, error_estimator)
        (
            mean_sym_mass_count,
            std_sym_mass_count,
            err_low_sym_mass_count,
            err_high_sym_mass_count,
        ) = _metric_stats(sym_mass_count_values, error_estimator)
        (
            mean_clamp_shift,
            std_clamp_shift,
            err_low_clamp_shift,
            err_high_clamp_shift,
        ) = _metric_stats(clamp_mass_values, error_estimator)
        (
            mean_clamp_count,
            std_clamp_count,
            err_low_clamp_count,
            err_high_clamp_count,
        ) = _metric_stats(clamp_mass_count_values, error_estimator)

        entry = {
            "corruption_target": target,
            "num_corruptions": int(count),
            "blend_ratio": ratio_value,
            "corruption_strength": float(strength) if strength is not None else None,
            "corruption_mode": mode,
            "mean_symmetric_kl": mean_skl,
            "std_symmetric_kl": std_skl,
            "symmetric_kl_err_low": err_low_skl,
            "symmetric_kl_err_high": err_high_skl,
            "min_symmetric_kl": float(values.min()),
            "max_symmetric_kl": float(values.max()),
            "num_samples": len(rows),
            "mean_cross_entropy": mean_cross,
            "std_cross_entropy": std_cross,
            "cross_entropy_err_low": err_low_cross,
            "cross_entropy_err_high": err_high_cross,
            "mean_entropy_original": float(entropy_orig_values.mean()),
            "mean_entropy_corrupted": float(entropy_corr_values.mean()),
            "mean_wasserstein_distance": mean_wasser,
            "std_wasserstein_distance": std_wasser,
            "wasserstein_distance_err_low": err_low_wasser,
            "wasserstein_distance_err_high": err_high_wasser,
            "mean_total_variation": mean_tv,
            "std_total_variation": std_tv,
            "total_variation_err_low": err_low_tv,
            "total_variation_err_high": err_high_tv,
            "mean_mass_shift_selected": mean_mass_shift,
            "std_mass_shift_selected": std_mass_shift,
            "mass_shift_selected_err_low": err_low_mass_shift,
            "mass_shift_selected_err_high": err_high_mass_shift,
            "mean_symmetric_mass_shift": mean_sym_mass_shift,
            "std_symmetric_mass_shift": std_sym_mass_shift,
            "symmetric_mass_shift_err_low": err_low_sym_mass_shift,
            "symmetric_mass_shift_err_high": err_high_sym_mass_shift,
            "mean_symmetric_mass_count": mean_sym_mass_count,
            "std_symmetric_mass_count": std_sym_mass_count,
            "symmetric_mass_count_err_low": err_low_sym_mass_count,
            "symmetric_mass_count_err_high": err_high_sym_mass_count,
            "mean_clamped_mass_shift": mean_clamp_shift,
            "std_clamped_mass_shift": std_clamp_shift,
            "clamped_mass_shift_err_low": err_low_clamp_shift,
            "clamped_mass_shift_err_high": err_high_clamp_shift,
            "mean_clamped_mass_count": mean_clamp_count,
            "std_clamped_mass_count": std_clamp_count,
            "clamped_mass_count_err_low": err_low_clamp_count,
            "clamped_mass_count_err_high": err_high_clamp_count,
        }
        if ratio is not None:
            effective = [
                r.get("blend_ratio_effective", ratio)
                for r in rows
                if (
                    r.get("blend_ratio_effective") is not None
                    and not (isinstance(r.get("blend_ratio_effective"), float) and np.isnan(r.get("blend_ratio_effective")))
                )
            ]
            if effective:
                entry["mean_effective_blend_ratio"] = float(np.mean(effective))
        summary.append(entry)
    summary.sort(
        key=lambda item: (
            item["corruption_target"],
            item["corruption_strength"] if item["corruption_strength"] is not None else float("inf"),
            item["num_corruptions"],
            item["corruption_mode"],
            -1.0 if item["blend_ratio"] is None else item["blend_ratio"],
        )
    )
    return summary


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify_fragment(text: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "-", text.strip())
    sanitized = sanitized.strip("-_.")
    return sanitized or "none"


def _format_value_for_slug(value) -> str:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "nan"
        formatted = f"{float(value):.4f}".rstrip("0").rstrip(".")
        return formatted or "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return ensure_output_dir(args.output_dir)

    components = [
        ("model", args.model_name),
        ("dataset", args.dataset_name),
    ]
    if args.dataset_config:
        components.append(("config", args.dataset_config))
    if args.dataset_split:
        components.append(("split", args.dataset_split))
    if args.corruption_modes:
        modes = "-".join(sorted(set(str(mode) for mode in args.corruption_modes)))
        components.append(("modes", modes))
    if args.corruption_targets:
        targets = "-".join(sorted(set(str(target) for target in args.corruption_targets)))
        components.append(("targets", targets))
    if args.corruption_strengths:
        strengths = "-".join(
            _format_value_for_slug(value) for value in sorted(set(args.corruption_strengths))
        )
        components.append(("strengths", strengths))
    else:
        components.append(("strength", _format_value_for_slug(args.corruption_strength)))
    if args.blend_ratios:
        ratios = "-".join(_format_value_for_slug(value) for value in args.blend_ratios)
        components.append(("blend", ratios))

    base_dir = Path("artifacts") / "next_token_corruption"
    folder = "__".join(
        f"{key}-{_slugify_fragment(str(value))}"
        for key, value in components
        if value is not None and str(value).strip()
    )
    folder = folder or "default"
    return ensure_output_dir(base_dir / folder)


def maybe_create_plots(
    results_df,
    output_dir: Path,
    mass_shift_threshold: float,
    mass_clamp_value: float,
    log_x: bool = False,
    log_y: bool = False,
    error_estimator: str = "std",
    estimator_tag: str = "",
    suffix: str = "",
):
    if pd is None or sns is None or plt is None:
        LOGGER.warning("Skipping plots because pandas/seaborn/matplotlib are unavailable")
        return []
    if results_df.empty:
        LOGGER.warning("No results to plot")
        return []
    plot_df = results_df.copy()
    if "corruption_strength" not in plot_df.columns:
        LOGGER.warning("Results missing corruption_strength column; skipping plots")
        return []
    if "blend_ratio" not in plot_df.columns:
        plot_df["blend_ratio"] = np.nan
    if "corruption_mode" not in plot_df.columns:
        plot_df["corruption_mode"] = "scale"
    plot_df["corruption_mode"] = plot_df["corruption_mode"].astype(str)
    for missing_col in (
        "symmetric_mass_shift",
        "symmetric_mass_count",
        "mass_shift_selected",
        "clamped_mass_shift",
        "clamped_mass_count",
    ):
        if missing_col not in plot_df.columns:
            plot_df[missing_col] = 0.0
    plot_df["blend_ratio"] = plot_df["blend_ratio"].astype(float)
    plot_df["corruption_strength"] = plot_df["corruption_strength"].astype(float)

    plot_df["blend_ratio_key"] = plot_df["blend_ratio"].fillna(-1.0)
    group_columns = [
        "corruption_target",
        "blend_ratio_key",
        "corruption_mode",
        "corruption_strength",
        "num_corruptions",
    ]
    stats_metrics = [
        ("symmetric_kl", "symmetric_kl"),
        ("cross_entropy", "cross_entropy"),
        ("wasserstein_distance", "wasserstein"),
        ("total_variation", "total_variation"),
        ("mass_shift_selected", "mass_shift"),
        ("symmetric_mass_shift", "symmetric_mass_shift"),
        ("symmetric_mass_count", "symmetric_mass_count"),
        ("clamped_mass_shift", "clamped_mass_shift"),
        ("clamped_mass_count", "clamped_mass_count"),
    ]
    agg_rows: list[dict] = []
    grouped = plot_df.groupby(group_columns, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record["sample_count"] = int(group["sample_id"].nunique())
        entropy_orig_mean, _, _, _ = _metric_stats(
            group["entropy_original"].to_numpy(dtype=np.float64),
            error_estimator,
        )
        entropy_corr_mean, _, _, _ = _metric_stats(
            group["entropy_corrupted"].to_numpy(dtype=np.float64),
            error_estimator,
        )
        record["entropy_original_mean"] = entropy_orig_mean
        record["entropy_corrupted_mean"] = entropy_corr_mean
        for source, alias in stats_metrics:
            mean_val, std_val, err_low, err_high = _metric_stats(
                group[source].to_numpy(dtype=np.float64),
                error_estimator,
            )
            record[f"{alias}_mean"] = mean_val
            record[f"{alias}_std"] = std_val
            record[f"{alias}_err_low"] = err_low
            record[f"{alias}_err_high"] = err_high
        agg_rows.append(record)
    agg_df = pd.DataFrame(agg_rows)
    if agg_df.empty:
        LOGGER.warning("Aggregated results empty; skipping plot generation")
        return []
    for _, alias in stats_metrics:
        agg_df[f"{alias}_std"] = agg_df[f"{alias}_std"].fillna(0.0)
        agg_df[f"{alias}_err_low"] = agg_df[f"{alias}_err_low"].fillna(0.0)
        agg_df[f"{alias}_err_high"] = agg_df[f"{alias}_err_high"].fillna(0.0)
    agg_df["blend_ratio"] = agg_df["blend_ratio_key"].replace({-1.0: np.nan})
    agg_df.drop(columns=["blend_ratio_key"], inplace=True)

    LOG_EPS = 1e-12

    def _annotate_facets(grid, labels, caption_y=0.06):
        pairs = []
        for idx, ax in enumerate(grid.axes.flat):
            if idx >= len(labels):
                continue
            letter = ascii_uppercase[idx]
            ax.set_title(letter)
            pairs.append(f"{letter}: {labels[idx]}")
        if pairs:
            caption = "Facet legend – " + ", ".join(pairs)
            grid.figure.text(0.5, caption_y, caption, ha="center", fontsize=10)

    def _descriptor(row):
        mode_label = row.get("corruption_mode", "scale")
        if row["corruption_target"] == "blend" and not np.isnan(row["blend_ratio"]):
            return f"{mode_label}: blend(r={row['blend_ratio']:.2f})"
        return f"{mode_label}: {row['corruption_target']}"

    agg_df["corruption_descriptor"] = agg_df.apply(_descriptor, axis=1)
    if "corruption_strength" not in agg_df.columns:
        agg_df["corruption_strength"] = 0.0
    agg_df["strength_label"] = agg_df["corruption_strength"].apply(lambda val: f"strength={float(val):.2f}")

    col_labels = list(dict.fromkeys(agg_df["strength_label"].tolist()))

    def _plot_path(basename: str) -> Path:
        base_path = Path(basename)
        suffix_parts: List[str] = []
        if estimator_tag:
            suffix_parts.append(estimator_tag)
        if suffix:
            suffix_parts.append(suffix)
        combined = "".join(suffix_parts)
        if not combined:
            return output_dir / basename
        return output_dir / f"{base_path.stem}{combined}{base_path.suffix}"

    def _prepare_y(values):
        arr = np.asarray(values, dtype=float)
        if log_y:
            arr = np.clip(arr, LOG_EPS, None)
        return arr

    def _mean_with_bounds(ordered, mean_key, std_key):
        mean_raw = ordered[mean_key].to_numpy(dtype=float)
        std = np.nan_to_num(ordered[std_key].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        mean = mean_raw.copy()
        lower_key = mean_key.replace("_mean", "_err_low")
        upper_key = mean_key.replace("_mean", "_err_high")
        if lower_key in ordered.columns and upper_key in ordered.columns:
            err_low = np.nan_to_num(
                ordered[lower_key].to_numpy(dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            err_high = np.nan_to_num(
                ordered[upper_key].to_numpy(dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            lower = mean_raw - err_low
            upper = mean_raw + err_high
        else:
            lower = mean_raw - std
            upper = mean_raw + std
        if log_y:
            mean = np.clip(mean, LOG_EPS, None)
            lower = np.clip(lower, LOG_EPS, None)
            upper = np.clip(upper, LOG_EPS, None)
        return mean, lower, upper

    def _prepare_plot(x, *arrays):
        x_arr = np.asarray(x, dtype=float)
        value_arrays = [np.asarray(arr, dtype=float) for arr in arrays]
        mask = np.isfinite(x_arr)
        if log_x:
            mask &= x_arr > 0.0
        for arr in value_arrays:
            arr_mask = np.isfinite(arr)
            if log_y:
                arr_mask &= arr > 0.0
            mask &= arr_mask
        if not np.any(mask):
            empty = x_arr[:0]
            return empty, [arr[:0] for arr in value_arrays]
        x_arr = x_arr[mask]
        value_arrays = [arr[mask] for arr in value_arrays]
        return x_arr, value_arrays

    def _finalize_grid(grid, x_label, y_label, caption_y=0.12):
        def _label(base: str, apply_log: bool) -> str:
            return f"{base} (log scale)" if apply_log else base

        grid.set_axis_labels(_label(x_label, log_x), _label(y_label, log_y))
        grid.set_titles("")
        grid.figure.tight_layout()
        grid.figure.subplots_adjust(top=0.83, bottom=0.22)
        _annotate_facets(grid, col_labels, caption_y=caption_y)
        if log_x or log_y:
            for ax in grid.axes.flat:
                if log_x:
                    ax.set_xscale("log")
                if log_y:
                    ax.set_yscale("log")

    cross_plot_path = _plot_path("corruption_cross_entropy_grid.png")

    def _plot_cross_entropy(data, color, label, **kwargs):
        ax = plt.gca()
        ordered = data.sort_values("num_corruptions")
        x = ordered["num_corruptions"].to_numpy(dtype=float)
        ce_mean, lower, upper = _mean_with_bounds(ordered, "cross_entropy_mean", "cross_entropy_std")
        ent_orig = _prepare_y(ordered["entropy_original_mean"].to_numpy())
        ent_corr = _prepare_y(ordered["entropy_corrupted_mean"].to_numpy())
        x, filtered = _prepare_plot(x, ce_mean, lower, upper, ent_orig, ent_corr)
        if x.size == 0:
            return
        ce_mean, lower, upper, ent_orig, ent_corr = filtered
        cross_label = f"{label} cross-entropy"
        ent_orig_label = f"{label} entropy (original)"
        ent_corr_label = f"{label} entropy (corrupted)"
        ax.plot(x, ce_mean, color=color, label=cross_label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.2)
        ax.plot(x, ent_orig, color=color, linestyle="--", label=ent_orig_label)
        ax.plot(x, ent_corr, color=color, linestyle=":", label=ent_corr_label)

    g = sns.FacetGrid(
        agg_df,
        col="strength_label",
        hue="corruption_descriptor",
        col_order=col_labels,
        sharex=True,
        sharey=True,
        height=4.0,
        aspect=1.3,
    )
    g.map_dataframe(_plot_cross_entropy)
    g.add_legend(title="Corruption regime")
    g.figure.suptitle("Impact of Corrupting LM Next-Token Distributions", fontsize=14)
    _finalize_grid(g, "Corrupted tokens", "Cross entropy and entropy (nats)", caption_y=0.12)
    g.figure.savefig(cross_plot_path, dpi=200)
    LOGGER.info("Wrote cross-entropy plot to %s", cross_plot_path)

    def _plot_symmetric_kl(data, color, label, **kwargs):
        ax = plt.gca()
        ordered = data.sort_values("num_corruptions")
        x = ordered["num_corruptions"].to_numpy(dtype=float)
        skl_mean, lower, upper = _mean_with_bounds(ordered, "symmetric_kl_mean", "symmetric_kl_std")
        x, filtered = _prepare_plot(x, skl_mean, lower, upper)
        if x.size == 0:
            return
        skl_mean, lower, upper = filtered
        ax.plot(x, skl_mean, color=color, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.2)

    sym_plot_path = _plot_path("corruption_symmetric_kl_grid.png")

    g2 = sns.FacetGrid(
        agg_df,
        col="strength_label",
        hue="corruption_descriptor",
        col_order=col_labels,
        sharex=True,
        sharey=True,
        height=4.0,
        aspect=1.3,
    )
    g2.map_dataframe(_plot_symmetric_kl)
    g2.add_legend(title="Corruption regime")
    g2.figure.suptitle("Symmetric KL Under Distribution Corruption", fontsize=14)
    _finalize_grid(g2, "Corrupted tokens", "Symmetric KL divergence", caption_y=0.12)
    g2.figure.savefig(sym_plot_path, dpi=200)
    LOGGER.info("Wrote symmetric-KL plot to %s", sym_plot_path)

    def _plot_wasserstein(data, color, label, **kwargs):
        ax = plt.gca()
        ordered = data.sort_values("num_corruptions")
        x = ordered["num_corruptions"].to_numpy(dtype=float)
        w_mean, lower, upper = _mean_with_bounds(ordered, "wasserstein_mean", "wasserstein_std")
        x, filtered = _prepare_plot(x, w_mean, lower, upper)
        if x.size == 0:
            return
        w_mean, lower, upper = filtered
        ax.plot(x, w_mean, color=color, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.2)

    wasser_plot_path = _plot_path("corruption_wasserstein_grid.png")

    g3 = sns.FacetGrid(
        agg_df,
        col="strength_label",
        hue="corruption_descriptor",
        col_order=col_labels,
        sharex=True,
        sharey=True,
        height=4.0,
        aspect=1.3,
    )
    g3.map_dataframe(_plot_wasserstein)
    g3.add_legend(title="Corruption regime")
    g3.figure.suptitle("Wasserstein Distance Under Distribution Corruption", fontsize=14)
    _finalize_grid(g3, "Corrupted tokens", "Wasserstein distance (W1)", caption_y=0.12)
    g3.figure.savefig(wasser_plot_path, dpi=200)
    LOGGER.info("Wrote Wasserstein plot to %s", wasser_plot_path)

    def _plot_total_variation(data, color, label, **kwargs):
        ax = plt.gca()
        ordered = data.sort_values("num_corruptions")
        x = ordered["num_corruptions"].to_numpy(dtype=float)
        tv_mean, lower, upper = _mean_with_bounds(ordered, "total_variation_mean", "total_variation_std")
        x, filtered = _prepare_plot(x, tv_mean, lower, upper)
        if x.size == 0:
            return
        tv_mean, lower, upper = filtered
        ax.plot(x, tv_mean, color=color, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.2)

    tv_plot_path = _plot_path("corruption_total_variation_grid.png")

    g4 = sns.FacetGrid(
        agg_df,
        col="strength_label",
        hue="corruption_descriptor",
        col_order=col_labels,
        sharex=True,
        sharey=True,
        height=4.0,
        aspect=1.3,
    )
    g4.map_dataframe(_plot_total_variation)
    g4.add_legend(title="Corruption regime")
    g4.figure.suptitle("Total Variation Under Distribution Corruption", fontsize=14)
    _finalize_grid(g4, "Corrupted tokens", "Total variation distance", caption_y=0.12)
    g4.figure.savefig(tv_plot_path, dpi=200)
    LOGGER.info("Wrote total variation plot to %s", tv_plot_path)

    def _plot_mass_shift(data, color, label, **kwargs):
        ax = plt.gca()
        ordered = data.sort_values("num_corruptions")
        x = ordered["num_corruptions"].to_numpy(dtype=float)
        ms_mean, lower, upper = _mean_with_bounds(ordered, "mass_shift_mean", "mass_shift_std")
        x, filtered = _prepare_plot(x, ms_mean, lower, upper)
        if x.size == 0:
            return
        ms_mean, lower, upper = filtered
        ax.plot(x, ms_mean, color=color, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.2)

    mass_shift_plot_path = _plot_path("corruption_mass_shift_grid.png")

    g5 = sns.FacetGrid(
        agg_df,
        col="strength_label",
        hue="corruption_descriptor",
        col_order=col_labels,
        sharex=True,
        sharey=True,
        height=4.0,
        aspect=1.3,
    )
    g5.map_dataframe(_plot_mass_shift)
    g5.add_legend(title="Corruption regime")
    g5.figure.suptitle("Per-Selection Mass Shift Under Corruption", fontsize=14)
    _finalize_grid(g5, "Corrupted tokens", "Selected mass |Δ| sum", caption_y=0.12)
    g5.figure.savefig(mass_shift_plot_path, dpi=200)
    LOGGER.info("Wrote mass shift plot to %s", mass_shift_plot_path)

    def _plot_symmetric_mass_shift(data, color, label, **kwargs):
        ax = plt.gca()
        ordered = data.sort_values("num_corruptions")
        x = ordered["num_corruptions"].to_numpy(dtype=float)
        sms_mean, lower, upper = _mean_with_bounds(ordered, "symmetric_mass_shift_mean", "symmetric_mass_shift_std")
        x, filtered = _prepare_plot(x, sms_mean, lower, upper)
        if x.size == 0:
            return
        sms_mean, lower, upper = filtered
        ax.plot(x, sms_mean, color=color, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.2)

    sym_mass_plot_path = _plot_path("corruption_symmetric_mass_shift_grid.png")

    g6 = sns.FacetGrid(
        agg_df,
        col="strength_label",
        hue="corruption_descriptor",
        col_order=col_labels,
        sharex=True,
        sharey=True,
        height=4.0,
        aspect=1.3,
    )
    g6.map_dataframe(_plot_symmetric_mass_shift)
    g6.add_legend(title="Corruption regime")
    g6.figure.suptitle(
        f"Symmetric Mass Shift (threshold={mass_shift_threshold:.1e})",
        fontsize=14,
    )
    _finalize_grid(g6, "Corrupted tokens", "Symmetric mass shift (|Δp| ≥ threshold)", caption_y=0.12)
    g6.figure.savefig(sym_mass_plot_path, dpi=200)
    LOGGER.info("Wrote symmetric mass shift plot to %s", sym_mass_plot_path)

    def _plot_clamped_mass_shift(data, color, label, **kwargs):
        ax = plt.gca()
        ordered = data.sort_values("num_corruptions")
        x = ordered["num_corruptions"].to_numpy(dtype=float)
        cms_mean, lower, upper = _mean_with_bounds(ordered, "clamped_mass_shift_mean", "clamped_mass_shift_std")
        x, filtered = _prepare_plot(x, cms_mean, lower, upper)
        if x.size == 0:
            return
        cms_mean, lower, upper = filtered
        ax.plot(x, cms_mean, color=color, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.2)

    clamp_mass_plot_path = _plot_path("corruption_clamped_mass_shift_grid.png")

    g7 = sns.FacetGrid(
        agg_df,
        col="strength_label",
        hue="corruption_descriptor",
        col_order=col_labels,
        sharex=True,
        sharey=True,
        height=4.0,
        aspect=1.3,
    )
    g7.map_dataframe(_plot_clamped_mass_shift)
    g7.add_legend(title="Corruption regime")
    g7.figure.suptitle(
        f"Clamped Mass Shift (τ={mass_clamp_value:.2f})",
        fontsize=14,
    )
    _finalize_grid(g7, "Corrupted tokens", "Clamped mass shift (τ)", caption_y=0.12)
    g7.figure.savefig(clamp_mass_plot_path, dpi=200)
    LOGGER.info("Wrote clamped mass shift plot to %s", clamp_mass_plot_path)

    return [
        cross_plot_path,
        sym_plot_path,
        wasser_plot_path,
        tv_plot_path,
        mass_shift_plot_path,
        sym_mass_plot_path,
        clamp_mass_plot_path,
    ]



def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir = resolve_output_dir(args)
    args.output_dir = output_dir
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    counts = resolve_counts(args)
    LOGGER.info("Evaluating corruption counts: %s", counts)
    strengths = resolve_strengths(args)
    LOGGER.info("Evaluating corruption strengths: %s", strengths)
    blend_ratios = resolve_blend_ratios(args)
    if blend_ratios:
        LOGGER.info("Evaluating blend ratios: %s", blend_ratios)

    if not args.corruption_modes:
        raise ValueError("At least one corruption mode must be specified via --corruption-modes")
    if args.transfer_delta_min < 0 or args.transfer_delta_max < 0:
        raise ValueError("Transfer deltas must be non-negative")
    if args.transfer_delta_min > args.transfer_delta_max:
        raise ValueError("--transfer-delta-min cannot exceed --transfer-delta-max")

    estimator_suffix = "" if args.error_estimator == "std" else f"_{args.error_estimator}"

    metadata_config = {
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name or args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "corruption_counts": counts,
        "corruption_strengths": strengths,
        "corruption_targets": list(args.corruption_targets),
        "corruption_modes": list(args.corruption_modes),
        "blend_ratios": blend_ratios,
        "mass_shift_threshold": args.mass_shift_threshold,
        "mass_clamp_value": args.mass_clamp_value,
        "error_estimator": args.error_estimator,
        "transfer_delta_min": args.transfer_delta_min,
        "transfer_delta_max": args.transfer_delta_max,
        "seed": args.seed,
        "artifact_name": args.wandb_artifact_name,
        "plot_log": args.plot_log,
        "plot_log_log": args.plot_log_log,
    }

    if args.local_reload:
        if args.wandb_action in {"reload", "both"}:
            raise ValueError("--local-reload is incompatible with wandb-action=reload/both")
        results_df, summary_df, plot_paths, artifact_files = reload_from_local(
            output_dir,
            args.no_plots,
            args.mass_shift_threshold,
            args.mass_clamp_value,
            args.plot_log,
            args.plot_log_log,
            args.error_estimator,
        )
        if not artifact_files:
            LOGGER.warning("No local artifacts found in %s; nothing to reload", output_dir)
            return
        wandb_run = init_wandb(args, metadata_config)
        log_to_wandb(
            wandb_run,
            results_df,
            summary_df,
            plot_paths,
            artifact_files,
            metadata_config,
        )
        finish_wandb(wandb_run)
        LOGGER.info("Local reload completed from %s", output_dir)
        return

    if args.wandb_action == "reload":
        reloaded_df = reload_from_wandb(args, output_dir)
        if reloaded_df is not None and pd is not None and not args.no_plots:
            maybe_create_plots(
                reloaded_df,
                output_dir,
                args.mass_shift_threshold,
                args.mass_clamp_value,
                log_x=False,
                log_y=False,
                error_estimator=args.error_estimator,
                estimator_tag=estimator_suffix,
            )
            if args.plot_log:
                maybe_create_plots(
                    reloaded_df,
                    output_dir,
                    args.mass_shift_threshold,
                    args.mass_clamp_value,
                    log_x=True,
                    log_y=False,
                    error_estimator=args.error_estimator,
                    estimator_tag=estimator_suffix,
                    suffix="_logx",
                )
            if args.plot_log_log:
                maybe_create_plots(
                    reloaded_df,
                    output_dir,
                    args.mass_shift_threshold,
                    args.mass_clamp_value,
                    log_x=True,
                    log_y=True,
                    error_estimator=args.error_estimator,
                    estimator_tag=estimator_suffix,
                    suffix="_log",
                )
        LOGGER.info("Reload action requested; exiting after artifact download")
        return

    wandb_run = init_wandb(args, metadata_config)

    prompts = load_prompts(args)
    model, tokenizer, device = load_model_and_tokenizer(args)

    distributions = compute_next_token_distributions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_length=args.prompt_length,
        batch_size=args.batch_size,
    )
    LOGGER.info("Collected %d next-token distributions", len(distributions))

    records: List[dict] = []
    transfer_settings = (args.transfer_delta_min, args.transfer_delta_max)
    unsupported_seen: set[Tuple[str, str]] = set()
    # TQDM with different bar levels for each loop
    for dist in tqdm.tqdm(distributions, desc="Processing distributions", leave=False, position=0):
        for strength in tqdm.tqdm(strengths, desc="Processing strengths", leave=False, position=1):
            for mode in tqdm.tqdm(args.corruption_modes, desc="Processing modes", leave=False, position=2):
                for target in tqdm.tqdm(args.corruption_targets, desc="Processing targets", leave=False, position=3):
                    for count in tqdm.tqdm(counts, desc="Processing counts", leave=False, position=4):
                        if target in TRANSFER_ONLY_TARGETS and mode != "transfer":
                            key = (mode, target)
                            if key not in unsupported_seen:
                                LOGGER.warning("Skipping unsupported combination: mode=%s target=%s", mode, target)
                                unsupported_seen.add(key)
                            continue
                        corrupted, metadata = corrupt_distribution(
                            base_probs=dist.probs,
                            num_corruptions=count,
                            target=target,
                            strength=strength,
                            rng=rng,
                            mode=mode,
                            transfer_settings=transfer_settings,
                        )
                        skl = compute_symmetric_kl(dist.probs, corrupted)
                        delta = np.abs(dist.probs - corrupted)
                        sym_mask = delta >= args.mass_shift_threshold
                        symmetric_mass_shift = float(delta[sym_mask].sum())
                        symmetric_mass_count = int(sym_mask.sum())
                        clamped_mass_shift = float(np.minimum(delta, args.mass_clamp_value).sum())
                        clamped_mass_count = int((delta >= args.mass_clamp_value).sum())
                        record = {
                            "sample_id": dist.sample_id,
                            "input_length": dist.input_length,
                            "corruption_target": target,
                            "num_corruptions": int(count),
                            "corruption_strength": float(strength),
                            "corruption_mode": mode,
                            "symmetric_kl": skl,
                            "cross_entropy": compute_cross_entropy(dist.probs, corrupted),
                            "entropy_original": compute_entropy(dist.probs),
                            "entropy_corrupted": compute_entropy(corrupted),
                            "wasserstein_distance": compute_wasserstein_distance(dist.probs, corrupted),
                            "total_variation": compute_total_variation(dist.probs, corrupted),
                            "symmetric_mass_shift": symmetric_mass_shift,
                            "symmetric_mass_count": symmetric_mass_count,
                            "clamped_mass_shift": clamped_mass_shift,
                            "clamped_mass_count": clamped_mass_count,
                            "mass_shift_selected": metadata["mass_shift_selected"],
                            "selected_mass_before": metadata["selected_mass_before"],
                            "selected_mass_after": metadata["selected_mass_after"],
                            "selection_size": metadata["selection_size"],
                            "blend_ratio": np.nan,
                            "blend_ratio_effective": np.nan,
                        }
                        if args.include_prompts:
                            record["prompt"] = dist.prompt_text
                        records.append(record)
                for ratio in tqdm.tqdm(blend_ratios, desc="Processing ratios", leave=False, position=3):
                    for count in tqdm.tqdm(counts, desc="Processing counts", leave=False, position=4):
                        corrupted, metadata = corrupt_distribution_blend(
                            base_probs=dist.probs,
                            num_corruptions=count,
                            high_fraction=ratio,
                            strength=strength,
                            rng=rng,
                            mode=mode,
                            transfer_settings=transfer_settings,
                        )
                        skl = compute_symmetric_kl(dist.probs, corrupted)
                        delta = np.abs(dist.probs - corrupted)
                        sym_mask = delta >= args.mass_shift_threshold
                        symmetric_mass_shift = float(delta[sym_mask].sum())
                        symmetric_mass_count = int(sym_mask.sum())
                        clamped_mass_shift = float(np.minimum(delta, args.mass_clamp_value).sum())
                        clamped_mass_count = int((delta >= args.mass_clamp_value).sum())
                        record = {
                            "sample_id": dist.sample_id,
                            "input_length": dist.input_length,
                            "corruption_target": "blend",
                            "num_corruptions": int(count),
                            "corruption_strength": float(strength),
                            "corruption_mode": mode,
                            "symmetric_kl": skl,
                            "cross_entropy": compute_cross_entropy(dist.probs, corrupted),
                            "entropy_original": compute_entropy(dist.probs),
                            "entropy_corrupted": compute_entropy(corrupted),
                            "wasserstein_distance": compute_wasserstein_distance(dist.probs, corrupted),
                            "total_variation": compute_total_variation(dist.probs, corrupted),
                            "symmetric_mass_shift": symmetric_mass_shift,
                            "symmetric_mass_count": symmetric_mass_count,
                            "clamped_mass_shift": clamped_mass_shift,
                            "clamped_mass_count": clamped_mass_count,
                            "mass_shift_selected": metadata["mass_shift_selected"],
                            "selected_mass_before": metadata["selected_mass_before"],
                            "selected_mass_after": metadata["selected_mass_after"],
                            "selection_size": metadata["selection_size"],
                            "blend_ratio": metadata["requested_high_fraction"],
                            "blend_ratio_effective": metadata["effective_high_fraction"],
                            "high_selection_size": metadata["high_count"],
                            "low_selection_size": metadata["low_count"],
                            "high_mass_before": metadata["high_mass_before"],
                            "high_mass_after": metadata["high_mass_after"],
                            "low_mass_before": metadata["low_mass_before"],
                            "low_mass_after": metadata["low_mass_after"],
                        }
                        if args.include_prompts:
                            record["prompt"] = dist.prompt_text
                        records.append(record)
    LOGGER.info("Generated %d corrupted distributions", len(records))

    results_df = None
    summary_df = None
    artifact_files: List[Path] = []
    plot_paths: List[Path] = []

    if pd is not None:
        results_df = pd.DataFrame.from_records(records)
        results_path = output_dir / f"corruption_results{estimator_suffix}.parquet"
        results_df.to_parquet(results_path, index=False)
        LOGGER.info("Wrote detailed results to %s", results_path)
        artifact_files.append(results_path)
        summary = summarize_results(records, error_estimator=args.error_estimator)
        summary_df = pd.DataFrame(summary)
        summary_path = output_dir / f"corruption_summary{estimator_suffix}.csv"
        summary_df.to_csv(summary_path, index=False)
        LOGGER.info("Wrote summary table to %s", summary_path)
        artifact_files.append(summary_path)
        if not args.no_plots:
            plot_paths = maybe_create_plots(
                results_df,
                output_dir,
                args.mass_shift_threshold,
                args.mass_clamp_value,
                log_x=False,
                log_y=False,
                error_estimator=args.error_estimator,
                estimator_tag=estimator_suffix,
            )
            if args.plot_log:
                plot_paths.extend(
                    maybe_create_plots(
                        results_df,
                        output_dir,
                        args.mass_shift_threshold,
                        args.mass_clamp_value,
                        log_x=True,
                        log_y=False,
                        error_estimator=args.error_estimator,
                        estimator_tag=estimator_suffix,
                        suffix="_logx",
                    )
                )
            if args.plot_log_log:
                plot_paths.extend(
                    maybe_create_plots(
                        results_df,
                        output_dir,
                        args.mass_shift_threshold,
                        args.mass_clamp_value,
                        log_x=True,
                        log_y=True,
                        error_estimator=args.error_estimator,
                        estimator_tag=estimator_suffix,
                        suffix="_log",
                    )
                )
            artifact_files.extend(plot_paths)
    else:
        summary = summarize_results(records, error_estimator=args.error_estimator)
        summary_path = output_dir / f"corruption_summary{estimator_suffix}.txt"
        ensure_output_dir(output_dir)
        with summary_path.open("w", encoding="utf-8") as sink:
            for row in summary:
                sink.write(str(row) + "\n")
        LOGGER.info("Wrote summary text to %s", summary_path)
        artifact_files.append(summary_path)

    log_to_wandb(
        wandb_run,
        results_df,
        summary_df,
        plot_paths,
        artifact_files,
        metadata_config,
    )

    run_path_str = None
    if wandb_run is not None and getattr(wandb_run, "path", None):
        path_components = wandb_run.path
        if isinstance(path_components, (list, tuple)):
            run_path_str = "/".join(path_components)
        else:
            run_path_str = str(path_components)

    finish_wandb(wandb_run)

    if args.wandb_action == "both":
        reload_target = args.wandb_source_run or run_path_str
        if reload_target is None:
            LOGGER.warning("Unable to reload from W&B because no run path is available")
        else:
            reloaded_df = reload_from_wandb(args, output_dir, run_path=reload_target)
            if reloaded_df is not None and pd is not None and not args.no_plots:
                maybe_create_plots(
                    reloaded_df,
                    output_dir,
                    args.mass_shift_threshold,
                    args.mass_clamp_value,
                    log_x=False,
                    log_y=False,
                )
                if args.plot_log:
                    maybe_create_plots(
                        reloaded_df,
                        output_dir,
                        args.mass_shift_threshold,
                        args.mass_clamp_value,
                        log_x=True,
                        log_y=False,
                        suffix="_logx",
                    )
                if args.plot_log_log:
                    maybe_create_plots(
                        reloaded_df,
                        output_dir,
                        args.mass_shift_threshold,
                        args.mass_clamp_value,
                        log_x=True,
                        log_y=True,
                        suffix="_log",
                    )

    LOGGER.info("Done")


if __name__ == "__main__":
    main()
