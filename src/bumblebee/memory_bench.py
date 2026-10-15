"""Memory / scale benchmark: CEC footprint vs modern LLM serving estimates.

Measures local CEC disk + RSS + params, then compares against catalog footprints
for common dense LLMs (not loaded — published FP16/INT8 estimates).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# Catalog: typical serving memory (weights only), not including KV cache / framework overhead.
# FP16 ≈ 2 bytes/param; INT8 ≈ 1 byte/param. Numbers are order-of-magnitude honest estimates.
LLM_CATALOG: List[Dict[str, Any]] = [
    {"id": "gpt2", "params_b": 0.124, "fp16_gb": 0.25, "int8_gb": 0.13, "class": "small"},
    {"id": "gpt2-medium", "params_b": 0.355, "fp16_gb": 0.71, "int8_gb": 0.36, "class": "small"},
    {"id": "SmolLM-360M", "params_b": 0.36, "fp16_gb": 0.72, "int8_gb": 0.36, "class": "small"},
    {"id": "Pythia-410M", "params_b": 0.41, "fp16_gb": 0.82, "int8_gb": 0.41, "class": "small"},
    {"id": "Llama-3.2-1B", "params_b": 1.0, "fp16_gb": 2.0, "int8_gb": 1.0, "class": "1B"},
    {"id": "Llama-3.2-3B", "params_b": 3.0, "fp16_gb": 6.0, "int8_gb": 3.0, "class": "3B"},
    {"id": "CEC-macfit-1.5B-class", "params_b": 1.5, "fp16_gb": 3.0, "int8_gb": 1.5, "class": "cec"},
    {"id": "Mistral-7B", "params_b": 7.0, "fp16_gb": 14.0, "int8_gb": 7.0, "class": "7B"},
    {"id": "Llama-3.1-8B", "params_b": 8.0, "fp16_gb": 16.0, "int8_gb": 8.0, "class": "8B"},
    {"id": "Llama-3.1-70B", "params_b": 70.0, "fp16_gb": 140.0, "int8_gb": 70.0, "class": "70B"},
    {"id": "Llama-3.1-405B", "params_b": 405.0, "fp16_gb": 810.0, "int8_gb": 405.0, "class": "frontier"},
    {
        "id": "GPT-4-class (public est.)",
        "params_b": 1000.0,
        "fp16_gb": 2000.0,
        "int8_gb": 1000.0,
        "class": "frontier",
        "note": "Order-of-magnitude; not an official OpenAI figure",
    },
]


def _rss_gb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except Exception:
        return float("nan")


def _file_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024**3)


def _dir_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024**3)


def measure_cec_memory(root: Path, *, n_asks: int = 8) -> dict:
    """Load CEC runtime, run a few asks, report peak RSS and artifact sizes."""
    from bumblebee.data import load_evals, load_passages
    from bumblebee.train import ask_question_fast, clear_runtime_cache, load_runtime

    clear_runtime_cache()
    baseline = _rss_gb()
    t0 = time.perf_counter()
    loaded = load_runtime(root)
    model = loaded[0]
    n_params = int(sum(p.numel() for p in model.parameters()))
    after_load = _rss_gb()

    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    questions = [e.question for e in evals if e.answerable][:n_asks]
    if len(questions) < n_asks:
        questions += ["What is Bumblebee?", "Who are you?", "What can you do?"]
        questions = questions[:n_asks]

    peak = after_load
    latencies: List[float] = []
    for q in questions:
        t1 = time.perf_counter()
        ask_question_fast(root, q)
        latencies.append(time.perf_counter() - t1)
        peak = max(peak, _rss_gb())

    passages = load_passages(root / "data" / "corpus")
    cfg_path = root / "artifacts" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    return {
        "baseline_rss_gb": round(baseline, 4),
        "after_load_rss_gb": round(after_load, 4),
        "peak_rss_gb": round(peak, 4),
        "delta_rss_gb": round(peak - baseline, 4),
        "load_seconds": round(time.perf_counter() - t0, 3),
        "ask_latency_mean_s": round(sum(latencies) / max(len(latencies), 1), 4),
        "ask_latency_p50_s": round(sorted(latencies)[len(latencies) // 2], 4) if latencies else None,
        "n_asks": len(latencies),
        "n_params": n_params,
        "n_params_m": round(n_params / 1e6, 3),
        "checkpoint_gb": round(_file_gb(root / "artifacts" / "cec.pt"), 4),
        "corpus_gb": round(_file_gb(root / "data" / "corpus" / "passages.jsonl"), 4),
        "data_dir_gb": round(_dir_gb(root / "data"), 4),
        "artifacts_dir_gb": round(_dir_gb(root / "artifacts"), 4),
        "n_passages": len(passages),
        "config": {
            "dim": cfg.get("dim"),
            "arch": cfg.get("arch", "cec"),
            "mass_fn": cfg.get("mass_fn"),
            "n_params_cfg": cfg.get("n_params"),
        },
        "honesty": (
            "CEC peak RSS includes Python/torch + corpus index; "
            "LLM catalog rows are weight-only estimates and understate KV-cache cost."
        ),
    }


def compare_to_llms(cec: dict) -> dict:
    """Where does CEC peak RSS sit vs catalog LLM weight footprints?"""
    peak = float(cec.get("peak_rss_gb") or 0.0)
    smaller_fp16 = [m for m in LLM_CATALOG if m["fp16_gb"] > peak]
    larger_fp16 = [m for m in LLM_CATALOG if m["fp16_gb"] <= peak]
    # "beats" = CEC uses less memory than this model's FP16 weights
    beats = [m["id"] for m in LLM_CATALOG if peak < m["fp16_gb"]]
    closest = min(LLM_CATALOG, key=lambda m: abs(m["fp16_gb"] - peak))
    return {
        "cec_peak_rss_gb": peak,
        "cec_params_m": cec.get("n_params_m"),
        "beats_fp16_weight_footprint": beats,
        "closest_catalog_fp16": closest,
        "ratio_vs_mistral7b_fp16": round(peak / 14.0, 4) if peak else None,
        "ratio_vs_llama70b_fp16": round(peak / 140.0, 4) if peak else None,
        "ratio_vs_405b_fp16": round(peak / 810.0, 4) if peak else None,
        "note": (
            "Ratios use catalog FP16 weight estimates only. Real LLM serving "
            "(KV cache, batch) is higher; CEC sparse-hash params ≠ dense FLOPs."
        ),
        "n_catalog_smaller_or_eq": len(larger_fp16),
        "n_catalog_larger": len(smaller_fp16),
    }


def run_memory_bench(root: Path, *, n_asks: int = 8) -> dict:
    cec = measure_cec_memory(root, n_asks=n_asks)
    cmp_ = compare_to_llms(cec)
    report = {
        "version": "v1_memory_scale",
        "cec": cec,
        "llm_catalog": LLM_CATALOG,
        "comparison": cmp_,
        "reading": (
            "This bench answers 'how far does CEC scale on memory vs modern LLMs?' "
            "It does not claim answer-quality parity with frontier models."
        ),
    }
    out = root / "artifacts" / "bench_memory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_memory_md(root, report)
    return report


def _write_memory_md(root: Path, report: dict) -> Path:
    cec = report["cec"]
    cmp_ = report["comparison"]
    lines = [
        "# Memory / scale bench",
        "",
        "Generated by `python -m bumblebee.cli bench --mode memory`.",
        "",
        report.get("reading", ""),
        "",
        "## CEC measured footprint",
        "",
        f"| Metric | Value |",
        f"| --- | ---: |",
        f"| Params | {cec.get('n_params_m')} M |",
        f"| Checkpoint (`cec.pt`) | {cec.get('checkpoint_gb')} GB |",
        f"| Corpus (`passages.jsonl`) | {cec.get('corpus_gb')} GB |",
        f"| Passages | {cec.get('n_passages')} |",
        f"| Peak RSS (load + {cec.get('n_asks')} asks) | {cec.get('peak_rss_gb')} GB |",
        f"| Δ RSS vs baseline | {cec.get('delta_rss_gb')} GB |",
        f"| Mean ask latency | {cec.get('ask_latency_mean_s')} s |",
        "",
        f"_{cec.get('honesty')}_",
        "",
        "## vs modern LLM weight catalogs (not loaded)",
        "",
        "| Model | Params (B) | FP16 weights (GB) | INT8 (GB) | CEC peak RSS smaller? |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    peak = float(cec.get("peak_rss_gb") or 0.0)
    for m in report["llm_catalog"]:
        smaller = "yes" if peak < float(m["fp16_gb"]) else "no"
        lines.append(
            f"| {m['id']} | {m['params_b']} | {m['fp16_gb']} | {m['int8_gb']} | {smaller} |"
        )
    lines.extend(
        [
            "",
            "## Scale reading",
            "",
            f"- CEC peak RSS **{peak:.3f} GB** vs Mistral-7B FP16 weights (~14 GB): "
            f"**{cmp_.get('ratio_vs_mistral7b_fp16')}×**",
            f"- vs Llama-3.1-70B FP16 (~140 GB): **{cmp_.get('ratio_vs_llama70b_fp16')}×**",
            f"- vs 405B-class FP16 (~810 GB): **{cmp_.get('ratio_vs_405b_fp16')}×**",
            f"- Closest catalog FP16 footprint: `{cmp_.get('closest_catalog_fp16', {}).get('id')}`",
            "",
            "CEC **beats** these FP16 weight footprints on raw memory:",
            "",
        ]
    )
    for mid in cmp_.get("beats_fp16_weight_footprint") or []:
        lines.append(f"- {mid}")
    lines.extend(
        [
            "",
            "## Honesty",
            "",
            "- Catalog rows are **weight-only**; real LLM inference adds KV cache and framework overhead.",
            "- CEC headline params may include sparse hash rows; do not equate to dense transformer FLOPs.",
            "- Quality is scored separately (`bench --mode all`); this card is **memory scale only**.",
            "",
            "## Artifacts",
            "",
            "- `artifacts/bench_memory.json`",
            "- `docs/MEMORY_BENCH.md`",
            "",
        ]
    )
    path = root / "docs" / "MEMORY_BENCH.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
