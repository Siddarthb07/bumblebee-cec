#!/usr/bin/env python3
"""Probe max dense CEC that fits RAM, train it, then run benches.

Replaces sparse-hash 1.5B theater with real dense FLOPs at the largest dim that fits.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["PYTHONUNBUFFERED"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)


def _avail_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / (1024**3)


def estimate_dense_params(vocab: int, dim: int) -> int:
    """Rough CECModel param count (dense emb + dual GRU paths + heads)."""
    emb = vocab * dim
    # LocalStateEncoder: emb + GRUCell(dim,dim) ≈ 3*dim^2 + 3*dim, used twice encode paths share weights
    gru = 3 * dim * dim + 3 * dim
    correlator = 2 * (dim * dim + dim)
    gate = 2 * dim + 2
    decoder = 2 * (dim + 1)
    # GenerativeAnswerDecoder: emb share or separate — model has own answer emb
    ans_emb = vocab * dim
    ans_gru = 3 * dim * dim + 3 * dim
    ans_out = dim * vocab + vocab
    cond = 2 * dim * dim + dim
    return int(emb + gru + correlator + gate + decoder + ans_emb + ans_gru + ans_out + cond)


def probe_max_dim(vocab: int, *, headroom_gb: float = 0.8) -> tuple[int, int, float]:
    """
    Find largest dim where Adam state (~12 bytes/param) fits.
    Uses max(available-headroom, total-3GB) so pressure from leftover processes
    does not collapse us to a tiny model when the machine is 8GB-class.
    """
    import psutil

    vm = psutil.virtual_memory()
    avail = vm.available / (1024**3)
    total = vm.total / (1024**3)
    budget = max(avail - headroom_gb, min(4.5, total - 3.0))
    print(f"probe budget_gb={budget:.2f} (avail={avail:.2f} total={total:.2f})", flush=True)
    best = (128, estimate_dense_params(vocab, 128), budget)
    for dim in (128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024):
        n = estimate_dense_params(vocab, dim)
        need = n * 12 / (1024**3)  # w+m+v fp32-ish
        if need > budget:
            print(f"probe dim={dim} params={n/1e6:.1f}M need={need:.2f}GB > budget — stop", flush=True)
            break
        try:
            import torch
            from bumblebee.model import CECModel

            gc.collect()
            m = CECModel(vocab, dim=dim, pad_id=0)
            n_real = sum(p.numel() for p in m.parameters())
            del m
            gc.collect()
            best = (dim, n_real, need)
            print(f"probe dim={dim} params={n_real/1e6:.1f}M est_adam_gb={need:.2f} ok", flush=True)
        except Exception as exc:
            print(f"probe dim={dim} FAIL {exc}", flush=True)
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--dim", type=int, default=0, help="0 = auto-probe max dense")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    import torch
    from bumblebee.data import load_evals, load_passages
    from bumblebee.tokenize import Vocab
    from bumblebee.train import clear_runtime_cache, train_model, evaluate
    from bumblebee.calibration import fit_gate_temperature

    print(f"avail_gb={_avail_gb():.2f}", flush=True)
    passages = load_passages(ROOT / "data" / "corpus")
    evals = load_evals(ROOT / "data" / "evals" / "questions.jsonl")
    texts = [p.text for p in passages] + [e.question for e in evals] + [e.answer or "" for e in evals]
    # Cap vocab for denser emb at fixed RAM (still dense FLOPs on used tokens)
    vocab = Vocab.build(texts, min_count=2)
    # If still huge, rebuild with higher min_count
    if len(vocab.itos) > 40_000:
        vocab = Vocab.build(texts, min_count=3)
    print(f"vocab={len(vocab.itos)} passages={len(passages)}", flush=True)

    if args.dim and args.dim > 0:
        dim = args.dim
        n = estimate_dense_params(len(vocab.itos), dim)
        print(f"forced dim={dim} est_params={n/1e6:.1f}M", flush=True)
    else:
        dim, n, need = probe_max_dim(len(vocab.itos))
        print(f"selected dense dim={dim} params≈{n/1e6:.1f}M adam≈{need:.2f}GB", flush=True)

    # Persist probe choice
    probe = {
        "arch": "dense_cec",
        "dim": dim,
        "vocab": len(vocab.itos),
        "est_params": n if args.dim else n,
        "avail_gb": _avail_gb(),
        "note": "Real dense FLOPs CEC — not sparse-hash param theater",
    }
    (ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / "dense_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")

    clear_runtime_cache()
    # train_model rebuilds vocab from data — patch by writing hint in config after
    print(f"=== DENSE TRAIN dim={dim} epochs={args.epochs} ===", flush=True)
    # Lower batch pressure: monkey via env consumed in train if we add it; for now use smaller max lens
    path = train_model(
        ROOT,
        epochs=args.epochs,
        dim=dim,
        max_p=128 if dim >= 384 else 192,
        max_q=48 if dim >= 384 else 64,
        lr=8e-4 if dim >= 256 else 1e-3,
    )
    print(f"=== TRAIN DONE {path} ===", flush=True)

    cfg_path = ROOT / "artifacts" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg["arch"] = "dense_cec"
    cfg["dense_probe"] = probe
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if args.skip_bench:
        return

    clear_runtime_cache()
    print("=== CALIBRATE ===", flush=True)
    print(fit_gate_temperature(ROOT), flush=True)
    clear_runtime_cache()
    print("=== EVAL ===", flush=True)
    rep = evaluate(ROOT)
    print("eval in_domain_f1", rep.get("in_domain_f1"), flush=True)
    from bumblebee.results import bench_all
    from bumblebee.skills_bench import run_skills_bench
    from bumblebee.memory_bench import run_memory_bench

    print("=== BENCH ALL ===", flush=True)
    bench_all(ROOT)
    print("=== SKILLS ===", flush=True)
    sk = run_skills_bench(ROOT, mode="auto")
    print("skills", sk.get("passed"), "/", sk.get("n"), flush=True)
    print("=== MEMORY ===", flush=True)
    mem = run_memory_bench(ROOT, n_asks=8)
    print(mem["cec"]["n_params_m"], "M peak_rss", mem["cec"]["peak_rss_gb"], flush=True)
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
