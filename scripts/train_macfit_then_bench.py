#!/usr/bin/env python3
"""Full mac-fit CEC train (~500M–1B sparse-hash params), then benches.

  python scripts/train_macfit_then_bench.py
  python scripts/train_macfit_then_bench.py --target-params 1000000000 --epochs 4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ["PYTHONUNBUFFERED"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-params", type=int, default=1_500_000_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    from bumblebee.train import clear_runtime_cache, train_model_mac_fit, evaluate
    from bumblebee.calibration import fit_gate_temperature

    clear_runtime_cache()
    print(f"=== MAC-FIT TRAIN target={args.target_params:,} epochs={args.epochs} ===", flush=True)
    path = train_model_mac_fit(
        ROOT,
        target_params=args.target_params,
        epochs=args.epochs,
    )
    print(f"=== TRAIN DONE saved={path} ===", flush=True)

    if args.skip_bench:
        return

    clear_runtime_cache()
    print("=== CALIBRATE ===", flush=True)
    print(fit_gate_temperature(ROOT), flush=True)

    clear_runtime_cache()
    print("=== EVAL ===", flush=True)
    rep = evaluate(ROOT)
    print("eval in_domain_f1", rep.get("in_domain_f1"), "n", rep.get("n"), flush=True)

    print("=== BENCH ALL ===", flush=True)
    from bumblebee.results import bench_all

    bench_all(ROOT)

    print("=== SKILLS ===", flush=True)
    from bumblebee.skills_bench import run_skills_bench

    sk = run_skills_bench(ROOT, mode="auto")
    print("skills", sk.get("passed"), "/", sk.get("n"), flush=True)

    print("=== MEMORY ===", flush=True)
    from bumblebee.memory_bench import run_memory_bench

    mem = run_memory_bench(ROOT, n_asks=8)
    print(
        "peak_rss",
        mem["cec"]["peak_rss_gb"],
        "params_m",
        mem["cec"]["n_params_m"],
        "passages",
        mem["cec"]["n_passages"],
        "ratio_vs_7b",
        mem["comparison"]["ratio_vs_mistral7b_fp16"],
        flush=True,
    )

    print("=== ADVERSARIAL ===", flush=True)
    from bumblebee.adversarial import run_adversarial_bench, write_adversarial_file
    from bumblebee.gate import AskResult
    from bumblebee.train import ask_question_fast

    write_adversarial_file(ROOT)
    clear_runtime_cache()

    def ask_fn(ex):
        d = ask_question_fast(ROOT, ex.question)
        return AskResult(
            d["status"],
            d.get("answer") or "",
            float(d.get("confidence") or 0.0),
            d.get("reason"),
            d.get("citation_span") or "",
            int(d.get("citation_start") or 0),
            int(d.get("citation_end") or 0),
        )

    run_adversarial_bench(ROOT, ask_fn)
    print("=== ALL DONE ===", flush=True)
    print("See docs/RESULTS.md docs/MEMORY_BENCH.md docs/SKILLS_BENCH.md", flush=True)


if __name__ == "__main__":
    main()
