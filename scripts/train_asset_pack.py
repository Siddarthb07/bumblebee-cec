#!/usr/bin/env python3
"""One-command CEC asset-pack train (transformer-free).

  python scripts/train_asset_pack.py
  python scripts/train_asset_pack.py --skip-ingest --epochs 3 --dim 128

Downloads Dolly/WikiText/Cosmopedia/SmolLM/FineMath/FineWeb-Edu/Wikipedia/HotpotQA/NQ
slices via `datasets` only — never loads HuggingFace transformer weights.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + train CEC asset pack (no transformers)")
    ap.add_argument("--skip-ingest", action="store_true", help="Reuse existing data/ pack")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--mac-fit", action="store_true", help="Use LargeCEC hash model")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-demo", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    if not (root / "pyproject.toml").exists():
        raise SystemExit("Run from the bumblebee repo root")

    # Avoid OpenMP hangs on small Macs; unbuffer child CLI progress
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ["PYTHONUNBUFFERED"] = "1"

    py = [sys.executable, "-u", "-m", "bumblebee.cli"]

    if not args.skip_ingest:
        run(py + ["ingest-assets"])
    else:
        run(py + ["seal"])
        run(py + ["grow-entity-gap", "--target-holdout", "30"])

    train_cmd = py + ["train", "--epochs", str(args.epochs), "--dim", str(args.dim)]
    if args.mac_fit:
        train_cmd.append("--mac-fit")
    run(train_cmd)
    run(py + ["calibrate"])
    run(py + ["eval"])
    if not args.skip_bench:
        run(py + ["bench", "--mode", "all"])
        run(py + ["bench", "--mode", "skills"])
        run(py + ["bench", "--mode", "memory"])
    if not args.skip_demo:
        run(py + ["demo"])
    print("\nAsset pack train complete. Dash: python -m bumblebee.cli dash")
    print("Honesty: CEC correlator (no transformer backend). See docs/RESULTS.md + docs/MEMORY_BENCH.md")


if __name__ == "__main__":
    main()
