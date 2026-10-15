#!/usr/bin/env python3
"""One-command demo: seal → (optional train) → calibrate → eval → bench → demo path.

Usage:
  python scripts/one_command_demo.py
  python scripts/one_command_demo.py --train --epochs 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Retrain small CEC before benches")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--skip-adversarial", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true", help="Ablation is slow; skip by default unless --ablation")
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()
    root = Path.cwd()
    if not (root / "data" / "corpus" / "passages.jsonl").exists():
        raise SystemExit("Run from repo root with data/corpus present")

    py = [sys.executable, "-m", "bumblebee.cli"]
    run(py + ["ingest"])
    run(py + ["seal"])
    run(py + ["grow-entity-gap", "--target-holdout", "30"])
    if args.train or not (root / "artifacts" / "cec.pt").exists():
        run(py + ["train", "--epochs", str(args.epochs), "--dim", str(args.dim)])
        run(py + ["calibrate"])
    run(py + ["eval"])
    run(py + ["bench", "--mode", "all"])
    run(py + ["bench", "--mode", "skills"])
    if not args.skip_adversarial:
        run(py + ["bench", "--mode", "adversarial"])
    if args.ablation and not args.skip_ablation:
        run(py + ["bench", "--mode", "ablation", "--ablation-epochs", "1"])
    run(py + ["demo"])
    print("\nDone. Open dash with: python -m bumblebee.cli dash")
    print("Pack builder: python -m bumblebee.cli pack-build ./my_notes")


if __name__ == "__main__":
    main()
