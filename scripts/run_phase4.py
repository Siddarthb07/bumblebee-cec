#!/usr/bin/env python
"""Phase-4 packaging: refresh metrics, RESULTS.md, and depth PDF."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=ROOT)


def main() -> None:
    run([sys.executable, "-m", "pytest", "-q"])
    if not (ROOT / "artifacts" / "cec.pt").exists():
        run([sys.executable, "-m", "bumblebee.cli", "train", "--epochs", "12"])
    run([sys.executable, "-m", "bumblebee.cli", "eval"])
    run([sys.executable, "-m", "bumblebee.cli", "bench", "--mode", "all"])
    run([sys.executable, "scripts/build_depth_pdf.py"])
    print("Phase 4 packaging complete.")
    print(" - docs/RESULTS.md")
    print(" - output/pdf/bumblebee_depth.pdf")
    print(" - artifacts/bench_all.json")


if __name__ == "__main__":
    main()
