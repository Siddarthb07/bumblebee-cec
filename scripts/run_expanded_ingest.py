#!/usr/bin/env python3
"""Run expanded asset-pack ingest (no transformer weights)."""
from __future__ import annotations

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

print("start ingest", flush=True)
from bumblebee.asset_ingest import ingest_assets_to_project

summary = ingest_assets_to_project(ROOT)
print("DONE passages=", summary.get("n_passages"), "evals=", summary.get("n_evals"), flush=True)
print("ERRORS", summary.get("errors"), flush=True)
for s in summary.get("sources", []):
    print(" SRC", s.get("source"), "chunks=", s.get("n_chunks"), "evals=", s.get("n_evals"), flush=True)
if summary.get("seal"):
    print(" SEAL", summary["seal"], flush=True)
