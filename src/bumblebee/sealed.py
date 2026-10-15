"""Sealed pack protocol — checksums for passages, eval IDs, holdout salt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bumblebee.data import load_evals, load_passages, sha256_file
from bumblebee.eval_protocol import calibration_split, is_holdout, train_eval_split


SEAL_VERSION = 1
HOLDOUT_SALT = "bumblebee-holdout-v1"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_text(text: str) -> str:
    return _hash_bytes(text.encode("utf-8"))


def build_seal(root: Path) -> Dict[str, Any]:
    corpus = root / "data" / "corpus"
    evals_path = root / "data" / "evals" / "questions.jsonl"
    passages = load_passages(corpus)
    evals = load_evals(evals_path)
    train, test = train_eval_split(evals)
    tr, cal, te = calibration_split(evals)

    passage_ids = sorted(p.id for p in passages)
    eval_ids = sorted(e.id for e in evals)
    test_ids = sorted(e.id for e in test)
    cal_ids = sorted(e.id for e in cal)
    holdout_ids = sorted(e.id for e in evals if is_holdout(e.id))

    seal = {
        "version": SEAL_VERSION,
        "holdout_salt": HOLDOUT_SALT,
        "n_passages": len(passages),
        "n_evals": len(evals),
        "n_train": len(train),
        "n_test": len(test),
        "n_cal": len(cal),
        "passages_sha256": sha256_file(corpus / "passages.jsonl"),
        "evals_sha256": sha256_file(evals_path),
        "passage_ids_sha256": _hash_text("\n".join(passage_ids)),
        "eval_ids_sha256": _hash_text("\n".join(eval_ids)),
        "test_ids_sha256": _hash_text("\n".join(test_ids)),
        "cal_ids_sha256": _hash_text("\n".join(cal_ids)),
        "holdout_ids_sha256": _hash_text("\n".join(holdout_ids)),
        "note": (
            "Reproducibility seal — not external validity. "
            "Checksums detect drift; they do not prove no leakage by themselves."
        ),
    }
    seal["seal_sha256"] = _hash_text(
        json.dumps({k: v for k, v in seal.items() if k != "seal_sha256"}, sort_keys=True)
    )
    return seal


def write_seal(root: Path) -> Path:
    sealed_dir = root / "data" / "sealed"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    seal = build_seal(root)
    path = sealed_dir / "SEAL.json"
    path.write_text(json.dumps(seal, indent=2), encoding="utf-8")
    # Convenience copies of ID lists (no gold answers)
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    _, test = train_eval_split(evals)
    (sealed_dir / "test_ids.txt").write_text(
        "\n".join(sorted(e.id for e in test)) + "\n", encoding="utf-8"
    )
    return path


def verify_seal(root: Path) -> Tuple[bool, List[str]]:
    path = root / "data" / "sealed" / "SEAL.json"
    problems: List[str] = []
    if not path.exists():
        return False, ["missing data/sealed/SEAL.json — run: bumblebee seal"]
    expected = json.loads(path.read_text(encoding="utf-8"))
    current = build_seal(root)
    for key in (
        "passages_sha256",
        "evals_sha256",
        "passage_ids_sha256",
        "eval_ids_sha256",
        "test_ids_sha256",
        "cal_ids_sha256",
        "holdout_ids_sha256",
        "n_passages",
        "n_evals",
    ):
        if expected.get(key) != current.get(key):
            problems.append(f"seal mismatch on {key}: sealed={expected.get(key)} now={current.get(key)}")
    return (len(problems) == 0), problems
