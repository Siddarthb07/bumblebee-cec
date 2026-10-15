from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Passage:
    id: str
    title: str
    text: str


@dataclass
class EvalExample:
    id: str
    question: str
    split: str  # in_domain | paraphrase_ood | entity_gap | far_ood
    answerable: bool
    answer: Optional[str]
    passage_id: Optional[str]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_passages(corpus_dir: Path) -> List[Passage]:
    path = corpus_dir / "passages.jsonl"
    out: List[Passage] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(Passage(id=row["id"], title=row.get("title", ""), text=row["text"]))
    return out


def load_evals(evals_path: Path) -> List[EvalExample]:
    out: List[EvalExample] = []
    for line in evals_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(
            EvalExample(
                id=row["id"],
                question=row["question"],
                split=row["split"],
                answerable=bool(row["answerable"]),
                answer=row.get("answer"),
                passage_id=row.get("passage_id"),
            )
        )
    return out


def passage_index(passages: List[Passage]) -> Dict[str, Passage]:
    return {p.id: p for p in passages}


def write_manifest(corpus_dir: Path) -> Path:
    files = sorted(p for p in corpus_dir.glob("*") if p.is_file())
    manifest = {
        "files": {p.name: sha256_file(p) for p in files if p.name != "MANIFEST.json"},
    }
    out = corpus_dir / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def verify_manifest(corpus_dir: Path) -> Tuple[bool, List[str]]:
    """Return (ok, list of mismatch messages). Missing MANIFEST → not ok."""
    path = corpus_dir / "MANIFEST.json"
    if not path.exists():
        return False, ["MANIFEST.json missing"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("files", {})
    problems: List[str] = []
    for name, digest in expected.items():
        fpath = corpus_dir / name
        if not fpath.exists():
            problems.append(f"missing file: {name}")
            continue
        got = sha256_file(fpath)
        if got != digest:
            problems.append(f"hash mismatch: {name}")
    for fpath in corpus_dir.glob("*"):
        if fpath.is_file() and fpath.name != "MANIFEST.json" and fpath.name not in expected:
            problems.append(f"untracked file: {fpath.name}")
    return (len(problems) == 0), problems
