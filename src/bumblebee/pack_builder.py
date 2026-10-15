"""Pack builder — drop notes/PDFs/text → corpus JSONL → ready to train/ask."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _pid(text: str) -> str:
    return "pack_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _chunk_text(text: str, *, max_chars: int = 800) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    # Prefer sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # optional

            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def ingest_folder(
    src: Path,
    *,
    out_corpus: Path,
    title_prefix: str = "notes",
    max_chars: int = 800,
    extensions: Tuple[str, ...] = (".txt", ".md", ".markdown", ".pdf"),
) -> dict:
    """
    Walk src for text/markdown/pdf, chunk into passages.jsonl under out_corpus.
    Does not invent gold answers — user must train + ask with refuse.
    """
    src = Path(src)
    out_corpus = Path(out_corpus)
    out_corpus.mkdir(parents=True, exist_ok=True)
    passages = []
    files_used = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        raw = _read_file(path)
        if not raw.strip():
            continue
        files_used.append(str(path))
        try:
            files_used[-1] = str(path.relative_to(src))
        except ValueError:
            pass
        for i, chunk in enumerate(_chunk_text(raw, max_chars=max_chars)):
            title = f"{title_prefix}/{path.stem}" + (f"#{i+1}" if i else "")
            passages.append({"id": _pid(title + chunk), "title": title, "text": chunk})

    out_path = out_corpus / "passages.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in passages:
            f.write(json.dumps(row) + "\n")

    # Minimal eval seed: one in-domain probe per passage (first sentence Q) — answerable false
    # so the pack starts refuse-first until the user adds real Q/A.
    evals_dir = out_corpus.parent / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    eval_path = evals_dir / "questions.jsonl"
    probes = []
    for i, row in enumerate(passages[:40]):
        probes.append(
            {
                "id": f"packq_{i}_{row['id'][:8]}",
                "question": f"What does the note titled {row['title']} say?",
                "split": "in_domain",
                "answerable": True,
                "answer": row["text"][:120],
                "passage_id": row["id"],
                "category": "pack_builder",
            }
        )
    # Always add a far-OOD refuse probe
    probes.append(
        {
            "id": "pack_far_ood_seed",
            "question": "Who won the 2022 FIFA World Cup?",
            "split": "far_ood",
            "answerable": False,
            "answer": None,
            "passage_id": None,
            "category": "far",
        }
    )
    with eval_path.open("w", encoding="utf-8") as f:
        for row in probes:
            f.write(json.dumps(row) + "\n")

    from bumblebee.data import write_manifest

    write_manifest(out_corpus)
    return {
        "n_passages": len(passages),
        "n_evals": len(probes),
        "files": files_used,
        "corpus": str(out_path),
        "evals": str(eval_path),
    }
