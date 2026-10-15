"""Miss log — record product asks for targeted pack growth (no transformers)."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from bumblebee.answer_craft import focus_terms


def _path(root: Path) -> Path:
    d = Path(root) / "artifacts" / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d / "miss_log.jsonl"


def record_ask(root: Path, question: str, out: dict) -> None:
    """Append one ask row; mark misses for pack growth."""
    status = out.get("status")
    source = out.get("source") or out.get("source_label") or ""
    miss = status == "refuse" or source in {"web_unverified", "refuse"}
    row = {
        "ts": time.time(),
        "question": (question or "")[:240],
        "status": status,
        "source": source,
        "source_label": out.get("source_label") or source,
        "reason": out.get("reason"),
        "confidence": float(out.get("confidence") or 0),
        "passage_id": out.get("passage_id") or "",
        "miss": bool(miss),
        "focus": focus_terms(question)[:12],
    }
    path = _path(root)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_misses(root: Path, *, limit: int = 500) -> List[dict]:
    path = _path(root)
    if not path.exists():
        return []
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("miss"):
            rows.append(row)
    return rows[-limit:]


def miss_focus_report(root: Path, *, top_k: int = 40) -> dict:
    """Aggregate missing focus terms for targeted ingest."""
    misses = load_misses(root)
    ctr: Counter = Counter()
    for row in misses:
        for t in row.get("focus") or []:
            if isinstance(t, str) and len(t) >= 4:
                ctr[t.lower()] += 1
    top = ctr.most_common(top_k)
    return {
        "n_misses": len(misses),
        "top_focus_terms": [{"term": t, "count": c} for t, c in top],
        "path": str(_path(root)),
    }


def suggest_pack_growth(root: Path, *, top_k: int = 25) -> Path:
    """
    Write artifacts/pack_growth_suggestions.json from miss focus terms.

    Operators can use these terms to pull Wikipedia/NQ slices — no auto HF
    download here (keeps offline/transformer-free).
    """
    report = miss_focus_report(root, top_k=top_k)
    out = Path(root) / "artifacts" / "pack_growth_suggestions.json"
    payload = {
        **report,
        "note": (
            "Targeted pack growth from product misses. Prefer ingesting passages "
            "that contain these focus terms (wiki leads / NQ contexts). "
            "Do not bump CEC dim. No transformers."
        ),
        "commands": [
            "python -m bumblebee.cli miss-report",
            "python -m bumblebee.cli ingest-assets  # then filter by focus terms",
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
