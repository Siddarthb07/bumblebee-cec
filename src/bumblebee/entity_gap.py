"""Grow entity-gap holdout probes from in-domain templates (unanswerable)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List

from bumblebee.data import EvalExample, load_evals, load_passages
from bumblebee.eval_protocol import is_holdout


def _gid(seed: str) -> str:
    return "gap_x_" + hashlib.sha256(seed.encode()).hexdigest()[:10]


def synthesize_entity_gaps(root: Path, *, target_holdout: int = 30) -> List[dict]:
    """
    Create additional entity-gap questions so holdout count >= target_holdout.
    Strategy: swap a proper-looking token in in-domain questions with another
    passage title — unanswerable by construction.
    """
    passages = load_passages(root / "data" / "corpus")
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    existing = [e for e in evals if e.split == "entity_gap"]
    holdout_n = sum(1 for e in existing if is_holdout(e.id))
    need = max(0, target_holdout - holdout_n)
    if need == 0:
        return []

    titles = [p.title.strip() for p in passages if p.title and len(p.title.strip()) > 2]
    # Prefer short title heads as fake entities
    entities = []
    for t in titles:
        head = t.split()[0]
        if head[:1].isupper() and len(head) > 2:
            entities.append(head)
    entities = list(dict.fromkeys(entities))
    in_dom = [e for e in evals if e.split == "in_domain" and e.answerable]

    rows: List[dict] = []
    # Keep minting until we have enough that hash into holdout, or exhaust attempts
    attempts = 0
    made_holdout = 0
    for ex in in_dom:
        if made_holdout >= need or attempts > need * 40:
            break
        attempts += 1
        ent = entities[attempts % max(len(entities), 1)] if entities else f"ZetaCorp{attempts}"
        q = re.sub(r"\b([A-Z][a-zA-Z]{2,})\b", ent, ex.question, count=1)
        if q == ex.question:
            q = f"What is the official score of {ent} according to the pack?"
        eid = _gid(ex.id + ent + str(attempts))
        row = {
            "id": eid,
            "question": q,
            "split": "entity_gap",
            "answerable": False,
            "answer": None,
            "passage_id": None,
            "category": "entity_gap_synth",
        }
        rows.append(row)
        if is_holdout(eid):
            made_holdout += 1
    return rows


def append_entity_gaps(root: Path, *, target_holdout: int = 30) -> dict:
    path = root / "data" / "evals" / "questions.jsonl"
    evals = load_evals(path)
    before = sum(1 for e in evals if e.split == "entity_gap" and is_holdout(e.id))
    new_rows = synthesize_entity_gaps(root, target_holdout=target_holdout)
    if new_rows:
        with path.open("a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(json.dumps(row) + "\n")
    evals2 = load_evals(path)
    after = sum(1 for e in evals2 if e.split == "entity_gap" and is_holdout(e.id))
    total = sum(1 for e in evals2 if e.split == "entity_gap")
    return {
        "added": len(new_rows),
        "entity_gap_total": total,
        "entity_gap_holdout_before": before,
        "entity_gap_holdout_after": after,
        "target_holdout": target_holdout,
    }
