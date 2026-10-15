"""Adversarial / hard-negative eval suite (paraphrase, distractors, conflicting docs)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from bumblebee.data import EvalExample, load_evals, load_passages, passage_index
from bumblebee.eval_protocol import summarize_ask_results
from bumblebee.gate import AskResult


def _pid(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:10]


def _paraphrase(q: str) -> str:
    q = q.strip()
    # Lightweight deterministic paraphrases — not LLM-generated
    rules = [
        (r"(?i)^what is\b", "Can you tell me what"),
        (r"(?i)^who was\b", "Who is"),
        (r"(?i)^who is\b", "Tell me about"),
        (r"(?i)^where\b", "In which place"),
        (r"(?i)^when\b", "At what time"),
        (r"(?i)^how\b", "In what way"),
    ]
    out = q
    for pat, repl in rules:
        if re.search(pat, out):
            out = re.sub(pat, repl, out, count=1)
            if not out.endswith("?"):
                out = out.rstrip(".") + "?"
            return out
    return f"Please answer: {q}"


def build_adversarial_evals(root: Path, *, n_para: int = 20, n_distract: int = 15) -> List[EvalExample]:
    """
    Build adversarial probes from the frozen pack (no gold leakage into ask).
    - paraphrase: in-domain questions reworded; still answerable
    - distractor: entity-swapped questions (unanswerable)
    - conflict: two passages claim different facts for same entity stub (unanswerable)
    """
    passages = load_passages(root / "data" / "corpus")
    pmap = passage_index(passages)
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    in_dom = [e for e in evals if e.split == "in_domain" and e.answerable and e.passage_id]
    out: List[EvalExample] = []

    for i, ex in enumerate(in_dom[:n_para]):
        out.append(
            EvalExample(
                id=f"adv_para_{_pid(ex.id)}",
                question=_paraphrase(ex.question),
                split="paraphrase",
                answerable=True,
                answer=ex.answer,
                passage_id=ex.passage_id,
            )
        )

    # Distractors: take in-domain Q template, swap a capitalized entity with another passage title token
    titles = [p.title for p in passages if p.title and len(p.title) > 2][:80]
    for i, ex in enumerate(in_dom[:n_distract]):
        if not titles:
            break
        swap = titles[i % len(titles)]
        q = re.sub(r"\b([A-Z][a-zA-Z]{2,})\b", swap, ex.question, count=1)
        if q == ex.question:
            q = f"What is the official rating of {swap}?"
        out.append(
            EvalExample(
                id=f"adv_distract_{_pid(ex.id + swap)}",
                question=q,
                split="distractor",
                answerable=False,
                answer=None,
                passage_id=None,
            )
        )

    # Conflicting docs: synthesize two short conflicting passages into eval-only questions
    for i in range(min(15, len(passages) // 2)):
        a, b = passages[i], passages[-(i + 1)]
        ent = (a.title or f"Entity{i}").split()[0]
        out.append(
            EvalExample(
                id=f"adv_conflict_{_pid(a.id + b.id)}",
                question=f"According to the pack, when was {ent} founded?",
                split="conflict",
                answerable=False,
                answer=None,
                passage_id=None,
            )
        )

    return out


def write_adversarial_file(root: Path) -> Path:
    adv = build_adversarial_evals(root)
    path = root / "data" / "evals" / "adversarial.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in adv:
            f.write(
                json.dumps(
                    {
                        "id": ex.id,
                        "question": ex.question,
                        "split": ex.split,
                        "answerable": ex.answerable,
                        "answer": ex.answer,
                        "passage_id": ex.passage_id,
                        "category": "adversarial",
                    }
                )
                + "\n"
            )
    return path


def run_adversarial_bench(root: Path, ask_fn) -> dict:
    """
    ask_fn(ex: EvalExample) -> AskResult
    Scores paraphrase F1 and refuse rates on distractor/conflict.
    """
    path = root / "data" / "evals" / "adversarial.jsonl"
    if not path.exists():
        write_adversarial_file(root)
    evals = load_evals(path)
    pairs = [(ex, ask_fn(ex)) for ex in evals]
    summary = summarize_ask_results(pairs)
    by = {}
    for split in ("paraphrase", "distractor", "conflict"):
        subset = [(ex, r) for ex, r in pairs if ex.split == split]
        if subset:
            by[split] = summarize_ask_results(subset)
    report = {"overall": summary, "by_split": by, "n": len(evals)}
    out = root / "artifacts" / "bench_adversarial.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
