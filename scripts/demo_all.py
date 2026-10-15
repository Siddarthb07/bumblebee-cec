#!/usr/bin/env python
"""Demo against the current HF (SQuAD) closed corpus checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from rich import print as rprint
from rich.panel import Panel

from bumblebee.data import load_evals
from bumblebee.train import ask_question


def main() -> None:
    root = Path.cwd()
    if not (root / "artifacts" / "cec.pt").exists():
        raise SystemExit("Missing artifacts/cec.pt — run: python -m bumblebee.cli train --epochs 6")

    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    by_split = {}
    for ex in evals:
        by_split.setdefault(ex.split, []).append(ex)

    cases = []
    for ex in by_split.get("in_domain", [])[:2]:
        cases.append(("IN-DOMAIN", ex.question, ex.passage_id, ex.answer))
    for ex in by_split.get("entity_gap", [])[:1]:
        cases.append(("ENTITY-GAP", ex.question, None, None))
    for ex in by_split.get("far_ood", [])[:1]:
        cases.append(("FAR-OOD", ex.question, None, None))

    rprint("[bold]Bumblebee demo[/bold] (SQuAD closed subset)\n")
    for label, q, passage_id, gold in cases:
        result = ask_question(root, q, passage_id=passage_id)
        if result["status"] == "answer":
            extra = f"\ngold: {gold}" if gold else ""
            body = (
                f"[bold]{label}[/bold]  conf={result['confidence']:.3f}\n"
                f"Q: {q}\n"
                f"A: {result['answer']}\n"
                f"citation [{result['passage_id']}]: {result['citation_span']}"
                f"{extra}"
            )
            rprint(Panel(body, border_style="green"))
        else:
            body = (
                f"[bold]{label}[/bold]  conf={result['confidence']:.3f}\n"
                f"Q: {q}\n"
                f"OUTSIDE_DATASET reason={result['reason']}"
            )
            rprint(Panel(body, border_style="yellow"))

    rprint("\n[dim]Try your own:[/dim]")
    rprint('  python -m bumblebee.cli ask "Your question here"')


if __name__ == "__main__":
    main()
