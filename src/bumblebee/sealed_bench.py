"""Sealed product benches — frozen usability v1 + strict Dolly slice (no transformers)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from bumblebee.usability_bench import REAL_CASES, RealCase, _ok_answer


def freeze_usability_v1(root: Path) -> Path:
    """Write data/sealed/usability_v1.jsonl — immutable product v1 set."""
    sealed = Path(root) / "data" / "sealed"
    sealed.mkdir(parents=True, exist_ok=True)
    path = sealed / "usability_v1.jsonl"
    lines = []
    for c in REAL_CASES:
        lines.append(
            json.dumps(
                {
                    "id": c.id,
                    "category": c.category,
                    "question": c.question,
                    "expect": c.expect,
                    "must_any": list(c.must_any),
                    "must_all": list(c.must_all),
                    "forbid": list(c.forbid),
                    "min_words": c.min_words,
                    "allow_web": c.allow_web,
                    "notes": c.notes,
                    "frozen": True,
                    "version": "v1",
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_frozen_usability(root: Path) -> List[RealCase]:
    path = Path(root) / "data" / "sealed" / "usability_v1.jsonl"
    if not path.exists():
        freeze_usability_v1(root)
    out: List[RealCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(
            RealCase(
                id=row["id"],
                category=row["category"],
                question=row["question"],
                expect=row["expect"],
                must_any=tuple(row.get("must_any") or ()),
                must_all=tuple(row.get("must_all") or ()),
                forbid=tuple(row.get("forbid") or ()),
                min_words=int(row.get("min_words") or 5),
                allow_web=bool(row.get("allow_web", True)),
                notes=row.get("notes") or "",
            )
        )
    return out


def run_frozen_usability(root: Path) -> dict:
    """Score the frozen v1 set — never regenerate cases mid-run."""
    from bumblebee.router import ask_hybrid
    from bumblebee.train import clear_runtime_cache

    clear_runtime_cache()
    cases = load_frozen_usability(root)
    rows = []
    t0 = time.time()
    for case in cases:
        t1 = time.time()
        try:
            out = ask_hybrid(root, case.question, mode="auto")  # type: ignore[arg-type]
        except Exception as exc:
            out = {
                "status": "refuse",
                "answer": "",
                "reason": f"error:{type(exc).__name__}",
                "source": "error",
                "confidence": 0.0,
            }
        ok, why = _ok_answer(
            case,
            out.get("status") or "refuse",
            out.get("answer") or "",
            out.get("source") or "",
        )
        rows.append(
            {
                "id": case.id,
                "category": case.category,
                "question": case.question,
                "ok": ok,
                "why": why,
                "status": out.get("status"),
                "source": out.get("source"),
                "source_label": out.get("source_label"),
                "confidence": float(out.get("confidence") or 0),
                "answer": out.get("answer") or "",
                "answer_preview": (out.get("answer") or "")[:200],
                "latency_ms": round((time.time() - t1) * 1000, 1),
            }
        )
    passed = sum(1 for r in rows if r["ok"])
    report = {
        "protocol": "frozen usability v1; no case regen; product ask_hybrid",
        "n": len(rows),
        "passed": passed,
        "accuracy": passed / max(len(rows), 1),
        "elapsed_s": round(time.time() - t0, 2),
        "failures": [r for r in rows if not r["ok"]],
        "cases": rows,
        "transformers": False,
    }
    outp = Path(root) / "artifacts" / "bench_usability_v1_frozen.json"
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_dolly_strict_slice(root: Path, *, n: int = 40) -> Path:
    """
    Sealed Dolly-like held-out questions with gold passage in pack.
    No preflight filter — honest difficulty.
    """
    from bumblebee.data import load_evals, load_passages
    from bumblebee.eval_protocol import train_eval_split
    from bumblebee.train import get_bm25_index

    evals = load_evals(Path(root) / "data" / "evals" / "questions.jsonl")
    _, test = train_eval_split(evals)
    passages = load_passages(Path(root) / "data" / "corpus")
    pmap = {p.id: p for p in passages}
    index = get_bm25_index(root, passages)

    rows = []
    for ex in test:
        if len(rows) >= n:
            break
        if not ex.answerable or not ex.answer or not ex.passage_id:
            continue
        if ex.passage_id not in pmap:
            continue
        if not str(ex.id).startswith("dolly"):
            continue
        q = (ex.question or "").strip()
        if len(q.split()) < 6 or len(q.split()) > 40:
            continue
        top = [h.passage_id for h in index.score(q)[:10]]
        rows.append(
            {
                "id": ex.id,
                "question": q,
                "gold_answer": ex.answer,
                "passage_id": ex.passage_id,
                "gold_in_bm25_top10": ex.passage_id in top,
                "allow_web": False,
            }
        )

    sealed = Path(root) / "data" / "sealed"
    sealed.mkdir(parents=True, exist_ok=True)
    path = sealed / "dolly_strict_v1.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


def run_dolly_strict(root: Path, *, n: int = 40) -> dict:
    """Product path, allow_web=False — refuse or pack answer only."""
    from bumblebee.gate import token_f1
    from bumblebee.router import ask_hybrid
    from bumblebee.train import clear_runtime_cache

    path = Path(root) / "data" / "sealed" / "dolly_strict_v1.jsonl"
    if not path.exists():
        build_dolly_strict_slice(root, n=n)

    clear_runtime_cache()
    rows = []
    t0 = time.time()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        q = ex["question"]
        gold = ex.get("gold_answer") or ""
        out = ask_hybrid(root, q, mode="corpus")  # type: ignore[arg-type]  # corpus = no web promote preferred
        # Force: if web slipped through, count as fail for strict slice
        status = out.get("status")
        source = out.get("source") or ""
        answer = out.get("answer") or ""
        if source.startswith("web"):
            ok = False
            why = "web_not_allowed"
            f1 = 0.0
        elif status == "answer":
            f1 = token_f1(answer, gold)
            # Pass if decent F1 OR gold content words appear in answer
            gold_bits = [t for t in gold.lower().split() if len(t) >= 4][:3]
            hit = any(b in answer.lower() for b in gold_bits) if gold_bits else False
            ok = f1 >= 0.15 or hit
            why = "ok" if ok else f"low_f1_{f1:.2f}"
        else:
            ok = False
            why = f"refuse:{out.get('reason')}"
            f1 = 0.0
        rows.append(
            {
                "id": ex["id"],
                "question": q,
                "ok": ok,
                "why": why,
                "f1": round(f1, 4),
                "status": status,
                "source": source,
                "source_label": out.get("source_label"),
                "answer": answer,
                "answer_preview": answer[:180],
                "gold_answer": gold,
                "gold_in_bm25_top10": ex.get("gold_in_bm25_top10"),
            }
        )

    passed = sum(1 for r in rows if r["ok"])
    report = {
        "protocol": "dolly_strict_v1; mode=corpus; allow_web=False; no preflight",
        "n": len(rows),
        "passed": passed,
        "accuracy": passed / max(len(rows), 1),
        "mean_f1_answered": (
            sum(r["f1"] for r in rows if r["status"] == "answer")
            / max(sum(1 for r in rows if r["status"] == "answer"), 1)
        ),
        "elapsed_s": round(time.time() - t0, 2),
        "failures": [r for r in rows if not r["ok"]][:20],
        "cases": rows,
        "transformers": False,
    }
    outp = Path(root) / "artifacts" / "bench_dolly_strict.json"
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")

    doc = Path(root) / "docs" / "DOLLY_STRICT_BENCH.md"
    doc.write_text(
        "\n".join(
            [
                "# Dolly strict slice (sealed)",
                "",
                "Product path with `mode=corpus` — **no web**. Honest held-out Dolly.",
                "",
                f"**{report['passed']}/{report['n']} ({report['accuracy']:.0%})** "
                f"in {report['elapsed_s']}s · mean F1 answered={report['mean_f1_answered']:.3f}",
                "",
                "Artifact: `artifacts/bench_dolly_strict.json`",
                "Seal: `data/sealed/dolly_strict_v1.jsonl`",
                "",
                "No transformers.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report
