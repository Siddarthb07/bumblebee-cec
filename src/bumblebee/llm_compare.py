"""Compare Bumblebee CEC against a real dense transformer LLM (~350–410M).

Optional dependency: transformers. Used only for offline/local peer benchmarking —
not an open-LM product sidecar.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from bumblebee.data import EvalExample, load_evals, load_passages, passage_index
from bumblebee.eval_protocol import refuse_confusion, train_eval_split
from bumblebee.gate import AskResult, token_f1
from bumblebee.router import ask_hybrid
from bumblebee.skills_bench import SKILL_CASES, SkillCase


# Models in the ~350–500M peer class (dense transformers) with safetensors on the Hub.
# Prefer safetensors: torch<2.6 cannot load legacy .bin under current transformers.
PEER_MODELS = {
    "gpt2-medium": "openai-community/gpt2-medium",  # 355M classic peer
    "smollm-360m": "HuggingFaceTB/SmolLM-360M",
    "pythia-410m": "EleutherAI/pythia-410m-deduped",
    "opt-350m": "facebook/opt-350m",  # may lack safetensors — use only if torch>=2.6
    "gpt2": "openai-community/gpt2",  # ~124M fallback
}

_REFUSE_MARKERS = (
    "outside_dataset",
    "outside dataset",
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "cannot answer",
    "can't answer",
    "no information",
    "not enough information",
    "unknown",
)


@dataclass
class PeerCaseResult:
    id: str
    suite: str
    question: str
    expect: str  # answer | refuse
    gold: Optional[str]
    cec_status: str
    cec_answer: str
    cec_ok: bool
    cec_f1: float
    cec_ms: float
    llm_status: str
    llm_answer: str
    llm_ok: bool
    llm_f1: float
    llm_ms: float
    llm_model: str


class LocalCausalLM:
    """Thin HuggingFace causal-LM wrapper for peer benches."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        max_new_tokens: int = 48,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        # Prefer CPU for deterministic offline benches on small Macs (MPS can stall).
        if device:
            self.device = device
        else:
            self.device = "cpu"

        print(f"[llm-compare] loading {model_id} on {self.device} …", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[llm-compare] ready params={self.n_params:,}", flush=True)

    def generate(self, prompt: str) -> str:
        torch = self.torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def _looks_like_refuse(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    return any(m in t for m in _REFUSE_MARKERS)


def _closed_prompt(question: str, notes: str) -> str:
    notes = (notes or "").strip() or "(empty)"
    return (
        "You answer ONLY from the NOTES below.\n"
        "If the notes do not contain enough information, reply with exactly: OUTSIDE_DATASET\n\n"
        f"NOTES:\n{notes[:1800]}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )


def _open_prompt(question: str) -> str:
    return (
        "Answer the question briefly. If you do not know, reply exactly: OUTSIDE_DATASET\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )


def llm_ask(
    lm: LocalCausalLM,
    question: str,
    *,
    notes: str = "",
    open_ended: bool = False,
) -> AskResult:
    prompt = _open_prompt(question) if open_ended else _closed_prompt(question, notes)
    t0 = time.perf_counter()
    raw = lm.generate(prompt)
    ms = (time.perf_counter() - t0) * 1000.0
    # Keep first paragraph / line as answer
    answer = raw.split("\n")[0].strip()
    if len(answer) > 400:
        answer = answer[:400].rsplit(" ", 1)[0]
    if _looks_like_refuse(answer):
        return AskResult(
            status="refuse",
            answer="",
            confidence=0.2,
            reason="llm_outside_or_unknown",
            citation_span="",
            citation_start=0,
            citation_end=0,
        )
    # Stash latency on reason for logging (AskResult has no ms field)
    return AskResult(
        status="answer",
        answer=answer,
        confidence=0.55,
        reason=f"llm_gen:{ms:.1f}ms",
        citation_span=answer[:120],
        citation_start=0,
        citation_end=0,
    )


def _score_skill(case: SkillCase, status: str, answer: str) -> Tuple[bool, float]:
    if case.expect == "refuse":
        return status == "refuse", 0.0
    if status != "answer" or not case.gold:
        return False, 0.0
    f1 = token_f1(answer, case.gold)
    ok = f1 >= case.min_f1 or case.gold.lower() in (answer or "").lower()
    return ok, f1


def _cec_ask(root: Path, question: str) -> Tuple[dict, float]:
    t0 = time.perf_counter()
    result = ask_hybrid(root, question)
    return result, (time.perf_counter() - t0) * 1000.0


def run_skills_compare(root: Path, lm: LocalCausalLM) -> List[PeerCaseResult]:
    rows: List[PeerCaseResult] = []
    for i, case in enumerate(SKILL_CASES, 1):
        print(f"[llm-compare] skills {i}/{len(SKILL_CASES)} {case.id}", flush=True)
        cec, cec_ms = _cec_ask(root, case.question)
        cec_ok, cec_f1 = _score_skill(case, cec.get("status", ""), cec.get("answer") or "")

        # Skills: math/code/chat/live are open-ish; corpus uses closed notes empty
        open_ended = case.category in {"chat", "math", "code", "live", "refuse"}
        t0 = time.perf_counter()
        llm = llm_ask(lm, case.question, open_ended=open_ended)
        llm_ms = (time.perf_counter() - t0) * 1000.0
        llm_ok, llm_f1 = _score_skill(case, llm.status, llm.answer)

        rows.append(
            PeerCaseResult(
                id=case.id,
                suite="skills",
                question=case.question,
                expect=case.expect,
                gold=case.gold,
                cec_status=cec.get("status", ""),
                cec_answer=(cec.get("answer") or cec.get("reason") or "")[:240],
                cec_ok=cec_ok,
                cec_f1=round(cec_f1, 4),
                cec_ms=round(cec_ms, 2),
                llm_status=llm.status,
                llm_answer=(llm.answer or llm.reason or "")[:240],
                llm_ok=llm_ok,
                llm_f1=round(llm_f1, 4),
                llm_ms=round(llm_ms, 2),
                llm_model=lm.model_id,
            )
        )
    return rows


def run_heldout_compare(
    root: Path,
    lm: LocalCausalLM,
    *,
    max_in_domain: int = 20,
    max_ood: int = 20,
) -> List[PeerCaseResult]:
    passages = load_passages(root / "data" / "corpus")
    pmap = passage_index(passages)
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    _, test = train_eval_split(evals)

    in_dom = [e for e in test if e.answerable and e.split == "in_domain"][:max_in_domain]
    ood = [e for e in test if not e.answerable or e.split != "in_domain"][:max_ood]
    sample: List[EvalExample] = in_dom + ood

    rows: List[PeerCaseResult] = []
    for i, ex in enumerate(sample, 1):
        print(f"[llm-compare] heldout {i}/{len(sample)} {ex.id}", flush=True)
        notes = ""
        if ex.passage_id and ex.passage_id in pmap:
            notes = pmap[ex.passage_id].text

        cec, cec_ms = _cec_ask(root, ex.question)
        expect = "answer" if ex.answerable else "refuse"
        cec_answered = cec.get("status") == "answer"
        cec_ok = (expect == "answer" and cec_answered) or (expect == "refuse" and not cec_answered)
        cec_f1 = token_f1(cec.get("answer") or "", ex.answer or "") if ex.answerable else 0.0

        t0 = time.perf_counter()
        # Fair closed-notes protocol for both answerable and OOD
        llm = llm_ask(lm, ex.question, notes=notes, open_ended=False)
        llm_ms = (time.perf_counter() - t0) * 1000.0
        llm_answered = llm.status == "answer"
        llm_ok = (expect == "answer" and llm_answered) or (expect == "refuse" and not llm_answered)
        llm_f1 = token_f1(llm.answer, ex.answer or "") if ex.answerable else 0.0

        rows.append(
            PeerCaseResult(
                id=ex.id,
                suite="heldout",
                question=ex.question,
                expect=expect,
                gold=ex.answer,
                cec_status=cec.get("status", ""),
                cec_answer=(cec.get("answer") or cec.get("reason") or "")[:240],
                cec_ok=cec_ok,
                cec_f1=round(cec_f1, 4),
                cec_ms=round(cec_ms, 2),
                llm_status=llm.status,
                llm_answer=(llm.answer or llm.reason or "")[:240],
                llm_ok=llm_ok,
                llm_f1=round(llm_f1, 4),
                llm_ms=round(llm_ms, 2),
                llm_model=lm.model_id,
            )
        )
    return rows


def _summarize(rows: Sequence[PeerCaseResult], suite: str) -> dict:
    sub = [r for r in rows if r.suite == suite]
    if not sub:
        return {"n": 0}
    cec_pass = sum(1 for r in sub if r.cec_ok)
    llm_pass = sum(1 for r in sub if r.llm_ok)
    ood = [r for r in sub if r.expect == "refuse"]
    ans = [r for r in sub if r.expect == "answer"]

    def refuse_stats(prefix: str) -> dict:
        # prefix cec_ or llm_
        tp = sum(1 for r in ood if getattr(r, f"{prefix}status") == "refuse")
        fn = sum(1 for r in ood if getattr(r, f"{prefix}status") == "answer")
        return {
            "ood_n": len(ood),
            "refuse_recall": tp / max(len(ood), 1),
            "forced_answer_rate": fn / max(len(ood), 1),
        }

    return {
        "n": len(sub),
        "cec_passed": cec_pass,
        "cec_accuracy": cec_pass / len(sub),
        "llm_passed": llm_pass,
        "llm_accuracy": llm_pass / len(sub),
        "cec_mean_f1_answerable": sum(r.cec_f1 for r in ans) / max(len(ans), 1),
        "llm_mean_f1_answerable": sum(r.llm_f1 for r in ans) / max(len(ans), 1),
        "cec_mean_ms": sum(r.cec_ms for r in sub) / len(sub),
        "llm_mean_ms": sum(r.llm_ms for r in sub) / len(sub),
        "cec_refuse": refuse_stats("cec_"),
        "llm_refuse": refuse_stats("llm_"),
    }


def resolve_model_id(name_or_id: str) -> str:
    key = (name_or_id or "").strip()
    return PEER_MODELS.get(key, key or PEER_MODELS["gpt2-medium"])


def run_llm_compare(
    root: Path,
    *,
    model: str = "gpt2-medium",
    max_in_domain: int = 20,
    max_ood: int = 20,
    device: Optional[str] = None,
) -> dict:
    """
    Two-phase compare to fit small Macs:
      1) CEC answers for all cases (may load LargeCEC once)
      2) free CEC runtime cache, then load dense LLM peer
    """
    import gc

    from bumblebee.train import clear_runtime_cache

    model_id = resolve_model_id(model)

    # ---- Phase 1: CEC only ----
    print("[llm-compare] phase 1/2: CEC answers", flush=True)
    passages = load_passages(root / "data" / "corpus")
    pmap = passage_index(passages)
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    _, test = train_eval_split(evals)
    in_dom = [e for e in test if e.answerable and e.split == "in_domain"][:max_in_domain]
    ood = [e for e in test if not e.answerable or e.split != "in_domain"][:max_ood]

    jobs: List[dict] = []
    for case in SKILL_CASES:
        jobs.append(
            {
                "id": case.id,
                "suite": "skills",
                "question": case.question,
                "expect": case.expect,
                "gold": case.gold,
                "min_f1": case.min_f1,
                "category": case.category,
                "notes": "",
                "open_ended": case.category in {"chat", "math", "code", "live", "refuse"},
            }
        )
    for ex in in_dom + ood:
        notes = ""
        if ex.passage_id and ex.passage_id in pmap:
            notes = pmap[ex.passage_id].text[:1200]
        jobs.append(
            {
                "id": ex.id,
                "suite": "heldout",
                "question": ex.question,
                "expect": "answer" if ex.answerable else "refuse",
                "gold": ex.answer,
                "min_f1": 0.0,
                "category": ex.split,
                "notes": notes,
                "open_ended": False,
            }
        )

    cec_side: Dict[str, dict] = {}
    for i, job in enumerate(jobs, 1):
        print(f"[llm-compare] CEC {i}/{len(jobs)} {job['id']}", flush=True)
        result, ms = _cec_ask(root, job["question"])
        status = result.get("status", "")
        answer = result.get("answer") or ""
        if job["suite"] == "skills":
            case = SkillCase(
                id=job["id"],
                category=job["category"],
                question=job["question"],
                expect=job["expect"],
                gold=job["gold"],
                min_f1=float(job["min_f1"]),
            )
            ok, f1 = _score_skill(case, status, answer)
        else:
            answered = status == "answer"
            ok = (job["expect"] == "answer" and answered) or (
                job["expect"] == "refuse" and not answered
            )
            f1 = token_f1(answer, job["gold"] or "") if job["expect"] == "answer" else 0.0
        cec_side[job["id"]] = {
            "status": status,
            "answer": (answer or result.get("reason") or "")[:240],
            "ok": ok,
            "f1": round(f1, 4),
            "ms": round(ms, 2),
        }

    clear_runtime_cache()
    gc.collect()
    print("[llm-compare] freed CEC runtime cache", flush=True)

    # ---- Phase 2: dense LLM peer ----
    print("[llm-compare] phase 2/2: dense LLM peer", flush=True)
    lm = LocalCausalLM(model_id, device=device or "cpu")
    llm_n_params = lm.n_params
    llm_device = lm.device
    rows: List[PeerCaseResult] = []
    for i, job in enumerate(jobs, 1):
        print(f"[llm-compare] LLM {i}/{len(jobs)} {job['id']}", flush=True)
        t0 = time.perf_counter()
        llm = llm_ask(
            lm,
            job["question"],
            notes=job["notes"],
            open_ended=bool(job["open_ended"]),
        )
        llm_ms = (time.perf_counter() - t0) * 1000.0
        if job["suite"] == "skills":
            case = SkillCase(
                id=job["id"],
                category=job["category"],
                question=job["question"],
                expect=job["expect"],
                gold=job["gold"],
                min_f1=float(job["min_f1"]),
            )
            llm_ok, llm_f1 = _score_skill(case, llm.status, llm.answer)
        else:
            answered = llm.status == "answer"
            llm_ok = (job["expect"] == "answer" and answered) or (
                job["expect"] == "refuse" and not answered
            )
            llm_f1 = (
                token_f1(llm.answer, job["gold"] or "") if job["expect"] == "answer" else 0.0
            )
        c = cec_side[job["id"]]
        rows.append(
            PeerCaseResult(
                id=job["id"],
                suite=job["suite"],
                question=job["question"],
                expect=job["expect"],
                gold=job["gold"],
                cec_status=c["status"],
                cec_answer=c["answer"],
                cec_ok=c["ok"],
                cec_f1=c["f1"],
                cec_ms=c["ms"],
                llm_status=llm.status,
                llm_answer=(llm.answer or llm.reason or "")[:240],
                llm_ok=llm_ok,
                llm_f1=round(llm_f1, 4),
                llm_ms=round(llm_ms, 2),
                llm_model=lm.model_id,
            )
        )

    del lm
    gc.collect()

    cec_cfg = {}
    cfg_path = root / "artifacts" / "config.json"
    if cfg_path.exists():
        cec_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    report = {
        "version": "v1_llm_compare",
        "protocol": (
            "Two-phase Mac-fit compare. Skills + held-out sample. "
            "LLM gets closed-NOTES + OUTSIDE_DATASET instruction on held-out; "
            "open prompt on skills. Gold never used for decoding."
        ),
        "cec": {
            "model_id": "cec",
            "arch": cec_cfg.get("arch"),
            "n_params": cec_cfg.get("n_params"),
            "note": "Sparse hash emb dominates params; core is narrow GRU+correlator.",
        },
        "llm_peer": {
            "model_id": model_id,
            "n_params": llm_n_params,
            "device": llm_device,
            "peer_class": "~350–500M dense transformer (or smaller fallback)",
        },
        "skills": _summarize(rows, "skills"),
        "heldout_sample": _summarize(rows, "heldout"),
        "cases": [asdict(r) for r in rows],
    }

    out = root / "artifacts" / "bench_llm_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(root, report)
    return report


def _write_md(root: Path, report: dict) -> Path:
    cec = report["cec"]
    llm = report["llm_peer"]
    sk = report["skills"]
    ho = report["heldout_sample"]
    lines = [
        "# LLM peer comparison",
        "",
        "Generated by `python -m bumblebee.cli bench --mode llm-compare`.",
        "",
        report["protocol"],
        "",
        "## Models",
        "",
        f"| Side | ID | Params | Notes |",
        f"| --- | --- | ---: | --- |",
        f"| CEC | `{cec.get('arch')}` | {cec.get('n_params')} | {cec.get('note')} |",
        f"| LLM peer | `{llm.get('model_id')}` | {llm.get('n_params')} | device={llm.get('device')} |",
        "",
        "## Skills suite (same questions)",
        "",
        f"| System | Passed | Acc | Mean F1 (answerable) | Mean latency ms | OOD refuse recall | Forced-answer rate |",
        f"| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| CEC | {sk.get('cec_passed')}/{sk.get('n')} | {sk.get('cec_accuracy', 0):.2f} | "
            f"{sk.get('cec_mean_f1_answerable', 0):.3f} | {sk.get('cec_mean_ms', 0):.1f} | "
            f"{(sk.get('cec_refuse') or {}).get('refuse_recall', 0):.2f} | "
            f"{(sk.get('cec_refuse') or {}).get('forced_answer_rate', 0):.2f} |"
        ),
        (
            f"| LLM | {sk.get('llm_passed')}/{sk.get('n')} | {sk.get('llm_accuracy', 0):.2f} | "
            f"{sk.get('llm_mean_f1_answerable', 0):.3f} | {sk.get('llm_mean_ms', 0):.1f} | "
            f"{(sk.get('llm_refuse') or {}).get('refuse_recall', 0):.2f} | "
            f"{(sk.get('llm_refuse') or {}).get('forced_answer_rate', 0):.2f} |"
        ),
        "",
        "## Held-out sample (closed NOTES + OUTSIDE_DATASET instruction for LLM)",
        "",
        f"| System | Correct behavior | Acc | Mean F1 (answerable) | Mean latency ms | OOD refuse recall | Forced-answer rate |",
        f"| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| CEC | {ho.get('cec_passed')}/{ho.get('n')} | {ho.get('cec_accuracy', 0):.2f} | "
            f"{ho.get('cec_mean_f1_answerable', 0):.3f} | {ho.get('cec_mean_ms', 0):.1f} | "
            f"{(ho.get('cec_refuse') or {}).get('refuse_recall', 0):.2f} | "
            f"{(ho.get('cec_refuse') or {}).get('forced_answer_rate', 0):.2f} |"
        ),
        (
            f"| LLM | {ho.get('llm_passed')}/{ho.get('n')} | {ho.get('llm_accuracy', 0):.2f} | "
            f"{ho.get('llm_mean_f1_answerable', 0):.3f} | {ho.get('llm_mean_ms', 0):.1f} | "
            f"{(ho.get('llm_refuse') or {}).get('refuse_recall', 0):.2f} | "
            f"{(ho.get('llm_refuse') or {}).get('forced_answer_rate', 0):.2f} |"
        ),
        "",
        "## Reading",
        "",
        "- Dense ~350M LLMs are trained for continuation; even with an OUTSIDE_DATASET instruction they often force-answer.",
        "- CEC params are sparse-hash heavy; do not claim FLOPs parity with OPT-350M.",
        "- Skills wins for CEC often come from packs/tools — that is the product architecture under test.",
        "",
        "## Artifacts",
        "",
        "- `artifacts/bench_llm_compare.json` — full case-level dump",
        "- `docs/LLM_COMPARE.md` — this summary",
        "",
    ]
    path = root / "docs" / "LLM_COMPARE.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
