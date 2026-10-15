"""Full multi-model rundown: CEC / Linear / BM25 / embed / memory / multi-peer LLMs.

One CEC pass, then each dense peer in turn. Writes artifacts/bench_full_rundown.json
+ docs/FULL_RUNDOWN.md. Honesty-first: quality ≠ memory ≠ usability scoreboards stay separate.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from bumblebee.data import load_evals, load_passages, passage_index
from bumblebee.eval_protocol import train_eval_split
from bumblebee.gate import token_f1
from bumblebee.llm_compare import (
    PEER_MODELS,
    LocalCausalLM,
    SkillCase,
    _cec_ask,
    _score_skill,
    llm_ask,
    resolve_model_id,
)
from bumblebee.skills_bench import SKILL_CASES
from bumblebee.train import clear_runtime_cache


DEFAULT_PEERS = ("gpt2", "gpt2-medium", "smollm-360m")


def _load(path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _f(v, digits=3):
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return None


def collect_static_scoreboards(root: Path) -> dict:
    """Fold existing measured cards into one place (no retrain)."""
    art = root / "artifacts"
    usability = _load(art / "bench_usability.json") or {}
    skills = _load(art / "bench_skills.json") or {}
    memory = _load(art / "bench_memory.json") or {}
    bench_all = _load(art / "bench_all.json") or {}
    eval_r = _load(art / "eval_report.json") or {}
    ablation = _load(art / "ablation_linear.json") or {}
    sweep = _load(art / "param_sweep_linear.json") or {}
    sealed = _load(art / "bench_sealed.json") or _load(art / "bench_usability_sealed.json")
    dolly = _load(art / "bench_dolly_strict.json")
    prior_llm = _load(art / "bench_llm_compare.json")

    retrieval = {}
    for k in ("cec", "bm25", "embed", "bm25_raw", "embed_raw"):
        src = bench_all.get(k) or (eval_r if k == "cec" else None)
        if not src:
            continue
        retrieval[k] = {
            "in_domain_f1": _f(src.get("in_domain_f1_mean")),
            "f1_n": src.get("in_domain_f1_n") or src.get("f1_n"),
            "ood_refuse_p": _f(src.get("ood_refuse_precision")),
            "ood_refuse_r": _f(src.get("ood_refuse_recall")),
            "aurc": _f(src.get("aurc")),
            "ece_answered": _f(src.get("ece_answered", src.get("ece"))),
        }

    linear_arms = {}
    for name, r in (ablation.get("by_arm") or {}).items():
        linear_arms[name] = {
            "in_domain_f1": _f(r.get("in_domain_f1_mean")),
            "ood_refuse_r": _f(r.get("ood_refuse_recall")),
            "aurc": _f(r.get("aurc")),
            "ece_answered": _f(r.get("ece_answered")),
            "n_params": r.get("n_params"),
            "train_mode": r.get("train_mode"),
        }

    return {
        "retrieval_holdout": retrieval,
        "linear_ablation": {
            "arms": linear_arms,
            "verdict": ablation.get("verdict"),
            "protocol": ablation.get("protocol"),
        },
        "param_sweep_winners": sweep.get("winners_by_f1"),
        "usability": {
            "passed": usability.get("passed") or usability.get("n_passed"),
            "n": usability.get("n") or usability.get("n_total"),
            "accuracy": _f(usability.get("accuracy")),
            "by_category": usability.get("by_category"),
        },
        "skills": {
            "product": skills.get("product") or skills.get("by_board", {}).get("product"),
            "core": skills.get("core") or skills.get("by_board", {}).get("core"),
            "accuracy": _f(skills.get("accuracy")),
            "passed": skills.get("passed"),
            "n": skills.get("n"),
        },
        "memory": memory.get("cec") or memory,
        "memory_vs_llms": (memory.get("comparison") or {}),
        "sealed": sealed,
        "dolly_strict": dolly,
        "prior_llm_compare": {
            "peer": (prior_llm or {}).get("llm_model") or (prior_llm or {}).get("peer"),
            "skills": (prior_llm or {}).get("skills") or (prior_llm or {}).get("summary", {}).get("skills"),
            "heldout": (prior_llm or {}).get("heldout") or (prior_llm or {}).get("summary", {}).get("heldout"),
        }
        if prior_llm
        else None,
    }


def _build_jobs(
    root: Path,
    *,
    max_in_domain: int,
    max_ood: int,
    include_skills: bool,
    max_skills: Optional[int] = None,
) -> List[dict]:
    jobs: List[dict] = []
    if include_skills:
        cases = list(SKILL_CASES)
        if max_skills is not None and max_skills > 0:
            cases = cases[: int(max_skills)]
        for case in cases:
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
    passages = load_passages(root / "data" / "corpus")
    pmap = passage_index(passages)
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    _, test = train_eval_split(evals)
    in_dom = [e for e in test if e.answerable and e.split == "in_domain"][:max_in_domain]
    ood = [e for e in test if not e.answerable or e.split != "in_domain"][:max_ood]
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
    return jobs


def _score_job(job: dict, status: str, answer: str) -> tuple:
    if job["suite"] == "skills":
        case = SkillCase(
            id=job["id"],
            category=job["category"],
            question=job["question"],
            expect=job["expect"],
            gold=job["gold"],
            min_f1=float(job["min_f1"]),
        )
        return _score_skill(case, status, answer)
    answered = status == "answer"
    ok = (job["expect"] == "answer" and answered) or (job["expect"] == "refuse" and not answered)
    f1 = token_f1(answer, job["gold"] or "") if job["expect"] == "answer" else 0.0
    return ok, f1


def _suite_summary(rows: List[dict], prefix: str) -> dict:
    """prefix is 'cec' or 'llm'."""
    if not rows:
        return {}
    n = len(rows)
    ok = sum(1 for r in rows if r[f"{prefix}_ok"])
    ans = [r for r in rows if r["expect"] == "answer"]
    ood = [r for r in rows if r["expect"] == "refuse"]
    f1s = [r[f"{prefix}_f1"] for r in ans]
    forced = 0
    refuse_hit = 0
    for r in ood:
        st = r[f"{prefix}_status"]
        if st == "answer":
            forced += 1
        elif st == "refuse":
            refuse_hit += 1
    return {
        "n": n,
        "correct": ok,
        "accuracy": round(ok / n, 4) if n else 0.0,
        "mean_f1_answerable": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "mean_ms": round(sum(r[f"{prefix}_ms"] for r in rows) / n, 1) if n else 0.0,
        "ood_n": len(ood),
        "ood_refuse_recall": round(refuse_hit / len(ood), 4) if ood else None,
        "forced_answer_rate": round(forced / len(ood), 4) if ood else None,
    }


def run_multi_peer_quality(
    root: Path,
    *,
    peers: Sequence[str] = DEFAULT_PEERS,
    max_in_domain: int = 8,
    max_ood: int = 8,
    include_skills: bool = True,
    max_skills: int = 12,
    max_new_tokens: int = 32,
    device: str = "cpu",
) -> dict:
    """
    One CEC pass over skills+heldout sample, then each peer LM.
    Caps keep Mac/CPU runs finite; not a full holdout claim.
    """
    jobs = _build_jobs(
        root,
        max_in_domain=max_in_domain,
        max_ood=max_ood,
        include_skills=include_skills,
        max_skills=max_skills,
    )
    print(f"[rundown] jobs={len(jobs)} peers={list(peers)}", flush=True)

    print("[rundown] CEC phase", flush=True)
    cec_side: Dict[str, dict] = {}
    for i, job in enumerate(jobs, 1):
        print(f"[rundown] CEC {i}/{len(jobs)} {job['id']}", flush=True)
        result, ms = _cec_ask(root, job["question"])
        status = result.get("status", "")
        answer = result.get("answer") or ""
        ok, f1 = _score_job(job, status, answer)
        cec_side[job["id"]] = {
            "status": status,
            "answer": (answer or result.get("reason") or "")[:240],
            "ok": ok,
            "f1": round(float(f1), 4),
            "ms": round(ms, 2),
        }

    clear_runtime_cache()
    gc.collect()

    peer_reports: Dict[str, dict] = {}
    for peer in peers:
        model_id = resolve_model_id(peer)
        print(f"[rundown] peer={peer} id={model_id}", flush=True)
        try:
            lm = LocalCausalLM(model_id, device=device, max_new_tokens=max_new_tokens)
        except Exception as e:
            peer_reports[peer] = {"error": str(e), "model_id": model_id}
            continue
        rows = []
        for i, job in enumerate(jobs, 1):
            print(f"[rundown] {peer} {i}/{len(jobs)} {job['id']}", flush=True)
            t0 = time.perf_counter()
            llm = llm_ask(
                lm,
                job["question"],
                notes=job["notes"],
                open_ended=bool(job["open_ended"]),
            )
            ms = (time.perf_counter() - t0) * 1000.0
            ok, f1 = _score_job(job, llm.status, llm.answer)
            cec = cec_side[job["id"]]
            rows.append(
                {
                    "id": job["id"],
                    "suite": job["suite"],
                    "expect": job["expect"],
                    "question": job["question"][:160],
                    "cec_status": cec["status"],
                    "cec_ok": cec["ok"],
                    "cec_f1": cec["f1"],
                    "cec_ms": cec["ms"],
                    "llm_status": llm.status,
                    "llm_ok": ok,
                    "llm_f1": round(float(f1), 4),
                    "llm_ms": round(ms, 2),
                    "llm_answer": (llm.answer or llm.reason or "")[:200],
                }
            )
        skills_rows = [r for r in rows if r["suite"] == "skills"]
        held_rows = [r for r in rows if r["suite"] == "heldout"]
        peer_reports[peer] = {
            "model_id": model_id,
            "n_params": lm.n_params,
            "device": lm.device,
            "skills": {
                "cec": _suite_summary(skills_rows, "cec"),
                "llm": _suite_summary(skills_rows, "llm"),
            },
            "heldout": {
                "cec": _suite_summary(held_rows, "cec"),
                "llm": _suite_summary(held_rows, "llm"),
            },
            "cases": rows,
        }
        del lm
        gc.collect()

    # CEC aggregate (same across peers)
    all_rows_cec = []
    for job in jobs:
        c = cec_side[job["id"]]
        all_rows_cec.append(
            {
                "suite": job["suite"],
                "expect": job["expect"],
                "cec_status": c["status"],
                "cec_ok": c["ok"],
                "cec_f1": c["f1"],
                "cec_ms": c["ms"],
            }
        )
    return {
        "protocol": (
            f"one CEC pass; peers={list(peers)}; skills≤{max_skills if include_skills else 0}; "
            f"heldout in={max_in_domain} ood={max_ood}; max_new_tokens={max_new_tokens}; "
            "LLM closed-NOTES+OUTSIDE_DATASET on heldout; open-ish on skills"
        ),
        "n_jobs": len(jobs),
        "cec_skills": _suite_summary([r for r in all_rows_cec if r["suite"] == "skills"], "cec"),
        "cec_heldout": _suite_summary([r for r in all_rows_cec if r["suite"] == "heldout"], "cec"),
        "peers": peer_reports,
        "peer_catalog": PEER_MODELS,
    }


def _verdict_block(static: dict, peers: dict) -> dict:
    """Plain-language good/bad cards — not a single scalar winner."""
    ret = static.get("retrieval_holdout") or {}
    cec = ret.get("cec") or {}
    bm25 = ret.get("bm25") or {}
    linear = (static.get("linear_ablation") or {}).get("arms") or {}
    usability = static.get("usability") or {}
    mem = static.get("memory_vs_llms") or {}

    wins = []
    losses = []
    ties = []

    # Usability
    uacc = usability.get("accuracy")
    if isinstance(uacc, (int, float)) and uacc >= 0.95:
        wins.append(
            f"Product usability {usability.get('passed')}/{usability.get('n')} "
            f"({uacc:.0%}) on natural asks — craft/tools/refuse path."
        )

    # Retrieval F1
    cf1, bf1 = cec.get("in_domain_f1"), bm25.get("in_domain_f1")
    if isinstance(cf1, (int, float)) and isinstance(bf1, (int, float)):
        if bf1 > cf1:
            losses.append(
                f"Retrieved in-domain F1: BM25 ({bf1:.3f}) beats Classic CEC ({cf1:.3f})."
            )
        else:
            wins.append(f"Retrieved F1: CEC ({cf1:.3f}) ≥ BM25 ({bf1:.3f}).")

    # Linear
    classic = linear.get("classic") or {}
    scratch = linear.get("linear_scratch") or {}
    if isinstance(classic.get("in_domain_f1"), (int, float)) and isinstance(
        scratch.get("in_domain_f1"), (int, float)
    ):
        if scratch["in_domain_f1"] < classic["in_domain_f1"]:
            losses.append(
                f"Linear CEC scratch F1 {scratch['in_domain_f1']:.3f} < classic "
                f"{classic['in_domain_f1']:.3f} under short ablation — not promoted."
            )
        else:
            wins.append("Linear scratch matched/beat classic under ablation budget.")

    # Memory
    if mem.get("beats_fp16_weight_footprint"):
        wins.append(
            "Peak RSS undercuts FP16 weight footprints of many ≥1B LLMs "
            "(catalog compare; KV cache not included for peers)."
        )

    # Peers
    for name, rep in (peers.get("peers") or {}).items():
        if rep.get("error"):
            losses.append(f"Peer {name} failed to load: {rep['error'][:120]}")
            continue
        sk = (rep.get("skills") or {}).get("llm") or {}
        ck = (rep.get("skills") or {}).get("cec") or {}
        if sk.get("accuracy") is not None and ck.get("accuracy") is not None:
            if ck["accuracy"] > sk["accuracy"]:
                wins.append(
                    f"Skills suite: CEC acc {ck['accuracy']:.2f} > {name} {sk['accuracy']:.2f} "
                    f"(packs/tools matter; LLM forced-answer rate={sk.get('forced_answer_rate')})."
                )
            else:
                losses.append(
                    f"Skills suite: {name} acc {sk['accuracy']:.2f} ≥ CEC {ck['accuracy']:.2f}."
                )
        hk = (rep.get("heldout") or {}).get("llm") or {}
        hc = (rep.get("heldout") or {}).get("cec") or {}
        if hk.get("mean_f1_answerable") is not None and hc.get("mean_f1_answerable") is not None:
            if hk["mean_f1_answerable"] > hc["mean_f1_answerable"]:
                losses.append(
                    f"Held-out sample F1: {name} ({hk['mean_f1_answerable']:.3f}) > "
                    f"CEC ({hc['mean_f1_answerable']:.3f}) when LLM gets gold passage NOTES."
                )
            else:
                wins.append(
                    f"Held-out sample F1: CEC ({hc['mean_f1_answerable']:.3f}) ≥ "
                    f"{name} ({hk['mean_f1_answerable']:.3f})."
                )
        if hk.get("forced_answer_rate") == 1.0 or (hk.get("forced_answer_rate") or 0) >= 0.8:
            wins.append(
                f"{name} forced-answer rate on OOD sample ≈{hk.get('forced_answer_rate')} — "
                "continuation bias; CEC refuse is architectural."
            )
        if sk.get("mean_ms") and ck.get("mean_ms") and sk["mean_ms"] > 5 * ck["mean_ms"]:
            wins.append(
                f"Latency: CEC skills ~{ck['mean_ms']:.0f}ms vs {name} ~{sk['mean_ms']:.0f}ms "
                "(CPU peer generate)."
            )

    if not wins and not losses:
        ties.append("Insufficient peer data — see static scoreboards only.")

    return {
        "where_strong": wins,
        "where_weak": losses,
        "notes": ties
        + [
            "Usability ≠ retrieved F1 ≠ peer LLM open chat.",
            "Linear CEC experimental; product default remains GRU+classic.",
            "Peer held-out gives LLM the gold passage as NOTES — favors extractive continuation.",
            "Do not equate CEC hash/GRU params with dense transformer FLOPs.",
        ],
    }


def run_full_rundown(
    root: Path,
    *,
    peers: Sequence[str] = DEFAULT_PEERS,
    max_in_domain: int = 8,
    max_ood: int = 8,
    include_skills: bool = True,
    max_skills: int = 12,
    skip_peers: bool = False,
) -> dict:
    static = collect_static_scoreboards(root)
    peer_block: dict = {"skipped": True, "peers": {}}
    if not skip_peers:
        peer_block = run_multi_peer_quality(
            root,
            peers=peers,
            max_in_domain=max_in_domain,
            max_ood=max_ood,
            include_skills=include_skills,
            max_skills=max_skills,
        )
    verdict = _verdict_block(static, peer_block)
    out = {
        "title": "Bumblebee full multi-model rundown",
        "static": static,
        "peer_quality": peer_block,
        "verdict": verdict,
        "transformers_used_for_peers": not skip_peers,
        "product_default": "gru+classic SoftCorrelator",
    }
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / "bench_full_rundown.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_md(root, out)
    return out


def _write_md(root: Path, out: dict) -> None:
    static = out.get("static") or {}
    peers = out.get("peer_quality") or {}
    verdict = out.get("verdict") or {}
    lines = [
        "# Full multi-model rundown",
        "",
        "How good / bad Bumblebee is across **separate scoreboards** "
        "(retrieval, usability, linear ablation, memory, dense LLM peers).",
        "",
        "## Verdict (plain language)",
        "",
        "### Where it is strong",
        "",
    ]
    for w in verdict.get("where_strong") or ["—"]:
        lines.append(f"- {w}")
    lines.extend(["", "### Where it is weak", ""])
    for w in verdict.get("where_weak") or ["—"]:
        lines.append(f"- {w}")
    lines.extend(["", "### Reading notes", ""])
    for w in verdict.get("notes") or []:
        lines.append(f"- {w}")

    lines.extend(["", "## 1. Retrieved holdout (Classic CEC vs BM25 / embed)", ""])
    lines.append("| System | In-domain F1 | OOD refuse R | AURC | ECE answered |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for k, r in (static.get("retrieval_holdout") or {}).items():
        lines.append(
            f"| {k} | {r.get('in_domain_f1')} | {r.get('ood_refuse_r')} | "
            f"{r.get('aurc')} | {r.get('ece_answered')} |"
        )

    lines.extend(["", "## 2. Linear CEC ablation (experimental)", ""])
    lines.append("| Arm | F1 | OOD R | AURC | ECE |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for k, r in ((static.get("linear_ablation") or {}).get("arms") or {}).items():
        lines.append(
            f"| {k} | {r.get('in_domain_f1')} | {r.get('ood_refuse_r')} | "
            f"{r.get('aurc')} | {r.get('ece_answered')} |"
        )
    v = (static.get("linear_ablation") or {}).get("verdict") or {}
    lines.append("")
    lines.append(
        f"Verdict: prefer={v.get('prefer_under_budget')}, "
        f"beats_classic={v.get('beats_classic_f1')}, promote={v.get('promote_to_product')}"
    )

    u = static.get("usability") or {}
    lines.extend(
        [
            "",
            "## 3. Product usability",
            "",
            f"- Score: **{u.get('passed')}/{u.get('n')}** (acc={u.get('accuracy')})",
            "- Path: ask_hybrid (tools/pack/craft/gate) — not core F1.",
            "",
            "## 4. Dense LLM peer quality (sampled)",
            "",
            peers.get("protocol") or "_peers skipped_",
            "",
        ]
    )
    lines.append("| Peer | Params | Skills CEC | Skills LLM | Heldout CEC F1 | Heldout LLM F1 | LLM forced-OOD |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    cec_sk = peers.get("cec_skills") or {}
    cec_hd = peers.get("cec_heldout") or {}
    for name, rep in (peers.get("peers") or {}).items():
        if rep.get("error"):
            lines.append(f"| {name} | — | — | ERROR | — | — | — |")
            continue
        sk = rep.get("skills") or {}
        hd = rep.get("heldout") or {}
        lines.append(
            f"| {name} | {rep.get('n_params')} | "
            f"{(sk.get('cec') or {}).get('accuracy')} | {(sk.get('llm') or {}).get('accuracy')} | "
            f"{(hd.get('cec') or {}).get('mean_f1_answerable')} | "
            f"{(hd.get('llm') or {}).get('mean_f1_answerable')} | "
            f"{(hd.get('llm') or {}).get('forced_answer_rate')} |"
        )
    if cec_sk:
        lines.append("")
        lines.append(
            f"CEC skills on this sample: acc={cec_sk.get('accuracy')} "
            f"mean_ms={cec_sk.get('mean_ms')}; heldout F1={cec_hd.get('mean_f1_answerable')}."
        )

    mem = static.get("memory") or {}
    cmp_ = static.get("memory_vs_llms") or {}
    lines.extend(
        [
            "",
            "## 5. Memory vs LLM catalogs",
            "",
            f"- CEC peak RSS: **{mem.get('peak_rss_gb')} GB** · params≈{mem.get('n_params') or mem.get('n_params_m')}",
            f"- Ratio vs Mistral-7B FP16: **{cmp_.get('ratio_vs_mistral7b_fp16')}**",
            f"- Beats FP16 footprints (partial list): {', '.join((cmp_.get('beats_fp16_weight_footprint') or [])[:8])}",
            "",
            "## Artifacts",
            "",
            "- `artifacts/bench_full_rundown.json`",
            "- `output/pdf/bumblebee_full_rundown.pdf` (if built)",
            "",
        ]
    )
    (root / "docs" / "FULL_RUNDOWN.md").write_text("\n".join(lines), encoding="utf-8")
