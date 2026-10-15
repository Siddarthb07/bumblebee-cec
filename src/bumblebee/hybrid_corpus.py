"""BM25 shortlist → intent sentence craft → CEC coverage gate.

Council plan: readable answers from BM25+craft; CEC decides answer vs refuse
on the shortlist (cite/coverage) — no transformer sidecar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bumblebee.answer_craft import craft_or_none, entities_covered
from bumblebee.data import Passage, load_passages
from bumblebee.tokenize import tokenize
from bumblebee.train import (
    _STOP,
    _apply_cal,
    _score_passage,
    _soft_adjust_conf,
    _unpack_runtime,
    get_bm25_index,
    lexical_grounded,
    load_runtime,
    novel_content_ratio,
)


def hybrid_corpus_ask(
    root: Path,
    question: str,
    *,
    top_k: int = 12,
    require_cec_gate: bool = True,
) -> Optional[dict]:
    """
    Product corpus path (no transformers):

    1. BM25 shortlist
    2. Intent-aware sentence craft (readable extract)
    3. CEC CoverageGate on the winning passage — refuse if ungated / ungrounded

    Returns ask-result dict or None if no grounded craft hit (caller may CEC-cascade / refuse).
    """
    root = Path(root)
    passages = load_passages(root / "data" / "corpus")
    if not passages:
        return None

    # Novel OOD first — refuse before BM25 zero-hit fallthrough to CEC scrapes
    corpus_tokens = {t for p in passages for t in tokenize(p.text) if t not in _STOP}
    novel = novel_content_ratio(question, corpus_tokens)
    if novel >= 0.75:
        return {
            "status": "refuse",
            "answer": "",
            "confidence": 0.15,
            "reason": "outside_pack_novel",
            "citation_span": "",
            "citation_start": 0,
            "citation_end": 0,
            "passage_id": "",
            "source": "refuse",
            "source_label": "refuse",
            "multihop": False,
            "mode": "corpus",
            "model_id": "hybrid_bm25_cec",
            "skill": "corpus",
            "version": "v3",
            "hint": "Too many question terms are absent from my frozen pack — refusing instead of guessing.",
        }

    index = get_bm25_index(root, passages)
    pmap = {p.id: p for p in passages}
    hits = [h for h in index.score(question)[:top_k] if h.passage_id in pmap and h.score > 0]
    if not hits:
        return None
    cands = [pmap[h.passage_id] for h in hits]

    crafted = craft_or_none(question, cands)
    if crafted is None:
        return None
    ans, pid, cite = crafted
    passage = pmap.get(pid) or next((p for p in cands if p.id == pid), cands[0])

    if not entities_covered(question, passage.text) and not lexical_grounded(
        question, tokenize(passage.text)
    ):
        return {
            "status": "refuse",
            "answer": "",
            "confidence": 0.2,
            "reason": "weak_lexical_support",
            "citation_span": cite[:160],
            "citation_start": 0,
            "citation_end": 0,
            "passage_id": passage.id,
            "source": "refuse",
            "source_label": "refuse",
            "multihop": False,
            "mode": "corpus",
            "model_id": "hybrid_bm25_cec",
            "skill": "corpus",
            "version": "v3",
        }

    loaded = load_runtime(root)
    model, vocab, _passages, threshold, device, max_q, max_p, temperature, platt_a, platt_b, use_platt = (
        _unpack_runtime(loaded)
    )
    threshold = float(threshold)

    cec_conf = 0.55
    raw = 0.55
    mass: list = []
    mass_tokens: list = []
    gated = require_cec_gate
    if require_cec_gate:
        try:
            out = _score_passage(model, vocab, question, passage, device, max_q, max_p)
            raw = float(out["confidence"][0])
            p_toks = tokenize(passage.text)[:max_p]
            concentration = float(out["concentration"][0])
            cec_conf = _soft_adjust_conf(
                question=question,
                passage_tokens=p_toks,
                conf=raw,
                concentration=concentration,
                evidence_tokens={t for t in p_toks if t not in _STOP},
                multihop=False,
                chatty=False,
            )
            cec_conf = _apply_cal(cec_conf, temperature, platt_a, platt_b, use_platt)
            mass = out["mass"][0].detach().cpu().tolist()[: len(p_toks)]
            mass_tokens = p_toks
        except Exception:
            gated = False
            cec_conf = 0.5

    if gated and cec_conf < threshold:
        return {
            "status": "refuse",
            "answer": "",
            "confidence": float(cec_conf),
            "reason": "low_coverage",
            "citation_span": cite[:160],
            "citation_start": 0,
            "citation_end": max(0, len(cite.split()) - 1),
            "passage_id": passage.id,
            "source": "refuse",
            "source_label": "refuse",
            "multihop": False,
            "mass": mass,
            "mass_tokens": mass_tokens,
            "mode": "corpus",
            "model_id": "hybrid_bm25_cec",
            "skill": "corpus",
            "version": "v3",
            "hint": "I found a candidate sentence but coverage was too low to answer.",
            "craft_preview": ans[:160],
        }

    bm25 = next((float(h.score) for h in hits if h.passage_id == passage.id), 0.0)
    import math

    bm25_conf = min(0.88, 0.45 + (1.0 - math.exp(-max(bm25, 0.0) / 8.0)) * 0.4)
    conf = 0.55 * float(cec_conf) + 0.45 * bm25_conf if gated else bm25_conf

    return {
        "status": "answer",
        "answer": ans,
        "confidence": float(conf),
        "reason": None,
        "citation_span": cite[:200],
        "citation_start": 0,
        "citation_end": max(0, len(cite.split()) - 1),
        "passage_id": passage.id,
        "source": "sentence",
        "source_label": "sentence+cec_gate" if gated else "sentence",
        "multihop": False,
        "mass": mass,
        "mass_tokens": mass_tokens,
        "mode": "corpus",
        "model_id": "hybrid_bm25_cec",
        "skill": "corpus",
        "version": "v3",
        "cec_raw": float(raw),
        "cec_conf": float(cec_conf),
        "bm25": bm25,
    }


def ensure_source_label(out: dict) -> dict:
    """Guarantee source_label on every product response."""
    if out.get("source_label"):
        return out
    src = (out.get("source") or "").lower()
    status = out.get("status")
    if status == "refuse":
        out["source_label"] = "refuse"
    elif src in {"tool", "pack", "knowledge", "sentence", "cec", "web_unverified", "chat_fallback"}:
        out["source_label"] = src
    elif out.get("reason") == "live_tool":
        out["source_label"] = "tool"
    elif out.get("reason") == "skill_pack":
        out["source_label"] = "pack"
    elif out.get("reason") == "knowledge_pack":
        out["source_label"] = "knowledge"
    else:
        out["source_label"] = src or ("refuse" if status == "refuse" else "unknown")
    return out
