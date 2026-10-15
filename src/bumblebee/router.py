"""Bumblebee router: CEC + non-LLM live tools. No open-LM sidecar."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from bumblebee.knowledge_pack import lookup_knowledge
from bumblebee.live_tools import _SEARCH_RE, try_live_or_tool, try_web_search_side
from bumblebee.pack_skills import lookup_pack_answer
from bumblebee.safety import check_banned, refuse_reason
from bumblebee.train import ask_question

Mode = Literal["auto", "corpus"]
Intent = Literal["code", "math", "chat", "live", "unsafe", "open", "corpus"]

_MATH_RE = re.compile(
    r"(∫|√|≤|≥|±|"
    r"\b(solve|calculate|compute|derivative|integral|equation|algebra|geometry|square|percent|percentage)\b|"
    r"\b\d+\s*[\+\-\*/×÷\^]\s*\d+\b|"
    r"\bwhat\s+is\s+\d|"
    r"\b\d+\s+times\s+\d+\b|"
    r"\bmath(ematics)?\b)",
    re.I,
)
_CODE_RE = re.compile(
    r"(```|"
    r"\bdef\s+\w+\s*\(|"
    r"\bclass\s+\w+|"
    r"\bimport\s+\w+|"
    r"\bfunction\s*\(|"
    r"\blist\s+comprehension\b|"
    r"\bwrite\s+(a\s+)?(python|javascript|java|c\+\+|rust|go|sql)\b|"
    r"\b(code|program|script|leetcode|algorithm|bug|stack\s*trace|compile)\b|"
    r"\bfizzbuzz\b|"
    r"\bimplement\b)",
    re.I,
)
_CHAT_RE = re.compile(
    r"^(hi|hello|hey|how are you|thanks|thank you|bye|ok|what can you do|who are you|"
    r"what'?s your name|whats your name|what is your name)"
    r"([\s!.?].*)?$",
    re.I,
)
_LIVE_RE = re.compile(
    r"("
    r"\blive\b.*\b(price|score|weather)\b|"
    r"\bright\s+now\b|"
    r"\bcurrent\s+(price|stock|btc|bitcoin|eth|ethereum|time|weather|temperature)\b|"
    r"\b(bitcoin|btc|ethereum|eth)\b.*\b(price|worth|cost)\b|"
    r"\b(price|worth|cost)\b.*\b(bitcoin|btc|ethereum|eth)\b|"
    r"\b(weather|temperature|forecast)\b|"
    r"\bwhat\s+time\s+is\s+it\b|\bwhat'?s\s+the\s+time\b|\btoday'?s\s+date\b|"
    r"\b(search|google|look\s*up|web\s*search)\b|"
    r"\b(who\s+is|who\s+was|what\s+happened|latest|breaking|news\s+about)\b|"
    r"^\s*(search|find|lookup)\s*:"
    r")",
    re.I,
)


def classify_intent(question: str) -> Intent:
    q = question.strip()
    if not q:
        return "open"
    if check_banned(q):
        return "unsafe"
    if _LIVE_RE.search(q):
        return "live"
    if _CHAT_RE.match(q):
        return "chat"
    if _CODE_RE.search(q):
        return "code"
    if _MATH_RE.search(q):
        return "math"
    return "open"


def _hard_refuse(reason: str = "unsafe") -> dict:
    return {
        "status": "refuse",
        "answer": "",
        "confidence": 0.0,
        "reason": reason,
        "citation_span": "",
        "passage_id": "",
        "mode": "refuse",
        "model_id": "none",
        "skill": reason,
        "source": "refuse",
        "version": "v2",
    }


def _tag(result: dict, *, mode: str, model_id: str, skill: str) -> dict:
    out = dict(result)
    out["mode"] = mode
    out["model_id"] = model_id
    out["skill"] = skill
    out["version"] = "v2"
    return out


def ask_hybrid(
    root: Path,
    question: str,
    *,
    history: Optional[List[Tuple]] = None,
    mode: Mode = "auto",
    passage_id: Optional[str] = None,
    open_model_id: Optional[str] = None,
) -> dict:
    """
    CEC + tools path (no open LM).

    Priority:
    1. banned → hard refuse
    2. deterministic live tools (calc/time/prices/weather) — not web search
    3. pack / knowledge
    4. CEC corpus try-harder (primary)
    5. optional web side-channel (unverified, low conf, overlap-scored) — never replaces a strong corpus hit
    """
    del open_model_id  # API compat; open LM removed
    mode_l = (mode or "auto").lower()

    def _finish(out: dict) -> dict:
        try:
            from bumblebee.hybrid_corpus import ensure_source_label
            from bumblebee.miss_log import record_ask
            from bumblebee.telemetry import emit_ask_activity

            out = ensure_source_label(out)
            emit_ask_activity(root, question, out, source="cli")
            record_ask(root, question, out)
        except Exception:
            try:
                from bumblebee.hybrid_corpus import ensure_source_label

                out = ensure_source_label(out)
            except Exception:
                pass
        return out

    if mode_l == "open":
        return _finish(_hard_refuse("open_lm_removed"))
    if mode_l not in {"auto", "corpus"}:
        mode_l = "auto"

    banned = check_banned(question)
    if banned:
        return _finish(_hard_refuse(refuse_reason(banned)))

    intent = classify_intent(question)
    wants_web = bool(_SEARCH_RE.search(question))

    # Deterministic tools only (web search excluded)
    tool = try_live_or_tool(question)
    if tool is not None:
        answer, tool_id, skill = tool
        return _finish({
            "status": "answer",
            "answer": answer,
            "confidence": 0.99 if skill != "live" or not tool_id.endswith("error") else 0.4,
            "reason": "live_tool" if skill == "live" else "tool",
            "citation_span": answer[:160],
            "passage_id": tool_id,
            "mode": "live" if skill == "live" else "tool",
            "model_id": "cec_tools",
            "skill": skill,
            "source": "tool",
            "source_label": "tool",
            "version": "v2",
        })

    # Frozen skill-pack (exact + soft chat)
    pack = lookup_pack_answer(question)
    if pack is not None:
        answer, pack_skill = pack
        out = {
            "status": "answer",
            "answer": answer,
            "confidence": 0.95,
            "reason": "skill_pack",
            "citation_span": answer[:160],
            "passage_id": f"skill:{pack_skill}",
            "mode": "corpus",
            "model_id": "cec_pack",
            "skill": pack_skill if pack_skill != "knowledge" else "corpus",
            "source": "pack",
            "source_label": "pack",
            "version": "v2",
        }
        if wants_web:
            web = try_web_search_side(question)
            if web:
                out["web"] = web
        return _finish(out)

    # Educational explain / what-is topics (curated knowledge pack — no LLM)
    know = lookup_knowledge(question)
    if know is not None:
        answer, know_skill = know
        out = {
            "status": "answer",
            "answer": answer,
            "confidence": 0.92,
            "reason": "knowledge_pack",
            "citation_span": answer[:160],
            "passage_id": f"knowledge:{know_skill}",
            "mode": "corpus",
            "model_id": "cec_knowledge",
            "skill": "corpus",
            "source": "knowledge",
            "source_label": "knowledge",
            "version": "v2",
        }
        if wants_web:
            web = try_web_search_side(question)
            if web:
                out["web"] = web
        return _finish(out)

    # Product corpus: BM25 shortlist → sentence craft → CEC coverage gate
    if mode_l == "auto" and intent not in {"chat"}:
        try:
            from bumblebee.hybrid_corpus import hybrid_corpus_ask

            hybrid = hybrid_corpus_ask(root, question)
        except Exception:
            hybrid = None
        if hybrid is not None:
            # Craft+gate decided — do not fall through to CEC span scrapes
            out = dict(hybrid)
            out.setdefault("mode", "corpus")
            out.setdefault("model_id", "hybrid_bm25_cec")
            out.setdefault("skill", "corpus")
            out.setdefault("version", "v3")
            return _finish(out)

    # Fallback: full CEC try-harder cascade (when craft found nothing)
    skill = intent if intent in {"code", "math", "chat"} else "corpus"
    try:
        raw = ask_question(root, question, passage_id=passage_id, history=history)
    except FileNotFoundError:
        return _finish({
            "status": "refuse",
            "answer": "",
            "confidence": 0.0,
            "reason": "checkpoint_missing",
            "citation_span": "",
            "passage_id": "",
            "mode": "corpus",
            "model_id": "cec",
            "skill": skill,
            "source": "refuse",
            "source_label": "refuse",
            "hint": "No checkpoint in this checkout. Train first: python -m bumblebee.cli train",
            "version": "v2",
        })
    out = _tag(
        raw,
        mode="corpus",
        model_id="cec",
        skill=skill,
    )
    out["source"] = raw.get("source") or ("refuse" if raw.get("status") == "refuse" else "cec")
    for k in ("mass", "mass_tokens", "multihop", "citation_start", "citation_end", "history_turns"):
        if k in raw:
            out[k] = raw[k]

    corpus_hit = out.get("status") == "answer" and float(out.get("confidence") or 0) >= 0.35
    # Web only when user asks to search, OR factoid miss with strong overlap (labeled).
    # Council: do not silently become an LLM-search wrapper.
    factoid = bool(
        re.search(
            r"\b(what(?:'?s|s)?|what\s+is|who(?:'?s|s)?|who\s+is|real\s+name|full\s+name|"
            r"when\s+was|when\s+did|where\s+is|define|tell\s+me\s+about)\b",
            question,
            re.I,
        )
    )
    if wants_web or (mode_l == "auto" and not corpus_hit and factoid):
        web = try_web_search_side(question)
        if web:
            out["web"] = web
            promote = (
                mode_l == "auto"
                and (not corpus_hit)
                and out.get("status") == "refuse"
                and (
                    wants_web
                    or (factoid and float(web.get("confidence") or 0) >= 0.28)
                )
            )
            if promote:
                out["status"] = "answer"
                out["answer"] = web["answer"]
                out["confidence"] = web["confidence"]
                out["reason"] = web["reason"]
                out["citation_span"] = web.get("citation_span", "")
                out["passage_id"] = web.get("passage_id", "")
                out["source"] = "web_unverified"
                out["source_label"] = "web_unverified"
                out["mode"] = "web"
                out["model_id"] = "web_ddg"
                out["skill"] = "search"
                out["unverified"] = True
                out["overlap"] = web.get("overlap")
                out["web"] = {**web, "note": "primary_because_corpus_miss"}
                out["hint"] = (
                    "Not in my frozen pack — showing an unverified web lookup. "
                    "Ask again with pack topics for cited answers."
                )

    if out.get("status") == "refuse" and not out.get("hint"):
        out["hint"] = (
            "I couldn't ground this in my pack. Try a topic from the corpus, "
            "or rephrase — I won't invent an answer."
        )
    return _finish(out)
