from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Set

from bumblebee.tokenize import tokenize

_GREETING_RE = re.compile(
    r"\b(hi|hello|hey|welcome|thanks|thank|bye|bumblebee)\b",
    re.I,
)


@dataclass
class AskResult:
    status: str  # "answer" | "refuse"
    answer: str
    confidence: float
    reason: Optional[str]
    citation_span: str
    citation_start: int
    citation_end: int


def span_tokens(tokens: Sequence[str], start: int, end: int) -> str:
    if not tokens:
        return ""
    end = min(end, len(tokens) - 1)
    start = max(0, min(start, end))
    return " ".join(tokens[start : end + 1])


def faithfulness_ok(answer: str, evidence_text: str, min_overlap: float = 0.15) -> bool:
    """Soft grounding: answer should share some content with evidence/passage."""
    a = {t for t in tokenize(answer) if len(t) > 2}
    c = {t for t in tokenize(evidence_text) if len(t) > 2}
    if not a:
        return False
    # Short chat templates: whole-word greetings only (not substring "hi" in "ship")
    if len(a) <= 8 and _GREETING_RE.search(answer):
        return True
    overlap = len(a & c) / max(len(a), 1)
    return overlap >= min_overlap


def looks_like_garbage(answer: str, question: str = "") -> bool:
    """Reject decode collapse / question-echo / template junk."""
    a = (answer or "").strip()
    if not a:
        return True
    al = a.lower()
    if "known answer" in al or "ground truth" in al:
        return True
    if re.search(r"\b(answer\s*:|passage\s*:|context\s*:)\b", al):
        return True
    # Question echoed at the start of the answer
    q = re.sub(r"[^\w\s]", " ", (question or "").lower())
    q = re.sub(r"\s+", " ", q).strip()
    a0 = re.sub(r"[^\w\s]", " ", al)
    a0 = re.sub(r"\s+", " ", a0).strip()
    if q and len(q) >= 8 and a0.startswith(q):
        return True
    q_toks = [t for t in tokenize(question) if len(t) > 2]
    a_toks = tokenize(a)
    if len(q_toks) >= 3 and len(a_toks) >= 3:
        head = a_toks[: len(q_toks)]
        if head == q_toks[: len(head)] and len(head) >= 3:
            return True
    return False


def decide(
    *,
    passage_tokens: Sequence[str],
    best_start: int,
    best_end: int,
    confidence: float,
    threshold: float,
    generated_answer: Optional[str] = None,
    passage_text: Optional[str] = None,
    question: Optional[str] = None,
    prefer_extractive: bool = True,
) -> AskResult:
    """Decide answer vs refuse. Never substitutes gold labels — score those separately."""
    citation = span_tokens(passage_tokens, best_start, best_end)
    evidence = passage_text or citation
    if confidence < threshold:
        return AskResult(
            status="refuse",
            answer="",
            confidence=float(confidence),
            reason="low_coverage",
            citation_span=citation,
            citation_start=best_start,
            citation_end=best_end,
        )

    gen = (generated_answer or "").strip()
    if gen and looks_like_garbage(gen, question or ""):
        gen = ""

    # Extractive-first: citation is the default answer; GRU polish only if faithful
    if prefer_extractive:
        if citation.strip() and not looks_like_garbage(citation, question or ""):
            answer = citation.strip()
            if (
                gen
                and faithfulness_ok(gen, evidence, min_overlap=0.2)
                and not looks_like_garbage(gen, question or "")
                and len(gen.split()) >= 3
            ):
                # Optional polish when clearly grounded
                answer = gen
        elif gen and faithfulness_ok(gen, evidence):
            answer = gen
        else:
            answer = citation.strip()
    else:
        answer = gen
        if not answer:
            answer = citation.strip()
        if answer and looks_like_garbage(answer, question or ""):
            return AskResult(
                status="refuse",
                answer="",
                confidence=float(confidence),
                reason="garbage_decode",
                citation_span=citation,
                citation_start=best_start,
                citation_end=best_end,
            )

    if answer and looks_like_garbage(answer, question or ""):
        return AskResult(
            status="refuse",
            answer="",
            confidence=float(confidence),
            reason="garbage_decode",
            citation_span=citation,
            citation_start=best_start,
            citation_end=best_end,
        )
    if not answer:
        return AskResult(
            status="refuse",
            answer="",
            confidence=float(confidence),
            reason="empty_decode",
            citation_span=citation,
            citation_start=best_start,
            citation_end=best_end,
        )
    if not faithfulness_ok(answer, evidence):
        if citation.strip() and faithfulness_ok(citation, evidence, min_overlap=0.5):
            if not looks_like_garbage(citation, question or ""):
                answer = citation
            else:
                return AskResult(
                    status="refuse",
                    answer="",
                    confidence=float(confidence),
                    reason="unfaithful_decode",
                    citation_span=citation,
                    citation_start=best_start,
                    citation_end=best_end,
                )
        else:
            return AskResult(
                status="refuse",
                answer="",
                confidence=float(confidence),
                reason="unfaithful_decode",
                citation_span=citation,
                citation_start=best_start,
                citation_end=best_end,
            )

    return AskResult(
        status="answer",
        answer=answer,
        confidence=float(confidence),
        reason=None,
        citation_span=citation,
        citation_start=best_start,
        citation_end=best_end,
    )


def token_f1(pred: str, gold: str) -> float:
    p: Set[str] = set(tokenize(pred))
    g: Set[str] = set(tokenize(gold))
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = len(p & g)
    precision = overlap / len(p)
    recall = overlap / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
