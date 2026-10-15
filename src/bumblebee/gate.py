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


def _token_char_spans(text: str, tokens: Sequence[str]) -> list:
    """Greedy left-to-right locate of tokens inside raw passage text."""
    spans: list = []
    pos = 0
    lower = text.lower()
    for tok in tokens:
        idx = lower.find(tok, pos)
        if idx < 0:
            idx = lower.find(tok)
            if idx < 0:
                spans.append((pos, pos))
                continue
        spans.append((idx, idx + len(tok)))
        pos = idx + len(tok)
    return spans


def readable_extract(
    passage_text: str,
    tokens: Sequence[str],
    start: int,
    end: int,
    *,
    max_chars: int = 360,
) -> str:
    """
    Expand a token span to a full sentence (or two) in the raw passage.

    Token streams drop punctuation/casing, so bare span_tokens reads like scrapes.
    Product answers should read like English while staying extractive.
    """
    text = (passage_text or "").strip()
    if not text or not tokens:
        return span_tokens(tokens, start, end)
    spans = _token_char_spans(text, tokens)
    if not spans:
        return span_tokens(tokens, start, end)
    end = min(end, len(spans) - 1)
    start = max(0, min(start, end))
    c0, c1 = spans[start][0], spans[end][1]
    # sentence start: after previous terminator, else passage start
    left = 0
    for m in re.finditer(r"[.!?…][\"')\]]*\s+", text[:c0]):
        left = m.end()
    # sentence end: through next terminator
    right = len(text)
    m = re.search(r"[.!?…][\"')\]]*", text[c1:])
    if m:
        right = c1 + m.end()
    chunk = text[left:right].strip()
    # If the model span was mid-list / mid-clause and sentence is huge, tighten
    if len(chunk) > max_chars:
        pad_l = max(left, c0 - 48)
        pad_r = min(right, c1 + 120)
        chunk = text[pad_l:pad_r].strip()
        # avoid starting mid-word
        if pad_l > left and " " in chunk:
            chunk = chunk.split(" ", 1)[-1]
    chunk = re.sub(r"\s+", " ", chunk).strip(" ,;:-")
    # Drop leading conjunction scraps if still mid-clause
    chunk = re.sub(r"^(?:and|or|but|which|who|that|with)\s+", "", chunk, flags=re.I)
    if chunk and chunk[0].islower():
        chunk = chunk[0].upper() + chunk[1:]
    return chunk or span_tokens(tokens, start, end)


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
    # User-facing extract: full sentence(s) from raw passage when available
    if passage_text and passage_tokens:
        readable = readable_extract(passage_text, passage_tokens, best_start, best_end)
    else:
        readable = citation
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

    # Extractive-first: readable sentence is the default; GRU polish only if faithful
    if prefer_extractive:
        if readable.strip() and not looks_like_garbage(readable, question or ""):
            answer = readable.strip()
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
            answer = readable.strip() or citation.strip()
    else:
        answer = gen
        if not answer:
            answer = readable.strip() or citation.strip()
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
        if readable.strip() and faithfulness_ok(readable, evidence, min_overlap=0.5):
            if not looks_like_garbage(readable, question or ""):
                answer = readable
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
        elif citation.strip() and faithfulness_ok(citation, evidence, min_overlap=0.5):
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
