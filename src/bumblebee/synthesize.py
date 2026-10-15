"""Multi-passage stitch + span synthesis helpers (no new model)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from bumblebee.data import Passage
from bumblebee.gate import span_tokens
from bumblebee.tokenize import tokenize


def build_stitch_passages(candidates: Sequence[Passage]) -> List[Passage]:
    """Build 2- and 3-passage concat stitches from BM25 top candidates."""
    out: List[Passage] = []
    if len(candidates) < 2:
        return out
    a, b = candidates[0], candidates[1]
    out.append(
        Passage(
            id=f"{a.id}+{b.id}",
            title=f"multihop:{a.title}+{b.title}",
            text=a.text + " || " + b.text,
        )
    )
    if len(candidates) >= 3:
        c = candidates[2]
        out.append(
            Passage(
                id=f"{a.id}+{b.id}+{c.id}",
                title="multihop3",
                text=a.text + " || " + b.text + " || " + c.text,
            )
        )
    return out


def extract_span_text(
    passage: Passage,
    best_start: int,
    best_end: int,
    max_p: int,
) -> str:
    toks = tokenize(passage.text)[:max_p]
    return span_tokens(toks, best_start, best_end).strip()


def synthesize_from_spans(
    scored: Sequence[Tuple[Passage, int, int, float]],
    *,
    max_p: int,
    top_k: int = 3,
    min_span_chars: int = 8,
) -> Optional[Passage]:
    """
    Merge high-mass citation spans from top-K scored passages into one evidence blob.

    ``scored`` items are (passage, best_start, best_end, confidence), highest conf first.
    """
    if not scored:
        return None
    ordered = sorted(scored, key=lambda x: x[3], reverse=True)[: max(1, top_k)]
    spans: List[str] = []
    ids: List[str] = []
    titles: List[str] = []
    seen_norm: set = set()
    for psg, s, e, _conf in ordered:
        span = extract_span_text(psg, s, e, max_p)
        if len(span) < min_span_chars:
            # Fall back to a short head of the passage if span is empty/tiny
            span = " ".join(tokenize(psg.text)[: min(48, max_p)])
        norm = " ".join(span.lower().split())
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        spans.append(span)
        # Only leaf ids (avoid nested multihop ids)
        for part in psg.id.split("+"):
            if part and part not in ids:
                ids.append(part)
        titles.append(psg.title or psg.id)
    if not spans:
        return None
    text = " || ".join(spans)
    # Cap by approximate token budget
    toks = tokenize(text)
    if len(toks) > max_p:
        text = " ".join(toks[:max_p])
    return Passage(
        id="+".join(ids) if ids else "synthesize",
        title="synthesize:" + "+".join(titles[:3]),
        text=text,
    )


def union_content_tokens(passages: Sequence[Passage], stop: set) -> set:
    """Content tokens across a set of passages (for novelty/support vs union)."""
    out: set = set()
    for p in passages:
        for t in tokenize(p.text):
            if t not in stop:
                out.add(t)
    return out
