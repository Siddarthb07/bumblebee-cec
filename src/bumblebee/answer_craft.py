"""Product answer crafting — usable sentences, not mid-span scrapes.

General rules (not per-entity hardcodes):
- Entity phrases must co-occur (tony+stark), not surname-only hits
- Sentence ranking is intent-aware (definition / identity / date-fact)
- Weak or off-intent hits return None so router can refuse or web-fallback
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

from bumblebee.data import Passage
from bumblebee.tokenize import tokenize

# Question scaffolding — not entities to "match" in passages
_Q_STOP = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "being", "what", "whats", "what's", "which",
    "who", "whom", "whos", "who's", "how", "hows", "how's", "why", "when",
    "where", "wheres", "do", "does", "did", "can", "could", "should", "would",
    "with", "as", "by", "from", "that", "this", "these", "those", "it", "its",
    "at", "into", "about", "please", "tell", "me", "know", "define", "meaning",
    "name", "real", "true", "called", "known", "full", "give", "get", "say",
    "said", "explain", "describe", "list", "show", "find", "look", "up",
    "brief", "briefly", "synopsis", "summary", "according", "following", "paragraph",
    "reference", "text", "family", "likelihood", "cause", "belong", "belongs",
    "given", "based", "passage", "extract", "names", "some", "per", "appear",
    "appeared", "first", "next", "years", "inputs", "outputs", "contribution",
    "occurring", "ocurring", "inefficiency", "operations", "make", "made",
    "using", "used", "into", "over", "under", "than", "then", "also", "just",
    "like", "many", "much", "more", "most", "such", "only", "even", "still",
})

_FRAME_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"what(?:'?s|s| is| are)|who(?:'?s|s| is| are)|when(?:'?s|s| was| were| did)|"
    r"where(?:'?s|s| is)|how(?:'?s|s| did| do| does)|"
    r"tell\s+me|give\s+me|explain|define|describe|"
    r"the\s+real\s+name\s+of|real\s+name\s+of|full\s+name\s+of|"
    r"when\s+did|when\s+was|"
    r"given\s+the\s+(?:reference\s+)?text(?:\s+about)?|"
    r"according\s+to\s+the\s+(?:following\s+)?(?:paragraph|passage|text)|"
    r"based\s+on\s+the\s+text"
    r")\s+",
    re.I,
)

_PROPER_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"  # Larry David, Borussia Dortmund
    r"|\b([A-Z]{2,}(?:-[A-Z0-9]+)+)\b"  # AES-GCM
    r"|\b([A-Z][a-z]+)\b"  # Cascadia, Holi (single proper)
)

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_DATEISH = re.compile(
    r"^(?:in\s+)?(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|\d{1,4})(?:\s+\d{1,4})?\.?$",
    re.I,
)
_TEMPLATE_JUNK = re.compile(
    r"\b(known answer|ground truth|passage\s*:|context\s*:|answer\s*:)\b|^\s*topic\s*:",
    re.I,
)
_YEAR_RE = re.compile(r"\b((?:1[0-9]{3})|(?:20[0-9]{2}))\b")
_IDENTITY_CUES = (
    "real name", "full name", "also known", "known as", "alter ego",
    "secret identity", "born ", " whose name", "fictional character",
    "portrayed by", "played by", " is a ", " was a ",
)
_DATE_FACT_CUES = (
    "founded", "born", "established", "created", "released", "opened",
    "started", "established in", "founded in", "born in", "built in",
)


def content_terms(question: str) -> List[str]:
    return [t for t in tokenize(question) if t not in _Q_STOP and len(t) > 1]


def focus_terms(question: str) -> List[str]:
    """
    Entity/topic focus for grounding.

    Prefers proper nouns / tech tokens from the raw question so Dolly wrappers
    like 'Given the reference text about…' do not drown 'Bityla sericea'.
    """
    raw = (question or "").strip()
    proper_toks: List[str] = []
    for m in _PROPER_RE.finditer(raw):
        chunk = next(g for g in m.groups() if g)
        if chunk.lower() in _Q_STOP or chunk.lower() in {
            "given", "what", "when", "where", "how", "please", "name", "tell",
            "based", "according", "following", "paragraph", "passage",
        }:
            continue
        proper_toks.extend(tokenize(chunk))

    q = _FRAME_RE.sub("", raw)
    q = re.sub(
        r"\b(given the reference text about|based on the text|as per the passage|"
        r"according to the following paragraph on|brief synopsis of)\b",
        " ",
        q,
        flags=re.I,
    )
    q = re.sub(r"[?!.]+$", "", q).strip()
    residual = [t for t in tokenize(q) if t not in _Q_STOP and len(t) > 2]

    seen: set = set()
    out: List[str] = []
    for t in proper_toks + residual:
        if t in seen or t in _Q_STOP:
            continue
        # If we have proper nouns, drop short generic residuals
        if proper_toks and t not in proper_toks and len(t) < 5 and "-" not in t:
            continue
        seen.add(t)
        out.append(t)
    return out or content_terms(question)


def entity_bigrams(terms: Sequence[str]) -> List[Tuple[str, str]]:
    return [(terms[i], terms[i + 1]) for i in range(len(terms) - 1)]


def detect_intents(question: str) -> Set[str]:
    """Coarse question intents for ranking — not entity-specific."""
    q = (question or "").lower()
    intents: Set[str] = set()
    if re.search(r"\b(real\s+name|full\s+name|also\s+known\s+as|alter\s+ego)\b", q):
        intents.add("identity")
    if re.search(r"\bwho\s+(?:is|was|are)\b", q):
        intents.add("identity")
    if re.search(
        r"\bwhen\b.+\b(found|born|invent|establish|start|creat|releas|open|built)\w*\b"
        r"|\b(found|born|invent|establish)\w*\b.+\bwhen\b"
        r"|\bwhen\s+(?:was|were|did)\b",
        q,
    ):
        intents.add("date_fact")
    if re.search(r"\b(what\s+is|what\s+are|define|meaning\s+of|tell\s+me\s+about)\b", q):
        intents.add("definition")
    if re.search(
        r"\b(what\s+would\s+cause|what\s+causes|why\s+(?:would|does|do|is|did)|"
        r"cause\s+\w+\s+to|inefficiency|what\s+family|belong\s+to)\b",
        q,
    ):
        intents.add("cause")
    if not intents:
        intents.add("generic")
    return intents


def entities_covered(question: str, text: str) -> bool:
    """
    Require real entity coverage — not a single shared surname/street token.

    Uses focus terms (frames stripped) so 'real name of tony stark' requires
    tony+stark together, not 'name' scaffolding.
    """
    terms = focus_terms(question)
    if not terms:
        terms = content_terms(question)
    if not terms:
        return False
    tl = (text or "").lower()
    toks = set(tokenize(text))
    anchors = [t for t in terms if len(t) >= 4]
    if not anchors:
        return all(t in toks for t in terms)

    bigrams = entity_bigrams(anchors) if len(anchors) >= 2 else []
    if bigrams:
        window = tokenize(text)
        ok_bi = False
        for a, b in bigrams:
            if f"{a} {b}" in tl or f"{a}-{b}" in tl:
                ok_bi = True
                break
            for i, tok in enumerate(window):
                if tok != a:
                    continue
                neigh = set(window[max(0, i - 3) : i + 4])
                if b in neigh:
                    ok_bi = True
                    break
            if ok_bi:
                break
        hit_n = sum(1 for t in anchors if t in toks)
        if ok_bi:
            return hit_n >= max(1, (len(anchors) + 1) // 2)
        # Soft person/topic: majority of anchors incl. at least one long token
        # (Larry David + Seinfeld when passage only has David…Seinfeld)
        if hit_n >= min(2, len(anchors)) and any(len(t) >= 5 and t in toks for t in anchors):
            return True
        return False

    return anchors[0] in toks


def entities_covered_loose(
    question: str,
    sentence: str,
    *,
    passage_text: str = "",
) -> bool:
    """
    Like entities_covered, but allows identity follow-ons:
    passage has 'Tony Stark…' and sentence is 'His full name is Anthony Edward Stark'.
    """
    if entities_covered(question, sentence):
        return True
    if not passage_text or not entities_covered(question, passage_text):
        return False
    intents = detect_intents(question)
    if "identity" not in intents and "date_fact" not in intents:
        return False
    sl = sentence.lower()
    if "identity" in intents and not re.search(
        r"\b(full name is|real name is|also known as|known as)\b", sl
    ):
        return False
    if "date_fact" in intents and not (
        any(c in sl for c in _DATE_FACT_CUES) and _YEAR_RE.search(sl)
    ):
        return False
    anchors = [t for t in focus_terms(question) if len(t) >= 4]
    if not anchors:
        return False
    # Surname / last focus token must appear in the follow-on sentence
    return anchors[-1] in set(tokenize(sentence))


def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "") if p and p.strip()]
    if parts:
        return parts
    t = (text or "").strip()
    return [t] if t else []


def is_usable_answer(answer: str, question: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return False
    if _TEMPLATE_JUNK.search(a):
        return False
    words = a.split()
    if len(words) < 5:
        return False
    if len(a) < 24:
        return False
    if _DATEISH.match(a):
        return False
    if re.match(r"^(?:in|on|at|of|for|and|or|the|a|an)\s+\d", a, re.I):
        return False
    if not entities_covered(question, a):
        terms = focus_terms(question)
        anchors = [t for t in terms if len(t) >= 4]
        if anchors:
            return False
    return True


def answers_intent(question: str, sentence: str) -> bool:
    """Hard filter: sentence must speak to the question type, not merely mention entities."""
    intents = detect_intents(question)
    sl = (sentence or "").lower()
    if _TEMPLATE_JUNK.search(sl):
        return False

    if "date_fact" in intents:
        has_cue = any(c in sl for c in _DATE_FACT_CUES)
        has_year = bool(_YEAR_RE.search(sl))
        return has_cue and has_year

    if "identity" in intents:
        ql = (question or "").lower()
        # "real/full name of X" needs an explicit identity cue — not "X who is still…"
        if re.search(r"\b(real\s+name|full\s+name|alter\s+ego|secret\s+identity)\b", ql):
            return bool(
                re.search(
                    r"\b(real name|full name|also known|known as|alter ego|"
                    r"secret identity|born\b|whose name|full name is|real name is)\b",
                    sl,
                )
            )
        # "who is X" — definitional identity OK
        if any(c in sl for c in _IDENTITY_CUES):
            return True
        anchors = [t for t in focus_terms(question) if len(t) >= 4]
        head = sl[:100]
        if anchors and any(a in head for a in anchors) and re.search(
            r"\b(?:is|was|are|were)\s+(?:a|an|the)\b", sl
        ):
            return True
        return False

    if "definition" in intents:
        anchors = [t for t in focus_terms(question) if len(t) >= 4]
        if not anchors:
            return " is " in sl or " are " in sl
        head = sl[:140]
        early = any(a in head for a in anchors)
        has_be = bool(re.search(r"\b(?:is|are|was|were)\b", sl))
        # taxonomy / membership questions
        if re.search(r"\b(family|genus|species|belong|member)\b", (question or "").lower()):
            return early and bool(
                re.search(r"\b(family|genus|species|belong|endemic|classified)\b", sl)
            )
        return early and has_be

    if "cause" in intents:
        return bool(
            re.search(
                r"\b(cause|because|would|if|dependency|blocking|prevent|lead(?:s|ing)?\s+to)\b",
                sl,
            )
        )

    return True


def score_sentence(question: str, sentence: str) -> float:
    terms = focus_terms(question)
    if not terms or not sentence:
        return 0.0
    st = tokenize(sentence)
    if not st:
        return 0.0
    hits = sum(1 for t in terms if t in st)
    support = hits / max(len(terms), 1)
    sl = sentence.lower()
    bonus = 0.0
    intents = detect_intents(question)

    if "date_fact" in intents:
        if any(c in sl for c in _DATE_FACT_CUES):
            bonus += 0.35
        if _YEAR_RE.search(sl):
            bonus += 0.25
    if "identity" in intents:
        for cue in _IDENTITY_CUES:
            if cue in sl:
                bonus += 0.12
    if "definition" in intents:
        if re.search(r"\b(?:is|are)\s+(?:a|an|the)\b", sl):
            bonus += 0.2
        for cue in (" company", " corporation", " language", " festival", " currency", " family "):
            if cue in sl:
                bonus += 0.1
    if "cause" in intents:
        for cue in (" blocking", " would ", " because", " dependency", " if ", " cause"):
            if cue in sl:
                bonus += 0.25

    # Prefer focus entity near the start (subject position)
    anchors = [t for t in terms if len(t) >= 4]
    for a in anchors:
        idx = sl.find(a)
        if idx < 0:
            continue
        if idx < 48:
            bonus += 0.35
        elif idx < 120:
            bonus += 0.1
        else:
            bonus -= 0.15  # late name-drop
        break

    # Phrase bigram bonus
    for a, b in entity_bigrams(anchors):
        if f"{a} {b}" in sl:
            bonus += 0.35

    n = len(st)
    if 8 <= n <= 50:
        bonus += 0.1
    elif n < 5:
        bonus -= 0.4
    elif n > 80:
        bonus -= 0.2

    if _TEMPLATE_JUNK.search(sl):
        bonus -= 1.0

    return support + bonus


def best_sentence_answer(
    question: str,
    passages: Sequence[Passage],
    *,
    min_score: float = 0.85,
) -> Optional[Tuple[str, str, str]]:
    """
    Pick the best intent-matching, entity-covering sentence across passages.

    Returns None when nothing clears the bar (caller may refuse / web).
    """
    best: Optional[Tuple[float, str, str, str, str]] = None  # score, ans, pid, cite, passage
    for psg in passages:
        if not entities_covered(question, psg.text):
            continue
        for sent in split_sentences(psg.text):
            if _TEMPLATE_JUNK.search(sent):
                continue
            if not entities_covered_loose(question, sent, passage_text=psg.text):
                continue
            if not answers_intent(question, sent):
                continue
            usable = is_usable_answer(sent, question)
            if not usable:
                if len(sent.split()) < 4:
                    continue
                intents = detect_intents(question)
                ok_follow = False
                if "identity" in intents and re.search(
                    r"\bfull name is\b|\breal name is\b", sent, re.I
                ):
                    ok_follow = True
                if "date_fact" in intents and entities_covered_loose(
                    question, sent, passage_text=psg.text
                ):
                    ok_follow = True
                if not ok_follow:
                    continue
            sc = score_sentence(question, sent)
            if best is None or sc > best[0]:
                best = (sc, sent.strip(), psg.id, sent.strip(), psg.text)
    if best is None or best[0] < min_score:
        return None
    ans = best[1]
    if len(ans) > 420:
        ans = ans[:417].rsplit(" ", 1)[0] + "…"
    if not answers_intent(question, ans):
        return None
    if not is_usable_answer(ans, question) and not entities_covered_loose(
        question, ans, passage_text=best[4]
    ):
        return None
    return ans, best[2], best[3]


def craft_or_none(
    question: str,
    passages: Sequence[Passage],
    *,
    fallback_answer: str = "",
) -> Optional[Tuple[str, str, str]]:
    """Prefer best sentence; accept fallback only if usable + intent-ok."""
    hit = best_sentence_answer(question, passages)
    if hit is not None:
        return hit
    if (
        fallback_answer
        and is_usable_answer(fallback_answer, question)
        and answers_intent(question, fallback_answer)
    ):
        return fallback_answer.strip(), "", fallback_answer.strip()
    return None


def try_sentence_corpus(root, question: str, *, top_k: int = 12) -> Optional[dict]:
    """
    Product-first corpus answer: BM25 → best intent-matched sentence.

    Returns None when evidence does not actually answer the question type.
    """
    from bumblebee.data import load_passages
    from bumblebee.train import get_bm25_index

    root = Path(root)
    passages = load_passages(root / "data" / "corpus")
    if not passages:
        return None
    index = get_bm25_index(root, passages)
    pmap = {p.id: p for p in passages}
    hits = index.score(question)[:top_k]
    cands = [pmap[h.passage_id] for h in hits if h.passage_id in pmap and h.score > 0]
    if not cands:
        return None
    crafted = craft_or_none(question, cands)
    if crafted is None:
        return None
    ans, pid, cite = crafted
    bm25 = 0.0
    for h in hits:
        if h.passage_id == pid:
            bm25 = float(h.score)
            break
    conf = min(0.88, 0.45 + (1.0 - math.exp(-max(bm25, 0.0) / 8.0)) * 0.4)
    return {
        "status": "answer",
        "answer": ans,
        "confidence": conf,
        "reason": None,
        "citation_span": cite[:200],
        "citation_start": 0,
        "citation_end": max(0, len(cite.split()) - 1),
        "passage_id": pid,
        "source": "sentence",
        "multihop": False,
        "mode": "corpus",
        "model_id": "bm25_sentence",
        "skill": "corpus",
        "version": "v2",
    }
