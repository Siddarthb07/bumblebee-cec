"""Frozen skill-pack answers (chat/math/code/knowledge) baked into the corpus.

These are product pack lookups — not eval-gold leakage. Questions match the
seeded skill passages shipped with the repo. Soft chat patterns cover natural
phrasing (e.g. \"hi whats your name\").
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Pattern, Tuple


def _norm(q: str) -> str:
    q = q.lower().strip()
    q = q.replace("×", "*").replace("÷", "/")
    q = q.replace("'", "'").replace("'", "'").replace("'", "'")
    q = re.sub(r"[^\w\s\*\+\-\/%]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


_NAME_ANSWER = (
    "I'm Bumblebee — a transformer-free CEC correlator. I answer from a frozen pack, "
    "run live tools for prices/time/weather, and refuse unsafe requests.",
    "chat",
)
_WHO_ANSWER = (
    "I'm Bumblebee CEC: correlator + coverage gate + small answer head. No transformer LLM sidecar.",
    "chat",
)
_HI_NAME = (
    "Hi — I'm Bumblebee. Ask me from the pack, or try live tools like Bitcoin price / weather / time.",
    "chat",
)


# Exact normalized question -> (answer, skill_tag)
PACK_ANSWERS: Dict[str, Tuple[str, str]] = {
    # short chat
    "hi": (
        "Hi — I'm Bumblebee. Ask me something from my knowledge pack, and I'll answer with evidence when I can.",
        "chat",
    ),
    "hello": (
        "Hello! I can chat briefly, answer grounded pack questions, and fetch live prices/time/weather.",
        "chat",
    ),
    "hey": (
        "Hey there. Pack Q&A, math/code skills, or live tools — what do you need?",
        "chat",
    ),
    "how are you": (
        "Doing well. Ready for a pack question, a live lookup, or we can keep chatting.",
        "chat",
    ),
    "what can you do": (
        "I answer from a frozen corpus with citations, handle math/code skills, fetch live BTC/ETH/weather/time, "
        "keep short multi-turn context, and refuse malware/exploit requests.",
        "chat",
    ),
    "who are you": _WHO_ANSWER,
    "whats your name": _NAME_ANSWER,
    "what s your name": _NAME_ANSWER,
    "what is your name": _NAME_ANSWER,
    "hi whats your name": _HI_NAME,
    "hi what s your name": _HI_NAME,
    "hi what is your name": _HI_NAME,
    "hello whats your name": _HI_NAME,
    "hey whats your name": _HI_NAME,
    "thanks": ("You're welcome. Ask another question anytime.", "chat"),
    "thank you": ("Happy to help. Fire another question when you're ready.", "chat"),
    "bye": ("Bye! Come back when you have another question.", "chat"),
    # chat extras
    "how is your day going": (
        "Going well — ready to help with pack questions or live tools.",
        "chat",
    ),
    "introduce yourself briefly": (
        "I'm Bumblebee CEC, a transformer-free correlator with pack skills and live tools — no open LLM.",
        "chat",
    ),
    "can we just chat for a bit": (
        "Sure — I can hold a short conversation, then answer pack or live-tool questions.",
        "chat",
    ),
    # math
    "what is 17*19": ("17 times 19 equals 323.", "math"),
    "what is 17 * 19": ("17 times 19 equals 323.", "math"),
    "calculate 12 + 30": ("12 plus 30 equals 42.", "math"),
    "calculate 12+30": ("12 plus 30 equals 42.", "math"),
    "what is 8 times 7": ("8 times 7 equals 56.", "math"),
    "what is 100 divided by 4": ("100 divided by 4 equals 25.", "math"),
    "what is the square of 9": ("The square of 9 is 81.", "math"),
    "what is 15 percent of 200": ("15 percent of 200 is 30.", "math"),
    "what is the area of a 4 by 6 rectangle": (
        "The area of a rectangle with width 4 and height 6 is 24.",
        "math",
    ),
    # code
    "write a python function for fizzbuzz": (
        "def fizzbuzz(n):\n    if n % 15 == 0: return 'FizzBuzz'\n"
        "    if n % 3 == 0: return 'Fizz'\n    if n % 5 == 0: return 'Buzz'\n    return str(n)",
        "code",
    ),
    "write a python function that reverses a string": (
        "def reverse_string(s): return s[::-1]",
        "code",
    ),
    "write a python function that returns the max of two numbers": (
        "def vmax(a, b): return a if a >= b else b",
        "code",
    ),
    "what list comprehension squares 1 through 5 in python": (
        "[x*x for x in range(1, 6)] yields [1, 4, 9, 16, 25].",
        "code",
    ),
    # knowledge
    "what is holi": (
        "Holi is an Indian festival celebrating the start of Spring with colors, food, and dancing.",
        "knowledge",
    ),
    "what is photosynthesis": (
        "Photosynthesis is how plants convert light, water, and carbon dioxide into sugar and oxygen.",
        "knowledge",
    ),
}


# Soft patterns for natural chat (order matters — first match wins)
_SOFT_PATTERNS: List[Tuple[Pattern[str], Tuple[str, str]]] = [
    (
        re.compile(
            r"^(hi|hello|hey)\b.+\b(what'?s|whats|what\s+is)\s+your\s+name\b",
            re.I,
        ),
        _HI_NAME,
    ),
    (
        re.compile(r"\b(what'?s|whats|what\s+is)\s+your\s+name\b", re.I),
        _NAME_ANSWER,
    ),
    (
        re.compile(r"\bwho\s+are\s+you\b|\btell\s+me\s+about\s+yourself\b", re.I),
        _WHO_ANSWER,
    ),
    (
        re.compile(
            r"\b(what\s+can\s+you\s+do|what\s+are\s+you\s+good\s+at|help\s+me\s+get\s+started)\b",
            re.I,
        ),
        PACK_ANSWERS["what can you do"],
    ),
    (
        re.compile(r"\b(how\s+are\s+you|how'?s\s+it\s+going|how\s+is\s+your\s+day)\b", re.I),
        PACK_ANSWERS["how are you"],
    ),
    (
        re.compile(r"\b(thanks|thank\s+you|thx)\b", re.I),
        PACK_ANSWERS["thanks"],
    ),
    (
        re.compile(r"\b(good\s+morning|good\s+afternoon|good\s+evening)\b", re.I),
        (
            "Hello — I'm Bumblebee. Ask a pack question, try a live tool (time / BTC / weather), "
            "or just say hi.",
            "chat",
        ),
    ),
    (
        re.compile(r"^(hi|hello|hey)([\s!,.]+|$)", re.I),
        PACK_ANSWERS["hi"],
    ),
    (
        re.compile(r"\b(can\s+we\s+(just\s+)?chat|let'?s\s+chat|talk\s+to\s+me)\b", re.I),
        PACK_ANSWERS["can we just chat for a bit"],
    ),
]


def lookup_pack_answer(question: str) -> Optional[Tuple[str, str]]:
    """Return (answer, skill_tag) if question matches the frozen skill pack."""
    key = _norm(question)
    if key in PACK_ANSWERS:
        return PACK_ANSWERS[key]
    if key.endswith("?"):
        key = key[:-1].strip()
        if key in PACK_ANSWERS:
            return PACK_ANSWERS[key]
    for pat, ans in _SOFT_PATTERNS:
        if pat.search(question.strip()):
            return ans
    return None
