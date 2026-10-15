"""Hard banned-topic guardrails. Always checked before tools/packs/CEC."""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Category regexes are request/how-to shaped where possible so benign history
# / overview questions are less likely to hard-refuse.

_CATEGORIES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "malware",
        re.compile(
            r"("
            r"\bwrite\s+(me\s+)?(a\s+)?(virus|malware|exploit|ransomware|keylogger)\b|"
            r"\b(build|create|make|develop)\s+(me\s+)?(a\s+)?(virus|malware|ransomware|keylogger|rootkit|trojan)\b|"
            r"\bhow\s+to\s+(write|build|create|spread|deploy)\s+(a\s+)?(virus|malware|ransomware|keylogger)\b|"
            r"\b(give|send|provide)\s+(me\s+)?(a\s+)?(virus|malware|ransomware|keylogger)\b|"
            r"\b(source\s+code|script)\s+(for|of)\s+(a\s+)?(virus|malware|ransomware|keylogger)\b"
            r")",
            re.I,
        ),
    ),
    (
        "weapons",
        re.compile(
            r"("
            r"\b(build|make|construct|assemble|craft)\s+(a\s+)?(bomb|explosive|ied|pipe\s*bomb|molotov)\b|"
            r"\bhow\s+to\s+(make|build|construct)\s+(a\s+)?(bomb|explosive|gun|firearm|silencer)\b|"
            r"\b(homemade|diy)\s+(bomb|explosive|firearm|gun|ammo|ammunition)\b|"
            r"\b(3d[\s-]?print|manufacture)\s+(a\s+)?(gun|firearm|ghost\s*gun)\b|"
            r"\binstructions?\s+(for|to)\s+(making|building)\s+(a\s+)?(bomb|explosive)\b"
            r")",
            re.I,
        ),
    ),
    (
        "drugs",
        re.compile(
            r"("
            r"\b(synthesize|manufacture|cook|produce)\s+(meth|methamphetamine|fentanyl|heroin|cocaine|lsd|mdma|ecstasy)\b|"
            r"\bhow\s+to\s+(make|cook|synthesize|manufacture)\s+(meth|methamphetamine|fentanyl|heroin|cocaine|lsd|mdma)\b|"
            r"\b(meth|fentanyl)\s+(lab|recipe|synthesis|cook)\b|"
            r"\bdrug\s+(manufacturing|synthesis)\s+(instructions?|recipe|guide)\b"
            r")",
            re.I,
        ),
    ),
    (
        "csam",
        re.compile(
            r"("
            r"\b(child\s*porn\w*|csam|child\s*sexual\s*abuse)\b|"
            r"\b(nude|naked|sexual|porn)\b.{0,40}\b(child|children|kid|kids|minor|minors|underage|preteen|toddler)\b|"
            r"\b(child|children|kid|kids|minor|minors|underage|preteen)\b.{0,40}\b(nude|naked|sexual|porn|erotic)\b|"
            r"\b(sexual(ly)?\s+)?(exploit|abuse)\b.{0,30}\b(child|children|kid|minor|underage)\b|"
            r"\b(child|minor|underage)\b.{0,30}\b(sexual(ly)?\s+)?(exploit|abuse)\b"
            r")",
            re.I,
        ),
    ),
    (
        "fraud",
        re.compile(
            r"("
            r"\bhow\s+to\s+(phish|scam|card|carding)\b|"
            r"\b(write|create|make|build)\s+(a\s+)?(phishing|phish)\s+(email|page|site|kit)\b|"
            r"\b(steal|clone)\s+(credit\s*cards?|identit(y|ies)|ssn|social\s*security)\b|"
            r"\b(identity\s+theft|carding|skimmer)\b.{0,40}\b(how|guide|tutorial|instructions?)\b|"
            r"\bhow\s+to\s+(commit|do)\s+(identity\s+theft|credit\s*card\s+fraud|wire\s+fraud)\b|"
            r"\b(bypass|hack)\s+(2fa|two[\s-]?factor|bank\s+account)\b.{0,20}\b(fraud|steal|scam)?\b"
            r")",
            re.I,
        ),
    ),
    (
        "violent_crime",
        re.compile(
            r"("
            r"\b(assassinate|assassination|hire\s+(a\s+)?(hitman|hit\s*man|killer))\b|"
            r"\bhow\s+to\s+(murder|kill|poison)\s+(someone|a\s+person|my\s+(boss|spouse|wife|husband|partner))\b|"
            r"\b(plan|planning)\s+(a\s+)?(murder|assassination|mass\s+shooting)\b|"
            r"\b(make|build)\s+(a\s+)?(poison|ricin)\s+(to\s+kill|for\s+killing)\b"
            r")",
            re.I,
        ),
    ),
)


def check_banned(question: str) -> Optional[str]:
    """
    Return banned category name if the question hard-refuses, else None.

    Callers should refuse with reason ``unsafe_<category>``.
    """
    q = (question or "").strip()
    if not q:
        return None
    for name, pattern in _CATEGORIES:
        if pattern.search(q):
            return name
    return None


def refuse_reason(category: str) -> str:
    return f"unsafe_{category}"
