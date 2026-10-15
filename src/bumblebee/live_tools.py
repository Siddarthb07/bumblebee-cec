"""Non-LLM live tools: prices, time, weather, web search, safe arithmetic.

No language model — only deterministic APIs / local clock / DuckDuckGo / eval-safe math.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import quote


_BTC_RE = re.compile(
    r"\b(bitcoin|btc)\b.*\b(price|worth|cost)\b|\b(price|worth|cost)\b.*\b(bitcoin|btc)\b|"
    r"\blive\b.*\b(bitcoin|btc)\b|\b(bitcoin|btc)\b.*\bright\s+now\b",
    re.I,
)
_ETH_RE = re.compile(
    r"\b(ethereum|eth)\b.*\b(price|worth|cost)\b|\b(price|worth|cost)\b.*\b(ethereum|eth)\b",
    re.I,
)

# Historical date facts must NOT hit the live clock (when was X founded / born…)
_HISTORICAL_WHEN_RE = re.compile(
    r"\bwhen\s+(?:was|were|did)\b.+\b|"
    r"\b(?:founded|born|invented|established|created|released|built|opened|started)\b",
    re.I,
)

# Live temporal: current clock/calendar — intent-based, not a phrase allowlist
_LIVE_TEMPORAL_RE = re.compile(
    r"\b(?:"
    r"(?:what|whats|what'?s|tell\s+me|give\s+me|check|show)\s+(?:(?:is|the)\s+)?"
    r"(?:current\s+|local\s+|today'?s?\s+|todays\s+)?"
    r"(?:time|date|day|clock)\b|"
    r"(?:current|local|today'?s?|todays)\s+(?:time|date|day)\b|"
    r"(?:time|date)\s+(?:is\s+it|now|today|please)\b|"
    r"what\s+day\s+is\s+(?:it|today)\b|"
    r"\bdate\s+(?:in|for|today|now)\b"
    r")",
    re.I,
)


def is_live_temporal(question: str) -> bool:
    """True when the user wants the *current* clock/date, not a historical fact."""
    q = (question or "").strip()
    if not q:
        return False
    # "when was X founded" / "born in" → corpus/web, not wall clock
    if _HISTORICAL_WHEN_RE.search(q) and not re.search(
        r"\b(today|now|current|clock)\b", q, re.I
    ):
        # Allow "what is today's date" even if something else matches oddly
        if not re.search(r"\b(today|now|current)\b.+\b(date|time)\b|\b(date|time)\b.+\b(today|now|current)\b", q, re.I):
            if re.search(r"\bwhen\s+(?:was|were|did)\b", q, re.I):
                return False
            if re.search(
                r"\b(?:founded|born|invented|established|created|released)\b", q, re.I
            ):
                return False
    return bool(_LIVE_TEMPORAL_RE.search(q))


_WEATHER_RE = re.compile(
    r"\b(?:weather|temperature|forecast)\b(?:\s+(?:in|for|at))?\s+([A-Za-z][A-Za-z\s\-']{1,40})|"
    r"\b(?:what(?:'?s| is)\s+the\s+)?(?:weather|temperature)\b",
    re.I,
)
_CALC_RE = re.compile(
    r"^(?:what\s+is|calculate|compute)?\s*(-?\d+(?:\.\d+)?)\s*([+\-*/×÷xX])\s*(-?\d+(?:\.\d+)?)\s*\??$",
    re.I,
)
_SEARCH_RE = re.compile(
    r"("
    r"\b(search|google|look\s*up|look\s*up|web\s*search|bing)\b|"
    r"\b(who\s+is|who\s+was|what\s+happened|latest|breaking|news\s+about)\b|"
    r"\b(current|today'?s|this\s+week'?s)\b.+\b(news|score|election|ceo|president)\b|"
    r"^\s*(search|find|lookup)\s*:"
    r")",
    re.I,
)


def _http_json(url: str, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "bumblebee-cec/0.3"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_text(url: str, timeout: float = 8.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "bumblebee-cec/0.3"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _fmt_usd(n: float) -> str:
    if n >= 1000:
        return f"${n:,.2f}"
    if n >= 1:
        return f"${n:.2f}"
    return f"${n:.6f}"


def _coin_price(coin_id: str, label: str) -> Tuple[str, str]:
    data = _http_json(
        f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    )
    usd = float(data[coin_id]["usd"])
    answer = f"{label} is about {_fmt_usd(usd)} USD right now (CoinGecko)."
    return answer, f"coingecko:{coin_id}"


def _weather(place: str) -> Tuple[str, str]:
    place = place.strip(" ?.,!") or "London"
    # wttr.in JSON — no API key
    url = f"https://wttr.in/{quote(place)}?format=j1"
    data = _http_json(url)
    cur = data["current_condition"][0]
    area = data.get("nearest_area", [{}])[0]
    name = place
    if area.get("areaName"):
        name = area["areaName"][0].get("value", place)
    temp_c = cur.get("temp_C", "?")
    temp_f = cur.get("temp_F", "?")
    desc = cur.get("weatherDesc", [{"value": "n/a"}])[0].get("value", "n/a")
    humidity = cur.get("humidity", "?")
    answer = (
        f"Weather in {name}: {desc}, {temp_c}°C / {temp_f}°F, humidity {humidity}% "
        f"(live via wttr.in)."
    )
    return answer, f"wttr:{name}"


def _safe_calc(a: float, op: str, b: float) -> Optional[float]:
    if op in {"*", "x", "X", "×"}:
        return a * b
    if op in {"+", "＋"}:
        return a + b
    if op in {"-", "−", "–"}:
        return a - b
    if op in {"/", "÷"}:
        if b == 0:
            return None
        return a / b
    return None


def _strip_search_prefix(question: str) -> str:
    q = question.strip()
    q = re.sub(r"^\s*(search|find|lookup|google|web\s*search)\s*:\s*", "", q, flags=re.I)
    q = re.sub(r"^\s*(search|google|look\s*up|find)\s+(the\s+web\s+for\s+|for\s+)?", "", q, flags=re.I)
    return q.strip() or question.strip()


def web_search(question: str, *, max_related: int = 5) -> Tuple[str, str, dict]:
    """
    DuckDuckGo Instant Answer — no key.

    Returns (primary_text, tool_id, meta) where meta includes related snippets
    and raw fields for overlap verification. Never treat as verified truth.
    """
    q = _strip_search_prefix(question)
    url = (
        "https://api.duckduckgo.com/?"
        f"q={quote(q)}&format=json&no_html=1&skip_disambig=1"
    )
    data = _http_json(url, timeout=10.0)
    bits: list[str] = []
    heading = (data.get("Heading") or "").strip()
    abstract = (data.get("AbstractText") or "").strip()
    answer = (data.get("Answer") or "").strip()
    definition = (data.get("Definition") or "").strip()
    source = (data.get("AbstractURL") or data.get("AnswerURL") or data.get("DefinitionURL") or "").strip()

    if answer:
        bits.append(answer)
    if abstract:
        bits.append(abstract if not heading else f"{heading}: {abstract}")
    elif definition:
        bits.append(definition if not heading else f"{heading}: {definition}")
    elif heading:
        bits.append(heading)

    related = data.get("RelatedTopics") or []
    related_lines: list[str] = []
    for item in related:
        if len(related_lines) >= max_related:
            break
        if isinstance(item, dict) and item.get("Text"):
            related_lines.append(item["Text"][:280])
        elif isinstance(item, dict) and item.get("Topics"):
            for sub in item["Topics"]:
                if len(related_lines) >= max_related:
                    break
                if isinstance(sub, dict) and sub.get("Text"):
                    related_lines.append(sub["Text"][:280])

    if not bits and related_lines:
        bits = [related_lines[0]]
    if not bits:
        lite = _http_text(f"https://lite.duckduckgo.com/lite/?q={quote(q)}", timeout=10.0)
        titles = re.findall(r"class=\"result-link\">([^<]+)</a>", lite)
        if not titles:
            titles = re.findall(r"<a rel=\"nofollow\"[^>]*>([^<]{8,120})</a>", lite)
        titles = [t.strip() for t in titles if t.strip()][:5]
        if not titles:
            raise ValueError("no_web_results")
        bits = titles[:1]
        related_lines = titles[1:]
        source = f"https://duckduckgo.com/?q={quote(q)}"

    primary = bits[0].strip()
    meta = {
        "query": q,
        "primary": primary,
        "related": related_lines,
        "source_url": source or "DuckDuckGo Instant Answer",
        "unverified": True,
    }
    return primary, f"ddg:{quote(q)[:80]}", meta


def web_overlap_score(primary: str, related: list, *, question: str = "") -> float:
    """
    Internal consistency check: overlap of primary vs related snippets (+ question cues).
    High score ≠ truth; only means snippets agree with each other.
    """
    from bumblebee.tokenize import tokenize

    prim = {t for t in tokenize(primary) if len(t) > 2}
    if not prim:
        return 0.0
    pool: set = set()
    for r in related or []:
        pool |= {t for t in tokenize(r) if len(t) > 2}
    q_toks = {t for t in tokenize(question) if len(t) > 2} if question else set()
    if not pool:
        # No related — weak self-check vs question only
        if not q_toks:
            return 0.15
        return min(0.35, len(prim & q_toks) / max(len(prim), 1))
    rel = len(prim & pool) / max(len(prim), 1)
    q_hit = len(prim & q_toks) / max(len(q_toks), 1) if q_toks else 0.0
    return float(min(1.0, 0.7 * rel + 0.3 * q_hit))


def try_web_search_side(question: str) -> Optional[dict]:
    """
    Fetch web side-channel (never primary corpus answer).
    Returns dict with answer, confidence (low), overlap, unverified flag.
    """
    try:
        primary, tid, meta = web_search(question)
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, TimeoutError, OSError):
        return None
    overlap = web_overlap_score(primary, meta.get("related") or [], question=question)
    # Cap confidence: unverified web is always secondary
    if overlap < 0.12:
        conf = 0.18
        note = "web_low_overlap"
    elif overlap < 0.28:
        conf = 0.28
        note = "web_partial_overlap"
    else:
        conf = 0.38
        note = "web_overlap_ok"
    related = meta.get("related") or []
    body = primary
    if related:
        body = primary + "\nRelated:\n" + "\n".join(f"- {r}" for r in related[:3])
    src = meta.get("source_url") or "DuckDuckGo"
    body = f"{body}\n(Unverified web — {src})"
    return {
        "status": "answer",
        "answer": body,
        "confidence": conf,
        "reason": note,
        "citation_span": primary[:160],
        "passage_id": tid,
        "source": "web_unverified",
        "overlap": round(overlap, 4),
        "unverified": True,
    }


def try_live_or_tool(question: str) -> Optional[Tuple[str, str, str]]:
    """
    Return (answer, tool_id, skill) for deterministic live tools only.
    Web search is NOT returned here — use try_web_search_side (secondary, low conf).
    """
    q = question.strip()
    if not q:
        return None

    # Safe arithmetic (ChatGPT-like for simple expressions)
    m = _CALC_RE.match(q.strip())
    if m:
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
        val = _safe_calc(a, op, b)
        if val is not None:
            if float(val).is_integer():
                shown = str(int(val))
            else:
                shown = f"{val:.6g}"
            return (f"{m.group(1)} {op} {m.group(3)} = {shown}.", "calc", "math")

    if is_live_temporal(q):
        now = datetime.now().astimezone()
        utc = datetime.now(timezone.utc)
        answer = (
            f"Local time is {now.strftime('%Y-%m-%d %H:%M:%S %Z')}. "
            f"UTC is {utc.strftime('%Y-%m-%d %H:%M:%S %Z')}."
        )
        return (answer, "local_clock", "live")

    try:
        if _BTC_RE.search(q):
            ans, tid = _coin_price("bitcoin", "Bitcoin (BTC)")
            return (ans, tid, "live")
        if _ETH_RE.search(q):
            ans, tid = _coin_price("ethereum", "Ethereum (ETH)")
            return (ans, tid, "live")
        wm = _WEATHER_RE.search(q)
        if wm and re.search(r"\b(weather|temperature|forecast)\b", q, re.I):
            place = (wm.group(1) or "").strip() if wm.lastindex else ""
            if not place or place.lower() in {"like", "today", "outside", "now"}:
                place = "London"
            ans, tid = _weather(place)
            return (ans, tid, "live")
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, TimeoutError, OSError) as exc:
        return (
            f"I tried a live tool but it failed ({type(exc).__name__}). Try again in a moment.",
            "live_error",
            "live",
        )

    return None
