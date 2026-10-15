from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from bumblebee.gate import AskResult, faithfulness_ok
from bumblebee.tokenize import tokenize


def _stable_hash(text: str) -> int:
    """Process-stable hash (unlike Python's salted hash())."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _bm25_idf(df: int, n_docs: int) -> float:
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


@dataclass
class RetrievalHit:
    passage_id: str
    score: float
    text: str


class BM25Index:
    def __init__(self, passages: Sequence[Tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids = [p[0] for p in passages]
        self.texts = [p[1] for p in passages]
        self.docs = [tokenize(t) for t in self.texts]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(len(self.docs), 1)
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: _bm25_idf(c, len(self.docs)) for t, c in df.items()}

    # Query-side stops: drop interrogatives so rare URL tokens (e.g. "whats" in a
    # citation link) cannot dominate retrieval for "whats <typo>" asks.
    _QUERY_STOP = frozenset({
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are",
        "was", "were", "be", "been", "being", "what", "whats", "what's", "which",
        "who", "whom", "whos", "who's", "how", "hows", "how's", "why", "when",
        "whens", "where", "wheres", "where's", "do", "does", "did", "can", "could",
        "should", "would", "with", "as", "by", "from", "that", "this", "these",
        "those", "it", "its", "at", "into", "about", "please", "tell", "me",
    })

    def score(self, query: str) -> List[RetrievalHit]:
        q = [t for t in tokenize(query) if t not in self._QUERY_STOP and len(t) > 1]
        if not q:
            q = tokenize(query)
        hits: List[RetrievalHit] = []
        for i, doc in enumerate(self.docs):
            tf = Counter(doc)
            s = 0.0
            dl = self.doc_len[i]
            for term in q:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                s += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            hits.append(RetrievalHit(self.ids[i], s, self.texts[i]))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits


class EmbedIndex:
    """Hashing-trick bag embeddings — no external model dependency."""

    def __init__(self, passages: Sequence[Tuple[str, str]], dim: int = 256):
        self.dim = dim
        self.ids = [p[0] for p in passages]
        self.texts = [p[1] for p in passages]
        self.vecs = [self._embed(t) for t in self.texts]

    def _embed(self, text: str):
        import numpy as np

        v = np.zeros(self.dim, dtype=np.float64)
        for tok in tokenize(text):
            h = _stable_hash(tok) % self.dim
            sign = 1.0 if (_stable_hash(tok + "#") % 2 == 0) else -1.0
            v[h] += sign
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def score(self, query: str) -> List[RetrievalHit]:
        import numpy as np

        q = self._embed(query)
        hits: List[RetrievalHit] = []
        for i, v in enumerate(self.vecs):
            s = float(np.dot(q, v))
            hits.append(RetrievalHit(self.ids[i], s, self.texts[i]))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits


def retrieval_ask(
    index,
    question: str,
    threshold: float,
    gold_answer: Optional[str] = None,
    *,
    faithfulness: bool = True,
    lexical_ood: bool = True,
    corpus_tokens: Optional[set] = None,
) -> AskResult:
    """
    Retrieve + extractive answer from the top hit.
    Same refuse family as CEC when faithfulness/lexical_ood enabled:
    low_coverage / unfaithful_decode / outside_pack (lexical).
    ``gold_answer`` is accepted for API compat but never used as the prediction.
    """
    del gold_answer  # never leak labels into predictions
    hits = index.score(question)
    if not hits:
        return AskResult("refuse", "", 0.0, "no_envelope_match", "", 0, 0)
    top = hits[0]
    # normalize crude confidence
    if isinstance(index, BM25Index):
        conf = 1.0 - math.exp(-max(top.score, 0.0) / 5.0)
    else:
        conf = max(0.0, min(1.0, (top.score + 1.0) / 2.0))
    citation = top.text
    toks = tokenize(citation)
    end = min(11, max(0, len(toks) - 1))

    if lexical_ood and corpus_tokens is not None:
        # Mirror CEC soft novel-content refuse (hard threshold for baselines)
        q_content = {t for t in tokenize(question) if len(t) > 2}
        if q_content:
            missing = sum(1 for t in q_content if t not in corpus_tokens)
            novel = missing / max(len(q_content), 1)
            if novel >= 0.5:
                return AskResult("refuse", "", conf * 0.35, "outside_pack", citation, 0, end)

    if conf < threshold:
        return AskResult("refuse", "", conf, "low_coverage", citation, 0, end)
    answer = " ".join(toks[:16])
    if faithfulness and not faithfulness_ok(answer, citation):
        return AskResult("refuse", "", conf, "unfaithful_decode", citation, 0, end)
    return AskResult("answer", answer, conf, None, citation, 0, end)
