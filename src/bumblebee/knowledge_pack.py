"""Educational knowledge pack for explain / what-is questions (no LLM).

Curated grounded explanations for common ML/systems topics so open phrasing
like \"explain LLM transformers to me\" does not fall into undertrained decode.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# topic keywords (any match) -> (answer, skill)
_TOPICS: List[Tuple[Tuple[str, ...], str, str]] = [
    (
        ("transformer", "transformers", "self-attention", "self attention"),
        "knowledge",
        "A transformer is a neural architecture (Vaswani et al., 2017) built from stacked "
        "self-attention and feed-forward blocks. Each token makes Query/Key/Value vectors; "
        "attention mixes values with weights softmax(QKᵀ/√d). Strength: any token can bind "
        "to any other in the window. Cost: vanilla attention is O(n²) in sequence length, and "
        "autoregressive decoding stores a growing KV cache. Large Language Models (LLMs) are "
        "usually transformers trained to predict the next token at web scale, then used for chat.",
    ),
    (
        ("llm", "large language model", "language model"),
        "knowledge",
        "An LLM (large language model) is typically a transformer trained to predict the next "
        "token given previous tokens. After training on huge text, the same next-token engine "
        "is prompted for chat, coding, and QA. Fluency comes from statistical continuation — "
        "the model is not required to abstain when evidence is missing, which drives many "
        "hallucinations (forced answering). Bumblebee challenges that default with a coverage gate.",
    ),
    (
        ("attention", "multi-head", "multihead"),
        "knowledge",
        "Attention lets a model weight which context tokens matter for the current position. "
        "In transformers, multi-head self-attention runs several QKV projections in parallel so "
        "different heads can specialize. Softmax attention is dense over the window; Bumblebee "
        "instead uses a SoftCorrelator with sparsemax mass over passage cells (not full QKᵀ).",
    ),
    (
        ("gru", "rnn", "recurrent"),
        "knowledge",
        "A GRU (Gated Recurrent Unit) updates a hidden state token-by-token with reset/update "
        "gates. Cost is O(n) in length, not O(n²). Bumblebee’s LocalStateEncoder and small answer "
        "head use GRUCells — transformer-free local state — then SoftCorrelator binds question "
        "state to passage cells.",
    ),
    (
        ("cec", "softcorrelator", "soft correlator", "coverage gate", "bumblebee"),
        "knowledge",
        "Bumblebee CEC (Correlated Evidence Cells) is a transformer-free QA stack: GRU local "
        "encoders → SoftCorrelator (sparsemax evidence mass) → CoverageGate (answer vs refuse) → "
        "extractive citation + small GRU answer head. Knowledge is a frozen pack plus optional "
        "non-LLM live tools (prices/time/weather). It is not a general open-world ChatGPT clone; "
        "it is evidence-gated correlator QA with coverage-gated refusal.",
    ),
    (
        ("sparsemax", "softmax"),
        "knowledge",
        "Softmax turns logits into a dense probability distribution (all entries > 0). Sparsemax "
        "can put exact zeros on irrelevant entries, yielding sparse mass. Bumblebee’s SoftCorrelator "
        "uses sparsemax so evidence concentrates on a few passage cells, which feeds the coverage gate.",
    ),
    (
        ("rag", "retrieval augmented", "retrieval-augmented"),
        "knowledge",
        "RAG (retrieval-augmented generation) fetches documents then lets a generator answer. "
        "Many RAG systems still answer when retrieval is weak. Bumblebee is closer to gated "
        "retrieval QA: if coverage/confidence is low, it refuses instead of inventing fluent text.",
    ),
    (
        ("hallucination", "forced answering", "abstain", "abstention"),
        "knowledge",
        "Hallucinations often come from forced answering: next-token models keep generating even "
        "without support. Selective prediction / abstention literature studies when to refuse. "
        "Bumblebee makes refusal first-class via CoverageGate + faithfulness checks + OOD eval "
        "(refuse precision/recall, ECE, AURC).",
    ),
    (
        ("kv cache", "kv-cache"),
        "knowledge",
        "During autoregressive transformer decoding, past Keys/Values are cached so each new token "
        "does not recompute full attention from scratch. The KV cache grows with sequence length and "
        "dominates serving memory. Bumblebee’s GRU path has no transformer KV cache; cost scales "
        "with recurrent steps and correlator mass over passage cells.",
    ),
    (
        ("embedding", "hash embedding", "token embedding"),
        "knowledge",
        "Token embeddings map discrete IDs to vectors. Bumblebee’s Mac-fit LargeCEC stores most "
        "parameters in a large sparse hash embedding (many buckets × emb_dim) while keeping a narrow "
        "GRU core — so headline parameter count can be ~500M while active dense compute stays small.",
    ),
    (
        ("photosynthesis",),
        "knowledge",
        "Photosynthesis is how plants convert light, water, and carbon dioxide into sugar and oxygen.",
    ),
    (
        ("holi",),
        "knowledge",
        "Holi is an Indian festival celebrating the start of Spring with colors, food, and dancing.",
    ),
]


_EXPLAIN_HINT = re.compile(
    r"\b(explain|describe|tell\s+me\s+about|what\s+are|what\s+is|how\s+do(?:es)?|how\s+does|"
    r"define|walk\s+me\s+through|break\s+down)\b",
    re.I,
)


def lookup_knowledge(question: str) -> Optional[Tuple[str, str]]:
    """Return (answer, skill) if an educational topic matches."""
    q = question.lower().strip()
    # Prefer longer keyword matches
    best: Optional[Tuple[int, str, str]] = None
    for keys, skill, answer in _TOPICS:
        for key in keys:
            if key in q:
                score = len(key)
                if best is None or score > best[0]:
                    best = (score, answer, skill)
    if best is None:
        return None
    # For generic "what is X" without our topics, caller handles refuse
    return best[1], best[2]


def is_explain_style(question: str) -> bool:
    return bool(_EXPLAIN_HINT.search(question or ""))
