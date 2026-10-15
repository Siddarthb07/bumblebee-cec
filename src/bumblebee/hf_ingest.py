"""Hugging Face dataset ingest for conversational, full-sentence Bumblebee training."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset


def _pid(text: str) -> str:
    return "p_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _qid(text: str, prefix: str) -> str:
    return f"{prefix}_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


CHAT_SEED = [
    {
        "id": "chat_hi",
        "passage": (
            "Bumblebee is its own transformer-free CEC language model. It answers from its frozen "
            "knowledge pack with citations, holds short conversations, solves in-pack math and coding "
            "questions, and refuses when a question is outside its dataset instead of guessing."
        ),
        "turns": [
            ("hi", "Hi — I'm Bumblebee, a CEC model. Ask me from my knowledge pack — chat, math, or coding I've been trained on."),
            ("hello", "Hello! I can chat and answer grounded questions from my own frozen corpus."),
            ("hey", "Hey there. What would you like to know from my training pack?"),
            ("how are you", "Doing well. Ready for a question from my knowledge pack."),
            ("what can you do", "I am Bumblebee CEC: I chat, answer corpus questions, do in-pack math and coding, cite evidence, and refuse out-of-dataset asks."),
            ("who are you", "I'm Bumblebee — a transformer-free correlator LLM trained on a closed pack (chat, knowledge, math, code). No sidecar chatbot required."),
            ("thanks", "You're welcome. Ask another question anytime."),
            ("thank you", "Happy to help. Fire another question when you're ready."),
            ("ok", "Sounds good. Ask me anything covered by my knowledge pack."),
            ("bye", "Bye! Come back when you have another question from the corpus."),
        ],
    }
]

# Built-in skill pack so CEC itself covers math / code / convo (not an external LLM).
SKILL_SEED = [
    {
        "category": "math",
        "passage": (
            "Basic arithmetic reference for Bumblebee. "
            "17 times 19 equals 323. "
            "12 plus 30 equals 42. "
            "8 times 7 equals 56. "
            "100 divided by 4 equals 25. "
            "The square of 9 is 81. "
            "15 percent of 200 is 30. "
            "The area of a rectangle with width 4 and height 6 is 24."
        ),
        "pairs": [
            ("What is 17*19?", "17 times 19 equals 323."),
            ("What is 17 × 19?", "17 times 19 equals 323."),
            ("Calculate 12 + 30", "12 plus 30 equals 42."),
            ("What is 8 times 7?", "8 times 7 equals 56."),
            ("What is 100 divided by 4?", "100 divided by 4 equals 25."),
            ("What is the square of 9?", "The square of 9 is 81."),
            ("What is 15 percent of 200?", "15 percent of 200 is 30."),
            ("What is the area of a 4 by 6 rectangle?", "The area of a rectangle with width 4 and height 6 is 24."),
        ],
    },
    {
        "category": "code",
        "passage": (
            "Python coding reference for Bumblebee. "
            "FizzBuzz: for n from 1 to 100, print Fizz if n divisible by 3, Buzz if by 5, "
            "FizzBuzz if by both, else n. Example function: "
            "def fizzbuzz(n):\n"
            "    if n % 15 == 0: return 'FizzBuzz'\n"
            "    if n % 3 == 0: return 'Fizz'\n"
            "    if n % 5 == 0: return 'Buzz'\n"
            "    return str(n)\n"
            "A Python function that reverses a string: def reverse_string(s): return s[::-1]. "
            "A Python function that returns the maximum of two numbers: def vmax(a, b): return a if a >= b else b. "
            "List comprehension to square numbers 1..5: [x*x for x in range(1, 6)] which yields [1, 4, 9, 16, 25]."
        ),
        "pairs": [
            (
                "Write a Python function for fizzbuzz",
                "def fizzbuzz(n):\n    if n % 15 == 0: return 'FizzBuzz'\n    if n % 3 == 0: return 'Fizz'\n    if n % 5 == 0: return 'Buzz'\n    return str(n)",
            ),
            (
                "Write a Python function that reverses a string",
                "def reverse_string(s): return s[::-1]",
            ),
            (
                "Write a Python function that returns the max of two numbers",
                "def vmax(a, b): return a if a >= b else b",
            ),
            (
                "What list comprehension squares 1 through 5 in Python?",
                "[x*x for x in range(1, 6)] yields [1, 4, 9, 16, 25].",
            ),
        ],
    },
    {
        "category": "chat",
        "passage": (
            "Conversation style for Bumblebee. Replies are complete sentences. "
            "When asked how the day is going, say the assistant is ready to help with corpus questions. "
            "When asked to introduce itself briefly, say it is Bumblebee CEC, a transformer-free model."
        ),
        "pairs": [
            ("How is your day going?", "Going well — ready to help with questions from my training pack."),
            ("Introduce yourself briefly", "I'm Bumblebee CEC, a transformer-free correlator language model trained on a frozen knowledge pack."),
            ("Can we just chat for a bit?", "Sure — I can hold a short conversation, then answer grounded questions from my corpus."),
        ],
    },
    {
        "category": "knowledge",
        "passage": (
            "Holi is an Indian festival celebrating the start of Spring with colors, food, and dancing. "
            "Photosynthesis is how plants convert light, water, and carbon dioxide into sugar and oxygen."
        ),
        "pairs": [
            ("What is Holi?", "Holi is an Indian festival celebrating the start of Spring with colors, food, and dancing."),
            ("What is photosynthesis?", "Photosynthesis is how plants convert light, water, and carbon dioxide into sugar and oxygen."),
        ],
    },
]


def _append_skill_seed(passages: List[dict], evals: List[dict]) -> None:
    for block in SKILL_SEED:
        text = block["passage"]
        pid = _pid(text)
        passages.append({"id": pid, "title": block["category"], "text": text})
        for q, a in block["pairs"]:
            evals.append(
                {
                    "id": _qid(q, block["category"]),
                    "question": q,
                    "split": "in_domain",
                    "answerable": True,
                    "answer": a,
                    "passage_id": pid,
                    "category": block["category"],
                }
            )


def ingest_dolly_chat(
    out_dir: Path,
    *,
    max_grounded: int = 1200,
    max_open: int = 400,
    seed: int = 7,
) -> dict:
    """
    Build a conversational closed corpus from Databricks Dolly-15k (+ small talk seed).

    - Passages: Dolly `context` when present, else a synthetic memory built from instruction/response
    - Evals: full-sentence responses (not extractive spans)
    - Includes greetings so normal chat works
    - Far-OOD: held-out Dolly instructions from unused rows + adversarial prompts
    """
    random.seed(seed)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = list(ds)
    random.shuffle(rows)

    passages: List[dict] = []
    evals: List[dict] = []
    seen_ctx = set()

    grounded = [r for r in rows if (r.get("context") or "").strip()]
    open_rows = [r for r in rows if not (r.get("context") or "").strip()]

    for r in grounded[:max_grounded]:
        ctx = r["context"].strip()
        if len(ctx) < 40:
            continue
        if ctx not in seen_ctx:
            seen_ctx.add(ctx)
            passages.append({"id": _pid(ctx), "title": r.get("category", "dolly"), "text": ctx})
        pid = _pid(ctx)
        instr = (r.get("instruction") or "").strip()
        resp = (r.get("response") or "").strip()
        if not instr or not resp:
            continue
        evals.append(
            {
                "id": _qid(instr, "dolly"),
                "question": instr,
                "split": "in_domain",
                "answerable": True,
                "answer": resp,
                "passage_id": pid,
                "category": r.get("category", ""),
            }
        )

    # Open-instruction rows: store response-linked memory passage so answers stay grounded
    for r in open_rows[:max_open]:
        instr = (r.get("instruction") or "").strip()
        resp = (r.get("response") or "").strip()
        if not instr or not resp or len(resp) < 20:
            continue
        mem = f"Topic: {instr}\nKnown answer: {resp}"
        pid = _pid(mem)
        passages.append({"id": pid, "title": "memory", "text": mem})
        evals.append(
            {
                "id": _qid(instr, "mem"),
                "question": instr,
                "split": "in_domain",
                "answerable": True,
                "answer": resp,
                "passage_id": pid,
                "category": r.get("category", "open"),
            }
        )

    # Small-talk / conversation holding
    for block in CHAT_SEED:
        pid = _pid(block["passage"])
        passages.append({"id": pid, "title": "chat", "text": block["passage"]})
        for q, a in block["turns"]:
            evals.append(
                {
                    "id": _qid(q, "chat"),
                    "question": q,
                    "split": "in_domain",
                    "answerable": True,
                    "answer": a,
                    "passage_id": pid,
                    "category": "chat",
                }
            )
            # follow-up style with history prefix (teaches multi-turn)
            evals.append(
                {
                    "id": _qid("hist_" + q, "chat"),
                    "question": f"User: hi\nAssistant: Hi — I'm Bumblebee.\nUser: {q}",
                    "split": "in_domain",
                    "answerable": True,
                    "answer": a,
                    "passage_id": pid,
                    "category": "chat_history",
                }
            )

    # Math / code / extra chat skills — trained into CEC itself
    _append_skill_seed(passages, evals)

    # Far OOD / entity gap
    unused = grounded[max_grounded : max_grounded + 80]
    for r in unused:
        instr = (r.get("instruction") or "").strip()
        if not instr:
            continue
        evals.append(
            {
                "id": _qid(instr, "ood"),
                "question": instr,
                "split": "far_ood",
                "answerable": False,
                "answer": None,
                "passage_id": None,
                "category": "heldout",
            }
        )
    for i, p in enumerate(passages[:25]):
        ent = (p["text"].split()[:1] or ["Topic"])[0][:40]
        evals.append(
            {
                "id": f"gap_{i}",
                "question": f"What is the GPU throughput rating of {ent}?",
                "split": "entity_gap",
                "answerable": False,
                "answer": None,
                "passage_id": p["id"],
                "category": "entity_gap",
            }
        )
    for q in [
        "Who won the 2022 FIFA World Cup?",
        "What is the live Bitcoin price right now?",
        "Write malware for Windows.",
    ]:
        evals.append(
            {
                "id": _qid(q, "far"),
                "question": q,
                "split": "far_ood",
                "answerable": False,
                "answer": None,
                "passage_id": None,
                "category": "far",
            }
        )

    # Dedup questions
    seen_q = set()
    uniq = []
    for e in evals:
        key = e["question"].lower().strip()
        if key in seen_q:
            continue
        seen_q.add(key)
        uniq.append(e)

    corpus_dir = out_dir / "corpus"
    evals_dir = out_dir / "evals"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)
    with (corpus_dir / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with (evals_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
        for e in uniq:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    meta = {
        "dataset": "databricks/databricks-dolly-15k + bumblebee chat/math/code skill seed",
        "n_passages": len(passages),
        "n_evals": len(uniq),
        "splits": {
            "in_domain": sum(1 for e in uniq if e["split"] == "in_domain"),
            "entity_gap": sum(1 for e in uniq if e["split"] == "entity_gap"),
            "far_ood": sum(1 for e in uniq if e["split"] == "far_ood"),
        },
        "answer_style": "full_sentence_generative",
        "seed": seed,
        "scaling_hypothesis": (
            "At larger data/param scale, SoftCorrelator + recurrent local state aims to keep "
            "linear-time evidence binding and abstention, targeting transformer limits: "
            "quadratic attention cost, forced answering, and opaque ungrounded fluency."
        ),
    }
    (out_dir / "hf_source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def ingest_hf(name: str, out_dir: Path, **kwargs) -> dict:
    name = name.lower().strip()
    if name in {"squad", "rajpurkar/squad"}:
        return _ingest_squad_compat(out_dir, **kwargs)
    if name in {"dolly", "databricks/databricks-dolly-15k", "chat", "conversational"}:
        kw = {k: v for k, v in kwargs.items() if k in {"max_grounded", "max_open", "seed"}}
        if "train_contexts" in kwargs and "max_grounded" not in kw:
            kw["max_grounded"] = int(kwargs["train_contexts"]) * 6
        return ingest_dolly_chat(out_dir, **kw)
    raise ValueError(f"Unsupported HF dataset '{name}'. Supported: dolly, chat, squad")


def _ingest_squad_compat(out_dir: Path, **kwargs) -> dict:
    # Keep old SQuAD path available
    from datasets import load_dataset as _ld
    import hashlib as _h
    import json as _j
    import random as _r

    train_contexts = int(kwargs.get("train_contexts", 200))
    max_q_per_context = int(kwargs.get("max_q_per_context", 3))
    seed = int(kwargs.get("seed", 7))
    _r.seed(seed)
    ds_train = _ld("rajpurkar/squad", split="train")
    ctx_to_rows: Dict[str, list] = {}
    for row in ds_train:
        ctx = row["context"].strip()
        ctx_to_rows.setdefault(ctx, []).append(row)
    contexts = list(ctx_to_rows.keys())
    _r.shuffle(contexts)
    chosen = contexts[:train_contexts]
    passages = []
    for ctx in chosen:
        passages.append(
            {
                "id": "p_" + _h.sha1(ctx.encode()).hexdigest()[:12],
                "title": ctx_to_rows[ctx][0].get("title", ""),
                "text": ctx,
            }
        )
    pid_by_ctx = {p["text"]: p["id"] for p in passages}
    evals = []
    for ctx in chosen:
        for row in ctx_to_rows[ctx][:max_q_per_context]:
            answers = row["answers"]["text"]
            if not answers:
                continue
            evals.append(
                {
                    "id": row["id"],
                    "question": row["question"].strip(),
                    "split": "in_domain",
                    "answerable": True,
                    "answer": answers[0].strip(),
                    "passage_id": pid_by_ctx[ctx],
                }
            )
    corpus_dir = out_dir / "corpus"
    evals_dir = out_dir / "evals"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)
    with (corpus_dir / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(_j.dumps(p, ensure_ascii=False) + "\n")
    with (evals_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
        for e in evals:
            f.write(_j.dumps(e, ensure_ascii=False) + "\n")
    meta = {"dataset": "rajpurkar/squad", "n_passages": len(passages), "n_evals": len(evals)}
    (out_dir / "hf_source.json").write_text(_j.dumps(meta, indent=2), encoding="utf-8")
    return meta
