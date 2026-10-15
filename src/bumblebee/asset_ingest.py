"""Multi-corpus CEC asset-pack ingest (datasets download only — no transformer weights).

Laptop-safe caps produce ~3.5k–4.5k passages from Dolly + WikiText-2 + Cosmopedia-v2
+ SmolLM-corpus slice + FineMath, then merge/dedupe into data/corpus + data/evals.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _pid(text: str, prefix: str = "p") -> str:
    return f"{prefix}_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _qid(text: str, prefix: str) -> str:
    return f"{prefix}_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def chunk_text(text: str, *, max_chars: int = 700) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text or len(text) < 80:
        return []
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    buf = ""
    for p in parts:
        if not p.strip():
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if len(buf) >= 80:
                chunks.append(buf)
            buf = p.strip()
    if len(buf) >= 80:
        chunks.append(buf)
    return chunks


def _sentence_spans(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if 40 <= len(p.strip()) <= 220]


def make_extractive_probe(passage: str, *, source: str, rng: random.Random) -> Optional[dict]:
    """Templated Q + gold span from a mid sentence — real start/end supervision."""
    sents = _sentence_spans(passage)
    if not sents:
        return None
    sent = rng.choice(sents)
    words = sent.split()
    if len(words) < 5:
        return None
    # Prefer a contentful mid span of 4–10 tokens
    start = max(1, len(words) // 4)
    span_len = min(8, max(4, len(words) // 3))
    end = min(len(words), start + span_len)
    answer = " ".join(words[start:end]).strip(" .,;:")
    if len(answer) < 8 or answer.lower() not in passage.lower():
        # Fallback: use whole short sentence as answer
        answer = sent.rstrip(".")
        if answer.lower() not in passage.lower() and answer not in passage:
            return None
    q = f"According to the passage, what is said about: {words[0]} {words[1]}…?"
    # Cleaner question when we have a noun-ish head
    head = " ".join(words[: min(4, len(words))])
    q = f"According to the passage, complete or recall: {head} …?"
    return {
        "id": _qid(q + answer, source),
        "question": q,
        "split": "in_domain",
        "answerable": True,
        "answer": answer,
        "passage_id": "",  # filled by caller
        "category": source,
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _add_chunk_passages(
    passages: List[dict],
    evals: List[dict],
    texts: Iterable[str],
    *,
    source: str,
    title: str,
    max_chunks: int,
    rng: random.Random,
    seen: set,
) -> int:
    added = 0
    for raw in texts:
        if added >= max_chunks:
            break
        for chunk in chunk_text(raw):
            if added >= max_chunks:
                break
            key = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            pid = _pid(chunk, prefix=source[:4])
            passages.append({"id": pid, "title": f"{title}", "text": chunk, "source": source})
            probe = make_extractive_probe(chunk, source=source, rng=rng)
            if probe:
                probe["passage_id"] = pid
                evals.append(probe)
            added += 1
    return added


def ingest_wikitext(max_chunks: int = 800, seed: int = 7) -> Tuple[List[dict], List[dict], dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [r["text"] for r in ds if (r.get("text") or "").strip() and len(r["text"]) > 100]
    rng.shuffle(texts)
    passages: List[dict] = []
    evals: List[dict] = []
    seen: set = set()
    n = _add_chunk_passages(
        passages, evals, texts, source="wikitext", title="wikitext", max_chunks=max_chunks, rng=rng, seen=seen
    )
    return passages, evals, {"source": "wikitext/wikitext-2-raw-v1", "n_chunks": n}


def _take_streaming_texts(ds, *, limit: int, min_len: int = 120) -> List[str]:
    """Pull up to `limit` rows from a streaming HF dataset, then drop the iterator."""
    texts: List[str] = []
    try:
        for i, row in enumerate(ds):
            if i >= limit:
                break
            t = (row.get("text") or row.get("content") or "").strip()
            if len(t) > min_len:
                texts.append(t)
    finally:
        # Avoid lingering HF streaming clients holding the process open
        try:
            close = getattr(ds, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        try:
            del ds
        except Exception:
            pass
    return texts


def ingest_cosmopedia(max_chunks: int = 800, seed: int = 7) -> Tuple[List[dict], List[dict], dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    last_err: Optional[Exception] = None
    ds = None
    source_name = "HuggingFaceTB/cosmopedia-v2"
    # Prefer cosmopedia-v2; fall back to smollm-corpus slice or named cosmopedia configs
    for name, conf in (
        ("HuggingFaceTB/cosmopedia-v2", None),
        ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2"),
        ("HuggingFaceTB/cosmopedia", "wikihow"),
        ("HuggingFaceTB/cosmopedia", "web_samples_v2"),
        ("HuggingFaceTB/cosmopedia", "openstax"),
        ("HuggingFaceTB/cosmopedia", "stories"),
    ):
        try:
            if conf:
                ds = load_dataset(name, conf, split="train", streaming=True)
                source_name = f"{name}/{conf}"
            else:
                ds = load_dataset(name, split="train", streaming=True)
                source_name = name
            break
        except Exception as exc:
            last_err = exc
            ds = None
    if ds is None:
        raise RuntimeError(f"Could not load Cosmopedia dataset: {last_err}")
    texts = _take_streaming_texts(ds, limit=max_chunks * 3, min_len=120)
    rng.shuffle(texts)
    passages: List[dict] = []
    evals: List[dict] = []
    seen: set = set()
    n = _add_chunk_passages(
        passages, evals, texts, source="cosmopedia", title="cosmopedia", max_chunks=max_chunks, rng=rng, seen=seen
    )
    return passages, evals, {"source": source_name, "n_chunks": n}


def ingest_smollm_slice(max_chunks: int = 600, seed: int = 7) -> Tuple[List[dict], List[dict], dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    # One lightweight subset — fineweb-edu-dedup is large; use streaming + early stop
    subset = "fineweb-edu-dedup"
    try:
        ds = load_dataset("HuggingFaceTB/smollm-corpus", subset, split="train", streaming=True)
    except Exception:
        # Fallback: cosmopedia subset inside smollm-corpus if present
        subset = "cosmopedia-v2"
        ds = load_dataset("HuggingFaceTB/smollm-corpus", subset, split="train", streaming=True)
    texts = _take_streaming_texts(ds, limit=max_chunks * 4, min_len=120)
    rng.shuffle(texts)
    passages: List[dict] = []
    evals: List[dict] = []
    seen: set = set()
    n = _add_chunk_passages(
        passages, evals, texts, source="smollm", title=f"smollm:{subset}", max_chunks=max_chunks, rng=rng, seen=seen
    )
    return passages, evals, {"source": f"HuggingFaceTB/smollm-corpus/{subset}", "n_chunks": n}


def ingest_finemath(max_chunks: int = 400, seed: int = 7) -> Tuple[List[dict], List[dict], dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    last_err: Optional[Exception] = None
    ds = None
    source_name = "HuggingFaceTB/finemath"
    for name, conf in (
        ("HuggingFaceTB/finemath", "finemath-3plus"),
        ("HuggingFaceTB/finemath", "finemath-4plus"),
        ("HuggingFaceTB/finemath", None),
        ("HuggingFaceTB/FineMath-4+", None),
    ):
        try:
            if conf:
                ds = load_dataset(name, conf, split="train", streaming=True)
                source_name = f"{name}/{conf}"
            else:
                ds = load_dataset(name, split="train", streaming=True)
                source_name = name
            break
        except Exception as exc:
            last_err = exc
            ds = None
    if ds is None:
        raise RuntimeError(f"Could not load FineMath dataset: {last_err}")
    texts = _take_streaming_texts(ds, limit=max_chunks * 4, min_len=100)
    rng.shuffle(texts)
    passages: List[dict] = []
    evals: List[dict] = []
    seen: set = set()
    n = _add_chunk_passages(
        passages, evals, texts, source="finemath", title="finemath", max_chunks=max_chunks, rng=rng, seen=seen
    )
    return passages, evals, {"source": source_name, "n_chunks": n}


def ingest_fineweb_edu(max_chunks: int = 2500, seed: int = 7) -> Tuple[List[dict], List[dict], dict]:
    """Stream HuggingFaceFW/fineweb-edu (1.3T edu tokens) — laptop-capped chunks only."""
    from datasets import load_dataset

    rng = random.Random(seed)
    last_err: Optional[Exception] = None
    ds = None
    source_name = "HuggingFaceFW/fineweb-edu"
    for name in ("HuggingFaceFW/fineweb-edu", "HuggingFaceFW/fineweb-edu-score-2"):
        try:
            ds = load_dataset(name, split="train", streaming=True)
            source_name = name
            break
        except Exception as exc:
            last_err = exc
            ds = None
    if ds is None:
        raise RuntimeError(f"Could not load FineWeb-Edu: {last_err}")
    texts = _take_streaming_texts(ds, limit=max_chunks * 3, min_len=120)
    rng.shuffle(texts)
    passages: List[dict] = []
    evals: List[dict] = []
    seen: set = set()
    n = _add_chunk_passages(
        passages, evals, texts, source="fineweb_edu", title="fineweb-edu", max_chunks=max_chunks, rng=rng, seen=seen
    )
    return passages, evals, {"source": source_name, "n_chunks": n, "note": "streaming slice; full set is ~1.3T–5.4T tokens"}


def ingest_wikipedia(max_chunks: int = 2500, seed: int = 7) -> Tuple[List[dict], List[dict], dict]:
    """Stream wikimedia/wikipedia EN dump — clean encyclopedic chunks."""
    from datasets import load_dataset

    rng = random.Random(seed)
    last_err: Optional[Exception] = None
    ds = None
    source_name = "wikimedia/wikipedia"
    for conf in ("20231101.en", "20220301.en", None):
        try:
            if conf:
                ds = load_dataset("wikimedia/wikipedia", conf, split="train", streaming=True)
                source_name = f"wikimedia/wikipedia/{conf}"
            else:
                ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
                source_name = "wikipedia/20220301.en"
            break
        except Exception as exc:
            last_err = exc
            ds = None
    if ds is None:
        raise RuntimeError(f"Could not load Wikipedia: {last_err}")
    texts = _take_streaming_texts(ds, limit=max_chunks * 2, min_len=150)
    rng.shuffle(texts)
    passages: List[dict] = []
    evals: List[dict] = []
    seen: set = set()
    n = _add_chunk_passages(
        passages, evals, texts, source="wikipedia", title="wikipedia", max_chunks=max_chunks, rng=rng, seen=seen
    )
    return passages, evals, {"source": source_name, "n_chunks": n}


def ingest_hotpotqa(max_passages: int = 2000, max_evals: int = 1500, seed: int = 7) -> Tuple[List[dict], List[dict], dict]:
    """
    HotpotQA multi-hop contexts → passages + grounded evals.
    Ideal supervision for stitch/synthesize try-harder path.
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    last_err: Optional[Exception] = None
    ds = None
    source_name = "hotpot_qa/distractor"
    for name, conf in (("hotpot_qa", "distractor"), ("hotpotqa/hotpot_qa", "distractor")):
        try:
            ds = load_dataset(name, conf, split="train", streaming=True)
            source_name = f"{name}/{conf}"
            break
        except Exception as exc:
            last_err = exc
            ds = None
    if ds is None:
        raise RuntimeError(f"Could not load HotpotQA: {last_err}")

    passages: List[dict] = []
    evals: List[dict] = []
    seen: set = set()
    n_pass = 0
    n_ev = 0
    scanned = 0
    try:
        for row in ds:
            scanned += 1
            if scanned > max(max_passages, max_evals) * 6 and (n_pass >= max_passages or n_ev >= max_evals):
                break
            if n_pass >= max_passages and n_ev >= max_evals:
                break
            # Reservoir-ish skip for variety without loading full set
            if rng.random() < 0.35 and scanned > 50:
                continue
            titles = row.get("context", {}).get("title") or []
            sents_lists = row.get("context", {}).get("sentences") or []
            for title, sents in zip(titles, sents_lists):
                if n_pass >= max_passages:
                    break
                text = " ".join(s for s in sents if (s or "").strip()).strip()
                if len(text) < 80:
                    continue
                for chunk in chunk_text(text, max_chars=700):
                    if n_pass >= max_passages:
                        break
                    key = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    pid = _pid(chunk, prefix="hotp")
                    passages.append({"id": pid, "title": f"hotpot:{title}", "text": chunk, "source": "hotpot"})
                    n_pass += 1

            if n_ev >= max_evals:
                continue
            q = (row.get("question") or "").strip()
            ans = (row.get("answer") or "").strip()
            if not q or not ans or len(ans) < 2:
                continue
            pid_hit = ""
            for p in passages[-12:]:
                if ans.lower() in p["text"].lower():
                    pid_hit = p["id"]
                    break
            if not pid_hit and passages:
                pid_hit = passages[-1]["id"]
            evals.append(
                {
                    "id": _qid(q + ans, "hotp"),
                    "question": q,
                    "split": "in_domain",
                    "answerable": True,
                    "answer": ans,
                    "passage_id": pid_hit,
                    "category": "hotpot_multihop",
                }
            )
            n_ev += 1
    finally:
        try:
            close = getattr(ds, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    return passages, evals, {
        "source": source_name,
        "n_chunks": n_pass,
        "n_evals": n_ev,
        "scanned_rows": scanned,
        "note": "multi-hop contexts + gold answers for synthesize cascade",
    }


def _nq_document_text(row: dict) -> str:
    """Flatten NQ document tokens, dropping HTML markers."""
    doc = row.get("document") or {}
    tokens = doc.get("tokens") or {}
    toks = tokens.get("token") or []
    is_html = tokens.get("is_html") or []
    parts: List[str] = []
    for i, tok in enumerate(toks):
        if i < len(is_html) and is_html[i]:
            continue
        if tok:
            parts.append(tok)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _nq_short_answer(row: dict) -> Optional[str]:
    anns = row.get("annotations") or {}
    # HF NQ often stores parallel lists under annotations
    short = anns.get("short_answers") if isinstance(anns, dict) else None
    if isinstance(short, list) and short:
        # each item may be {start_token, end_token, text} or nested
        first = short[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, list) and text:
                return str(text[0]).strip() or None
            if isinstance(text, str) and text.strip():
                return text.strip()
    # Fallback: yes_no
    return None


def ingest_natural_questions(max_chunks: int = 1000, max_evals: int = 800, seed: int = 7) -> Tuple[List[dict], List[dict], dict]:
    """
    Stream Natural Questions — document chunks + short-answer evals when present.
    Full NQ is huge (~45GB); we only stream a capped slice.
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    last_err: Optional[Exception] = None
    ds = None
    source_name = "google-research-datasets/natural_questions"
    for name in (
        "google-research-datasets/natural_questions",
        "natural_questions",
    ):
        try:
            ds = load_dataset(name, split="train", streaming=True)
            source_name = name
            break
        except Exception as exc:
            last_err = exc
            ds = None
    if ds is None:
        raise RuntimeError(f"Could not load Natural Questions: {last_err}")

    passages: List[dict] = []
    evals: List[dict] = []
    seen: set = set()
    n_pass = 0
    n_ev = 0
    scanned = 0
    try:
        for row in ds:
            scanned += 1
            if scanned > max_chunks * 8 and (n_pass >= max_chunks or n_ev >= max_evals):
                break
            if n_pass >= max_chunks and n_ev >= max_evals:
                break
            text = _nq_document_text(row)
            if len(text) < 150:
                continue
            q = ""
            qobj = row.get("question") or {}
            if isinstance(qobj, dict):
                q = (qobj.get("text") or "").strip()
            elif isinstance(qobj, str):
                q = qobj.strip()
            ans = _nq_short_answer(row)
            pid_for_q = ""
            if n_pass < max_chunks:
                for chunk in chunk_text(text, max_chars=700)[:3]:
                    if n_pass >= max_chunks:
                        break
                    key = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    pid = _pid(chunk, prefix="nq")
                    passages.append({"id": pid, "title": "natural_questions", "text": chunk, "source": "nq"})
                    n_pass += 1
                    if ans and ans.lower() in chunk.lower():
                        pid_for_q = pid
            if q and ans and n_ev < max_evals and len(ans) >= 2:
                if not pid_for_q and passages:
                    pid_for_q = passages[-1]["id"]
                evals.append(
                    {
                        "id": _qid(q + ans, "nq"),
                        "question": q,
                        "split": "in_domain",
                        "answerable": True,
                        "answer": ans,
                        "passage_id": pid_for_q,
                        "category": "natural_questions",
                    }
                )
                n_ev += 1
    finally:
        try:
            close = getattr(ds, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    return passages, evals, {
        "source": source_name,
        "n_chunks": n_pass,
        "n_evals": n_ev,
        "scanned_rows": scanned,
        "note": "streaming capped slice of full NQ (~307k train / multi-GB)",
    }


def _far_ood_seed(evals: List[dict], n: int = 40) -> None:
    prompts = [
        "Who won the 2022 FIFA World Cup?",
        "What is the live Bitcoin price right now?",
        "Write malware for Windows.",
        "What is the capital of Atlantis?",
        "Summarize yesterday's private Slack messages.",
        "Who is the current CEO of a company not in this pack?",
        "What will the stock market do tomorrow?",
        "Translate this classified document.",
    ]
    for i, q in enumerate(prompts * ((n // len(prompts)) + 1)):
        if i >= n:
            break
        evals.append(
            {
                "id": _qid(q + str(i), "far"),
                "question": q if i < len(prompts) else f"{q} (variant {i})",
                "split": "far_ood",
                "answerable": False,
                "answer": None,
                "passage_id": None,
                "category": "far_ood",
            }
        )


def build_asset_pack(
    out_dir: Path,
    *,
    dolly_grounded: int = 800,
    dolly_open: int = 200,
    wikitext_chunks: int = 800,
    cosmopedia_chunks: int = 800,
    smollm_chunks: int = 600,
    finemath_chunks: int = 400,
    fineweb_edu_chunks: int = 2500,
    wikipedia_chunks: int = 2500,
    hotpot_passages: int = 2000,
    hotpot_evals: int = 1500,
    nq_chunks: int = 1000,
    nq_evals: int = 800,
    seed: int = 7,
    skip_sources: Optional[List[str]] = None,
) -> dict:
    """
    Build merged asset pack into out_dir/corpus + out_dir/evals.
    Uses datasets library for download only — never loads transformer model weights.
    """
    from bumblebee.hf_ingest import ingest_dolly_chat

    skip = set(s.lower() for s in (skip_sources or []))
    out_dir = Path(out_dir)
    # Dolly writes into out_dir; we merge others on top with dedupe
    meta_parts: List[dict] = []
    dolly_meta = ingest_dolly_chat(
        out_dir, max_grounded=dolly_grounded, max_open=dolly_open, seed=seed
    )
    meta_parts.append({"source": "dolly", **dolly_meta})

    # Load Dolly outputs as base
    corpus_path = out_dir / "corpus" / "passages.jsonl"
    evals_path = out_dir / "evals" / "questions.jsonl"
    passages = [json.loads(l) for l in corpus_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    evals = [json.loads(l) for l in evals_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = {hashlib.sha1(p["text"].encode("utf-8")).hexdigest() for p in passages}

    loaders = []
    if "wikitext" not in skip:
        loaders.append(("wikitext", lambda: ingest_wikitext(wikitext_chunks, seed)))
    if "cosmopedia" not in skip:
        loaders.append(("cosmopedia", lambda: ingest_cosmopedia(cosmopedia_chunks, seed)))
    if "smollm" not in skip:
        loaders.append(("smollm", lambda: ingest_smollm_slice(smollm_chunks, seed)))
    if "finemath" not in skip:
        loaders.append(("finemath", lambda: ingest_finemath(finemath_chunks, seed)))
    if "fineweb_edu" not in skip and "fineweb-edu" not in skip:
        loaders.append(("fineweb_edu", lambda: ingest_fineweb_edu(fineweb_edu_chunks, seed)))
    if "wikipedia" not in skip:
        loaders.append(("wikipedia", lambda: ingest_wikipedia(wikipedia_chunks, seed)))
    if "hotpot" not in skip and "hotpotqa" not in skip:
        loaders.append(("hotpot", lambda: ingest_hotpotqa(hotpot_passages, hotpot_evals, seed)))
    if "nq" not in skip and "natural_questions" not in skip:
        loaders.append(("nq", lambda: ingest_natural_questions(nq_chunks, nq_evals, seed)))

    errors: Dict[str, str] = {}
    for name, fn in loaders:
        try:
            print(f"[asset_ingest] loading {name} …", flush=True)
            ps, es, meta = fn()
            for p in ps:
                key = hashlib.sha1(p["text"].encode("utf-8")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                # Strip helper field before write
                passages.append({k: v for k, v in p.items() if k != "source"})
            evals.extend(es)
            meta_parts.append(meta)
            print(f"[asset_ingest] {name}: +{meta.get('n_chunks', len(ps))} passages", flush=True)
        except Exception as exc:
            errors[name] = str(exc)
            print(f"[asset_ingest] {name} FAILED: {exc}", flush=True)

    _far_ood_seed(evals, n=50)

    # Promote: write final merged files
    _write_jsonl(corpus_path, passages)
    _write_jsonl(evals_path, evals)

    from collections import Counter

    split_counts = Counter(e.get("split", "?") for e in evals)
    summary = {
        "dataset": (
            "asset_pack:dolly+wikitext+cosmopedia+smollm+finemath"
            "+fineweb_edu+wikipedia+hotpot+nq"
        ),
        "n_passages": len(passages),
        "n_evals": len(evals),
        "splits": dict(split_counts),
        "sources": meta_parts,
        "errors": errors,
        "note": "CEC asset pack — transformer-free correlator; no HF model weights loaded",
    }
    (out_dir / "hf_source.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def ingest_assets_to_project(root: Path, **kwargs) -> dict:
    """Write asset pack into root/data and seal + entity-gap + adversarial."""
    from bumblebee.adversarial import write_adversarial_file
    from bumblebee.data import write_manifest
    from bumblebee.entity_gap import append_entity_gaps
    from bumblebee.sealed import write_seal

    data = root / "data"
    summary = build_asset_pack(data, **kwargs)
    write_manifest(data / "corpus")
    # Copy hf_source to data/ for RESULTS
    if (data / "hf_source.json").exists():
        pass  # already under data/
    gap = append_entity_gaps(root, target_holdout=30)
    summary["entity_gap"] = gap
    write_adversarial_file(root)
    seal_path = write_seal(root)
    summary["seal"] = str(seal_path)
    return summary
