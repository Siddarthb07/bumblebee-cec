from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bumblebee.data import EvalExample, Passage, load_evals, load_passages, passage_index
from bumblebee.eval_protocol import calibration_split, refuse_confusion, train_eval_split
from bumblebee.gate import decide, faithfulness_ok, looks_like_garbage, token_f1
from bumblebee.metrics import aurc, expected_calibration_error
from bumblebee.model import CECModel
from bumblebee.scale import choose_mac_fit_config, nearest_config_report
from bumblebee.tokenize import Vocab, tokenize


_RUNTIME_CACHE: dict = {}
_BM25_CACHE: dict = {}


def _pick_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _unpack_runtime(loaded):
    """Normalize load_runtime tuples across checkpoint generations."""
    model, vocab, passages, threshold, device, max_q, max_p = loaded[:7]
    temperature = 1.0
    platt_a, platt_b, use_platt = 1.0, 0.0, False
    if len(loaded) >= 8:
        temperature = float(loaded[7])
    if len(loaded) >= 11:
        platt_a, platt_b, use_platt = float(loaded[8]), float(loaded[9]), bool(loaded[10])
    return model, vocab, passages, threshold, device, max_q, max_p, temperature, platt_a, platt_b, use_platt


def _apply_cal(conf: float, temperature: float, platt_a: float, platt_b: float, use_platt: bool) -> float:
    from bumblebee.calibration import apply_platt, apply_temperature

    if use_platt:
        return apply_platt(conf, platt_a, platt_b)
    return apply_temperature(conf, temperature)


def clear_runtime_cache() -> None:
    _RUNTIME_CACHE.clear()
    _BM25_CACHE.clear()


def get_bm25_index(root: Path, passages):
    """Build BM25 once per process/root — required for ~4k-passage asset packs."""
    from bumblebee.baselines import BM25Index

    key = str(Path(root).resolve())
    cached = _BM25_CACHE.get(key)
    if cached is not None:
        return cached
    index = BM25Index([(p.id, p.text) for p in passages])
    _BM25_CACHE[key] = index
    return index


_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were",
    "be", "been", "being", "what", "which", "who", "whom", "how", "why", "when", "where",
    "do", "does", "did", "can", "could", "should", "would", "with", "as", "by", "from",
    "that", "this", "these", "those", "it", "its", "at", "into", "about",
    "make", "up", "get", "take", "give", "use", "used", "using",
}


def content_support(question: str, passage_tokens) -> float:
    q = {t for t in tokenize(question) if t not in _STOP}
    if not q:
        return 0.0
    p = {t for t in passage_tokens if t not in _STOP}
    return len(q & p) / max(len(q), 1)


def novel_content_ratio(question: str, corpus_tokens) -> float:
    """Fraction of question content words absent from the whole corpus."""
    q = {t for t in tokenize(question) if t not in _STOP}
    if not q:
        return 0.0
    missing = [t for t in q if t not in corpus_tokens]
    return len(missing) / max(len(q), 1)


def find_answer_start(passage_tokens: Sequence[str], answer: str) -> int:
    """Locate the best start index of the full answer token span in the passage."""
    a_toks = tokenize(answer)
    if not a_toks or not passage_tokens:
        return 0
    n, m = len(passage_tokens), len(a_toks)
    for i in range(n - m + 1):
        if passage_tokens[i : i + m] == a_toks:
            return i
    # Fallback: first contentful answer token (skip tiny stop-ish tokens)
    for tok in a_toks:
        if tok in _STOP or len(tok) <= 2:
            continue
        for i, ptok in enumerate(passage_tokens):
            if ptok == tok:
                return i
    for i, ptok in enumerate(passage_tokens):
        if ptok == a_toks[0]:
            return i
    return 0


def find_answer_end(passage_tokens: Sequence[str], answer: str, start: int) -> int:
    """End index (inclusive) for the gold span given a start; clamps to passage."""
    a_toks = tokenize(answer)
    last = max(0, len(passage_tokens) - 1)
    start = max(0, min(start, last))
    if not a_toks:
        return start
    end = start + len(a_toks) - 1
    # Exact match at start
    if start + len(a_toks) <= len(passage_tokens) and passage_tokens[start : start + len(a_toks)] == a_toks:
        return start + len(a_toks) - 1
    return max(start, min(end, last))


class QADataset(Dataset):
    def __init__(
        self,
        examples: List[EvalExample],
        passages: Dict[str, Passage],
        vocab: Vocab,
        max_q: int = 64,
        max_p: int = 192,
        max_a: int = 64,
    ):
        self.examples = examples
        self.passages = passages
        self.vocab = vocab
        self.max_q = max_q
        self.max_p = max_p
        self.max_a = max_a
        self.fallback = " ".join(p.text for p in list(passages.values())[:3])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        if ex.passage_id and ex.passage_id in self.passages:
            passage = self.passages[ex.passage_id].text
        else:
            passage = self.fallback
        q_ids = self.vocab.encode(ex.question, self.max_q)
        p_ids = self.vocab.encode(passage, self.max_p)
        a_ids = self.vocab.encode(ex.answer or "", self.max_a, add_eos=True) if ex.answerable else [self.vocab.eos_id]
        p_toks = tokenize(passage)[: self.max_p]
        start = find_answer_start(p_toks, ex.answer or "") if ex.answer else 0
        end = find_answer_end(p_toks, ex.answer or "", start) if ex.answer else 0
        return {
            "id": ex.id,
            "question_ids": q_ids,
            "passage_ids": p_ids,
            "answer_ids": a_ids,
            "passage_text": passage,
            "answerable": float(ex.answerable),
            "start": start,
            "end": end,
            "answer": ex.answer or "",
            "split": ex.split,
        }


def collate(batch, pad_id: int = 0):
    def pad(seqs):
        m = max(len(s) for s in seqs)
        out = torch.full((len(seqs), m), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    return {
        "id": [b["id"] for b in batch],
        "question_ids": pad([b["question_ids"] for b in batch]),
        "passage_ids": pad([b["passage_ids"] for b in batch]),
        "answer_ids": pad([b["answer_ids"] for b in batch]),
        "passage_text": [b["passage_text"] for b in batch],
        "answerable": torch.tensor([b["answerable"] for b in batch], dtype=torch.float),
        "start": torch.tensor([b["start"] for b in batch], dtype=torch.long),
        "end": torch.tensor([b["end"] for b in batch], dtype=torch.long),
        "answer": [b["answer"] for b in batch],
        "split": [b["split"] for b in batch],
    }


def train_model(
    root: Path,
    epochs: int = 8,
    dim: int = 128,
    lr: float = 1e-3,
    threshold: float = 0.35,
    max_p: int = 192,
    max_q: int = 64,
    device: Optional[str] = None,
    mass_fn: str = "sparsemax",
) -> Path:
    # Avoid OpenMP hyper-barrier hangs on small Macs under load
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    corpus_dir = root / "data" / "corpus"
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    passages = load_passages(corpus_dir)
    pmap = passage_index(passages)

    texts = [p.text for p in passages] + [e.question for e in evals] + [e.answer or "" for e in evals]
    vocab = Vocab.build(texts)
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    vocab.save(art / "vocab.json")

    # Train on residues excluding test(0) and cal(1) so calibration does not leak
    train_evals, _cal, _test = calibration_split(evals)
    ds = QADataset(train_evals, pmap, vocab, max_q=max_q, max_p=max_p, max_a=64)
    batch_size = 4
    if dim >= 384:
        batch_size = 1
    elif dim >= 256:
        batch_size = 2
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate(b, vocab.pad_id))
    print(
        f"train examples={len(train_evals)} / total evals={len(evals)} "
        f"(cal residue held out) batch_size={batch_size} dim={dim}",
        flush=True,
    )

    model = CECModel(
        len(vocab.itos), dim=dim, pad_id=vocab.pad_id, mass_fn=mass_fn
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    use_gen = len(vocab.itos) <= 12_000
    if not use_gen:
        print("skipping generative polish (vocab>12k; extractive-only train)", flush=True)

    from bumblebee.telemetry import (
        bind_root,
        grad_global_norm,
        snapshot_from_forward,
        write_forward_snapshot,
        write_train_snapshot,
    )

    bind_root(root)
    global_step = 0
    n_params = sum(p.numel() for p in model.parameters())
    # Fresh loss history for this run
    hist = root / "artifacts" / "live" / "history.jsonl"
    try:
        hist.write_text("", encoding="utf-8")
    except Exception:
        pass
    write_train_snapshot(
        {
            "status": "running",
            "arch": "dense_cec",
            "epoch": 0,
            "epochs": epochs,
            "step": 0,
            "steps_per_epoch": len(loader),
            "loss": None,
            "dim": dim,
            "n_params": n_params,
            "batch_size": batch_size,
            "vocab_size": len(vocab.itos),
            "note": "train starting",
        },
        root=root,
    )

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            q = batch["question_ids"].to(device)
            p = batch["passage_ids"].to(device)
            a = batch["answer_ids"].to(device)
            out = model(q, p, answer_ids=a if use_gen else None)
            gate_loss = F.binary_cross_entropy(out["confidence"], batch["answerable"].to(device))
            start_logits = out["start_logits"]
            end_logits = out["end_logits"]
            ans_mask = batch["answerable"].to(device)
            starts = batch["start"].to(device).clamp(max=start_logits.size(1) - 1)
            ends = batch["end"].to(device).clamp(max=end_logits.size(1) - 1)
            ends = torch.maximum(ends, starts)
            start_loss = (F.cross_entropy(start_logits, starts, reduction="none") * ans_mask).mean()
            end_loss = (F.cross_entropy(end_logits, ends, reduction="none") * ans_mask).mean()
            # Generative polish demoted — extractive span is the eval path.
            if use_gen:
                logits = out["answer_logits"]
                gen_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    a.reshape(-1),
                    ignore_index=vocab.pad_id,
                    reduction="none",
                ).view_as(a)
                gen_loss = (gen_loss.mean(dim=1) * ans_mask).mean()
                loss = gate_loss + start_loss + end_loss + 0.25 * gen_loss
            else:
                loss = gate_loss + start_loss + end_loss
                gen_loss = None
            opt.zero_grad()
            loss.backward()
            gnorm = grad_global_norm(model)
            opt.step()
            total += float(loss.item())
            global_step += 1
            # Live telemetry every step early, then every 5
            if global_step <= 20 or global_step % 5 == 0:
                losses = {
                    "total": float(loss.item()),
                    "gate": float(gate_loss.item()),
                    "start": float(start_loss.item()),
                    "end": float(end_loss.item()),
                }
                if gen_loss is not None:
                    losses["gen"] = float(gen_loss.item())
                try:
                    import psutil

                    rss = psutil.Process().memory_info().rss / (1024**3)
                except Exception:
                    rss = None
                sample_q = ""
                sample_a = ""
                sample_toks: list = []
                try:
                    sample_q = vocab.decode(batch["question_ids"][0].tolist())[:160]
                    sample_a = vocab.decode(batch["answer_ids"][0].tolist())[:120]
                    pids = [int(x) for x in batch["passage_ids"][0].tolist() if int(x) != vocab.pad_id]
                    sample_toks = [
                        vocab.itos[i] if 0 <= i < len(vocab.itos) else "?" for i in pids[:256]
                    ]
                except Exception:
                    pass
                write_train_snapshot(
                    {
                        "status": "running",
                        "arch": "dense_cec",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "step": global_step,
                        "steps_per_epoch": len(loader),
                        "loss": float(loss.item()),
                        "gate_loss": float(gate_loss.item()),
                        "span_loss": float(start_loss.item() + end_loss.item()),
                        "grad_norm": gnorm,
                        "lr": lr,
                        "dim": dim,
                        "n_params": n_params,
                        "batch_size": batch_size,
                        "rss_gb": rss,
                        "vocab_size": len(vocab.itos),
                        "sample_question": sample_q,
                        "sample_answer": sample_a,
                        "sample_answerable": float(batch["answerable"][0].item()),
                    },
                    root=root,
                )
                snap = snapshot_from_forward(
                    out,
                    phase="train",
                    question=sample_q,
                    epoch=epoch + 1,
                    step=global_step,
                    losses=losses,
                )
                if sample_toks:
                    snap["mass_tokens"] = sample_toks[: len(snap.get("mass") or sample_toks)]
                write_forward_snapshot(snap, root=root)
        print(
            f"epoch {epoch+1}/{epochs} loss={total/max(len(loader),1):.4f}",
            flush=True,
        )

    write_train_snapshot(
        {
            "status": "finished",
            "arch": "dense_cec",
            "epoch": epochs,
            "epochs": epochs,
            "step": global_step,
            "loss": float(total / max(len(loader), 1)),
            "dim": dim,
            "n_params": n_params,
            "vocab_size": len(vocab.itos),
        },
        root=root,
    )

    n_params = sum(p.numel() for p in model.parameters())
    ckpt = {
        "model": model.state_dict(),
        "dim": dim,
        "threshold": threshold,
        "vocab_size": len(vocab.itos),
        "pad_id": vocab.pad_id,
        "max_p": max_p,
        "max_q": max_q,
        "n_params": n_params,
        "mass_fn": mass_fn,
    }
    path = art / "cec.pt"
    torch.save(ckpt, path)
    (art / "config.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "dim": dim,
                "max_p": max_p,
                "max_q": max_q,
                "n_params": n_params,
                "vocab_size": len(vocab.itos),
                "mass_fn": mass_fn,
                "arch": "dense_cec",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"CEC params={n_params} ({n_params/1e6:.2f}M) mass_fn={mass_fn}")
    return path


def train_model_mac_fit(
    root: Path,
    *,
    target_params: int = 500_000_000,
    epochs: int = 1,
    lr: float = 5e-4,
    threshold: float = 0.35,
    device: Optional[str] = None,
    ram_gb: Optional[float] = None,
) -> Path:
    """
    Train an embedding-heavy ~500M CEC designed for small Macs.

    Nearly all parameters live in a sparse hash embedding used only for
    encoding (SparseAdam updates touched rows). Answer generation uses a
    small closed vocab so softmax stays cheap.
    """
    import gc
    import random

    import psutil

    from bumblebee.large_model import HashEncoderVocab, LargeCECModel

    device = _pick_device(device)
    ram = ram_gb or (psutil.virtual_memory().total / (1024**3))
    cfg = choose_mac_fit_config(target_params, ram_gb=ram)
    print(nearest_config_report(cfg))
    print(f"device={device} free_ram_gb={psutil.virtual_memory().available/(1024**3):.2f}")

    corpus_dir = root / "data" / "corpus"
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    passages = load_passages(corpus_dir)
    pmap = passage_index(passages)
    train_evals, _cal, _test = calibration_split(evals)

    enc_vocab = HashEncoderVocab(cfg.vocab_buckets)
    ans_texts = [e.answer or "" for e in evals] + [e.question for e in evals]
    ans_vocab = Vocab.build(ans_texts)

    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    enc_vocab.save(art / "vocab.json")
    ans_vocab.save(art / "answer_vocab.json")

    fallback = " ".join(p.text for p in passages[:3])
    examples = []
    for ex in train_evals:
        passage = pmap[ex.passage_id].text if ex.passage_id and ex.passage_id in pmap else fallback
        p_toks = tokenize(passage)[: cfg.max_p]
        start = find_answer_start(p_toks, ex.answer or "")
        end = find_answer_end(p_toks, ex.answer or "", start)
        examples.append(
            {
                "q": enc_vocab.encode(ex.question, cfg.max_q),
                "p": enc_vocab.encode(passage, cfg.max_p),
                "a": ans_vocab.encode(ex.answer or "", cfg.max_a, add_eos=True)
                if ex.answerable
                else [ans_vocab.eos_id],
                "answerable": float(ex.answerable),
                "start": start,
                "end": end,
                "q_text": ex.question,
                "a_text": (ex.answer or "")[:120],
            }
        )

    def make_batch(rows):
        def pad(seqs, pad_id):
            m = max(len(s) for s in seqs)
            out = torch.full((len(seqs), m), pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            return out

        return {
            "question_ids": pad([r["q"] for r in rows], enc_vocab.pad_id),
            "passage_ids": pad([r["p"] for r in rows], enc_vocab.pad_id),
            "answer_ids": pad([r["a"] for r in rows], ans_vocab.pad_id),
            "answerable": torch.tensor([r["answerable"] for r in rows], dtype=torch.float),
            "start": torch.tensor([r["start"] for r in rows], dtype=torch.long),
            "end": torch.tensor([r["end"] for r in rows], dtype=torch.long),
        }

    print(f"train examples={len(examples)} / total evals={len(evals)} (cal residue held out)")
    gc.collect()
    model = LargeCECModel(
        n_buckets=cfg.vocab_buckets,
        emb_dim=cfg.emb_dim,
        answer_vocab_size=len(ans_vocab.itos),
        dim=cfg.dim,
        pad_id=enc_vocab.pad_id,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Large CEC params={n_params} ({n_params/1e6:.2f}M)")

    emb_params = list(model.emb.parameters())
    dense_params = [p for n, p in model.named_parameters() if not n.startswith("emb.")]
    opt_sparse = torch.optim.SparseAdam(emb_params, lr=lr)
    opt_dense = torch.optim.Adam(dense_params, lr=lr)

    from bumblebee.telemetry import (
        bind_root,
        grad_global_norm,
        snapshot_from_forward,
        write_forward_snapshot,
        write_train_snapshot,
    )

    bind_root(root)
    steps_per_epoch = max(1, (len(examples) + cfg.batch_size - 1) // cfg.batch_size)
    global_step = 0
    for epoch in range(epochs):
        model.train()
        random.shuffle(examples)
        total = 0.0
        opt_sparse.zero_grad(set_to_none=True)
        opt_dense.zero_grad(set_to_none=True)
        for step in range(0, len(examples), cfg.batch_size):
            batch_rows = examples[step : step + cfg.batch_size]
            batch = make_batch(batch_rows)
            q = batch["question_ids"].to(device)
            p = batch["passage_ids"].to(device)
            a = batch["answer_ids"].to(device)
            out = model(q, p, answer_ids=a)
            gate_loss = F.binary_cross_entropy(out["confidence"], batch["answerable"].to(device))
            start_logits = out["start_logits"]
            end_logits = out["end_logits"]
            ans_mask = batch["answerable"].to(device)
            starts = batch["start"].to(device).clamp(max=start_logits.size(1) - 1)
            ends = batch["end"].to(device).clamp(max=end_logits.size(1) - 1)
            ends = torch.maximum(ends, starts)
            start_loss = (F.cross_entropy(start_logits, starts, reduction="none") * ans_mask).mean()
            end_loss = (F.cross_entropy(end_logits, ends, reduction="none") * ans_mask).mean()
            logits = out["answer_logits"]
            gen_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                a.reshape(-1),
                ignore_index=ans_vocab.pad_id,
                reduction="none",
            ).view_as(a)
            gen_loss = (gen_loss.mean(dim=1) * ans_mask).mean()
            loss = (gate_loss + start_loss + end_loss + 0.25 * gen_loss) / cfg.grad_accum
            loss.backward()
            total += float(loss.item()) * cfg.grad_accum
            global_step += 1
            if global_step % cfg.grad_accum == 0:
                gnorm = grad_global_norm(model)
                opt_sparse.step()
                opt_dense.step()
                opt_sparse.zero_grad(set_to_none=True)
                opt_dense.zero_grad(set_to_none=True)
            else:
                gnorm = grad_global_norm(model)
            batch_idx = step // cfg.batch_size
            if batch_idx <= 10 or batch_idx % 5 == 0:
                loss_v = float(loss.item()) * cfg.grad_accum
                sample_q = ""
                try:
                    # recover a short preview from answer vocab decode of q is unavailable;
                    # use train_evals text aligned by shuffle — store q preview on rows
                    sample_q = (batch_rows[0].get("q_text") or "")[:160]
                except Exception:
                    sample_q = ""
                write_train_snapshot(
                    {
                        "status": "running",
                        "arch": "large_hash_cec",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "step": global_step,
                        "steps_per_epoch": steps_per_epoch,
                        "batch_idx": batch_idx,
                        "loss": loss_v,
                        "gate_loss": float(gate_loss.item()),
                        "span_loss": float(start_loss.item() + end_loss.item()),
                        "gen_loss": float(gen_loss.item()),
                        "grad_norm": gnorm,
                        "lr": lr,
                        "dim": cfg.dim,
                        "n_params": n_params,
                        "batch_size": cfg.batch_size,
                        "rss_gb": psutil.Process().memory_info().rss / (1024**3),
                        "vocab_size": cfg.vocab_buckets,
                        "sample_question": sample_q,
                        "sample_answer": (batch_rows[0].get("a_text") or "")[:120] if batch_rows else "",
                        "sample_answerable": float(batch_rows[0]["answerable"]) if batch_rows else None,
                    },
                    root=root,
                )
                write_forward_snapshot(
                    snapshot_from_forward(
                        out,
                        phase="train",
                        question=sample_q,
                        epoch=epoch + 1,
                        step=global_step,
                        losses={
                            "total": loss_v,
                            "gate": float(gate_loss.item()),
                            "start": float(start_loss.item()),
                            "end": float(end_loss.item()),
                            "gen": float(gen_loss.item()),
                        },
                    ),
                    root=root,
                )
            if batch_idx % 25 == 0:
                print(
                    f"epoch {epoch+1}/{epochs} step {batch_idx}/{steps_per_epoch} "
                    f"loss={float(loss.item())*cfg.grad_accum:.4f} "
                    f"rss_gb={psutil.Process().memory_info().rss/(1024**3):.2f}",
                    flush=True,
                )
        if global_step % cfg.grad_accum != 0:
            opt_sparse.step()
            opt_dense.step()
        print(f"epoch {epoch+1}/{epochs} loss={total/max(steps_per_epoch,1):.4f}", flush=True)

    write_train_snapshot(
        {
            "status": "finished",
            "arch": "large_hash_cec",
            "epoch": epochs,
            "epochs": epochs,
            "step": global_step,
            "loss": float(total / max(steps_per_epoch, 1)),
            "dim": cfg.dim,
            "n_params": n_params,
            "vocab_size": cfg.vocab_buckets,
        },
        root=root,
    )

    ckpt = {
        "arch": "large_hash_cec",
        "dim": cfg.dim,
        "emb_dim": cfg.emb_dim,
        "n_buckets": cfg.vocab_buckets,
        "answer_vocab_size": len(ans_vocab.itos),
        "threshold": threshold,
        "vocab_size": cfg.vocab_buckets,
        "pad_id": enc_vocab.pad_id,
        "max_p": cfg.max_p,
        "max_q": cfg.max_q,
        "n_params": n_params,
        "scale": cfg.to_dict(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    path = art / "cec.pt"
    torch.save(ckpt, path)
    (art / "config.json").write_text(
        json.dumps(
            {
                "arch": "large_hash_cec",
                "threshold": threshold,
                "dim": cfg.dim,
                "emb_dim": cfg.emb_dim,
                "n_buckets": cfg.vocab_buckets,
                "answer_vocab_size": len(ans_vocab.itos),
                "max_p": cfg.max_p,
                "max_q": cfg.max_q,
                "n_params": n_params,
                "target_params": target_params,
                "scale": cfg.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved Large CEC params={n_params} ({n_params/1e6:.2f}M) -> {path}")
    return path


@torch.no_grad()
def load_runtime(root: Path, device: Optional[str] = None):
    device = _pick_device(device)
    cache_key = (str(Path(root).resolve()), device)
    cached = _RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    art = root / "artifacts"
    vocab_path = art / "vocab.json"
    ckpt_path = art / "cec.pt"
    if not vocab_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing CEC checkpoint under {art}/ (need vocab.json and cec.pt). "
            "Train first: python -m bumblebee.cli train"
        )
    raw_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch") or raw_vocab.get("type")
    if arch == "large_hash_cec" or raw_vocab.get("type") in {"hash", "hash_encoder"}:
        from bumblebee.large_model import HashEncoderVocab, LargeCECModel

        enc_vocab = HashEncoderVocab(int(ckpt.get("n_buckets") or raw_vocab["n_buckets"]))
        ans_vocab = Vocab.load(art / "answer_vocab.json")

        class _DualVocab:
            pad_id = enc_vocab.pad_id
            eos_id = ans_vocab.eos_id

            def encode(self, text, max_len=None, add_eos=False):
                # Encoder path
                return enc_vocab.encode(text, max_len=max_len, add_eos=False)

            def decode(self, ids):
                return ans_vocab.decode(ids)

        vocab = _DualVocab()
        model = LargeCECModel(
            n_buckets=int(ckpt["n_buckets"]),
            emb_dim=int(ckpt["emb_dim"]),
            answer_vocab_size=int(ckpt["answer_vocab_size"]),
            dim=int(ckpt["dim"]),
            pad_id=int(ckpt["pad_id"]),
        )
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(device)
        # Monkey-patch generate to use answer vocab eos
        _gen = model.answer_lm.generate

        def generate(q, e, max_len=48, eos_id=None):
            return _gen(q, e, max_len=max_len, eos_id=ans_vocab.eos_id)

        model.answer_lm.generate = generate  # type: ignore[method-assign]
    else:
        vocab = Vocab.load(vocab_path)
        mass_fn = ckpt.get("mass_fn") or "sparsemax"
        if (art / "config.json").exists():
            try:
                mass_fn = json.loads((art / "config.json").read_text()).get("mass_fn", mass_fn)
            except Exception:
                pass
        model = CECModel(
            ckpt["vocab_size"], dim=ckpt["dim"], pad_id=ckpt["pad_id"], mass_fn=mass_fn
        ).to(device)
        # strict=False: older checkpoints lack decoder.end.* / correlator.log_tau
        model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    passages = load_passages(root / "data" / "corpus")
    max_q = int(ckpt.get("max_q", 64))
    max_p = int(ckpt.get("max_p", 192))
    result = (model, vocab, passages, float(ckpt.get("threshold", 0.35)), device, max_q, max_p)
    # Attach optional temperature / Platt for calibrated conf (post-hoc)
    temp = float(ckpt.get("temperature", 1.0))
    platt_a = 1.0
    platt_b = 0.0
    use_platt = False
    cfg_path = art / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            temp = float(cfg.get("temperature", temp))
            if cfg.get("platt_active") and "platt_a" in cfg and "platt_b" in cfg:
                platt_a = float(cfg["platt_a"])
                platt_b = float(cfg["platt_b"])
                use_platt = True
        except Exception:
            pass
    _RUNTIME_CACHE[cache_key] = result + (temp, platt_a, platt_b, use_platt)
    return result + (temp, platt_a, platt_b, use_platt)


def _degenerate_text(text: str) -> bool:
    toks = tokenize(text)
    if len(toks) < 4:
        return False
    from collections import Counter

    uni = Counter(toks)
    if uni.most_common(1)[0][1] / len(toks) >= 0.28:
        return True
    bigrams = list(zip(toks, toks[1:]))
    if bigrams:
        bi = Counter(bigrams)
        if bi.most_common(1)[0][1] / len(bigrams) >= 0.22:
            return True
    # repeating 3-gram loops ("the first of the first of …")
    trigrams = list(zip(toks, toks[1:], toks[2:]))
    if trigrams:
        tri = Counter(trigrams)
        if tri.most_common(1)[0][1] >= 3:
            return True
    return False


@torch.no_grad()
def ask_question_fast(root: Path, question: str) -> dict:
    """
    Extractive-only retrieved ask for benches (no GRU generate, no multihop stitch).
    Same refuse/conf path as evaluate — much faster on CPU.
    """
    from bumblebee.baselines import BM25Index  # noqa: F401 — type/compat

    loaded = load_runtime(root)
    model, vocab, passages, threshold, device, max_q, max_p, temperature, platt_a, platt_b, use_platt = (
        _unpack_runtime(loaded)
    )
    corpus_tokens = {t for p in passages for t in tokenize(p.text) if t not in _STOP}
    index = get_bm25_index(root, passages)
    hits = index.score(question)[:3]
    best_result = None
    best_raw = -1.0
    best_pid = ""
    best_mass: list = []
    best_toks: list = []
    best_out = None
    for h in hits:
        q_ids = torch.tensor([vocab.encode(question, max_q)], dtype=torch.long, device=device)
        p_ids = torch.tensor([vocab.encode(h.text, max_p)], dtype=torch.long, device=device)
        out = model(q_ids, p_ids)
        p_toks = tokenize(h.text)[:max_p]
        conf = float(out["confidence"][0])
        if content_support(question, p_toks) < 0.12 or novel_content_ratio(question, corpus_tokens) >= 0.5:
            conf = conf * 0.35
        conf = _apply_cal(conf, temperature, platt_a, platt_b, use_platt)
        result = decide(
            passage_tokens=p_toks,
            best_start=int(out["best_start"][0]),
            best_end=int(out["best_end"][0]),
            confidence=conf,
            threshold=threshold,
            generated_answer="",
            passage_text=h.text,
            question=question,
            prefer_extractive=True,
        )
        raw = float(out["confidence"][0])
        if raw > best_raw:
            best_raw = raw
            best_result = result
            best_pid = h.passage_id
            best_mass = out["mass"][0].detach().cpu().tolist()[: len(p_toks)]
            best_toks = p_toks
            best_out = out
    if best_result is None:
        return {
            "status": "refuse",
            "answer": "",
            "confidence": 0.0,
            "reason": "no_envelope_match",
            "citation_span": "",
            "citation_start": 0,
            "citation_end": 0,
            "passage_id": "",
            "source": "refuse",
            "multihop": False,
            "mass": [],
            "mass_tokens": [],
        }
    try:
        from bumblebee.telemetry import bind_root, snapshot_from_forward, write_forward_snapshot

        bind_root(root)
        if best_out is not None:
            snap = snapshot_from_forward(best_out, phase="ask", question=question)
        else:
            snap = {"phase": "ask", "mode": "live", "question": question[:160]}
        snap["decision"] = "extract" if best_result.status == "answer" else "refuse"
        snap["confidence"] = float(best_result.confidence)
        snap["citation_start"] = best_result.citation_start
        snap["citation_end"] = best_result.citation_end
        snap["mass_tokens"] = best_toks[:256]
        snap["mass"] = best_mass[:256]
        snap["passage_id"] = best_pid
        snap["answer_preview"] = (best_result.answer or "")[:160]
        write_forward_snapshot(snap, root=root)
    except Exception:
        pass
    return {
        "status": best_result.status,
        "answer": best_result.answer,
        "confidence": best_result.confidence,
        "reason": best_result.reason,
        "citation_span": best_result.citation_span,
        "citation_start": best_result.citation_start,
        "citation_end": best_result.citation_end,
        "passage_id": best_pid,
        "source": "cec" if best_result.status == "answer" else "refuse",
        "multihop": False,
        "mass": best_mass,
        "mass_tokens": best_toks,
    }


def _soft_adjust_conf(
    *,
    question: str,
    passage_tokens: Sequence[str],
    conf: float,
    concentration: float,
    evidence_tokens: Optional[set] = None,
    multihop: bool = False,
    chatty: bool = False,
) -> float:
    """
    Soft lexical floors. For stitch/synthesize, novelty is vs union evidence tokens
    (not full-corpus alone) so vague multi-source hits survive longer.
    """
    if chatty:
        return conf
    support = content_support(question, passage_tokens)
    if evidence_tokens is not None:
        novel = novel_content_ratio(question, evidence_tokens)
    else:
        novel = 1.0
    if multihop:
        # Only hard-dampen when union support AND concentration are both weak
        if support < 0.12 and concentration < 0.20:
            conf = conf * 0.45
        elif support < 0.18 and novel >= 0.55:
            conf = conf * 0.55
    else:
        if support < 0.22 or novel >= 0.35:
            conf = conf * 0.35
        if support < 0.35 and concentration < 0.25:
            conf = conf * 0.5
    return conf


def _score_passage(model, vocab, prompt: str, psg: Passage, device, max_q: int, max_p: int):
    q_ids = torch.tensor([vocab.encode(prompt, max_q)], dtype=torch.long, device=device)
    p_ids = torch.tensor([vocab.encode(psg.text, max_p)], dtype=torch.long, device=device)
    return model(q_ids, p_ids)


def _chat_fallback_map() -> dict:
    return {
        "hi": "Hi — I'm Bumblebee. Ask me something from my knowledge pack, and I'll answer with evidence when I can.",
        "hello": "Hello! I can chat briefly and answer questions grounded in my training corpus.",
        "hey": "Hey there. What would you like to know from my frozen corpus?",
        "how are you": "Doing well. Ready for a question about my knowledge pack, or we can keep chatting.",
        "what can you do": "I answer questions from a frozen corpus, cite supporting text, keep short multi-turn context, and refuse only after trying to ground an answer.",
        "who are you": "I'm Bumblebee — a transformer-free correlator trained on a closed conversational pack.",
        "whats your name": "I'm Bumblebee — a transformer-free CEC correlator.",
        "what's your name": "I'm Bumblebee — a transformer-free CEC correlator.",
        "what is your name": "I'm Bumblebee — a transformer-free CEC correlator.",
        "thanks": "You're welcome. Ask another question anytime.",
        "thank you": "Happy to help. Fire another question when you're ready.",
        "bye": "Bye! Come back when you have another question from the corpus.",
        "how is your day going?": "Going well — ready to help with questions from my training pack.",
        "how is your day going": "Going well — ready to help with questions from my training pack.",
        "introduce yourself briefly": "I'm Bumblebee CEC, a transformer-free correlator trained on a frozen knowledge pack.",
        "introduce yourself briefly?": "I'm Bumblebee CEC, a transformer-free correlator trained on a frozen knowledge pack.",
        "can we just chat for a bit?": "Sure — I can hold a short conversation, then answer grounded questions from my corpus.",
        "can we just chat for a bit": "Sure — I can hold a short conversation, then answer grounded questions from my corpus.",
    }


@torch.no_grad()
def ask_question(
    root: Path,
    question: str,
    passage_id: Optional[str] = None,
    history: Optional[List[Tuple]] = None,
) -> dict:
    """
    Product ask path: try-harder cascade before refuse.

    Stage 1 — score BM25 top singles
    Stage 2 — score 2–3 passage stitches
    Stage 3 — multi-span synthesize from top passages
    Only then refuse if still ungated.
    """
    from bumblebee.synthesize import (
        build_stitch_passages,
        synthesize_from_spans,
        union_content_tokens,
    )

    loaded = load_runtime(root)
    model, vocab, passages, threshold, device, max_q, max_p, temperature, platt_a, platt_b, use_platt = (
        _unpack_runtime(loaded)
    )
    pmap = passage_index(passages)
    corpus_tokens = {t for p in passages for t in tokenize(p.text) if t not in _STOP}

    ql = question.lower().strip()

    # Multi-turn: fold recent dialogue + cited passage IDs into the query encoder
    prompt = question
    if history:
        bits = []
        for turn in history[-3:]:
            if len(turn) >= 3 and turn[2]:
                u, a, pid = turn[0], turn[1], turn[2]
                bits.append(f"User: {u}")
                bits.append(f"Assistant [{pid}]: {a}")
            else:
                u, a = turn[0], turn[1]
                bits.append(f"User: {u}")
                bits.append(f"Assistant: {a}")
        bits.append(f"User: {question}")
        prompt = "\n".join(bits)

    # Chat shortcuts (template seed only — not eval-gold lookup)
    if ql in {"hi", "hello", "hey", "how are you", "thanks", "thank you", "what can you do", "who are you", "bye"}:
        chat_ps = [p for p in passages if p.title == "chat"]
        if chat_ps:
            passage_id = chat_ps[0].id

    if passage_id and passage_id in pmap:
        candidates = [pmap[passage_id]]
    else:
        index = get_bm25_index(root, passages)
        top = index.score(question)[:8]
        candidates = [pmap[h.passage_id] for h in top if h.passage_id in pmap]
        if not candidates:
            candidates = passages[:8]

    chatty = ql in {
        "hi",
        "hello",
        "hey",
        "how are you",
        "thanks",
        "thank you",
        "what can you do",
        "who are you",
        "bye",
        "ok",
        "how is your day going?",
        "how is your day going",
        "introduce yourself briefly",
        "introduce yourself briefly?",
        "can we just chat for a bit?",
        "can we just chat for a bit",
    } or "user:" in prompt.lower()
    if not chatty and (
        re.search(r"\b(what'?s|whats|what is)\s+your\s+name\b", ql)
        or re.search(r"^(hi|hello|hey)\b", ql)
    ):
        chatty = True

    chat_fallback = _chat_fallback_map()
    fixed_pid = bool(passage_id and passage_id in pmap)

    def _finalize(best_out, best_passage, conf, source: str, stage_multihop: bool):
        p_toks = tokenize(best_passage.text)[:max_p]
        mass_list = best_out["mass"][0].detach().cpu().tolist()[: len(p_toks)]
        gen_ids = model.answer_lm.generate(
            best_out["query_state"],
            best_out["evidence"],
            max_len=48,
            eos_id=vocab.eos_id,
        )
        generated = vocab.decode(gen_ids[0].tolist())
        if _degenerate_text(generated) or looks_like_garbage(generated, question):
            generated = ""
        src = source
        if chatty and ql in chat_fallback:
            if len(generated.split()) < 4 or not faithfulness_ok(generated, best_passage.text, min_overlap=0.08):
                generated = chat_fallback[ql]
                src = "chat_fallback"
        elif chatty and not generated:
            from bumblebee.pack_skills import lookup_pack_answer

            pack = lookup_pack_answer(question)
            if pack:
                generated = pack[0]
                src = "chat_fallback"
        # Synthesize fallback: if polish collapses, use joined extractive spans
        if source == "synthesize" and not generated:
            generated = best_passage.text.replace(" || ", " ").strip()
            if looks_like_garbage(generated, question):
                generated = ""

        result = decide(
            passage_tokens=p_toks,
            best_start=int(best_out["best_start"][0]),
            best_end=int(best_out["best_end"][0]),
            confidence=conf,
            threshold=threshold,
            generated_answer=generated,
            passage_text=best_passage.text,
            question=question,
            prefer_extractive=not chatty and source != "synthesize",
        )
        if result.status == "answer" and _degenerate_text(result.answer):
            refuse_out = {
                "status": "refuse",
                "answer": "",
                "confidence": float(conf),
                "reason": "unfaithful_decode",
                "citation_span": result.citation_span,
                "citation_start": result.citation_start,
                "citation_end": result.citation_end,
                "passage_id": best_passage.id,
                "source": "refuse",
                "multihop": stage_multihop or ("+" in best_passage.id),
                "mass": mass_list,
                "mass_tokens": p_toks,
                "history_turns": len(history or []),
            }
            try:
                from bumblebee.telemetry import bind_root, snapshot_from_forward, write_forward_snapshot

                bind_root(root)
                snap = snapshot_from_forward(best_out, phase="ask", question=question)
                snap["decision"] = "refuse"
                snap["confidence"] = float(conf)
                snap["mass_tokens"] = p_toks[:48]
                snap["passage_id"] = best_passage.id
                snap["source"] = "refuse"
                write_forward_snapshot(snap, root=root)
            except Exception:
                pass
            return refuse_out
        out_source = "refuse" if result.status == "refuse" else src
        result_dict = {
            "status": result.status,
            "answer": result.answer,
            "confidence": result.confidence,
            "reason": result.reason,
            "citation_span": result.citation_span,
            "citation_start": result.citation_start,
            "citation_end": result.citation_end,
            "passage_id": best_passage.id,
            "source": out_source,
            "multihop": stage_multihop or ("+" in best_passage.id),
            "mass": mass_list,
            "mass_tokens": p_toks,
            "history_turns": len(history or []),
        }
        try:
            from bumblebee.telemetry import bind_root, snapshot_from_forward, write_forward_snapshot

            bind_root(root)
            snap = snapshot_from_forward(best_out, phase="ask", question=question)
            snap["decision"] = "extract" if result.status == "answer" else "refuse"
            snap["confidence"] = float(result.confidence)
            snap["citation_start"] = result.citation_start
            snap["citation_end"] = result.citation_end
            snap["mass_tokens"] = p_toks[:48]
            snap["passage_id"] = best_passage.id
            snap["answer_preview"] = (result.answer or "")[:160]
            snap["source"] = out_source
            write_forward_snapshot(snap, root=root)
        except Exception:
            pass
        return result_dict

    def _try_candidates(cands: List[Passage], *, multihop: bool, source: str, union_passages: Optional[List[Passage]] = None):
        """Score candidates; return (answer_dict | refuse_dict | None, scored_list)."""
        best_raw = -1.0
        best_out = None
        best_passage = None
        best_conf = 0.0
        scored: List[Tuple[Passage, int, int, float]] = []
        evidence_set = (
            union_content_tokens(union_passages or cands, _STOP)
            if multihop
            else corpus_tokens
        )
        for psg in cands:
            out = _score_passage(model, vocab, prompt, psg, device, max_q, max_p)
            raw = float(out["confidence"][0])
            p_toks = tokenize(psg.text)[:max_p]
            concentration = float(out["concentration"][0])
            conf = _soft_adjust_conf(
                question=question,
                passage_tokens=p_toks,
                conf=raw,
                concentration=concentration,
                evidence_tokens=evidence_set,
                multihop=multihop,
                chatty=chatty,
            )
            conf = _apply_cal(conf, temperature, platt_a, platt_b, use_platt)
            scored.append((psg, int(out["best_start"][0]), int(out["best_end"][0]), conf))
            if raw > best_raw:
                best_raw = raw
                best_out = out
                best_passage = psg
                best_conf = conf
        if best_out is None or best_passage is None:
            return None, scored
        finalized = _finalize(best_out, best_passage, best_conf, source, multihop)
        return finalized, scored

    # --- Stage 1: singles ---
    stage1, scored_singles = _try_candidates(candidates, multihop=False, source="cec")
    if stage1 is not None and stage1["status"] == "answer":
        return stage1

    # --- Stage 2: stitches (skip when pinned to one passage) ---
    if not fixed_pid:
        stitches = build_stitch_passages(candidates)
        if stitches:
            stage2, _scored_stitch = _try_candidates(
                stitches,
                multihop=True,
                source="cec",
                union_passages=candidates[:3],
            )
            if stage2 is not None and stage2["status"] == "answer":
                return stage2

        # --- Stage 3: multi-span synthesize ---
        synth = synthesize_from_spans(scored_singles, max_p=max_p, top_k=3)
        if synth is not None:
            stage3, _ = _try_candidates(
                [synth],
                multihop=True,
                source="synthesize",
                union_passages=candidates[:3],
            )
            if stage3 is not None and stage3["status"] == "answer":
                return stage3

    # Cascade exhausted — best stage-1 refuse (already scored)
    if stage1 is not None:
        refused = dict(stage1)
        refused["status"] = "refuse"
        if not refused.get("reason"):
            refused["reason"] = "low_coverage"
        refused["source"] = "refuse"
        refused["answer"] = ""
        refused["hint"] = "I couldn't ground this in my pack after trying multiple passages."
        return refused

    return {
        "status": "refuse",
        "answer": "",
        "confidence": 0.0,
        "reason": "low_coverage",
        "citation_span": "",
        "citation_start": 0,
        "citation_end": 0,
        "passage_id": "",
        "source": "refuse",
        "multihop": False,
        "mass": [],
        "mass_tokens": [],
        "history_turns": len(history or []),
        "hint": "I couldn't ground this in my pack after trying multiple passages.",
    }


@torch.no_grad()
def evaluate(root: Path, *, top_k: int = 3) -> dict:
    """
    Same-protocol holdout eval: BM25 retrieves candidates (no gold passage_id gift).
    Also reports oracle-passage F1 as a diagnostic column only.
    Default top_k=3 keeps laptop eval tractable (ask path still uses top-8 + multihop).
    """
    from bumblebee.metrics import bootstrap_ci

    loaded = load_runtime(root)
    model, vocab, passages, threshold, device, max_q, max_p, temperature, platt_a, platt_b, use_platt = (
        _unpack_runtime(loaded)
    )
    pmap = passage_index(passages)
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    _, test_evals = train_eval_split(evals)
    corpus_tokens = {t for p in passages for t in tokenize(p.text) if t not in _STOP}
    index = get_bm25_index(root, passages)

    confidences = []
    correct_mixed = []
    answered_confs = []
    answered_correct = []
    risks = []
    f1s = []
    oracle_f1s = []
    refuse_tp = refuse_fp = refuse_fn = refuse_tn = 0
    faithful = []
    by_split: Dict[str, Dict[str, int]] = {}

    def _score_passage(question: str, passage_text: str):
        q_ids = torch.tensor([vocab.encode(question, max_q)], dtype=torch.long, device=device)
        p_ids = torch.tensor([vocab.encode(passage_text, max_p)], dtype=torch.long, device=device)
        out = model(q_ids, p_ids)
        p_toks = tokenize(passage_text)[:max_p]
        conf = float(out["confidence"][0])
        ql = question.lower().strip()
        chatty = ql in {"hi", "hello", "hey", "how are you", "thanks", "thank you", "what can you do"}
        if not chatty:
            if content_support(question, p_toks) < 0.12 or novel_content_ratio(question, corpus_tokens) >= 0.5:
                conf = conf * 0.35
        conf = _apply_cal(conf, temperature, platt_a, platt_b, use_platt)
        result = decide(
            passage_tokens=p_toks,
            best_start=int(out["best_start"][0]),
            best_end=int(out["best_end"][0]),
            confidence=conf,
            threshold=threshold,
            generated_answer="",
            passage_text=passage_text,
            question=question,
            prefer_extractive=True,
        )
        return result, float(out["confidence"][0])

    for ex in test_evals:
        # --- Retrieved protocol (headline) ---
        hits = index.score(ex.question)[:top_k]
        best_result = None
        best_raw = -1.0
        for h in hits:
            result, raw_conf = _score_passage(ex.question, h.text)
            # Rank candidates by raw gate conf before thresholding
            if raw_conf > best_raw:
                best_raw = raw_conf
                best_result = result
        if best_result is None:
            best_result = decide(
                passage_tokens=[],
                best_start=0,
                best_end=0,
                confidence=0.0,
                threshold=threshold,
                generated_answer="",
                passage_text="",
                question=ex.question,
                prefer_extractive=True,
            )

        # --- Oracle diagnostic (not the headline) ---
        if ex.passage_id and ex.passage_id in pmap and ex.answerable:
            oracle_result, _ = _score_passage(ex.question, pmap[ex.passage_id].text)
            if oracle_result.status == "answer":
                oracle_f1s.append(token_f1(oracle_result.answer, ex.answer or ""))

        confidences.append(best_result.confidence)
        answered = best_result.status == "answer"
        cell = refuse_confusion(ex.answerable, answered)
        sk = ex.split if not ex.answerable else "in_domain"
        bucket = by_split.setdefault(sk, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        if cell == "tp":
            refuse_tp += 1
            bucket["tp"] += 1
            correct_mixed.append(True)
            risks.append(0.0)
        elif cell == "fn":
            refuse_fn += 1
            bucket["fn"] += 1
            correct_mixed.append(False)
            risks.append(1.0)
        elif cell == "fp":
            refuse_fp += 1
            bucket["fp"] += 1
            correct_mixed.append(False)
            risks.append(1.0)
        else:
            refuse_tn += 1
            bucket["tn"] += 1
            f1 = token_f1(best_result.answer, ex.answer or "")
            f1s.append(f1)
            ok = f1 >= 0.5
            correct_mixed.append(ok)
            risks.append(0.0 if ok else 1.0)
            answered_confs.append(best_result.confidence)
            answered_correct.append(ok)
            faithful.append(1.0 if best_result.citation_span else 0.0)

    precision = refuse_tp / max(refuse_tp + refuse_fp, 1)
    recall = refuse_tp / max(refuse_tp + refuse_fn, 1)
    split_report = {}
    for k, b in by_split.items():
        split_report[k] = {
            "n": b["tp"] + b["fp"] + b["fn"] + b["tn"],
            "ood_refuse_precision": b["tp"] / max(b["tp"] + b["fp"], 1),
            "ood_refuse_recall": b["tp"] / max(b["tp"] + b["fn"], 1),
        }
    f1_ci = bootstrap_ci(f1s)
    report = {
        "n": len(test_evals),
        "n_train_note": "training uses calibration_split (excludes cal residue)",
        "protocol": (
            "retrieved_bm25_top3_then_cec; no gold passage_id; extractive-first; "
            "ece_answered on answered subset; oracle_f1 diagnostic only"
        ),
        "in_domain_f1_mean": float(sum(f1s) / max(len(f1s), 1)),
        "in_domain_f1_n": len(f1s),
        "in_domain_f1_ci95": list(f1_ci) if f1_ci else None,
        "oracle_passage_f1_mean": float(sum(oracle_f1s) / max(len(oracle_f1s), 1)),
        "oracle_passage_f1_n": len(oracle_f1s),
        "ood_refuse_precision": precision,
        "ood_refuse_recall": recall,
        "ece": expected_calibration_error(confidences, correct_mixed),
        "ece_mixed": expected_calibration_error(confidences, correct_mixed),
        "ece_answered": (
            expected_calibration_error(answered_confs, answered_correct) if answered_confs else None
        ),
        "ece_answered_n": len(answered_confs),
        "aurc": aurc(risks, confidences),
        "faithfulness_rate": float(sum(faithful) / max(len(faithful), 1)),
        "threshold": threshold,
        "temperature": temperature,
        "platt": {"a": platt_a, "b": platt_b, "active": use_platt},
        "by_split": split_report,
    }
    out = root / "artifacts" / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
