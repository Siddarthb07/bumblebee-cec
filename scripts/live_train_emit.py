#!/usr/bin/env python3
"""Emit live train telemetry from real CEC batches without overwriting cec.pt.

Used so the architecture monitor can show live loss/mass while a long dense
train (started before telemetry hooks) keeps running.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bumblebee.data import load_evals, load_passages, passage_index
from bumblebee.eval_protocol import calibration_split
from bumblebee.model import CECModel
from bumblebee.telemetry import (
    bind_root,
    grad_global_norm,
    snapshot_from_forward,
    write_forward_snapshot,
    write_train_snapshot,
)
from bumblebee.tokenize import Vocab
from bumblebee.train import QADataset, collate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    bind_root(root)
    device = "cpu"
    torch.set_num_threads(1)

    cfg_path = root / "artifacts" / "config.json"
    dim = 128
    mass_fn = "sparsemax"
    if cfg_path.exists():
        import json

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        dim = int(cfg.get("dim") or 128)
        # Cap for monitor so we don't fight the dense dim=640 job for RAM
        dim = min(dim, 160)
        mass_fn = cfg.get("mass_fn") or "sparsemax"

    corpus_dir = root / "data" / "corpus"
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    passages = load_passages(corpus_dir)
    pmap = passage_index(passages)
    texts = [p.text for p in passages] + [e.question for e in evals] + [e.answer or "" for e in evals]
    vocab_path = root / "artifacts" / "vocab.json"
    if vocab_path.exists():
        vocab = Vocab.load(vocab_path)
    else:
        vocab = Vocab.build(texts)

    train_evals, _, _ = calibration_split(evals)
    # Keep monitor light
    train_evals = train_evals[:800]
    ds = QADataset(train_evals, pmap, vocab, max_q=64, max_p=192, max_a=64)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=lambda b: collate(b, vocab.pad_id))

    model = CECModel(len(vocab.itos), dim=dim, pad_id=vocab.pad_id, mass_fn=mass_fn).to(device)
    ckpt = root / "artifacts" / "cec.pt"
    if ckpt.exists():
        try:
            blob = torch.load(ckpt, map_location=device)
            if int(blob.get("dim") or 0) == dim and blob.get("model"):
                model.load_state_dict(blob["model"], strict=False)
                print(f"loaded cec.pt dim={dim}", flush=True)
        except Exception as e:
            print(f"ckpt load skipped: {e}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())
    epochs = 2
    global_step = 0
    print(f"live train emit dim={dim} params={n_params} examples={len(train_evals)}", flush=True)

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            q = batch["question_ids"].to(device)
            p = batch["passage_ids"].to(device)
            a = batch["answer_ids"].to(device)
            out = model(q, p, answer_ids=None)
            gate_loss = F.binary_cross_entropy(out["confidence"], batch["answerable"].to(device))
            start_logits = out["start_logits"]
            end_logits = out["end_logits"]
            ans_mask = batch["answerable"].to(device)
            starts = batch["start"].to(device).clamp(max=start_logits.size(1) - 1)
            ends = batch["end"].to(device).clamp(max=end_logits.size(1) - 1)
            ends = torch.maximum(ends, starts)
            start_loss = (F.cross_entropy(start_logits, starts, reduction="none") * ans_mask).mean()
            end_loss = (F.cross_entropy(end_logits, ends, reduction="none") * ans_mask).mean()
            loss = gate_loss + start_loss + end_loss
            opt.zero_grad()
            loss.backward()
            gnorm = grad_global_norm(model)
            opt.step()
            global_step += 1

            sample_q = vocab.decode(batch["question_ids"][0].tolist())[:160]
            sample_a = vocab.decode(batch["answer_ids"][0].tolist())[:120]
            try:
                import psutil

                rss = psutil.Process().memory_info().rss / (1024**3)
            except Exception:
                rss = None

            write_train_snapshot(
                {
                    "status": "running",
                    "arch": "dense_cec_monitor",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "step": global_step,
                    "steps_per_epoch": len(loader),
                    "loss": float(loss.item()),
                    "gate_loss": float(gate_loss.item()),
                    "span_loss": float(start_loss.item() + end_loss.item()),
                    "grad_norm": gnorm,
                    "lr": 1e-3,
                    "dim": dim,
                    "n_params": n_params,
                    "batch_size": 2,
                    "rss_gb": rss,
                    "vocab_size": len(vocab.itos),
                    "sample_question": sample_q,
                    "sample_answer": sample_a,
                    "sample_answerable": float(batch["answerable"][0].item()),
                    "note": "monitor feed (does not overwrite cec.pt)",
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
                        "total": float(loss.item()),
                        "gate": float(gate_loss.item()),
                        "start": float(start_loss.item()),
                        "end": float(end_loss.item()),
                    },
                ),
                root=root,
            )
            if global_step % 10 == 0:
                print(
                    f"monitor step {global_step} loss={loss.item():.4f} q={sample_q[:60]!r}",
                    flush=True,
                )
            # Pace so the dash can be read; still live
            time.sleep(0.35)

    write_train_snapshot(
        {
            "status": "finished",
            "arch": "dense_cec_monitor",
            "epoch": epochs,
            "epochs": epochs,
            "step": global_step,
            "loss": float(loss.item()),
            "dim": dim,
            "n_params": n_params,
            "note": "monitor feed finished",
        },
        root=root,
    )
    print("live train emit done", flush=True)


if __name__ == "__main__":
    main()
