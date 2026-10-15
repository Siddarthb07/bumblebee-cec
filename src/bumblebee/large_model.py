"""Large Mac-fit CEC: ~500M sparse hash encoder emb + small answer head."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from bumblebee.model import CoverageGate, ExtractiveDecoder, build_correlator
from bumblebee.tokenize import Vocab, tokenize


def hash_token_id(token: str, n_buckets: int, *, offset: int = 3) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return offset + (int(h[:8], 16) % (n_buckets - offset))


class HashEncoderVocab:
    """Hash buckets for encoder inputs only (enables huge embedding tables)."""

    def __init__(self, n_buckets: int, pad_id: int = 0, unk_id: int = 1, eos_id: int = 2):
        self.n_buckets = n_buckets
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.eos_id = eos_id

    def __len__(self) -> int:
        return self.n_buckets

    def encode(self, text: str, max_len: int | None = None, add_eos: bool = False) -> List[int]:
        ids = [hash_token_id(t, self.n_buckets) for t in tokenize(text)]
        if not ids:
            ids = [self.unk_id]
        if add_eos:
            ids = ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
            if add_eos and ids and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"type": "hash_encoder", "n_buckets": self.n_buckets}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "HashEncoderVocab":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return cls(int(obj["n_buckets"]))


class HashLocalEncoder(nn.Module):
    def __init__(self, emb: nn.Embedding, emb_dim: int, dim: int, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = emb
        self.proj = nn.Linear(emb_dim, dim)
        self.cell = nn.GRUCell(dim, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.emb(token_ids))
        b, t, d = x.shape
        if t == 0:
            return self.out_norm(x.new_zeros(b, 1, d))

        def step_fn(h, x_t, m_t):
            return self.cell(x_t, h) * m_t

        h = x.new_zeros(b, d)
        states = []
        for step in range(t):
            m = (token_ids[:, step] != self.pad_id).float().unsqueeze(-1)
            if self.training:
                h = checkpoint(step_fn, h, x[:, step], m, use_reentrant=False)
            else:
                h = step_fn(h, x[:, step], m)
            states.append(h)
        return self.out_norm(torch.stack(states, dim=1))


class SmallAnswerLM(nn.Module):
    """Closed-vocab answer LM (keeps softmax cheap on Mac)."""

    def __init__(self, vocab_size: int, dim: int, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.cond = nn.Linear(dim * 2, dim)
        self.rnn = nn.GRUCell(dim, dim)
        self.out = nn.Linear(dim, vocab_size)
        self.hearth = None

    def enable_hearth(self):
        """Exact word-table argmax. Short list only when the residual certifies it."""
        from bumblebee.hearth import HearthHead

        self.hearth = HearthHead.from_linear(self.out)
        return self.hearth

    def teacher_force(self, query_state, evidence, answer_ids):
        b, ta = answer_ids.shape
        h = torch.tanh(self.cond(torch.cat([query_state, evidence], dim=-1)))
        inp = torch.cat(
            [torch.full((b, 1), self.pad_id, device=answer_ids.device, dtype=torch.long), answer_ids[:, :-1]],
            dim=1,
        )
        logits = []
        for t in range(ta):
            x = self.emb(inp[:, t])
            h = self.rnn(x, h)
            logits.append(self.out(h))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, query_state, evidence, max_len: int = 48, eos_id: int | None = None):
        b = query_state.size(0)
        h = torch.tanh(self.cond(torch.cat([query_state, evidence], dim=-1)))
        tok = torch.full((b,), self.pad_id, device=query_state.device, dtype=torch.long)
        outs = []
        self.hearth_seals = []
        for _ in range(max_len):
            x = self.emb(tok)
            h = self.rnn(x, h)
            if self.hearth is not None:
                tok = self.hearth.argmax(h)
                if self.hearth.seal is not None:
                    self.hearth_seals.append(self.hearth.seal)
            else:
                tok = self.out(h).argmax(dim=-1)
            outs.append(tok)
            if eos_id is not None and bool((tok == eos_id).all()):
                break
        return torch.stack(outs, dim=1)


class LargeCECModel(nn.Module):
    """
    Mac-fit ~500M CEC:
    - Sparse hash embedding holds nearly all parameters (encoder only).
    - Narrow GRU + SoftCorrelator core.
    - Small closed-vocab answer head (no million-way softmax).
    """

    def __init__(
        self,
        n_buckets: int,
        emb_dim: int,
        answer_vocab_size: int,
        dim: int = 192,
        pad_id: int = 0,
        span_len: int = 12,
        mass_fn: str = "sparsemax",
        correlator_kind: str = "classic",
    ):
        super().__init__()
        self.pad_id = pad_id
        self.n_buckets = n_buckets
        self.emb_dim = emb_dim
        self.dim = dim
        self.correlator_kind = (correlator_kind or "classic").lower()
        self.emb = nn.Embedding(n_buckets, emb_dim, padding_idx=pad_id, sparse=True)
        self.encoder = HashLocalEncoder(self.emb, emb_dim, dim, pad_id=pad_id)
        self.correlator = build_correlator(
            dim=dim, mass_fn=mass_fn, correlator_kind=self.correlator_kind, learn_temperature=True
        )
        self.gate = CoverageGate(dim=dim)
        self.decoder = ExtractiveDecoder(dim=dim, span_len=span_len)
        self.answer_lm = SmallAnswerLM(answer_vocab_size, dim, pad_id=pad_id)
        self.mass_fn = mass_fn

    def encode(self, token_ids: torch.Tensor):
        states = self.encoder(token_ids)
        mask = (token_ids != self.pad_id).float()
        lengths = mask.sum(dim=-1).long().clamp(min=1) - 1
        b = token_ids.size(0)
        final = states[torch.arange(b, device=token_ids.device), lengths]
        return states, final

    def forward(self, question_ids, passage_ids, answer_ids=None):
        _, q = self.encode(question_ids)
        p_states, _ = self.encode(passage_ids)
        p_mask = (passage_ids != self.pad_id).float()
        mass, evidence = self.correlator(q, p_states, p_mask)
        conf, concentration = self.gate(q, evidence, mass, p_mask)
        start_logits, end_logits, best_start, best_end = self.decoder(p_states, mass, p_mask)
        out = {
            "mass": mass,
            "evidence": evidence,
            "confidence": conf,
            "concentration": concentration,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "best_start": best_start,
            "best_end": best_end,
            "query_state": q,
        }
        if answer_ids is not None:
            out["answer_logits"] = self.answer_lm.teacher_force(q, evidence, answer_ids)
        return out
