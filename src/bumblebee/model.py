from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalStateEncoder(nn.Module):
    """Causal local recurrent encoder — no QK^T attention over the sequence."""

    def __init__(self, vocab_size: int, dim: int = 128, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.cell = nn.GRUCell(dim, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B, T]
        returns cell states: [B, T, D]
        """
        x = self.emb(token_ids)
        b, t, d = x.shape
        if t == 0:
            # Empty encode — keep shapes valid for downstream stack/bmm
            return self.out_norm(x.new_zeros(b, 1, d))
        h = x.new_zeros(b, d)
        states = []
        for step in range(t):
            h = self.cell(x[:, step], h)
            # zero out padded positions
            mask = (token_ids[:, step] != self.pad_id).float().unsqueeze(-1)
            h = h * mask
            states.append(h)
        return self.out_norm(torch.stack(states, dim=1))


class SelectiveStateEncoder(nn.Module):
    """
    Mamba-lite discrete selective SSM — transformer-free, O(T) sequential scan.
    Input-dependent Δ / B / C; fixed negative diagonal A. Same interface as LocalStateEncoder.
    """

    def __init__(
        self, vocab_size: int, dim: int = 128, pad_id: int = 0, delta_max: float = 8.0
    ):
        super().__init__()
        self.pad_id = pad_id
        self.dim = dim
        self.delta_max = float(delta_max)
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.delta_proj = nn.Linear(dim, dim)
        self.B_proj = nn.Linear(dim, dim)
        self.C_proj = nn.Linear(dim, dim)
        # log(-A) style: A = -softplus(param) so A < 0
        self.log_A_neg = nn.Parameter(torch.randn(dim) * 0.1)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(token_ids)
        b, t, d = x.shape
        if t == 0:
            return self.out_norm(x.new_zeros(b, 1, d))
        A = -F.softplus(self.log_A_neg)  # [D], negative
        h = x.new_zeros(b, d)
        states = []
        for step in range(t):
            xt = x[:, step]
            delta = F.softplus(self.delta_proj(xt)).clamp(min=1e-4, max=self.delta_max)
            B = self.B_proj(xt)
            C = self.C_proj(xt)
            # Discrete: h' = exp(delta * A) * h + (1 - exp(delta * A)) * (B ⊙ x)
            # A < 0 so delta*A is negative; clamp decay for stability
            decay = torch.exp((delta * A.unsqueeze(0)).clamp(min=-20.0, max=0.0))
            h = decay * h + (1.0 - decay) * (B * xt)
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            y = C * h
            mask = (token_ids[:, step] != self.pad_id).float().unsqueeze(-1)
            h = h * mask
            y = y * mask
            states.append(y)
        return self.out_norm(torch.stack(states, dim=1))


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo) — sparse differentiable alternative to softmax."""
    # Sort
    z_sorted, _ = torch.sort(logits, descending=True, dim=dim)
    # Cumsum of sorted
    cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.dim()
    shape[dim] = -1
    k = k.view(*shape)
    support = (1 + k * z_sorted) > cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)
    # Gather tau
    # support is bool; get last True index values via masked
    tau = (cumsum.gather(dim, (k_z - 1).long()) - 1) / k_z.to(logits.dtype)
    return torch.clamp(logits - tau, min=0.0)


def topk_softmax(logits: torch.Tensor, k: int = 8, dim: int = -1) -> torch.Tensor:
    """Softmax over the top-k logits; zeros elsewhere (renormalized)."""
    k = max(1, min(int(k), logits.size(dim)))
    topv, topi = torch.topk(logits, k=k, dim=dim)
    mask = torch.zeros_like(logits).scatter(dim, topi, 1.0)
    masked = logits.masked_fill(mask == 0, -1e4)
    return torch.softmax(masked, dim=dim)


class SoftCorrelator(nn.Module):
    """
    Differentiable evidence binder.
    mass_fn: sparsemax | softmax | topk
    learnable temperature via softplus(log_tau).
    """

    def __init__(
        self,
        dim: int = 128,
        temperature: float = 1.0,
        mass_fn: str = "sparsemax",
        topk: int = 8,
        learn_temperature: bool = True,
    ):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.mass_fn = (mass_fn or "sparsemax").lower()
        self.topk = topk
        if learn_temperature:
            self.log_tau = nn.Parameter(torch.tensor(float(math.log(max(temperature, 1e-3)))))
            self._fixed_tau = None
        else:
            self.register_parameter("log_tau", None)
            self._fixed_tau = float(temperature)

    @property
    def temperature(self) -> float:
        if self.log_tau is not None:
            return float(torch.nn.functional.softplus(self.log_tau).clamp(min=1e-3).item())
        return float(self._fixed_tau or 1.0)

    def forward(
        self, query_state: torch.Tensor, passage_states: torch.Tensor, passage_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query_state: [B, D] (final query cell)
        passage_states: [B, Tp, D]
        passage_mask: [B, Tp] 1=valid
        returns:
          mass: [B, Tp] evidence weights
          evidence: [B, D] mass-weighted sum of passage states
        """
        q = self.query_proj(query_state).unsqueeze(1)  # [B,1,D]
        k = self.key_proj(passage_states)  # [B,Tp,D]
        if self.log_tau is not None:
            tau = torch.nn.functional.softplus(self.log_tau).clamp(min=1e-3)
        else:
            tau = max(self._fixed_tau or 1.0, 1e-6)
        logits = (q * k).sum(-1) / tau  # [B,Tp]
        logits = logits.masked_fill(passage_mask == 0, -1e4)
        if self.mass_fn == "softmax":
            mass = torch.softmax(logits, dim=-1)
        elif self.mass_fn in {"topk", "top-k", "top_k"}:
            mass = topk_softmax(logits, k=self.topk, dim=-1)
        else:
            mass = sparsemax(logits, dim=-1)
        mass = mass * passage_mask
        mass = mass / mass.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        evidence = torch.bmm(mass.unsqueeze(1), passage_states).squeeze(1)
        return mass, evidence


class GatedDeltaCorrelator(nn.Module):
    """
    Query-conditioned gated-delta mix over passage states, then classic mass bind.
    Transformer-free linear mixer — SoftCorrelator API: (q, p_states, mask) -> mass, evidence.
    """

    def __init__(
        self,
        dim: int = 128,
        temperature: float = 1.0,
        mass_fn: str = "sparsemax",
        topk: int = 8,
        learn_temperature: bool = True,
        delta_max: float = 2.0,
    ):
        super().__init__()
        self.dim = dim
        self.delta_max = float(delta_max)
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.delta_proj = nn.Linear(dim * 2, dim)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.mass_fn = (mass_fn or "sparsemax").lower()
        self.topk = topk
        if learn_temperature:
            self.log_tau = nn.Parameter(torch.tensor(float(math.log(max(temperature, 1e-3)))))
            self._fixed_tau = None
        else:
            self.register_parameter("log_tau", None)
            self._fixed_tau = float(temperature)

    @property
    def temperature(self) -> float:
        if self.log_tau is not None:
            return float(torch.nn.functional.softplus(self.log_tau).clamp(min=1e-3).item())
        return float(self._fixed_tau or 1.0)

    def forward(
        self, query_state: torch.Tensor, passage_states: torch.Tensor, passage_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query_state: [B, D]
        passage_states: [B, Tp, D]
        passage_mask: [B, Tp]
        """
        b, tp, d = passage_states.shape
        h = passage_states.new_zeros(b, d)
        refined = []
        q_exp = query_state
        for t in range(tp):
            kt = passage_states[:, t]
            cat = torch.cat([q_exp, kt], dim=-1)
            g = torch.sigmoid(self.gate_proj(cat))
            # Bound delta — unbounded softplus + recurrent mix can explode conf/mass
            delta = F.softplus(self.delta_proj(cat)).clamp(min=1e-4, max=self.delta_max)
            h = (1.0 - g) * h + g * delta * (kt - h)
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            m = passage_mask[:, t].unsqueeze(-1)
            h = h * m
            refined.append(h)
        refined_states = torch.stack(refined, dim=1)  # [B,Tp,D]
        q = self.query_proj(query_state).unsqueeze(1)
        k = self.key_proj(refined_states)
        if self.log_tau is not None:
            tau = torch.nn.functional.softplus(self.log_tau).clamp(min=1e-3)
        else:
            tau = max(self._fixed_tau or 1.0, 1e-6)
        logits = (q * k).sum(-1) / tau
        logits = logits.masked_fill(passage_mask == 0, -1e4)
        if self.mass_fn == "softmax":
            mass = torch.softmax(logits, dim=-1)
        elif self.mass_fn in {"topk", "top-k", "top_k"}:
            mass = topk_softmax(logits, k=self.topk, dim=-1)
        else:
            mass = sparsemax(logits, dim=-1)
        mass = mass * passage_mask
        mass = mass / mass.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        evidence = torch.bmm(mass.unsqueeze(1), refined_states).squeeze(1)
        return mass, evidence


class CoverageGate(nn.Module):
    """Calibrated coverage from evidence mass concentration + raw score head."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        query_state: torch.Tensor,
        evidence: torch.Tensor,
        mass: torch.Tensor,
        passage_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns confidence in [0,1], and coverage entropy proxy.
        Entropy is normalized by valid (non-pad) length, not padded T.
        """
        eps = 1e-8
        if passage_mask is not None:
            m = mass * passage_mask
            m = m / m.sum(dim=-1, keepdim=True).clamp(min=eps)
            valid = passage_mask.sum(dim=-1).clamp(min=2.0)
        else:
            m = mass
            valid = (mass > eps).float().sum(dim=-1).clamp(min=2.0)
        ent = -(m * (m + eps).log()).sum(dim=-1)
        max_ent = valid.log().clamp(min=math.log(2.0))
        concentration = (1.0 - (ent / max_ent)).clamp(0.0, 1.0)
        logits = self.score(torch.cat([query_state, evidence], dim=-1)).squeeze(-1)
        conf = (torch.sigmoid(logits) * concentration).clamp(0.0, 1.0)
        conf = torch.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0)
        return conf, concentration


class ExtractiveDecoder(nn.Module):
    """Start + end scorers; inference end also biased by mass within max_span."""

    def __init__(self, dim: int = 128, span_len: int = 12, max_span: int = 48):
        super().__init__()
        self.span_len = span_len  # fallback / aux
        self.max_span = max_span
        self.start = nn.Linear(dim, 1)
        self.end = nn.Linear(dim, 1)

    def forward(
        self, passage_states: torch.Tensor, mass: torch.Tensor, passage_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns start_logits [B,Tp], end_logits [B,Tp], best_start [B], best_end [B]
        Inference end: argmax end_logits in [start, start+max_span), mass-biased.
        """
        start_logits = self.start(passage_states).squeeze(-1)
        end_logits = self.end(passage_states).squeeze(-1)
        start_logits = start_logits.masked_fill(passage_mask == 0, -1e4)
        end_logits = end_logits.masked_fill(passage_mask == 0, -1e4)
        combined = start_logits + mass.log().clamp(min=-20)
        best_start = combined.argmax(dim=-1)
        lengths = passage_mask.sum(dim=-1).long() - 1
        bsz = passage_states.size(0)
        best_ends = []
        for b in range(bsz):
            s = int(best_start[b].item())
            last = int(lengths[b].item())
            s = max(0, min(s, last))
            upper = min(last, s + self.max_span - 1)
            # Learned end score + mild mass bias inside the legal window
            window_logits = end_logits[b, s : upper + 1].clone()
            window_mass = mass[b, s : upper + 1].log().clamp(min=-20)
            scored = window_logits + 0.25 * window_mass
            best_e = s + int(scored.argmax().item())
            if best_e == s and last > s:
                best_e = min(last, s + self.span_len - 1)
            best_ends.append(best_e)
        best_end = torch.tensor(best_ends, device=passage_states.device, dtype=torch.long)
        best_end = torch.minimum(best_end, lengths)
        return start_logits, end_logits, best_start, best_end


class GenerativeAnswerDecoder(nn.Module):
    """
    Transformer-free answer generator: GRU language model conditioned on
    [query_state ; evidence] — fluent replies without QK^T attention.
    """

    def __init__(self, vocab_size: int, dim: int = 128, pad_id: int = 0):
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

    def teacher_force(
        self,
        query_state: torch.Tensor,
        evidence: torch.Tensor,
        answer_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        answer_ids: [B, Ta] including leading BOS-like first token target stream
        returns logits [B, Ta, V] predicting each answer token
        """
        b, ta = answer_ids.shape
        h = torch.tanh(self.cond(torch.cat([query_state, evidence], dim=-1)))
        # shift: input is <pad>/prev token, predict current
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
    def generate(
        self,
        query_state: torch.Tensor,
        evidence: torch.Tensor,
        max_len: int = 48,
        eos_id: int | None = None,
    ) -> torch.Tensor:
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


def build_encoder(
    vocab_size: int,
    dim: int,
    pad_id: int,
    encoder_kind: str = "gru",
    delta_max: float = 8.0,
) -> nn.Module:
    kind = (encoder_kind or "gru").lower()
    if kind in {"ssm", "selective", "mamba"}:
        return SelectiveStateEncoder(
            vocab_size, dim=dim, pad_id=pad_id, delta_max=delta_max
        )
    return LocalStateEncoder(vocab_size, dim=dim, pad_id=pad_id)


def build_correlator(
    dim: int,
    mass_fn: str = "sparsemax",
    correlator_kind: str = "classic",
    learn_temperature: bool = True,
    temperature: float = 1.0,
    topk: int = 8,
    delta_max: float = 2.0,
) -> nn.Module:
    kind = (correlator_kind or "classic").lower()
    if kind in {"gated_delta", "gated-delta", "delta"}:
        return GatedDeltaCorrelator(
            dim=dim,
            mass_fn=mass_fn,
            learn_temperature=learn_temperature,
            temperature=temperature,
            topk=topk,
            delta_max=delta_max,
        )
    return SoftCorrelator(
        dim=dim,
        mass_fn=mass_fn,
        learn_temperature=learn_temperature,
        temperature=temperature,
        topk=topk,
    )


class CECModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        pad_id: int = 0,
        span_len: int = 12,
        mass_fn: str = "sparsemax",
        learn_temperature: bool = True,
        encoder_kind: str = "gru",
        correlator_kind: str = "classic",
        temperature: float = 1.0,
        topk: int = 8,
        encoder_delta_max: float = 8.0,
        correlator_delta_max: float = 2.0,
    ):
        super().__init__()
        self.encoder_kind = (encoder_kind or "gru").lower()
        self.correlator_kind = (correlator_kind or "classic").lower()
        self.encoder = build_encoder(
            vocab_size,
            dim,
            pad_id,
            self.encoder_kind,
            delta_max=encoder_delta_max,
        )
        self.correlator = build_correlator(
            dim=dim,
            mass_fn=mass_fn,
            correlator_kind=self.correlator_kind,
            learn_temperature=learn_temperature,
            temperature=temperature,
            topk=topk,
            delta_max=correlator_delta_max,
        )
        self.gate = CoverageGate(dim=dim)
        self.decoder = ExtractiveDecoder(dim=dim, span_len=span_len)
        self.answer_lm = GenerativeAnswerDecoder(vocab_size, dim=dim, pad_id=pad_id)
        self.pad_id = pad_id
        self.mass_fn = mass_fn
        self.topk = topk
        self.encoder_delta_max = encoder_delta_max
        self.correlator_delta_max = correlator_delta_max

    def encode(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        states = self.encoder(token_ids)
        mask = (token_ids != self.pad_id).float()
        # final non-pad state
        lengths = mask.sum(dim=-1).long().clamp(min=1) - 1
        b = token_ids.size(0)
        final = states[torch.arange(b, device=token_ids.device), lengths]
        return states, final

    def forward(
        self,
        question_ids: torch.Tensor,
        passage_ids: torch.Tensor,
        answer_ids: torch.Tensor | None = None,
    ):
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
