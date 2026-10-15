"""Distill classic CEC → linear student (SSM + gated-delta). No transformer teacher."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _masked_mass_kl(
    student_mass: torch.Tensor,
    teacher_mass: torch.Tensor,
    passage_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL(student ‖ teacher) over valid passage positions."""
    m_s = student_mass * passage_mask
    m_t = teacher_mass.detach() * passage_mask
    m_s = m_s / m_s.sum(dim=-1, keepdim=True).clamp(min=eps)
    m_t = m_t / m_t.sum(dim=-1, keepdim=True).clamp(min=eps)
    # KL(s||t) = sum s * (log s - log t)
    kl = (m_s * ((m_s + eps).log() - (m_t + eps).log())).sum(dim=-1)
    return kl.mean()


def _logits_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Temperature-softened KL over masked positions (batch mean)."""
    t = max(float(temperature), 1e-3)
    s = student_logits / t
    teach = teacher_logits.detach() / t
    s = s.masked_fill(mask == 0, -1e4)
    teach = teach.masked_fill(mask == 0, -1e4)
    log_p = F.log_softmax(s, dim=-1)
    q = F.softmax(teach, dim=-1)
    # sum over tokens then mean batch; ignore pad via mask on q mass
    kl = (q * (q.clamp(min=1e-8).log() - log_p)).sum(dim=-1)
    # only average over rows that have any valid tokens
    valid = (mask.sum(dim=-1) > 0).float()
    return (kl * valid).sum() / valid.sum().clamp(min=1.0)


def distill_loss(
    teacher_out: Dict[str, Any],
    student_out: Dict[str, Any],
    *,
    passage_mask: torch.Tensor,
    answerable: Optional[torch.Tensor] = None,
    start: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
    lambda_mass: float = 1.0,
    lambda_span: float = 0.5,
    lambda_gate: float = 0.5,
    lambda_task: float = 0.5,
    span_temperature: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined KD + light task loss. Teacher tensors are detached inside.
    Returns (loss, metrics_dict).
    """
    mass_l = _masked_mass_kl(student_out["mass"], teacher_out["mass"], passage_mask)
    span_l = _logits_kl(
        student_out["start_logits"], teacher_out["start_logits"], passage_mask, span_temperature
    ) + _logits_kl(
        student_out["end_logits"], teacher_out["end_logits"], passage_mask, span_temperature
    )
    gate_l = F.mse_loss(student_out["confidence"], teacher_out["confidence"].detach())
    if "concentration" in student_out and "concentration" in teacher_out:
        gate_l = gate_l + F.mse_loss(
            student_out["concentration"], teacher_out["concentration"].detach()
        )

    task_l = student_out["confidence"].new_zeros(())
    if answerable is not None:
        task_l = task_l + F.binary_cross_entropy(
            student_out["confidence"].clamp(0.0, 1.0),
            answerable.to(dtype=student_out["confidence"].dtype),
        )
        if start is not None and end is not None:
            ans = answerable.to(device=student_out["start_logits"].device).float()
            # CE only on answerable rows
            sl = student_out["start_logits"]
            el = student_out["end_logits"]
            s_tgt = start.to(sl.device).clamp(max=sl.size(1) - 1)
            e_tgt = end.to(el.device).clamp(max=el.size(1) - 1)
            start_ce = F.cross_entropy(sl, s_tgt, reduction="none")
            end_ce = F.cross_entropy(el, e_tgt, reduction="none")
            span_ce = ((start_ce + end_ce) * ans).sum() / ans.sum().clamp(min=1.0)
            task_l = task_l + span_ce

    loss = (
        lambda_mass * mass_l
        + lambda_span * span_l
        + lambda_gate * gate_l
        + lambda_task * task_l
    )
    metrics = {
        "mass_kl": float(mass_l.detach().item()),
        "span_kl": float(span_l.detach().item()),
        "gate_mse": float(gate_l.detach().item()),
        "task": float(task_l.detach().item()) if task_l.numel() else 0.0,
        "loss": float(loss.detach().item()),
    }
    return loss, metrics


def copy_embedding_from_teacher(student: torch.nn.Module, teacher: torch.nn.Module) -> bool:
    """Copy encoder.emb when shapes match. Returns True if copied."""
    try:
        s_emb = student.encoder.emb
        t_emb = teacher.encoder.emb
        if s_emb.weight.shape == t_emb.weight.shape:
            s_emb.weight.data.copy_(t_emb.weight.data)
            return True
    except Exception:
        pass
    return False
