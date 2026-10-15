"""Live training / forward telemetry for the architecture monitor dash."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()
_ROOT: Optional[Path] = None


def bind_root(root: Path) -> None:
    global _ROOT
    _ROOT = Path(root)
    (_ROOT / "artifacts" / "live").mkdir(parents=True, exist_ok=True)


def _live_dir(root: Optional[Path] = None) -> Path:
    r = Path(root or _ROOT or ".")
    d = r / "artifacts" / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_train_snapshot(payload: dict, *, root: Optional[Path] = None) -> None:
    """Overwrite artifacts/live/train.json with current training state."""
    d = _live_dir(root)
    payload = dict(payload)
    payload["ts"] = time.time()
    payload["kind"] = "train"
    with _LOCK:
        _atomic_write(d / "train.json", payload)
        # Append compact history point
        hist = d / "history.jsonl"
        point = {
            "ts": payload["ts"],
            "epoch": payload.get("epoch"),
            "step": payload.get("step"),
            "loss": payload.get("loss"),
            "gate_loss": payload.get("gate_loss"),
            "span_loss": payload.get("span_loss"),
            "grad_norm": payload.get("grad_norm"),
            "lr": payload.get("lr"),
            "rss_gb": payload.get("rss_gb"),
        }
        with hist.open("a", encoding="utf-8") as f:
            f.write(json.dumps(point) + "\n")


def write_forward_snapshot(payload: dict, *, root: Optional[Path] = None) -> None:
    """Overwrite artifacts/live/forward.json with last real forward activations."""
    d = _live_dir(root)
    payload = dict(payload)
    payload["ts"] = time.time()
    payload["kind"] = "forward"
    with _LOCK:
        _atomic_write(d / "forward.json", payload)
        _atomic_write(d / "trace_latest.json", payload)


def write_event(message: str, *, root: Optional[Path] = None, **extra: Any) -> None:
    """Append a recent-event line for the architecture dash."""
    d = _live_dir(root)
    row = {"ts": time.time(), "message": str(message)[:240], **extra}
    with _LOCK:
        with (d / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_events(root: Path, *, limit: int = 40) -> List[dict]:
    path = _live_dir(root) / "events.jsonl"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        out: List[dict] = []
        for line in lines[-limit:]:
            if line.strip():
                out.append(json.loads(line))
        return out
    except Exception:
        return []


def emit_ask_activity(
    root: Path,
    question: str,
    result: dict,
    *,
    source: str = "cli",
) -> None:
    """
    Ensure every CLI/chat ask lights up the architecture dash.
    Prefer real CEC forward already written; otherwise synthesize from the question.
    """
    try:
        bind_root(root)
        status = str(result.get("status") or "?")
        conf = float(result.get("confidence") or 0.0)
        decision = "extract" if status == "answer" else "refuse"
        q = (question or "")[:160]
        # If ask_question already wrote a rich forward, refresh ts + annotate source
        existing: Dict[str, Any] = {}
        fwd_path = _live_dir(root) / "forward.json"
        if fwd_path.exists():
            try:
                existing = json.loads(fwd_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        same_q = (existing.get("question") or "")[:80] == q[:80]
        fresh = same_q and (time.time() - float(existing.get("ts") or 0)) < 8.0
        if fresh and existing.get("query_units"):
            existing["source"] = result.get("source") or source
            existing["cli"] = True
            existing["answer_preview"] = (result.get("answer") or result.get("reason") or "")[:160]
            existing["confidence"] = conf
            existing["decision"] = decision
            write_forward_snapshot(existing, root=root)
        else:
            # Synthetic CEC-shaped activity so pack/tool asks still animate the NN
            toks = [t for t in q.lower().replace("?", " ").split() if t][:24] or ["?"]
            n = max(32, min(96, len(toks) * 4))
            # deterministic pseudo-activations from token chars
            query_units = []
            for i in range(n):
                ch = toks[i % len(toks)]
                v = (sum(ord(c) for c in ch) * (i + 3) % 97) / 97.0
                query_units.append(0.15 + 0.85 * v)
            mass = result.get("mass") or []
            mass_tokens = result.get("mass_tokens") or toks
            if not mass:
                mass = [max(0.02, 1.0 / (1 + abs(i - len(mass_tokens) // 3))) for i in range(len(mass_tokens))]
                s = sum(mass) or 1.0
                mass = [m / s for m in mass]
            peak = max(mass) if mass else 0.2
            snap = {
                "phase": "ask_cli",
                "label": "CLI ASK",
                "mode": "live",
                "cli": True,
                "source": result.get("source") or source,
                "question": q,
                "answer_preview": (result.get("answer") or result.get("reason") or "")[:160],
                "confidence": conf,
                "decision": decision,
                "mass": mass[:256],
                "mass_tokens": list(mass_tokens)[:256],
                "query_units": query_units,
                "evidence_units": [min(1.0, u * (0.6 + conf)) for u in query_units],
                "activation": {
                    "encoder": 0.55 + 0.4 * conf,
                    "correlator": min(1.0, peak * 2.0),
                    "evidence": 0.45 + 0.5 * conf,
                    "gate": conf,
                    "extractive": 1.0 if decision == "extract" else 0.12,
                    "refuse": 1.0 if decision == "refuse" else 0.1,
                    "generative": 0.25,
                },
                "modules": {
                    "correlator": {"mass_peak": peak, "mass_nnz": sum(1 for x in mass if x > 0.01)},
                },
            }
            write_forward_snapshot(snap, root=root)
        write_event(
            f"ask [{result.get('source') or source}] {status} conf={conf:.3f} · {q[:80]}",
            root=root,
            kind="ask",
            status=status,
            source=result.get("source") or source,
        )
    except Exception:
        pass


def read_live_bundle(root: Path) -> dict:
    """Bundle for /api/live — train + forward + recent history."""
    d = _live_dir(root)
    train = {}
    forward = {}
    if (d / "train.json").exists():
        try:
            train = json.loads((d / "train.json").read_text(encoding="utf-8"))
        except Exception:
            train = {}
    if (d / "forward.json").exists():
        try:
            forward = json.loads((d / "forward.json").read_text(encoding="utf-8"))
        except Exception:
            forward = {}
    history: List[dict] = []
    hist_path = d / "history.jsonl"
    if hist_path.exists():
        try:
            lines = hist_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-240:]:
                if line.strip():
                    history.append(json.loads(line))
        except Exception:
            history = []
    now = time.time()
    train_age = (now - float(train["ts"])) if train.get("ts") else None
    fwd_age = (now - float(forward["ts"])) if forward.get("ts") else None
    return {
        "train": train,
        "forward": forward,
        "history": history,
        "events": read_events(root, limit=40),
        "train_live": train_age is not None and train_age < 15.0,
        "forward_live": fwd_age is not None and fwd_age < 60.0,
        "ts": now,
    }


def tensor_stats(t) -> dict:
    """CPU-safe summary of a tensor for the dash (no huge payloads)."""
    import torch

    if t is None:
        return {}
    x = t.detach()
    if x.numel() == 0:
        return {"numel": 0}
    flat = x.float().reshape(-1)
    # Sample if huge
    if flat.numel() > 4096:
        idx = torch.linspace(0, flat.numel() - 1, 4096).long()
        flat = flat[idx]
    return {
        "shape": list(x.shape),
        "numel": int(x.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "abs_mean": float(flat.abs().mean().item()),
        "max": float(flat.max().item()),
        "min": float(flat.min().item()),
        "l2": float(flat.norm().item()),
    }


def mass_vector(mass_t, cap: int = 192) -> List[float]:
    import torch

    m = mass_t.detach().float().reshape(-1)
    if m.numel() > cap:
        m = m[:cap]
    s = float(m.sum().item()) or 1.0
    return [float(x) / s for x in m.tolist()]


def unit_vector(t, cap: Optional[int] = None) -> List[float]:
    """Per-unit |activation| in [0,1] for 1:1 neuron rendering."""
    import torch

    if t is None:
        return []
    x = t.detach().float().reshape(-1)
    if cap is not None and x.numel() > cap:
        x = x[:cap]
    if x.numel() == 0:
        return []
    abs_x = x.abs()
    peak = float(abs_x.max().item()) or 1.0
    return [float(v) / peak for v in abs_x.tolist()]

def snapshot_from_forward(
    out: dict,
    *,
    phase: str,
    question: str = "",
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    losses: Optional[dict] = None,
) -> dict:
    """Build a forward snapshot from CECModel / LargeCEC forward outputs."""
    mass = out.get("mass")
    conf = out.get("confidence")
    conc = out.get("concentration")
    payload: Dict[str, Any] = {
        "phase": phase,
        "label": "LIVE FORWARD",
        "mode": "live",
        "question": (question or "")[:160],
        "epoch": epoch,
        "step": step,
        "modules": {},
        "decision": "unknown",
    }
    if mass is not None:
        # batch 0 — keep full passage mass (up to max_p) for 1:1 token nodes
        m0 = mass[0] if hasattr(mass, "dim") and mass.dim() > 1 else mass
        payload["mass"] = mass_vector(m0, cap=256)
        payload["modules"]["correlator"] = {
            "mass_entropy": _entropy(payload["mass"]),
            "mass_peak": max(payload["mass"]) if payload["mass"] else 0.0,
            "mass_nnz": sum(1 for x in payload["mass"] if x > 0.01),
        }
    if conf is not None:
        c = float(conf.reshape(-1)[0].item()) if hasattr(conf, "reshape") else float(conf)
        payload["confidence"] = c
    if conc is not None:
        payload["concentration"] = (
            float(conc.reshape(-1)[0].item()) if hasattr(conc, "reshape") else float(conc)
        )
    if out.get("evidence") is not None:
        ev = out["evidence"][0]
        payload["modules"]["evidence"] = tensor_stats(ev)
        payload["evidence_units"] = unit_vector(ev)
    if out.get("query_state") is not None:
        qs = out["query_state"][0]
        payload["modules"]["encoder"] = tensor_stats(qs)
        payload["query_units"] = unit_vector(qs)
    if out.get("start_logits") is not None:
        sl = out["start_logits"][0]
        el = out["end_logits"][0] if out.get("end_logits") is not None else None
        payload["modules"]["extractive"] = {
            "start": tensor_stats(sl),
            "end": tensor_stats(el) if el is not None else {},
            "best_start": int(out["best_start"][0].item()) if out.get("best_start") is not None else None,
            "best_end": int(out["best_end"][0].item()) if out.get("best_end") is not None else None,
        }
        payload["start_units"] = unit_vector(sl, cap=256)
        if el is not None:
            payload["end_units"] = unit_vector(el, cap=256)
    thr = 0.35
    cval = float(payload.get("confidence") or 0)
    payload["decision"] = "extract" if cval >= thr else "refuse"
    if losses:
        payload["losses"] = losses
    # Activation levels 0-1 for graph node brightness
    payload["activation"] = {
        "encoder": _norm01((payload.get("modules") or {}).get("encoder", {}).get("abs_mean")),
        "correlator": min(1.0, float((payload.get("modules") or {}).get("correlator", {}).get("mass_peak") or 0) * 1.5),
        "evidence": _norm01((payload.get("modules") or {}).get("evidence", {}).get("abs_mean")),
        "gate": min(1.0, cval),
        "extractive": 1.0 if payload["decision"] == "extract" else 0.15,
        "refuse": 1.0 if payload["decision"] == "refuse" else 0.1,
        "generative": 0.35,
    }
    return payload


def _entropy(mass: List[float]) -> float:
    import math

    h = 0.0
    for p in mass:
        if p > 1e-12:
            h -= p * math.log(p + 1e-12)
    return float(h)


def _norm01(v: Optional[float]) -> float:
    if v is None:
        return 0.0
    # soft squash typical activation magnitudes
    return float(min(1.0, abs(v) / (abs(v) + 0.5)))


def grad_global_norm(model) -> float:
    import torch

    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            total += float(g.norm().item() ** 2)
    return total ** 0.5
