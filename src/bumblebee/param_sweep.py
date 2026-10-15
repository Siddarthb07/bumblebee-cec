"""In-depth Linear CEC parameter probes + one-factor micro train sweeps."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from bumblebee.distill import distill_loss
from bumblebee.model import (
    CECModel,
    GatedDeltaCorrelator,
    SelectiveStateEncoder,
    SoftCorrelator,
)
from bumblebee.train import (
    clear_runtime_cache,
    evaluate,
    train_model,
    train_model_distill,
)


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def _mass_entropy(mass: torch.Tensor, mask: torch.Tensor) -> float:
    eps = 1e-8
    m = mass * mask
    m = m / m.sum(dim=-1, keepdim=True).clamp(min=eps)
    ent = -(m * (m + eps).log()).sum(dim=-1)
    return float(ent.mean().item())


def probe_behavioral(seed: int = 0) -> Dict[str, Any]:
    """
    No-train probes: each parameter must change a measurable quantity or stay stable.
    Returns pass/fail per probe with numeric evidence.
    """
    torch.manual_seed(seed)
    results: Dict[str, Any] = {}
    dim = 16
    q = torch.randn(2, dim)
    p = torch.randn(2, 10, dim)
    mask = torch.ones(2, 10)
    mask[:, -2:] = 0.0
    ids = torch.randint(1, 40, (2, 8))
    ids[:, -1] = 0

    # --- mass_fn ---
    ent = {}
    nnz = {}
    for mf in ("sparsemax", "softmax", "topk"):
        corr = SoftCorrelator(dim=dim, mass_fn=mf, topk=3, learn_temperature=False, temperature=1.0)
        mass, ev = corr(q, p, mask)
        assert mass.shape == (2, 10) and ev.shape == (2, dim)
        assert _finite(mass) and _finite(ev)
        ent[mf] = _mass_entropy(mass, mask)
        nnz[mf] = float((mass > 1e-6).float().sum(dim=-1).mean().item())
    results["mass_fn"] = {
        "entropy": ent,
        "mean_support": nnz,
        "pass": nnz["topk"] <= 3.01 and abs(ent["softmax"] - ent["sparsemax"]) > 1e-6,
        "note": "topk support ≤k; softmax vs sparsemax entropy differs",
    }

    # --- topk ---
    supports = {}
    for k in (2, 4, 8):
        corr = SoftCorrelator(dim=dim, mass_fn="topk", topk=k, learn_temperature=False)
        mass, _ = corr(q, p, mask)
        supports[k] = float((mass > 1e-6).float().sum(dim=-1).max().item())
    results["topk"] = {
        "max_support": supports,
        "pass": supports[2] <= 2.01 and supports[4] <= 4.01 and supports[8] <= 8.01,
        "note": "support cardinality respects topk",
    }

    # --- temperature / learn_temperature ---
    masses = {}
    for tau in (0.5, 1.0, 2.0, 4.0):
        corr = SoftCorrelator(dim=dim, mass_fn="softmax", learn_temperature=False, temperature=tau)
        mass, _ = corr(q, p, mask)
        masses[tau] = _mass_entropy(mass, mask)
    # Higher tau → softer → higher entropy (usually)
    results["temperature"] = {
        "entropy_by_tau": masses,
        "pass": masses[4.0] >= masses[0.5] - 1e-5,
        "note": "higher fixed τ softens softmax mass (entropy ↑)",
    }
    learn = SoftCorrelator(dim=dim, learn_temperature=True, temperature=1.0)
    fixed = SoftCorrelator(dim=dim, learn_temperature=False, temperature=1.0)
    results["learn_temperature"] = {
        "has_log_tau_param": learn.log_tau is not None,
        "fixed_has_no_param": fixed.log_tau is None,
        "pass": learn.log_tau is not None and fixed.log_tau is None,
        "note": "learn_temperature toggles log_tau Parameter",
    }

    # --- encoder_kind ---
    gru = CECModel(48, dim=dim, encoder_kind="gru", correlator_kind="classic")
    ssm = CECModel(48, dim=dim, encoder_kind="ssm", correlator_kind="classic")
    o_gru = gru(ids[:, :6], ids)
    o_ssm = ssm(ids[:, :6], ids)
    results["encoder_kind"] = {
        "gru_conf_finite": _finite(o_gru["confidence"]),
        "ssm_conf_finite": _finite(o_ssm["confidence"]),
        "states_differ": float(
            (gru.encode(ids)[0] - ssm.encode(ids)[0]).abs().mean().item()
        ),
        "pass": _finite(o_gru["confidence"])
        and _finite(o_ssm["confidence"])
        and float((gru.encode(ids)[0] - ssm.encode(ids)[0]).abs().mean().item()) > 1e-4,
        "note": "GRU vs SSM produce different states; both finite",
    }

    # --- correlator_kind ---
    classic = SoftCorrelator(dim=dim, learn_temperature=False)
    gated = GatedDeltaCorrelator(dim=dim, learn_temperature=False, delta_max=2.0)
    m_c, e_c = classic(q, p, mask)
    m_g, e_g = gated(q, p, mask)
    results["correlator_kind"] = {
        "mass_l1": float((m_c - m_g).abs().sum().item()),
        "evidence_l2": float((e_c - e_g).norm().item()),
        "gated_finite": _finite(m_g) and _finite(e_g),
        "pass": float((m_c - m_g).abs().sum().item()) > 1e-4 and _finite(m_g),
        "note": "gated_delta changes mass/evidence vs classic",
    }

    # --- correlator delta_max ---
    torch.manual_seed(1)
    # Amplify delta_proj so clamp binds
    dm = {}
    for dmax in (0.5, 2.0, 8.0):
        g = GatedDeltaCorrelator(dim=dim, learn_temperature=False, delta_max=dmax)
        with torch.no_grad():
            g.delta_proj.weight.mul_(5.0)
            g.delta_proj.bias.fill_(3.0)
        mass, ev = g(q, p, mask)
        dm[dmax] = {
            "mass_entropy": _mass_entropy(mass, mask),
            "ev_norm": float(ev.norm(dim=-1).mean().item()),
            "finite": _finite(mass) and _finite(ev),
        }
    results["correlator_delta_max"] = {
        "by_max": dm,
        "pass": all(v["finite"] for v in dm.values()),
        "note": "delta_max keeps gated mix finite under stressed weights",
    }

    # --- encoder delta_max ---
    enc_stats = {}
    for dmax in (0.5, 2.0, 8.0):
        enc = SelectiveStateEncoder(40, dim=dim, pad_id=0, delta_max=dmax)
        with torch.no_grad():
            enc.delta_proj.weight.mul_(4.0)
            enc.delta_proj.bias.fill_(2.0)
        out = enc(ids)
        enc_stats[dmax] = {
            "abs_mean": float(out.abs().mean().item()),
            "finite": _finite(out),
        }
    results["encoder_delta_max"] = {
        "by_max": enc_stats,
        "pass": all(v["finite"] for v in enc_stats.values()),
        "note": "SSM delta_max keeps encoder states finite under stress",
    }

    # --- distill lambdas (loss composition) ---
    teacher = CECModel(40, dim=dim, encoder_kind="gru", correlator_kind="classic")
    student = CECModel(40, dim=dim, encoder_kind="ssm", correlator_kind="gated_delta")
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    q_ids = torch.randint(1, 40, (2, 4))
    p_ids = torch.randint(1, 40, (2, 6))
    p_mask = (p_ids != 0).float()
    with torch.no_grad():
        t_out = teacher(q_ids, p_ids)
    s_out = student(q_ids, p_ids)
    base, base_m = distill_loss(
        t_out,
        s_out,
        passage_mask=p_mask,
        answerable=torch.ones(2),
        start=torch.zeros(2, dtype=torch.long),
        end=torch.ones(2, dtype=torch.long),
        lambda_mass=1.0,
        lambda_span=0.5,
        lambda_gate=0.5,
        lambda_task=0.5,
    )
    only_mass, _ = distill_loss(
        t_out,
        s_out,
        passage_mask=p_mask,
        lambda_mass=1.0,
        lambda_span=0.0,
        lambda_gate=0.0,
        lambda_task=0.0,
    )
    only_span, _ = distill_loss(
        t_out,
        s_out,
        passage_mask=p_mask,
        lambda_mass=0.0,
        lambda_span=1.0,
        lambda_gate=0.0,
        lambda_task=0.0,
    )
    results["distill_lambdas"] = {
        "base": float(base.item()),
        "mass_only": float(only_mass.item()),
        "span_only": float(only_span.item()),
        "components": base_m,
        "pass": _finite(base)
        and abs(float(only_mass.item()) - base_m["mass_kl"]) < 1e-4
        and float(only_span.item()) > 0,
        "note": "λ_mass / λ_span isolate mass_kl and span_kl terms",
    }

    # --- span_temperature ---
    st = {}
    for temp in (1.0, 2.0, 4.0):
        _, met = distill_loss(
            t_out,
            s_out,
            passage_mask=p_mask,
            lambda_mass=0.0,
            lambda_span=1.0,
            lambda_gate=0.0,
            lambda_task=0.0,
            span_temperature=temp,
        )
        st[temp] = met["span_kl"]
    results["span_temperature"] = {
        "span_kl_by_temp": st,
        "pass": all(math.isfinite(v) for v in st.values()),
        "note": "span KD defined for T∈{1,2,4}",
    }

    # --- conf clamp / coverage ---
    m = CECModel(40, dim=dim, encoder_kind="ssm", correlator_kind="gated_delta")
    out = m(q_ids, p_ids)
    conf = out["confidence"]
    results["confidence_bounds"] = {
        "min": float(conf.min().item()),
        "max": float(conf.max().item()),
        "pass": float(conf.min().item()) >= -1e-6 and float(conf.max().item()) <= 1.0 + 1e-6,
        "note": "CoverageGate conf ∈ [0,1]",
    }

    results["all_pass"] = all(
        bool(v.get("pass")) for k, v in results.items() if isinstance(v, dict) and "pass" in v
    )
    return results


def _arm_train_eval(
    root: Path,
    *,
    label: str,
    train_kwargs: dict,
    distill: bool = False,
    distill_kwargs: Optional[dict] = None,
    teacher_path: Optional[Path] = None,
    max_eval: int = 80,
) -> Dict[str, Any]:
    t0 = time.time()
    clear_runtime_cache()
    if distill:
        assert teacher_path is not None
        kw = dict(distill_kwargs or {})
        kw.setdefault("ckpt_name", "cec.pt")
        train_model_distill(root, teacher_path=teacher_path, **kw)
    else:
        train_model(root, **train_kwargs)
    clear_runtime_cache()
    report = evaluate(root, max_examples=max_eval)
    clear_runtime_cache()
    cfg = {}
    try:
        cfg = json.loads((root / "artifacts" / "config.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "label": label,
        "in_domain_f1_mean": report.get("in_domain_f1_mean"),
        "ood_refuse_recall": report.get("ood_refuse_recall"),
        "aurc": report.get("aurc"),
        "ece_answered": report.get("ece_answered"),
        "n_params": cfg.get("n_params"),
        "seconds": round(time.time() - t0, 1),
        "train": {**(train_kwargs if not distill else (distill_kwargs or {})), "distill": distill},
    }


def run_param_depth_sweep(
    root: Path,
    *,
    epochs: int = 1,
    dim: int = 32,
    max_train_examples: int = 96,
    max_eval: int = 80,
) -> dict:
    """
    One-factor-at-a-time micro train+eval across Linear CEC parameters.
    Restores production cec.pt / config / vocab afterward.
    """
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    ckpt = art / "cec.pt"
    cfg = art / "config.json"
    vocab = art / "vocab.json"
    backups = {
        ckpt: art / "cec.pt.param_sweep_backup",
        cfg: art / "config.json.param_sweep_backup",
        vocab: art / "vocab.json.param_sweep_backup",
    }
    for src, dst in backups.items():
        if not src.exists():
            continue
        # Never clobber a larger existing backup with a smaller/corrupt ckpt
        if dst.exists() and src.stat().st_size < dst.stat().st_size:
            continue
        shutil.copy2(src, dst)

    behavioral = probe_behavioral()
    axes: Dict[str, List[Dict[str, Any]]] = {}
    teacher = art / "cec_teacher_param_sweep.pt"

    try:
        # Shared classic teacher for distill axes
        clear_runtime_cache()
        train_model(
            root,
            epochs=epochs,
            dim=dim,
            encoder_kind="gru",
            correlator_kind="classic",
            max_train_examples=max_train_examples,
            ckpt_name="cec.pt",
        )
        shutil.copy2(ckpt, teacher)

        # --- encoder × correlator ---
        axis = []
        for enc, corr in (
            ("gru", "classic"),
            ("gru", "gated_delta"),
            ("ssm", "classic"),
            ("ssm", "gated_delta"),
        ):
            axis.append(
                _arm_train_eval(
                    root,
                    label=f"{enc}+{corr}",
                    train_kwargs=dict(
                        epochs=epochs,
                        dim=dim,
                        encoder_kind=enc,
                        correlator_kind=corr,
                        max_train_examples=max_train_examples,
                        ckpt_name="cec.pt",
                    ),
                    max_eval=max_eval,
                )
            )
        axes["encoder_x_correlator"] = axis

        # --- mass_fn on ssm+gated_delta ---
        axis = []
        for mf in ("sparsemax", "softmax", "topk"):
            axis.append(
                _arm_train_eval(
                    root,
                    label=mf,
                    train_kwargs=dict(
                        epochs=epochs,
                        dim=dim,
                        encoder_kind="ssm",
                        correlator_kind="gated_delta",
                        mass_fn=mf,
                        topk=4,
                        max_train_examples=max_train_examples,
                        ckpt_name="cec.pt",
                    ),
                    max_eval=max_eval,
                )
            )
        axes["mass_fn"] = axis

        # --- topk ---
        axis = []
        for k in (2, 4, 8):
            axis.append(
                _arm_train_eval(
                    root,
                    label=f"topk={k}",
                    train_kwargs=dict(
                        epochs=epochs,
                        dim=dim,
                        encoder_kind="ssm",
                        correlator_kind="gated_delta",
                        mass_fn="topk",
                        topk=k,
                        max_train_examples=max_train_examples,
                        ckpt_name="cec.pt",
                    ),
                    max_eval=max_eval,
                )
            )
        axes["topk"] = axis

        # --- temperature ---
        axis = []
        for learn, tau in ((True, 1.0), (False, 0.5), (False, 1.0), (False, 2.0)):
            axis.append(
                _arm_train_eval(
                    root,
                    label=f"learn={learn}_tau={tau}",
                    train_kwargs=dict(
                        epochs=epochs,
                        dim=dim,
                        encoder_kind="ssm",
                        correlator_kind="gated_delta",
                        learn_temperature=learn,
                        temperature=tau,
                        max_train_examples=max_train_examples,
                        ckpt_name="cec.pt",
                    ),
                    max_eval=max_eval,
                )
            )
        axes["temperature"] = axis

        # --- correlator_delta_max ---
        axis = []
        for dmax in (0.5, 2.0, 8.0):
            axis.append(
                _arm_train_eval(
                    root,
                    label=f"corr_delta_max={dmax}",
                    train_kwargs=dict(
                        epochs=epochs,
                        dim=dim,
                        encoder_kind="ssm",
                        correlator_kind="gated_delta",
                        correlator_delta_max=dmax,
                        max_train_examples=max_train_examples,
                        ckpt_name="cec.pt",
                    ),
                    max_eval=max_eval,
                )
            )
        axes["correlator_delta_max"] = axis

        # --- encoder_delta_max ---
        axis = []
        for dmax in (0.5, 2.0, 8.0):
            axis.append(
                _arm_train_eval(
                    root,
                    label=f"enc_delta_max={dmax}",
                    train_kwargs=dict(
                        epochs=epochs,
                        dim=dim,
                        encoder_kind="ssm",
                        correlator_kind="gated_delta",
                        encoder_delta_max=dmax,
                        max_train_examples=max_train_examples,
                        ckpt_name="cec.pt",
                    ),
                    max_eval=max_eval,
                )
            )
        axes["encoder_delta_max"] = axis

        # --- dim ---
        axis = []
        for d in (24, 32, 48):
            axis.append(
                _arm_train_eval(
                    root,
                    label=f"dim={d}",
                    train_kwargs=dict(
                        epochs=epochs,
                        dim=d,
                        encoder_kind="ssm",
                        correlator_kind="gated_delta",
                        max_train_examples=max_train_examples,
                        ckpt_name="cec.pt",
                    ),
                    max_eval=max_eval,
                )
            )
        axes["dim"] = axis

        # --- distill lambdas ---
        axis = []
        lambda_sets = [
            ("default", 1.0, 0.5, 0.5, 0.5),
            ("mass_heavy", 2.0, 0.25, 0.25, 0.25),
            ("span_heavy", 0.25, 2.0, 0.25, 0.25),
            ("gate_heavy", 0.25, 0.25, 2.0, 0.25),
            ("task_heavy", 0.25, 0.25, 0.25, 2.0),
            ("kd_only", 1.0, 0.5, 0.5, 0.0),
        ]
        for name, lm, ls, lg, lt in lambda_sets:
            axis.append(
                _arm_train_eval(
                    root,
                    label=name,
                    train_kwargs={},
                    distill=True,
                    teacher_path=teacher,
                    distill_kwargs=dict(
                        epochs=epochs,
                        encoder_kind="ssm",
                        correlator_kind="gated_delta",
                        max_train_examples=max_train_examples,
                        ckpt_name="cec.pt",
                        lambda_mass=lm,
                        lambda_span=ls,
                        lambda_gate=lg,
                        lambda_task=lt,
                    ),
                    max_eval=max_eval,
                )
            )
        axes["distill_lambdas"] = axis

        # --- span_temperature ---
        axis = []
        for st in (1.0, 2.0, 4.0):
            axis.append(
                _arm_train_eval(
                    root,
                    label=f"span_T={st}",
                    train_kwargs={},
                    distill=True,
                    teacher_path=teacher,
                    distill_kwargs=dict(
                        epochs=epochs,
                        encoder_kind="ssm",
                        correlator_kind="gated_delta",
                        max_train_examples=max_train_examples,
                        ckpt_name="cec.pt",
                        span_temperature=st,
                    ),
                    max_eval=max_eval,
                )
            )
        axes["span_temperature"] = axis

    finally:
        clear_runtime_cache()
        for src, bak in backups.items():
            if bak.exists():
                shutil.copy2(bak, src)
                bak.unlink()
        if teacher.exists():
            teacher.unlink()
        clear_runtime_cache()

    # Summaries: best F1 per axis
    winners = {}
    for axis_name, rows in axes.items():
        best = None
        for r in rows:
            f1 = r.get("in_domain_f1_mean")
            if not isinstance(f1, (int, float)):
                continue
            if best is None or f1 > best["in_domain_f1_mean"]:
                best = r
        winners[axis_name] = best["label"] if best else None

    out = {
        "protocol": (
            f"one-factor micro sweep dim≈{dim} epochs={epochs} "
            f"max_train={max_train_examples} max_eval={max_eval}; "
            "behavioral probes + train/eval; production cec.pt restored"
        ),
        "behavioral": behavioral,
        "axes": axes,
        "winners_by_f1": winners,
        "transformers": False,
    }
    (art / "param_sweep_linear.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_md(root, out)
    return out


def _write_md(root: Path, out: dict) -> None:
    lines = [
        "# Linear CEC parameter depth sweep",
        "",
        out["protocol"],
        "",
        "## Behavioral probes (no train)",
        "",
        "| parameter | pass | note |",
        "| --- | --- | --- |",
    ]
    beh = out.get("behavioral") or {}
    for k, v in beh.items():
        if k == "all_pass" or not isinstance(v, dict) or "pass" not in v:
            continue
        lines.append(f"| `{k}` | {'yes' if v.get('pass') else 'NO'} | {v.get('note', '')} |")
    lines.append(f"| **all** | {'yes' if beh.get('all_pass') else 'NO'} | |")
    lines.extend(["", "## Micro train+eval (one factor at a time)", ""])
    for axis, rows in (out.get("axes") or {}).items():
        lines.append(f"### `{axis}`")
        lines.append("")
        lines.append("| setting | F1 | OOD R | AURC | ECE | sec |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            f1 = r.get("in_domain_f1_mean")
            ood = r.get("ood_refuse_recall")
            aurc = r.get("aurc")
            ece = r.get("ece_answered")
            if isinstance(f1, (int, float)):
                lines.append(
                    f"| {r.get('label')} | {f1:.3f} | "
                    f"{ood:.3f} | {aurc:.3f} | {ece:.3f} | {r.get('seconds')} |"
                )
            else:
                lines.append(
                    f"| {r.get('label')} | — | — | — | — | {r.get('seconds')} |"
                )
        win = (out.get("winners_by_f1") or {}).get(axis)
        lines.append("")
        lines.append(f"Best F1 under this axis: **{win}**")
        lines.append("")
    lines.extend(
        [
            "## Honesty",
            "",
            "- Micro budget — do not treat winners as production promotions.",
            "- CIs will overlap; prefer directional signal + behavioral pass.",
            "- Product default remains GRU + classic SoftCorrelator.",
            "",
        ]
    )
    (root / "docs" / "PARAM_SWEEP_LINEAR.md").write_text("\n".join(lines), encoding="utf-8")
