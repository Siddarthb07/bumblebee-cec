"""SoftCorrelator mass_fn ablation — sparsemax vs softmax vs top-k (short train)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch

from bumblebee.eval_protocol import calibration_split
from bumblebee.train import evaluate, train_model, clear_runtime_cache


def run_mass_ablation(
    root: Path,
    *,
    mass_fns: List[str] | None = None,
    epochs: int = 2,
    dim: int = 64,
) -> dict:
    """
    Train a small CEC per mass_fn, evaluate same-protocol, write artifacts/ablation_mass.json.
    Does not claim a winner until CIs separate — publishes the card either way.
    """
    mass_fns = mass_fns or ["sparsemax", "softmax", "topk"]
    results: Dict[str, dict] = {}
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    # Preserve current checkpoint
    ckpt = art / "cec.pt"
    cfg = art / "config.json"
    backup = art / "cec.pt.ablation_backup"
    cfg_backup = art / "config.json.ablation_backup"
    if ckpt.exists():
        ckpt.replace(backup)
    if cfg.exists():
        cfg.replace(cfg_backup)

    try:
        for mf in mass_fns:
            clear_runtime_cache()
            train_model(root, epochs=epochs, dim=dim, mass_fn=mf)
            clear_runtime_cache()
            report = evaluate(root)
            results[mf] = {
                "in_domain_f1_mean": report.get("in_domain_f1_mean"),
                "in_domain_f1_ci95": report.get("in_domain_f1_ci95"),
                "ood_refuse_recall": report.get("ood_refuse_recall"),
                "aurc": report.get("aurc"),
                "ece_answered": report.get("ece_answered"),
                "n_params": json.loads((art / "config.json").read_text()).get("n_params"),
                "learned_tau": None,
            }
            # Read learned correlator temperature if present
            try:
                from bumblebee.train import load_runtime, _unpack_runtime

                model, *_ = _unpack_runtime(load_runtime(root))
                if hasattr(model, "correlator"):
                    results[mf]["learned_tau"] = float(model.correlator.temperature)
            except Exception:
                pass
            clear_runtime_cache()
    finally:
        # Restore prior checkpoint
        clear_runtime_cache()
        if backup.exists():
            if ckpt.exists():
                ckpt.unlink()
            backup.replace(ckpt)
        if cfg_backup.exists():
            if cfg.exists():
                cfg.unlink()
            cfg_backup.replace(cfg)

    out = {
        "protocol": "short train dim=64; same-protocol retrieved eval; publish win or lose",
        "epochs": epochs,
        "dim": dim,
        "by_mass_fn": results,
    }
    path = art / "ablation_mass.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = root / "docs" / "ABLATION_MASS.md"
    lines = [
        "# SoftCorrelator mass_fn ablation",
        "",
        f"Short train ({epochs} epochs, dim={dim}). Same-protocol retrieved eval.",
        "Do not overclaim a winner unless CI95 intervals separate.",
        "",
        "| mass_fn | F1 | F1 CI95 | OOD refuse R | AURC | ECE answered | learned τ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mf, r in results.items():
        ci = r.get("in_domain_f1_ci95")
        ci_s = f"[{ci[1]:.3f},{ci[2]:.3f}]" if ci and len(ci) == 3 else "—"
        tau = r.get("learned_tau")
        tau_s = f"{tau:.3f}" if isinstance(tau, (int, float)) else "—"
        lines.append(
            f"| {mf} | {r.get('in_domain_f1_mean', float('nan')):.3f} | {ci_s} | "
            f"{r.get('ood_refuse_recall', float('nan')):.3f} | "
            f"{r.get('aurc', float('nan')):.3f} | "
            f"{r.get('ece_answered', float('nan')):.3f} | {tau_s} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
