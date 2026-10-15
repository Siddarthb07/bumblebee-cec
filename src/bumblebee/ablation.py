"""SoftCorrelator mass_fn ablation + linear CEC (SSM / gated-delta / distill) ablation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

from bumblebee.train import (
    clear_runtime_cache,
    evaluate,
    train_model,
    train_model_distill,
)


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


def _arm_metrics(report: dict, art: Path) -> dict:
    cfg = {}
    try:
        cfg = json.loads((art / "config.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "in_domain_f1_mean": report.get("in_domain_f1_mean"),
        "in_domain_f1_ci95": report.get("in_domain_f1_ci95"),
        "ood_refuse_recall": report.get("ood_refuse_recall"),
        "aurc": report.get("aurc"),
        "ece_answered": report.get("ece_answered"),
        "n_params": cfg.get("n_params"),
        "encoder_kind": cfg.get("encoder_kind"),
        "correlator_kind": cfg.get("correlator_kind"),
        "train_mode": cfg.get("train_mode"),
    }


def run_linear_ablation(
    root: Path,
    *,
    epochs: int = 2,
    dim: int = 64,
    max_train_examples: int = 512,
) -> dict:
    """
    Same-budget compare: classic GRU+SoftCorrelator vs SSM+gated-delta scratch
    vs SSM+gated-delta distilled from classic. Restores production cec.pt after.

    Uses a capped train subset (default 512) so the card finishes on laptop hardware;
    protocol is still same-budget across arms. Re-run with max_train_examples=0 for full.
    """
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    ckpt = art / "cec.pt"
    cfg = art / "config.json"
    backup = art / "cec.pt.linear_ablation_backup"
    cfg_backup = art / "config.json.linear_ablation_backup"
    vocab = art / "vocab.json"
    vocab_backup = art / "vocab.json.linear_ablation_backup"
    teacher = art / "cec_teacher_ablation.pt"

    train_cap = None if max_train_examples is None or max_train_examples <= 0 else int(max_train_examples)

    if ckpt.exists():
        # Never replace a larger production backup with a smaller experimental ckpt
        if not (backup.exists() and ckpt.stat().st_size < backup.stat().st_size):
            shutil.copy2(ckpt, backup)
    if cfg.exists():
        shutil.copy2(cfg, cfg_backup)
    if vocab.exists():
        shutil.copy2(vocab, vocab_backup)

    results: Dict[str, dict] = {}
    try:
        # 1) classic scratch
        clear_runtime_cache()
        train_model(
            root,
            epochs=epochs,
            dim=dim,
            encoder_kind="gru",
            correlator_kind="classic",
            ckpt_name="cec.pt",
            max_train_examples=train_cap,
        )
        shutil.copy2(ckpt, teacher)
        clear_runtime_cache()
        results["classic"] = _arm_metrics(evaluate(root, max_examples=300), art)
        clear_runtime_cache()

        # 2) linear from scratch
        train_model(
            root,
            epochs=epochs,
            dim=dim,
            encoder_kind="ssm",
            correlator_kind="gated_delta",
            ckpt_name="cec.pt",
            max_train_examples=train_cap,
        )
        clear_runtime_cache()
        results["linear_scratch"] = _arm_metrics(evaluate(root, max_examples=300), art)
        clear_runtime_cache()

        # 3) linear distill from classic teacher (same dim/vocab as arm 1)
        train_model_distill(
            root,
            teacher_path=teacher,
            epochs=epochs,
            encoder_kind="ssm",
            correlator_kind="gated_delta",
            ckpt_name="cec.pt",
            max_train_examples=train_cap,
        )
        clear_runtime_cache()
        results["linear_distill"] = _arm_metrics(evaluate(root, max_examples=300), art)
        clear_runtime_cache()

        # Keep a copy of the distilled student for inspection (do NOT clobber product cec_linear.pt)
        if ckpt.exists():
            shutil.copy2(ckpt, art / "cec_linear_ablation_student.pt")
    finally:
        clear_runtime_cache()
        if backup.exists():
            shutil.copy2(backup, ckpt)
            backup.unlink()
        if cfg_backup.exists():
            shutil.copy2(cfg_backup, cfg)
            cfg_backup.unlink()
        if vocab_backup.exists():
            shutil.copy2(vocab_backup, vocab)
            vocab_backup.unlink()
        if teacher.exists():
            teacher.unlink()

    def _f1(arm: str) -> float:
        v = results.get(arm, {}).get("in_domain_f1_mean")
        return float(v) if isinstance(v, (int, float)) else float("nan")

    f_classic, f_scratch, f_distill = _f1("classic"), _f1("linear_scratch"), _f1("linear_distill")
    prefer_distill = f_distill >= f_scratch
    beats_classic = f_distill > f_classic if prefer_distill else f_scratch > f_classic
    verdict = {
        "prefer_under_budget": "linear_distill" if prefer_distill else "linear_scratch",
        "beats_classic_f1": bool(beats_classic),
        "promote_to_product": False,
        "note": (
            "Prefer distill if it beats scratch under the same budget; "
            "promote to product only if it also beats classic on retrieved F1 or refuse quality."
        ),
    }

    out = {
        "protocol": (
            f"short train dim={dim} epochs={epochs} max_train_examples={train_cap}; "
            "eval max_examples=300; classic vs linear_scratch vs linear_distill; "
            "same-protocol retrieved eval; production cec.pt restored"
        ),
        "epochs": epochs,
        "dim": dim,
        "max_train_examples": train_cap,
        "by_arm": results,
        "verdict": verdict,
        "transformers": False,
    }
    path = art / "ablation_linear.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = root / "docs" / "ABLATION_LINEAR.md"
    lines = [
        "# Linear CEC ablation (SSM + gated-delta + distill)",
        "",
        f"Short train ({epochs} epochs, dim={dim}, max_train_examples={train_cap}). "
        "Same-protocol retrieved eval.",
        "Arms: classic GRU+SoftCorrelator, linear scratch, linear distill from classic.",
        "No transformers. Production `cec.pt` restored after the run.",
        "",
        "| arm | F1 | F1 CI95 | OOD refuse R | AURC | ECE answered | params |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, r in results.items():
        ci = r.get("in_domain_f1_ci95")
        ci_s = f"[{ci[1]:.3f},{ci[2]:.3f}]" if ci and len(ci) == 3 else "—"
        f1 = r.get("in_domain_f1_mean")
        f1_s = f"{f1:.3f}" if isinstance(f1, (int, float)) else "—"
        ood = r.get("ood_refuse_recall")
        ood_s = f"{ood:.3f}" if isinstance(ood, (int, float)) else "—"
        aurc = r.get("aurc")
        aurc_s = f"{aurc:.3f}" if isinstance(aurc, (int, float)) else "—"
        ece = r.get("ece_answered")
        ece_s = f"{ece:.3f}" if isinstance(ece, (int, float)) else "—"
        lines.append(
            f"| {arm} | {f1_s} | {ci_s} | {ood_s} | {aurc_s} | {ece_s} | "
            f"{r.get('n_params', '—')} |"
        )
    lines.extend(
        [
            "",
            f"**Under-budget prefer:** `{verdict['prefer_under_budget']}`",
            f"**Beats classic F1:** {verdict['beats_classic_f1']}",
            f"**Promote to product:** {verdict['promote_to_product']} (manual only)",
            "",
            verdict["note"],
            "",
        ]
    )
    md.write_text("\n".join(lines), encoding="utf-8")
    return out
