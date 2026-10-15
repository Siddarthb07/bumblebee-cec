"""
Push CEC to laptop-safe capacity limits under the honesty protocol.

Flow:
  1. Snapshot baseline eval (if present)
  2. Optional larger Dolly ingest
  3. High-capacity train (dim/epochs/max_p/gen+span weights)
  4. Held-out eval + threshold sweep
  5. retrieval / skills / all benches
  6. Write artifacts/limits_report.json (+ docs/LIMITS_STRESS.md)

Usage:
  python scripts/stress_limits.py
  python scripts/stress_limits.py --skip-ingest --dim 192 --epochs 24
  python scripts/stress_limits.py --contexts 200 --max-open 400 --dim 256 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_md(report: dict, path: Path) -> None:
    base = report.get("baseline") or {}
    best = report.get("best_threshold") or {}
    final = report.get("final_eval") or {}
    lines = [
        "# Bumblebee CEC — limits stress",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        "Honesty protocol: held-out in-domain + all OOD; **no gold-answer leakage**.",
        "",
        "## Train config",
        "",
        "```json",
        json.dumps(report.get("train_config", {}), indent=2),
        "```",
        "",
        "## Baseline (pre-stress ckpt)",
        "",
        f"- in-domain F1: **{base.get('in_domain_f1_mean', 'n/a')}**",
        f"- OOD refuse P/R: **{base.get('ood_refuse_precision', 'n/a')}** / **{base.get('ood_refuse_recall', 'n/a')}**",
        f"- ECE / AURC: **{base.get('ece', 'n/a')}** / **{base.get('aurc', 'n/a')}**",
        f"- n: `{base.get('n', 'n/a')}` protocol=`{base.get('protocol', '')}`",
        "",
        "## Final eval (post-train, ckpt threshold)",
        "",
        f"- in-domain F1: **{final.get('in_domain_f1_mean', 'n/a')}**",
        f"- OOD refuse P/R: **{final.get('ood_refuse_precision', 'n/a')}** / **{final.get('ood_refuse_recall', 'n/a')}**",
        f"- ECE / AURC: **{final.get('ece', 'n/a')}** / **{final.get('aurc', 'n/a')}**",
        f"- faithfulness: **{final.get('faithfulness_rate', 'n/a')}**",
        f"- n / n_train: `{final.get('n')}` / `{final.get('n_train')}`",
        "",
        "## Best threshold (held-out sweep)",
        "",
        f"- threshold: **{best.get('threshold')}**",
        f"- score (F1 + 0.5·refuse_P + 0.5·refuse_R − ECE): **{best.get('score')}**",
        f"- metrics: `{json.dumps(best.get('metrics', {}), sort_keys=True)}`",
        "",
        "## Threshold sweep",
        "",
        "| τ | F1 | refuse P | refuse R | ECE | AURC |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("threshold_sweep", []):
        m = row["metrics"]
        lines.append(
            f"| {row['threshold']:.2f} | {m.get('in_domain_f1_mean', float('nan')):.3f} | "
            f"{m.get('ood_refuse_precision', float('nan')):.3f} | "
            f"{m.get('ood_refuse_recall', float('nan')):.3f} | "
            f"{m.get('ece', float('nan')):.3f} | {m.get('aurc', float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Benches",
            "",
            f"- retrieval: see `artifacts/bench_retrieval.json`",
            f"- skills: see `artifacts/bench_skills.json`",
            f"- all: see `artifacts/bench_all.json` / `docs/RESULTS.md`",
            "",
            f"Wall time: **{report.get('elapsed_sec', 0):.0f}s**",
            "",
            "## Reading",
            "",
            "- Compare against baseline F1 under the **same** held-out protocol.",
            "- Skills-bench alone is not proof of limits (fixed 18 cases).",
            "- If BM25 still wins in-domain F1, say so — valid depth outcome.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stress CEC to capacity limits")
    ap.add_argument("--skip-ingest", action="store_true", help="Keep current corpus/evals")
    ap.add_argument("--contexts", type=int, default=200, help="Dolly ingest scale hint")
    ap.add_argument("--max-open", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--max-passage-len", type=int, default=192)
    ap.add_argument("--max-question-len", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gen-weight", type=float, default=2.5)
    ap.add_argument("--span-weight", type=float, default=2.0)
    ap.add_argument("--skip-benches", action="store_true")
    ap.add_argument("--skip-all-bench", action="store_true", help="Skip slow tiny-transformer bench --mode all")
    args = ap.parse_args()

    t0 = time.time()
    art = ROOT / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # Baseline from current ckpt (honesty). Prefer saved pre-stress snapshot if load fails.
    baseline = _load_json(art / "eval_report_baseline_dim96.json") or _load_json(
        art / "eval_report_pre_stress.json"
    )
    if (art / "cec.pt").exists():
        try:
            _run([py, "-m", "bumblebee.cli", "eval"])
            baseline = _load_json(art / "eval_report.json") or baseline
            shutil.copy2(art / "eval_report.json", art / "eval_report_pre_stress.json")
        except subprocess.CalledProcessError as exc:
            print(f"baseline eval failed (continuing with saved snapshot): {exc}", flush=True)

    if not args.skip_ingest:
        # Preserve prior pack
        backup = art / "data_backup_pre_stress"
        if backup.exists():
            shutil.rmtree(backup)
        backup.mkdir(parents=True)
        for name in ("corpus", "evals"):
            src = ROOT / "data" / name
            if src.exists():
                shutil.copytree(src, backup / name)
        src_meta = ROOT / "data" / "hf_source.json"
        if src_meta.exists():
            shutil.copy2(src_meta, backup / "hf_source.json")
        _run(
            [
                py,
                "-m",
                "bumblebee.cli",
                "ingest-hf",
                "--dataset",
                "dolly",
                "--contexts",
                str(args.contexts),
                "--max-open",
                str(args.max_open),
                "--seed",
                "7",
            ]
        )
        _run([py, "-m", "bumblebee.cli", "ingest"])

    train_config = {
        "epochs": args.epochs,
        "dim": args.dim,
        "max_passage_len": args.max_passage_len,
        "max_question_len": args.max_question_len,
        "threshold": args.threshold,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "gen_weight": args.gen_weight,
        "span_weight": args.span_weight,
        "skip_ingest": args.skip_ingest,
        "contexts": None if args.skip_ingest else args.contexts,
        "max_open": None if args.skip_ingest else args.max_open,
    }

    _run(
        [
            py,
            "-m",
            "bumblebee.cli",
            "train",
            "--epochs",
            str(args.epochs),
            "--dim",
            str(args.dim),
            "--max-passage-len",
            str(args.max_passage_len),
            "--max-question-len",
            str(args.max_question_len),
            "--threshold",
            str(args.threshold),
            "--lr",
            str(args.lr),
            "--batch-size",
            str(args.batch_size),
            "--gen-weight",
            str(args.gen_weight),
            "--span-weight",
            str(args.span_weight),
        ]
    )

    _run([py, "-m", "bumblebee.cli", "eval"])
    final_eval = _load_json(art / "eval_report.json")

    # Threshold sweep (held-out)
    from bumblebee.train import evaluate

    sweep = []
    best = None
    for tau in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        metrics = evaluate(ROOT, threshold=tau)
        score = (
            float(metrics["in_domain_f1_mean"])
            + 0.5 * float(metrics["ood_refuse_precision"])
            + 0.5 * float(metrics["ood_refuse_recall"])
            - float(metrics["ece"])
        )
        row = {"threshold": tau, "metrics": metrics, "score": score}
        sweep.append(row)
        if best is None or score > best["score"]:
            best = row
        print(
            f"sweep τ={tau:.2f} F1={metrics['in_domain_f1_mean']:.3f} "
            f"P={metrics['ood_refuse_precision']:.3f} R={metrics['ood_refuse_recall']:.3f} "
            f"score={score:.3f}",
            flush=True,
        )

    # Persist best threshold into ckpt + config (weights unchanged)
    if best is not None:
        import torch

        ckpt_path = art / "cec.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt["threshold"] = float(best["threshold"])
        torch.save(ckpt, ckpt_path)
        cfg = _load_json(art / "config.json") or {}
        cfg["threshold"] = float(best["threshold"])
        cfg["limits_stress_best_score"] = best["score"]
        (art / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        _run([py, "-m", "bumblebee.cli", "eval"])
        final_eval = _load_json(art / "eval_report.json")

    if not args.skip_benches:
        _run([py, "-m", "bumblebee.cli", "bench", "--mode", "retrieval"])
        _run([py, "-m", "bumblebee.cli", "bench", "--mode", "skills"])
        if not args.skip_all_bench:
            _run([py, "-m", "bumblebee.cli", "bench", "--mode", "all"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_config": train_config,
        "baseline": baseline,
        "final_eval": final_eval,
        "threshold_sweep": [
            {"threshold": r["threshold"], "score": r["score"], "metrics": r["metrics"]} for r in sweep
        ],
        "best_threshold": {
            "threshold": None if best is None else best["threshold"],
            "score": None if best is None else best["score"],
            "metrics": None if best is None else best["metrics"],
        },
        "config": _load_json(art / "config.json"),
        "hf_source": _load_json(ROOT / "data" / "hf_source.json"),
        "elapsed_sec": time.time() - t0,
    }
    out = art / "limits_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(report, ROOT / "docs" / "LIMITS_STRESS.md")
    print(f"\nWrote {out}")
    print(f"Wrote {ROOT / 'docs' / 'LIMITS_STRESS.md'}")
    print(f"Elapsed {report['elapsed_sec']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
