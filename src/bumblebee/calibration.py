"""Post-hoc calibration for CoverageGate confidence (no weight retrain).

Fits on the true calibration residue (disjoint from train + test).
Primary metric: ece_answered (span-correct | answered). Mixed ECE is diagnostic only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from bumblebee.data import load_evals, load_passages
from bumblebee.eval_protocol import calibration_split, refuse_confusion
from bumblebee.gate import decide, token_f1
from bumblebee.metrics import expected_calibration_error
from bumblebee.tokenize import tokenize


def apply_temperature(conf: float, temperature: float) -> float:
    """Map conf in (0,1) through logit / T / sigmoid. Softens overconfident mass."""
    t = max(float(temperature), 1e-3)
    c = min(max(float(conf), 1e-6), 1.0 - 1e-6)
    logit = torch.log(torch.tensor(c / (1.0 - c)))
    scaled = torch.sigmoid(logit / t).item()
    return float(scaled)


def apply_platt(conf: float, a: float, b: float) -> float:
    """sigmoid(a * logit(conf) + b)."""
    c = min(max(float(conf), 1e-6), 1.0 - 1e-6)
    logit = math.log(c / (1.0 - c))
    return float(1.0 / (1.0 + math.exp(-(a * logit + b))))


def fit_temperature(
    confidences: Sequence[float],
    correct: Sequence[bool],
    grid: Sequence[float] | None = None,
) -> Tuple[float, float, float]:
    """
    Pick T minimizing ECE on (conf, correct).
    Returns (best_T, ece_before, ece_after).
    Grid extends past 5.0 so the old boundary artifact is visible if it recurs.
    """
    confidences = list(confidences)
    correct = list(correct)
    if not confidences:
        return 1.0, 0.0, 0.0
    before = expected_calibration_error(confidences, correct)
    grid = list(grid or [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    best_t, best_ece = 1.0, before
    for t in grid:
        scaled = [apply_temperature(c, t) for c in confidences]
        ece = expected_calibration_error(scaled, correct)
        if ece < best_ece:
            best_t, best_ece = float(t), ece
    return best_t, before, best_ece


def fit_platt(
    confidences: Sequence[float],
    correct: Sequence[bool],
) -> Tuple[float, float, float, float]:
    """
    Grid-search Platt a,b minimizing ECE. Returns (a, b, ece_before, ece_after).
    """
    confidences = list(confidences)
    correct = list(correct)
    if not confidences:
        return 1.0, 0.0, 0.0, 0.0
    before = expected_calibration_error(confidences, correct)
    best = (1.0, 0.0, before)
    for a in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        for b in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            scaled = [apply_platt(c, a, b) for c in confidences]
            ece = expected_calibration_error(scaled, correct)
            if ece < best[2]:
                best = (float(a), float(b), ece)
    return best[0], best[1], before, best[2]


@torch.no_grad()
def fit_gate_temperature(root: Path, device: str | None = None) -> dict:
    """
    Run CEC on the calibration split with BM25 retrieval (same protocol as evaluate),
    fit T (+ optional Platt) on answered-subset correctness, write artifacts.
    """
    from bumblebee.train import (
        _STOP,
        content_support,
        get_bm25_index,
        load_runtime,
        novel_content_ratio,
        _unpack_runtime,
    )

    model, vocab, passages, threshold, device, max_q, max_p, _temp, _pa, _pb, _up = _unpack_runtime(
        load_runtime(root, device)
    )
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    _train, cal, _test = calibration_split(evals)
    corpus_tokens = {t for p in passages for t in tokenize(p.text) if t not in _STOP}
    index = get_bm25_index(root, passages)

    answered_confs: List[float] = []
    answered_correct: List[bool] = []
    mixed_confs: List[float] = []
    mixed_correct: List[bool] = []

    for ex in cal:
        hits = index.score(ex.question)[:3]
        best_result = None
        best_raw = -1.0
        for h in hits:
            q_ids = torch.tensor([vocab.encode(ex.question, max_q)], dtype=torch.long, device=device)
            p_ids = torch.tensor([vocab.encode(h.text, max_p)], dtype=torch.long, device=device)
            out = model(q_ids, p_ids)
            p_toks = tokenize(h.text)[:max_p]
            conf = float(out["confidence"][0])
            if content_support(ex.question, p_toks) < 0.12 or novel_content_ratio(ex.question, corpus_tokens) >= 0.5:
                conf = conf * 0.35
            result = decide(
                passage_tokens=p_toks,
                best_start=int(out["best_start"][0]),
                best_end=int(out["best_end"][0]),
                confidence=conf,
                threshold=threshold,
                generated_answer="",
                passage_text=h.text,
                question=ex.question,
                prefer_extractive=True,
            )
            raw = float(out["confidence"][0])
            if raw > best_raw:
                best_raw = raw
                best_result = result
        if best_result is None:
            continue
        answered = best_result.status == "answer"
        cell = refuse_confusion(ex.answerable, answered)
        mixed_confs.append(best_result.confidence)
        if cell == "tp":
            mixed_correct.append(True)
        elif cell in {"fn", "fp"}:
            mixed_correct.append(False)
        else:
            ok = token_f1(best_result.answer, ex.answer or "") >= 0.5
            mixed_correct.append(ok)
            answered_confs.append(best_result.confidence)
            answered_correct.append(ok)

    # Fit on answered subset when possible; fall back to mixed if too few answers
    fit_confs = answered_confs if len(answered_confs) >= 8 else mixed_confs
    fit_correct = answered_correct if len(answered_confs) >= 8 else mixed_correct
    t, before_t, after_t = fit_temperature(fit_confs, fit_correct)
    a, b, before_p, after_p = fit_platt(fit_confs, fit_correct)

    # Reject Platt if it collapses answerability (pushes all cal confs under a soft floor)
    thr = float(threshold)
    platt_scaled = [apply_platt(c, a, b) for c in fit_confs]
    still_answerable = sum(1 for c in platt_scaled if c >= thr)
    use_platt = after_p < after_t and still_answerable >= max(3, len(fit_confs) // 5)
    if not use_platt:
        a, b = 1.0, 0.0

    report = {
        "temperature": t,
        "platt_a": a,
        "platt_b": b,
        "platt_active": use_platt,
        "platt_still_answerable": still_answerable,
        "ece_answered_before_T": before_t if len(answered_confs) >= 8 else None,
        "ece_answered_after_T": after_t if len(answered_confs) >= 8 else None,
        "ece_answered_after_platt": after_p if len(answered_confs) >= 8 else None,
        "ece_mixed_before": expected_calibration_error(mixed_confs, mixed_correct) if mixed_confs else None,
        "ece_before": before_t,
        "ece_after": after_t if not use_platt else after_p,
        "n_cal": len(cal),
        "n_answered": len(answered_confs),
        "n_scored": len(fit_confs),
        "fit_target": "answered" if len(answered_confs) >= 8 else "mixed_fallback",
        "protocol": (
            "calibration_split rem=1 disjoint from train; BM25 retrieve top-3; "
            "extractive-first; ece_answered preferred; Platt only if answerability preserved"
        ),
    }
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    (art / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cfg_path = art / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["temperature"] = t
        cfg["platt_a"] = a
        cfg["platt_b"] = b
        cfg["platt_active"] = use_platt
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return report
