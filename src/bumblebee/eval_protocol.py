"""Shared evaluation protocol helpers — refuse metrics and held-out splits."""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Tuple

from bumblebee.data import EvalExample
from bumblebee.gate import AskResult, token_f1


def is_holdout(example_id: str, *, holdout_mod: int = 5, holdout_rem: int = 0) -> bool:
    """Deterministic held-out membership from example id (stable across runs)."""
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % holdout_mod == holdout_rem


def train_eval_split(
    evals: Sequence[EvalExample],
    *,
    holdout_mod: int = 5,
    holdout_rem: int = 0,
) -> Tuple[List[EvalExample], List[EvalExample]]:
    """
    Honest disjoint split (no train/test leakage on OOD).

    Train: non-holdout in-domain answerable + non-holdout unanswerable (gate signal).
    Test:  holdout in-domain answerable + holdout unanswerable.

    Previously every unanswerable was in *both* lists, inflating refuse metrics.
    """
    train: List[EvalExample] = []
    test: List[EvalExample] = []
    for ex in evals:
        held = is_holdout(ex.id, holdout_mod=holdout_mod, holdout_rem=holdout_rem)
        if not ex.answerable:
            (test if held else train).append(ex)
            continue
        if ex.split != "in_domain":
            # Answerable but tagged OOD/entity — treat as test-only probes
            test.append(ex)
            continue
        (test if held else train).append(ex)
    return train, test


def calibration_split(
    evals: Sequence[EvalExample],
    *,
    holdout_mod: int = 5,
    test_rem: int = 0,
    cal_rem: int = 1,
) -> Tuple[List[EvalExample], List[EvalExample], List[EvalExample]]:
    """
    Train / calibration / test with disjoint membership.
    Calibration uses a different hash residue than test.
    Training MUST use this train set (not train_eval_split) when fitting calibration,
    otherwise residue cal_rem leaks into the fitted temperature/Platt params.
    """
    train: List[EvalExample] = []
    cal: List[EvalExample] = []
    test: List[EvalExample] = []
    for ex in evals:
        digest = hashlib.sha256(ex.id.encode("utf-8")).hexdigest()
        rem = int(digest[:8], 16) % holdout_mod
        if not ex.answerable or ex.split != "in_domain":
            if rem == test_rem:
                test.append(ex)
            elif rem == cal_rem:
                cal.append(ex)
            else:
                train.append(ex)
            continue
        # in-domain answerable
        if rem == test_rem:
            test.append(ex)
        elif rem == cal_rem:
            cal.append(ex)
        else:
            train.append(ex)
    return train, cal, test


def refuse_confusion(answerable: bool, answered: bool) -> str:
    """
    Positive class = refuse on unanswerable (OOD).
    Returns one of: tp, fn, fp, tn.
    """
    if not answerable:
        return "tp" if not answered else "fn"
    return "tn" if answered else "fp"


def summarize_ask_results(
    pairs: Iterable[Tuple[EvalExample, AskResult]],
) -> dict:
    """
    Score ask results without gold leakage — gold only used for F1 after the fact.

    ECE is reported two ways:
    - ece_mixed: legacy (correct refuse @ low conf counts as correct) — diagnostic only
    - ece_answered: P(correct | answered) vs confidence on the answered subset only
    """
    confidences: List[float] = []
    correct_mixed: List[bool] = []
    answered_confs: List[float] = []
    answered_correct: List[bool] = []
    risks: List[float] = []
    f1s: List[float] = []
    tp = fp = fn = tn = 0
    by_split: dict = {}

    for ex, result in pairs:
        confidences.append(result.confidence)
        answered = result.status == "answer"
        cell = refuse_confusion(ex.answerable, answered)
        split_key = ex.split if not ex.answerable else "in_domain"
        bucket = by_split.setdefault(
            split_key, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "f1s": []}
        )
        if cell == "tp":
            tp += 1
            bucket["tp"] += 1
            correct_mixed.append(True)
            risks.append(0.0)
        elif cell == "fn":
            fn += 1
            bucket["fn"] += 1
            correct_mixed.append(False)
            risks.append(1.0)
        elif cell == "fp":
            fp += 1
            bucket["fp"] += 1
            correct_mixed.append(False)
            risks.append(1.0)
        else:
            tn += 1
            bucket["tn"] += 1
            f1 = token_f1(result.answer, ex.answer or "")
            f1s.append(f1)
            bucket["f1s"].append(f1)
            ok = f1 >= 0.5
            correct_mixed.append(ok)
            risks.append(0.0 if ok else 1.0)
            answered_confs.append(result.confidence)
            answered_correct.append(ok)

    from bumblebee.metrics import aurc, bootstrap_ci, expected_calibration_error

    split_report = {}
    for k, b in by_split.items():
        split_report[k] = {
            "ood_refuse_precision": b["tp"] / max(b["tp"] + b["fp"], 1),
            "ood_refuse_recall": b["tp"] / max(b["tp"] + b["fn"], 1),
            "in_domain_f1_mean": float(sum(b["f1s"]) / max(len(b["f1s"]), 1)) if b["f1s"] else None,
            "n": b["tp"] + b["fp"] + b["fn"] + b["tn"],
        }

    f1_ci = bootstrap_ci(f1s)
    ece_answered = (
        expected_calibration_error(answered_confs, answered_correct) if answered_confs else None
    )
    return {
        "in_domain_f1_mean": float(sum(f1s) / max(len(f1s), 1)),
        "in_domain_f1_n": len(f1s),
        "in_domain_f1_ci95": list(f1_ci) if f1_ci else None,
        "ood_refuse_precision": tp / max(tp + fp, 1),
        "ood_refuse_recall": tp / max(tp + fn, 1),
        "ood_refuse_tp": tp,
        "ood_refuse_fp": fp,
        "ood_refuse_fn": fn,
        "ood_refuse_tn": tn,
        "ece": expected_calibration_error(confidences, correct_mixed),  # legacy alias
        "ece_mixed": expected_calibration_error(confidences, correct_mixed),
        "ece_answered": ece_answered,
        "ece_answered_n": len(answered_confs),
        "aurc": aurc(risks, confidences),
        "by_split": split_report,
        "protocol": "disjoint_holdout; no gold leakage; ece_answered on answered subset",
    }
