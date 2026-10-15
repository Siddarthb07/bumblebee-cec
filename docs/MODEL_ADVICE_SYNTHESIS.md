# Model advice synthesis

> **Historical advisory.** Quotes below still say F1 ~0.19 and ECE ~0.8. Those figures are from before the sealed pack in [RESULTS.md](RESULTS.md) (CEC F1 0.122, BM25 0.136, answered-subset ECE 0.915). Do not cite the quote as the current card.

Queried three models (Claude, GPT, Grok) on how to improve Bumblebee given benches at that time.
Full per-model writeups: sections below were saved from the advisory runs; see also this file’s history in chat.

## Consensus (all three agreed)

1. **Fix OOD train/test leakage** — `train_eval_split` puts every unanswerable into *both* train and test, so headline refuse P/R is partly “seen in training.” Hold out OOD; re-measure.
2. **Split the scoreboard** — Product path (packs/tools/router) vs **core CEC** (`ask_question` only). Don’t present 20/20 as learned freeform CEC.
3. **Calibrate the gate for real** — ECE ~0.8; stop post-hoc `conf = min(conf, τ−ε)` clamps that wreck ECE; temperature/Platt on a true calibration split.
4. **Prefer extractive+cite over freeform GRU decode** — BM25 still wins F1; train a real end-span head; demote generative polish.
5. **Hard negatives** — entity-gap / BM25-wrong-passage / paraphrase; far-OOD alone is too easy.
6. **Honest param card** — core ~few M dense + sparse hash table; never equate 472M sparse to dense GPT-2 Medium.
7. **Demo path that never loads 1.8GB** — pack/tool-only scripted demo; cache live APIs.

## Pitch-safe honesty line (Claude)

> Bumblebee’s measured advantage is abstention — it declines OOD questions that a 355M transformer answers anyway — while in-domain extractive F1 (~0.19) still trails BM25, confidence is not yet calibrated (ECE ~0.8), and polished skill demos come from curated packs and live tools rather than freeform generation.

## 2-week plan (merged)

| Days | Focus |
| --- | --- |
| 1–3 | Disjoint OOD holdout; re-bench refuse; label every response source (pack/tool/cec/refuse) |
| 4–6 | Calibration split + temperature scaling; separate answerability vs correctness metrics |
| 7–10 | BM25 shortlist → CEC rerank → extractive span (end pointer); hard negatives |
| 11–12 | Ablations: sparsemax vs softmax, floors on/off, small vs LargeCEC, latency/memory |
| 13–14 | Rewrite RESULTS around three tables; paraphrase skills slice; rebuild deep-dive PDF |

## What not to do (consensus)

- Lead with 20/20 or “472M ≈ GPT-2 Medium”
- Claim transformers inherently force-answer from one GPT-2 Medium run
- Spend two weeks chasing freeform GRU fluency
- Paper over demo gaps with more regex/pack entries
- Reintroduce a transformer sidecar (blurs the claim)
