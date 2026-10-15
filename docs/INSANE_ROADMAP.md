# Bumblebee roadmap — CEC asset pack scale

**Status (2026-09-19):** sealed multi-corpus asset pack live — **11568 passages / 10693 evals** (8500 train · 2193 test · 2114 cal; seal in `data/sealed/SEAL.json`) from Dolly + WikiText-2 + Cosmopedia-v2 (via `smollm-corpus/cosmopedia-v2`) + SmolLM fineweb-edu slice + FineMath-3plus + FineWeb-Edu + Wikipedia + HotpotQA + NQ. Train via `scripts/train_asset_pack.py` (OMP/MKL threads=1, dim=128, epochs=4). **No** transformer weights.

## Decision

Grow **CEC only** — SoftCorrelator + CoverageGate + cite/refuse. **No** HuggingFace transformer weights, **no** next-token SLM sidecar.

## Asset sources (laptop caps)

| Source | Cap |
| --- | --- |
| Dolly-15k + skill seed | ~800–1000 grounded |
| WikiText-2 raw | ~800 chunks |
| Cosmopedia-v2 | ~800 chunks |
| SmolLM fineweb-edu-dedup | ~600 chunks |
| FineMath-3plus | ~400 chunks |
| FineWeb-Edu (stream) | ~2500 chunks |
| Wikipedia EN (stream) | ~2500 chunks |
| HotpotQA contexts | ~2000 passages + multihop evals |
| Natural Questions (stream) | ~1000 chunks + short-answer evals |

Target after dedupe: ~10k–15k passages on laptop caps (full FineWeb-Edu/NQ are trillion-token / multi-GB — stream only). Entity-gap holdout ≥30; adversarial suite regenerated. Memory scale: `bench --mode memory` → `docs/MEMORY_BENCH.md`.

Excluded: C4, Dolma, OSCAR, The Stack, OMol, AG News, IMDB, MNIST, CIFAR.

## Commands

```bash
python scripts/train_asset_pack.py
python -m bumblebee.cli ingest-assets
python -m bumblebee.cli dash
```

## Honesty

Same-protocol scoreboard on the scaled pack (see `docs/RESULTS.md`): **BM25 leads CEC on in-domain F1**; CEC keeps higher OOD refuse precision on entity-gap / far-ood holdouts (N≥30 entity-gap). Product packs/tools ≠ core CEC skill. Scaled pack ≠ frontier LLM.
