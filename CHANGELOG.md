# Changelog

## 0.4.1 — 2026-10-16

Scoreboard and code from the research history at `fae3909` (dim=640, 127,106,806 params).

- Retrieved F1 is now CEC 0.131 (n=497) and BM25 0.133 (n=2079). BM25 still leads. CEC OOD refuse recall is 0.849.
- Core CEC on the skill questions is 10/17. Product path stays 20/20.
- Memory card is 127.1M params, peak RSS 0.717 GB, mean ask 8.6 s. Footprint only.
- Hearth is the decoder word table. It does not add parameters. `hearth-bench` is on the CLI.
- This package still does not ship `artifacts/cec.pt`. A missing checkpoint refuses with `checkpoint_missing` instead of a traceback.

## 0.4.0 — 2026-10-15

Public launch of the CEC correlator as `bumblebee-cec`.

- MIT license, security policy, contributing guide, and citation file.
- README quick start that works before training: pack and live-tool questions answer; a missing checkpoint refuses with a hint instead of a traceback.
- Docs aligned to `data/sealed/SEAL.json` (11568 passages, 10693 evals) and to `docs/RESULTS.md` (BM25 in-domain F1 0.136, CEC 0.122).
- Removed the stale "optional open LM" diagram edge. `mode=open` hard-refuses.
- Corpus slices called out as upstream-licensed data, not MIT code.
- Distribution name is `bumblebee-cec` because the PyPI name `bumblebee` is already taken. The import package and the `bumblebee` command are unchanged.

Model behavior on a trained checkpoint is the same 0.3.0 CEC stack.
