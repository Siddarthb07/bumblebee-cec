# Changelog

## 0.4.0 — 2026-10-15

Public launch of the CEC correlator as `bumblebee-cec`.

- MIT license, security policy, contributing guide, and citation file.
- README quick start that works before training: pack and live-tool questions answer; a missing checkpoint refuses with a hint instead of a traceback.
- Docs aligned to `data/sealed/SEAL.json` (11568 passages, 10693 evals) and to `docs/RESULTS.md` (BM25 in-domain F1 0.136, CEC 0.122).
- Removed the stale "optional open LM" diagram edge. `mode=open` hard-refuses.
- Corpus slices called out as upstream-licensed data, not MIT code.
- Distribution name is `bumblebee-cec` because the PyPI name `bumblebee` is already taken. The import package and the `bumblebee` command are unchanged.

Model behavior on a trained checkpoint is the same 0.3.0 CEC stack.
