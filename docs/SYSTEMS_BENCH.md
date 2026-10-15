# Systems bench — QUARANTINED

> **Do not cite these numbers.** This file is stale orphan output from a deleted
> TinyStories / systems-bench path (`bench --mode systems` removed). Perfect
> F1=1.000 / OOD=1.000 on n=40 is not reproducible with current code and
> contradicts `docs/RESULTS.md` (CEC in-domain F1 0.131, BM25 0.133, on the sealed pack).

Regenerate nothing here until a new systems opponent and protocol exist.
See `docs/INSANE_ROADMAP.md` (P4 scrap this cycle).

---

<details><summary>Archived stale table (for archaeology only)</summary>

Examples: **40** (stratified subset of the HF eval JSONL).

| System | Params (M) | Latency ms (mean) | Latency ms (p50) | Peak RSS MB | In-domain F1 | OOD P | OOD R | Ctx tokens/q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cec` | 6.47 | 48.6 | 47.4 | 432 | 1.000 | 1.000 | 1.000 | 256 |
| `artifacts/opponent_tinystories1m` | 3.75 | 324.6 | 299.6 | 882 | 0.053 | 0.000 | 0.000 | 470 |

</details>
