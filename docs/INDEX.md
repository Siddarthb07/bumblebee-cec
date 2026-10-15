# Documentation index

`data/sealed/SEAL.json` is the count lock. `docs/RESULTS.md` is the quality lock.
If a sentence disagrees with those two files, the sentence is stale.

| Doc | Standing |
| --- | --- |
| [RESULTS.md](RESULTS.md) | Current same-protocol scoreboard. BM25 leads in-domain F1 (0.133 vs 0.131, n=2079 vs n=497). |
| [SKILLS_BENCH.md](SKILLS_BENCH.md) | Current dual scoreboard. Product path 20/20. Core CEC 10/17. |
| [MEMORY_BENCH.md](MEMORY_BENCH.md) | Footprint only. 127.1M params, dim=640, peak RSS 0.717 GB. Not answer quality. |
| [HEARTH_BENCH.md](HEARTH_BENCH.md) | Decoder word table. Same 127,106,806 params. |
| [LLM_COMPARE.md](LLM_COMPARE.md) | One CPU run against gpt2-medium. Skills wins are mostly packs and tools. Held-out quality is not a CEC win. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Current product shape. No open-LM sidecar. 30B is a hypothesis. |
| [ARCHITECTURE_FLOWCHART.md](ARCHITECTURE_FLOWCHART.md) | Current flow. `mode=open` refuses. |
| [HOW_IT_WORKS.md](HOW_IT_WORKS.md) | Walkthrough and diagrams. |
| [INSANE_ROADMAP.md](INSANE_ROADMAP.md) | Scale notes for this pack. Not a ship promise. |
| [PRIOR_ART.md](PRIOR_ART.md) | Related work. This is not a first-abstention claim. |
| [PATENT_LETTER.md](PATENT_LETTER.md) | Counsel draft. **Nothing filed.** |
| [PATENT_APPENDIX.md](PATENT_APPENDIX.md) | Same posture, short form. |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | Unresolved. Not shipped behavior. |
| [MESSAGING.md](MESSAGING.md) | Telegram stub. WhatsApp is not shipped. |
| [MODEL_ADVICE.md](MODEL_ADVICE.md) | Historical advisory. Numbers in it predate this sealed pack. |
| [MODEL_ADVICE_SYNTHESIS.md](MODEL_ADVICE_SYNTHESIS.md) | Same advisory, merged. Use RESULTS for figures. |
| [SYSTEMS_BENCH.md](SYSTEMS_BENCH.md) | **Quarantined. Do not cite.** |
| [CONCEPTUAL_DEPTH.md](CONCEPTUAL_DEPTH.md) | How to talk about the measured split. |
| [PORTFOLIO.md](PORTFOLIO.md) | Short demo script. Pack answers are labeled as pack answers. |

Generated PDFs are local builds (`python scripts/build_depth_pdf.py`). They are not the scoreboard.
