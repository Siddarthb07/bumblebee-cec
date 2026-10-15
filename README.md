# Bumblebee CEC

Bumblebee is a **closed-corpus correlator**. You ask a question. It answers from a frozen pack and cites the passage, or it refuses. The binder is Correlated Evidence Cells (CEC): a GRU reads the question and the passage, a sparsemax SoftCorrelator puts mass on the cells that match, and a coverage gate decides whether that mass is enough to answer.

It is not a frontier next-token language model. There is no Hugging Face transformer in the answer path. Chat, math, and short code replies that feel polished come from a skill pack and a few live tools, then CEC if those miss.

**Standing result** on the dim=640 run (127,106,806 params), same retrieval protocol, no gold passage handed to the model: BM25 in-domain F1 **0.133** (n=2079), CEC **0.131** (n=497). BM25 still leads F1. CEC OOD refuse recall is **0.849**, BM25 **0.075**. The full card is [docs/RESULTS.md](docs/RESULTS.md). This clone does not ship `artifacts/cec.pt`. That checkpoint is in the research history via Git LFS.

## Install

Python 3.9 or newer.

```bash
git clone https://github.com/Siddarthb07/bumblebee-cec.git
cd bumblebee-cec
python -m pip install -e ".[dev]"
```

The PyPI project name `bumblebee` is already taken by a different package. This distribution is `bumblebee-cec`. The import and the command stay `bumblebee`.

```bash
python -m pip install "bumblebee-cec @ git+https://github.com/Siddarthb07/bumblebee-cec.git"
```

## Try it before you train

No checkpoint is shipped. Pack replies and live tools still run. A corpus question refuses with a hint instead of a stack trace.

```bash
python -m bumblebee.cli ask "who are you"
python -m bumblebee.cli ask "what is 17 * 19"
python -m bumblebee.cli ask "What is photosynthesis?"
python -m bumblebee.cli dash
```

`dash` serves a local chat UI at http://127.0.0.1:7860.

Those three asks are **not** core CEC. "who are you" and "What is photosynthesis?" are skill-pack replies (`source=pack`). "17 * 19" is the calculator tool (`source=tool`). The CLI prints `source=` on the answer line. Core CEC is `source=cec`, and it needs a trained checkpoint:

```bash
python -m bumblebee.cli train --epochs 4 --dim 128
python -m bumblebee.cli calibrate
python -m bumblebee.cli ask "What is photosynthesis?" --mode corpus
```

Training the sealed pack (11568 passages) on a laptop takes a while. `ingest` only hashes `data/corpus` into `MANIFEST.json`. It does not download data. The pack is already in the clone.

## What the numbers say

| Claim | Where it stands |
| --- | --- |
| In-domain F1, retrieved, no gold passage | BM25 0.133 (n=2079), CEC 0.131 (n=497). BM25 still leads. [docs/RESULTS.md](docs/RESULTS.md) |
| Product skills | 20/20. That is packs and tools. |
| Core CEC on the same skill questions | 10/17. [docs/SKILLS_BENCH.md](docs/SKILLS_BENCH.md) |
| Memory | 127.1M params, dim=640, peak RSS 0.717 GB, mean ask 8.6 s. Footprint only. [docs/MEMORY_BENCH.md](docs/MEMORY_BENCH.md) |
| Decoder word table | Hearth. Int8 table plus a certified residual. Same parameter count. [docs/HEARTH_BENCH.md](docs/HEARTH_BENCH.md) |
| Versus gpt2-medium | Skills wins are mostly packs and tools. Held-out quality is not a CEC win. [docs/LLM_COMPARE.md](docs/LLM_COMPARE.md) |
| Calibration | Answered-subset ECE is about 0.942. The gate is not calibrated. |
| OOD refuse on entity-gap / far-ood holdouts | Precision 1.0 on small N. Recall 0.767 entity-gap, 0.957 far-ood. |
| `docs/SYSTEMS_BENCH.md` | Quarantined. Do not cite. |
| Patent | Nothing filed. [docs/PATENT_LETTER.md](docs/PATENT_LETTER.md) |
| Scale to 30B | Architectural hypothesis. Not a measured run. |

Sealed pack (`data/sealed/SEAL.json`): **11568 passages / 10693 evals** (8500 train, 2193 test, 2114 calibration). Seal SHA-256 prefix `525f9d66…`. That seal checks that the files did not drift. It is not external validity.

Sources, download only, no transformer weights: Dolly, WikiText-2, Cosmopedia-v2, a SmolLM slice, FineMath, FineWeb-Edu, Wikipedia, HotpotQA, Natural Questions. Upstream licenses are in [NOTICE.md](NOTICE.md). The MIT license is for the code.

## CLI

| Command | What it does |
| --- | --- |
| `ask` | One question. Prints `source=` so you can see pack, tool, knowledge, or cec. |
| `chat` | Terminal chat. Requires a checkpoint. |
| `dash` | Local web UI on port 7860. |
| `arch-dash` | Architecture monitor on port 7861. |
| `train` / `calibrate` / `eval` | Fit CEC, temperature, held-out eval. |
| `hearth-bench` | Check the decoder word table against the dense argmax. |
| `bench --mode all` | Rewrites the scoreboard docs from artifacts. |
| `ingest` | Hash the corpus that is already on disk. |
| `ingest-assets` | Rebuild the multi-corpus pack. Needs `pip install -e ".[ingest]"` and a datasets download. |
| `seal` | Rewrite `data/sealed/SEAL.json` from the current files. |

`python -m bumblebee.cli --help` lists the rest.

## Docs

Start at [docs/INDEX.md](docs/INDEX.md). It says which file is current and which file is historical.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and questions use the issue templates. Vulnerabilities go through [SECURITY.md](SECURITY.md).

## License

Code: [MIT](LICENSE). Corpus text: [NOTICE.md](NOTICE.md).

Earlier research history, before this release packaging, is at [Siddarthb07/bumblebee](https://github.com/Siddarthb07/bumblebee).
