# Architecture

## v2 product shape

**Bumblebee = CEC.** Chat, math, and short code hit a frozen skill pack or a deterministic live tool first. If those miss, CEC `ask_question` runs. No open-LM sidecar. `mode=open` hard-refuses.

```text
User question
     |
     v
 live_or_unsafe? --yes--> hard refuse
     |
     no
     v
 CEC ask_question  (mode=auto|corpus)
     |
     +-- answer (model_id=cec)
     +-- refuse (outside pack)
```

## Goal (CEC core)

Bumblebee’s core is a **transformer-free** closed-corpus correlator: local recurrent state + SoftCorrelator evidence binding + coverage-gated generative answers. It answers from a frozen pack, holds short conversation, and refuses when coverage is low.

At demo scale this is a small CEC. The architectural bet (hypothesis, not a measured 30B result) is that the same stack scales to larger data/params with **linear-time** evidence binding, cheaper inference, and much larger effective context than quadratic attention — while keeping abstention as a first-class path.

## Data flow (CEC)

```text
Frozen passages.jsonl  (+ Dolly / chat seed)
        |
        v
 LocalStateEncoder (GRUCell per token)     Question (+ history) -> LocalStateEncoder
        |                                         |
        +---- SoftCorrelator (sparsemax mass) <---+
                        |
                        v
              CoverageGate (confidence)
                 /              \
          conf >= tau        conf < tau
               |                  |
   GenerativeAnswerDecoder   OUTSIDE_DATASET
   (GRU LM on [q; evidence])   (low_coverage)
               |
       FaithfulnessCheck vs passage
          /          \
        pass        fail -> unfaithful_decode
         |
   fluent answer + span citation + passage_id
```

## Modules (`src/bumblebee/`)

| Module | Role |
| --- | --- |
| `router.py` | Intent classify + CEC `ask_hybrid` (no open LM) |
| `tokenize.py` | Closed-corpus vocab; regex word tokens; `<eos>` |
| `model.py` | CEC stack + `GenerativeAnswerDecoder` |
| `gate.py` | Decision object, soft faithfulness, token F1 |
| `train.py` | Corpus train / `ask_question` / eval |
| `hf_ingest.py` | Dolly-15k + chat seed (optional `[ingest]` extra) |
| `baselines.py` | BM25+abstain, hashing-embed+abstain |
| `metrics.py` | ECE, risk-coverage, AURC |
| `cli.py` | `ingest`, `train`, `ask`, `eval`, `bench`, `chat`, `dash`, `arch-dash` |

## SoftCorrelator math (v1)

Given query final state \(q \in \mathbb{R}^{d}\) and passage cell states \(P \in \mathbb{R}^{T \times d}\):

\[
\mathrm{logits}_t = \frac{(W_q q)^\top (W_k P_t)}{\tau}, \quad
m = \mathrm{sparsemax}(\mathrm{logits}), \quad
e = \sum_t m_t P_t
\]

## CoverageGate

\[
\mathrm{conf} = \sigma(\mathrm{MLP}([q; e])) \cdot \left(1 - \frac{H(m)}{\log T_{\mathrm{valid}}}\right)
\]

where \(T_{\mathrm{valid}}\) is the non-pad passage length (not the padded batch width).
## Scale hypothesis (architectural, not measured at 30B)

Transformers hit cost/context/forced-fluency walls that CEC targets for **grounded** tasks. Skills expand by growing the **CEC skill pack**, not by calling another chatbot.

## What we are not claiming

- That CEC already replaces frontier transformer LLMs
- That any of this is **patented** or patent-pending (see [PATENT_LETTER.md](PATENT_LETTER.md))
- That 30B-scale CEC training has been run here
- That an open-LM sidecar is part of the product (removed)
