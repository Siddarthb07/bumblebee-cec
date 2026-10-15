# Patent / Invention Disclosure Letter (Draft)

**Status: NOTHING IS PATENTED.**  
No provisional, utility, or PCT application has been filed for Bumblebee / CEC.  
Project policy ([PATENT_APPENDIX.md](PATENT_APPENDIX.md)): **not pursued** for the current release after prior-art review. This document is an **invention disclosure draft** you could hand to counsel later — it is **not** a patent, **not** a filing, and **not** permission to market the work as “patented.”

---

## Letter

**To:** Patent counsel (to be designated)  
**From:** Siddarth Boggarapu  
**Re:** Invention disclosure — Correlated Evidence Cells (CEC) for closed-corpus abstaining QA  
**Date:** 14 September 2026  
**Project:** Bumblebee (`github.com/Siddarthb07` / local `bumblebee`)  
**Related docs:** [ARCHITECTURE_FLOWCHART.md](ARCHITECTURE_FLOWCHART.md), [PRIOR_ART.md](PRIOR_ART.md), [ARCHITECTURE.md](ARCHITECTURE.md)

### 1. Purpose of this letter

I request a formal prior-art search and freedom-to-operate / patentability opinion on the system described below. I do **not** authorize filing until counsel confirms a **narrow, novel, non-obvious** claim set that survives the checklist in PRIOR_ART.md.

Until then, public materials must **not** say “patented,” “patent pending,” or “never done before.”

### 2. Field

Machine learning systems for question answering over a frozen document corpus, with abstention when evidence coverage is low, using a transformer-free local recurrent encoder and a differentiable sparse evidence binder.

### 3. Problem addressed

Decoder-only transformer LMs (i) pay roughly quadratic attention cost in context length, (ii) are trained to always continue text (forced fluency / hallucination under uncertainty), and (iii) often lack an explicit, differentiable link from “how concentrated is my evidence?” to “should I refuse?”

Closed-corpus assistants need the opposite contract: **answer only with bound evidence, else refuse**, at low inference cost.

### 4. Summary of the alleged invention (system-level idea)

A method and system (**CEC — Correlated Evidence Cells**) comprising:

1. Encoding a query and candidate passage tokens with a **causal local recurrent encoder** (e.g., GRUCell) that does **not** compute full-sequence self-attention \(QK^\top\) over the token sequence.
2. Binding the query’s final state to passage **cell states** via a **SoftCorrelator**: learned projections, affinity logits, **sparsemax** mass over cells, and a mass-weighted evidence vector.
3. Computing a **CoverageGate** confidence as a function of (query, evidence) and **mass concentration** (entropy of the sparsemax mass).
4. Generating or extracting an answer conditioned on evidence, then applying a **faithfulness check** against the passage; if confidence is below a threshold or faithfulness fails, emitting an explicit **OUTSIDE_DATASET / refuse** decision instead of an answer.
5. Optionally shortlisting passages with classical retrieval (e.g., BM25) before SoftCorrelator scoring.
6. Training jointly with (a) answerability / gate loss, (b) optional extractive span loss for citations, (c) generative answer LM loss on gold answers from a frozen pack (including chat / math / code skill seeds).

Figures: full control-flow diagrams in ARCHITECTURE_FLOWCHART.md §§1–7.

### 5. What is **not** claimed as novel (prior art / unpatentable building blocks)

Counsel should treat these as **known** (see PRIOR_ART.md):

| Building block | Prior art family |
| --- | --- |
| Sparsemax | Martins & Astudillo (ICML 2016) |
| Selective prediction / reject option | Chow; Geifman & El-Yaniv; risk-coverage / AURC |
| RAG + score thresholding | Lewis et al. RAG; industry retrieval gates |
| Recurrent encoders / linear-time sequence models | GRU/LSTM; SSM / Mamba / RWKV class (related, not identical) |
| Extractive span QA | SQuAD-style start/end prediction |
| Memory networks | Sukhbaatar et al.; RETRO-scale external memory |

**Removed, and not part of Bumblebee:** a third-party instruct LM behind the router (`open_lm.py`). That file is gone. `mode=open` hard-refuses. It is **excluded** from any contemplated claims.

### 6. Hypothetical claim focus (for counsel only — **not filed**)

If — and only if — a search leaves room, the **narrowest** interesting combination to stress-test is:

> A **transformer-free** QA stack that **jointly** (i) binds evidence with **sparsemax mass over recurrent passage cells**, (ii) multiplies model score by **mass-concentration** to form confidence, and (iii) **hard-refuses** on low confidence **or** failed **faithfulness**, with citations derived from mass / span decode — evaluated under risk-coverage.

That is a **composition** claim candidate, not a claim that sparsemax, GRUs, or abstention are new.

**Broader claims to avoid / weaken:**

- “Replaces transformer LLMs”
- “Unbounded open-world AGI”
- “Any abstaining chatbot”
- “Any use of sparsemax in NLP”

### 7. Embodiment (as implemented)

- Code: `src/bumblebee/model.py` (LocalStateEncoder, SoftCorrelator, CoverageGate, GenerativeAnswerDecoder, CECModel); `gate.py`; `train.py`; `router.py`.
- Data: sealed pack in `data/corpus/passages.jsonl` and `data/evals/questions.jsonl` — 11568 passages / 10693 evals (`data/sealed/SEAL.json`). Dolly is one source among WikiText-2, Cosmopedia, SmolLM slices, FineMath, FineWeb-Edu, Wikipedia, HotpotQA, and Natural Questions, plus local skill seeds.
- Scale that has actually been trained here: headline about **24.3M** parameters at dim=128 on that pack. An earlier demo was about 2–6M. Mac-fit sparse-hash runs can print a much larger headline count; that count is not dense FLOPs. None of these is a 30B training run.

### 8. Inventor / ownership

Inventor: Siddarth Boggarapu.  
No assignment to an employer is asserted in this draft; confirm with counsel before filing.

### 9. Public disclosure / bar dates

Public GitHub / portfolio discussion may start novelty clocks in some jurisdictions. Counsel should calendar bar dates before any filing decision.

### 10. Requested counsel actions

1. Prior-art search against selective prediction + sparsemax attention + RAG-abstain + memory networks.  
2. Written opinion: patentable gap **yes/no**, with claim sketch if yes.  
3. If no: confirm **do not file**; keep engineering paper / portfolio framing.  
4. If yes: provisional draft only after explicit go-ahead; budget and venue (e.g., USPTO provisional) TBD.

### 11. Explicit marketing rule

Until a filing receipt exists:

- Say: “engineering system; patent **not pursued** / disclosure draft only.”
- Do **not** say: “patented architecture,” “patent-pending LLM killer,” or “unique never-before SoftCorrelator invention” without counsel sign-off.

---

**Signed (draft):** Siddarth Boggarapu  

---

## Plain-language: what is patented vs the full idea?

| Question | Answer |
| --- | --- |
| Is any of this patented today? | **No.** Zero filings. |
| Is it “patent pending”? | **No.** |
| What is the “full idea”? | CEC: recurrent encode → SoftCorrelator (sparsemax mass) → CoverageGate → generate/cite or **refuse**. |
| What *might* be claimable later? | Only a **narrow composition** of those pieces together for closed-corpus abstaining QA — **if** counsel finds a gap. |
| What is definitely **not** yours alone? | Sparsemax, abstention, RAG, GRUs, extractive QA, “being an LLM.” |
| Is the removed open-LM path part of a patent? | **No — excluded.** The sidecar was removed. It is not Bumblebee’s inventive concept. |

**Bottom line:** The flowchart is the real architecture. The letter is a **disclosure for optional future counsel**. The product story stays: measurable engineering depth — not a patent pitch.
