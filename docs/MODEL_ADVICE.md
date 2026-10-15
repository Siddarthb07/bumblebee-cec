# Model advice (raw)

> **Historical advisory.** The figures below (F1 ~0.19, ECE ~0.8, ~2M dense core, train/test OOD overlap) are from an earlier pack. The current scoreboard is [RESULTS.md](RESULTS.md): CEC in-domain F1 0.131 (n=497), BM25 0.133 (n=2079), answered-subset ECE 0.942, headline params 127.1M at dim=640. The advice about splitting the scoreboard and not equating sparse params with dense FLOPs still stands. The numbers in the quotes do not.

Advisory runs on improving Bumblebee, given benches at the time and Mac-fit LargeCEC constraints.
Synthesis: [MODEL_ADVICE_SYNTHESIS.md](MODEL_ADVICE_SYNTHESIS.md).

---

## GPT

## 1. Top 7 improvements

1. **Fix evaluation leakage and realism.** Unanswerable examples currently appear in both train and test, while evaluation can supply the gold passage. Create disjoint, topic-grouped train/calibration/test sets and evaluate the complete retrieval→CEC→gate path. Add confidence intervals across several splits.

2. **Separate product-path results.** Report tools, exact packs, knowledge packs, and freeform CEC independently. Rename 20/20 as a “product integration suite,” not a CEC model score.

3. **Prioritize extractive answer quality.** BM25 already beats CEC F1. Combine BM25 top-k retrieval with learned CEC reranking and sentence/span extraction. Make beating BM25 on genuinely held-out questions the next technical milestone.

4. **Repair calibration measurement.** Current confidence represents answerability, but ECE treats correct refusals as low-confidence errors, making the metric semantically mismatched. Track separate probabilities for answerability and answer correctness; fit temperature/isotonic calibration only on a held-out calibration split.

5. **Demote unconstrained generation.** Default to cited extraction or tightly controlled rewriting. Permit GRU freeform output only when it passes stronger entailment/copy-support checks; otherwise return the span or refuse.

6. **Train with realistic hard negatives.** Use BM25-retrieved wrong passages, entity swaps, near-domain unsupported questions, paraphrases, and partial-answer cases. These better train the gate than obvious far-OOD examples.

7. **Publish honest size/efficiency ablations.** Compare small CEC, LargeCEC, BM25, and GPT-2 on quality, latency, peak memory, active parameters, and training time. The 472M sparse-hash count is not equivalent to a dense 355M model; demonstrate whether it actually contributes.

## 2. What not to do

- Do not present 20/20 pack/tool routing as learned CEC reasoning.
- Do not claim GPT-2’s refusal failure proves transformers inherently force-answer.
- Do not claim dense-model parity from nominal parameter count.
- Do not claim long-context or linear-time superiority without measured scaling curves.
- Do not tune thresholds, packs, or examples against the final test set.
- Do not spend the next two weeks trying to make the GRU decoder broadly fluent.
- Do not reintroduce a transformer sidecar; it would blur the central claim.

## 3. Best two-week plan

**Days 1–3:** Build clean grouped train/calibration/test splits; remove OOD overlap; run the actual product path; label every response source.

**Days 4–6:** Add hard-negative evaluation and bootstrap confidence intervals. Correct calibration metrics and choose thresholds on calibration data only.

**Days 7–10:** Improve BM25 top-k→CEC reranking→extractive answer selection. Make citation support measurable rather than simple token overlap.

**Days 11–12:** Run small/large CEC ablations with latency, memory, active-parameter, F1, refusal, and risk-at-coverage results.

**Days 13–14:** Freeze artifacts and presentation. Lead with the selective grounded-QA result; show failures and pack/tool contributions explicitly.

## 4. Pitch-safe honesty line

“Bumblebee is currently a transformer-free, closed-corpus QA prototype that refuses unsupported questions reliably, while its learned freeform answer quality remains below a simple BM25 baseline and most polished skills come from explicit packs and tools.”

---

## Grok

**Lead with refuse-vs-force-answer, not 20/20 or 472M.** Skills 20/20 is packs/tools; BM25 still wins in-domain F1; ECE ~0.81 means the gate is not calibrated; ~99.5% of params are a hash table.

### 1. Top 7 (impact × feasibility)

1. Split the scoreboard (product vs core CEC).
2. Close the OOD train/test leak, then re-report refuse.
3. Temperature-scale the CoverageGate (kill ECE ~0.81).
4. Extractive+cite as the default; GRU LM as optional polish.
5. Train a real span head; use BM25 as shortlist only.
6. Demo-harden the product path (never load 1.8GB in the talk).
7. Param card + skills paraphrase slice.

### 2. What NOT to do

- Lead with 20/20 or 472M ≈ GPT-2 Medium.
- Say “calibrated” until ECE moves.
- Re-add an open-LM sidecar.
- Train more epochs hoping the GRU LM becomes fluent.
- Grow hash buckets toward a fake 1B.
- Market linear-time 30B / patent-pending / regex-malware as safety.
- Compare product-path CEC to bare GPT-2 as an architecture bake-off.

### 3. Best 2-week plan

Days 1–2 dual scoreboard + param card; 3–4 held-out OOD; 5–6 temperature scaling; 7–9 extractive-first + end-span; 10–11 paraphrases; 12–14 demo lock + honesty slide.

### 4. Pitch-safe honesty line

**Bumblebee is a coverage-gated correlator on a frozen pack plus tools — it abstains when evidence is thin, but freeform CEC decode is still weak, skills wins are mostly packs/tools, and BM25 still answers in-domain better.**

---

## Claude

**The blocker:** headline refuse numbers are measured on examples the gate trained on (`train_eval_split` puts every unanswerable into both train and test). Fix that before presenting.

### 1. Top 7

1. Split unanswerables; re-measure; report far_ood vs entity_gap separately; add hard entity_gap negatives.
2. Ablation table (sparsemax→softmax, floors off, mean-pool encoder, BM25-only, etc.).
3. Replace post-hoc conf clamps with a real calibrated gate; ECE 0.81 is partly self-inflicted.
4. End-pointer + consider SQuAD v2 (Dolly golds are abstractive; span_len=12 caps F1).
5. Shrink vanity hash emb; spend budget on wider/bi-dir core; report params-with-gradient.
6. Swap GRUCell Python loops for `nn.GRU`; scale correlator logits by 1/√d.
7. Latency/memory vs passage length plot (the transformer-free cost story).

### 2. What not to do

- Don’t headline CEC 20/20 vs GPT-2 3/20 as model capability.
- Don’t say “attention-free” (SoftCorrelator is single-step cross-attention).
- Don’t imply retrieval-free (BM25 is in the serving path).
- Don’t claim calibration / “we don’t hallucinate.”
- Don’t compare 472M to 355M as fair; lead with ~2M dense core.
- Don’t paper over demos with more regex/pack keys.
- Don’t add a transformer encoder “just for retrieval.”

### 3. Two-week plan

Week 1: honest splits, hard negatives, ablations, `nn.GRU` + cost plot.  
Week 2: end-pointer, bi-dir passage, calibration, rewrite RESULTS into three tables. Credibility items (1–3, 7) before metric items (4–6).

### 4. Pitch-safe honesty line

"Bumblebee's measured advantage is abstention — it declines out-of-distribution questions that a 355M transformer answers anyway — while its in-domain extractive F1 (~0.19) still trails a BM25 baseline, its confidence is not yet calibrated (ECE ~0.8), and its polished skill demos come from curated packs and live tools rather than freeform generation."
