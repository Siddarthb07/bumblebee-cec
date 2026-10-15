# Bumblebee CEC — Full Architecture Flowcharts

Complete system maps for training, inference, routing, and module ownership.  
Companion to [ARCHITECTURE.md](ARCHITECTURE.md). Patent posture: [PATENT_LETTER.md](PATENT_LETTER.md) (**nothing filed**).

---

## 1. System context (what Bumblebee is)

```mermaid
flowchart TB
  subgraph product [Bumblebee_product]
    User[User_question]
    CLI[CLI_chat_dash_ask]
    Router[router.ask_hybrid]
    CEC[CEC_core_transformer_free]
    Pack[Frozen_corpus_skill_pack]
    Gate[Coverage_and_faithfulness]
    OutAns[Answer_plus_citation]
    OutRef[Refuse_OUTSIDE_DATASET]
  end

  User --> CLI --> Router
  Router -->|live_or_unsafe| OutRef
  Router -->|auto_or_corpus| CEC
  Pack --> CEC
  CEC --> Gate
  Gate -->|conf_ok_and_faithful| OutAns
  Gate -->|low_coverage_or_unfaithful| OutRef
```

**Product path:** User → Router → **CEC only**. Open-LM sidecar removed.

---

## 2. End-to-end inference (full detail)

```mermaid
flowchart TD
  Q[Raw_question_plus_optional_history] --> Intent[classify_intent]
  Intent -->|live_or_unsafe| HardRefuse[Hard_refuse_live_or_unsafe]
  Intent --> ModeCheck{mode}

  ModeCheck -->|auto_or_corpus| Retrieve
  ModeCheck -->|open_removed| OpenGone[refuse_open_lm_removed]

  Retrieve[BM25_shortlist_top_passages] --> Loop

  subgraph Loop [Per_candidate_passage]
    EncQ[LocalStateEncoder_on_prompt]
    EncP[LocalStateEncoder_on_passage]
    SoftC[SoftCorrelator_sparsemax_mass]
    Cov[CoverageGate_confidence]
    EncQ --> SoftC
    EncP --> SoftC
    SoftC --> Cov
  end

  Loop --> Pick[Pick_passage_with_max_confidence]
  Pick --> LexFloor[Lexical_floors_content_support_novel_ratio]
  LexFloor --> Gen[GenerativeAnswerDecoder_GRU_LM]
  Gen --> DegCheck{Degenerate_text}
  DegCheck -->|yes| CapConf[Cap_confidence_below_threshold]
  DegCheck -->|no| Decide
  CapConf --> Decide

  Decide[gate.decide] --> Faith{Faithfulness_ok}
  Faith -->|conf_lt_tau| R1[refuse_low_coverage]
  Faith -->|empty| R2[refuse_empty_decode]
  Faith -->|unfaithful| R3[refuse_unfaithful_decode]
  Faith -->|pass| Ans[answer_plus_citation_span_passage_id]

  Ans --> Tag[Tag_mode_corpus_model_cec]
  OpenGone --> Done[Response_JSON]
  HardRefuse --> Done
  R1 --> Done
  R2 --> Done
  R3 --> Done
  Tag --> Done
```

---

## 3. CEC neural stack (forward pass)

```mermaid
flowchart LR
  subgraph inputs [Inputs]
    Qids[question_token_ids]
    Pids[passage_token_ids]
    Aids[answer_token_ids_train_only]
  end

  subgraph enc [LocalStateEncoder]
    Emb[Embedding]
    GRU[GRUCell_step_loop]
    LN[LayerNorm]
    Emb --> GRU --> LN
  end

  Qids --> enc
  Pids --> enc
  enc --> Qfinal[query_final_state_q]
  enc --> Pstates[passage_cell_states_P]

  subgraph soft [SoftCorrelator]
    Wq[W_q_linear]
    Wk[W_k_linear]
    Aff[affinity_logits]
    SpX[sparsemax_mass_m]
    Ev[evidence_e_equals_sum_m_t_P_t]
    Wq --> Aff
    Wk --> Aff
    Aff --> SpX --> Ev
  end

  Qfinal --> soft
  Pstates --> soft

  subgraph gate [CoverageGate]
    MLP[MLP_on_concat_q_e]
    Conc[mass_concentration_1_minus_H_m_over_logT]
    Conf[confidence_sigma_times_concentration]
    MLP --> Conf
    Conc --> Conf
  end

  Qfinal --> gate
  Ev --> gate
  SpX --> gate

  subgraph dec [Decoders]
    Ext[ExtractiveDecoder_start_span_citation]
    Gen[GenerativeAnswerDecoder_GRU_conditioned_on_q_e]
  end

  Pstates --> Ext
  SpX --> Ext
  Qfinal --> Gen
  Ev --> Gen
  Aids -.->|teacher_force| Gen

  Conf --> Out[Forward_outputs]
  Ext --> Out
  Gen --> Out
```

---

## 4. SoftCorrelator math (control flow)

```mermaid
flowchart TD
  q[q_in_R_d] --> Wq[W_q_q]
  P[P_in_R_T_x_d] --> Wk[W_k_P_t]
  Wq --> Dot["logits_t = dot / temperature"]
  Wk --> Dot
  Mask[passage_mask] --> Dot
  Dot --> Sp["m = sparsemax(logits)"]
  Sp --> Norm["renormalize_m_on_mask"]
  Norm --> E["e = sum_t m_t P_t"]
  Norm --> H["concentration = 1 - H(m)/log T"]
```

No full-sequence \(QK^\top\) attention over tokens. Mass is over **passage cells** only.

---

## 5. CoverageGate + decide (refusal logic)

```mermaid
flowchart TD
  qe["concat(q, e)"] --> MLP[MLP]
  MLP --> Raw["raw = sigma(MLP)"]
  Mass[m] --> Ent["H(m)"]
  Ent --> Conc["concentration"]
  Raw --> Mul["conf = raw * concentration"]
  Conc --> Mul

  Mul --> Lex["optional lexical floors"]
  Lex --> Thr{conf >= tau}

  Thr -->|no| LC[refuse_low_coverage]
  Thr -->|yes| GenText[generated_or_extractive_span]
  GenText --> Empty{non_empty}
  Empty -->|no| ED[refuse_empty_decode]
  Empty -->|yes| Faith[faithfulness_ok_vs_passage]
  Faith -->|no| UF[refuse_unfaithful_decode]
  Faith -->|yes| OK[status_answer]
```

---

## 6. Training pipeline

```mermaid
flowchart TD
  HF[HF_Dolly_plus_skill_seed] --> Ingest[hf_ingest]
  Ingest --> Passages[data/corpus/passages.jsonl]
  Ingest --> Evals[data/evals/questions.jsonl]
  Passages --> Vocab[Vocab.build]
  Evals --> Vocab
  Vocab --> DS[QADataset]
  Passages --> DS
  Evals --> DS
  DS --> Batch[collate_q_p_a]

  Batch --> CEC[CECModel.forward]
  CEC --> L1[gate_BCE_vs_answerable]
  CEC --> L2[start_CE_extractive_aux]
  CEC --> L3[gen_CE_answer_LM]
  L1 --> Loss["loss = L1 + L2 + 1.5*L3"]
  L2 --> Loss
  L3 --> Loss
  Loss --> Adam[Adam_step]
  Adam --> Ckpt[artifacts/cec.pt_plus_vocab_config]
```

Skill categories ingested into the same pack: **chat, math, code, knowledge** (+ Dolly grounded/open memories).

---

## 7. Data / skill pack composition

```mermaid
flowchart LR
  subgraph pack [Frozen_pack]
    DollyG[Dolly_grounded_contexts]
    DollyO[Dolly_open_memories]
    Chat[CHAT_SEED]
    Math[SKILL_math]
    Code[SKILL_code]
    Know[SKILL_knowledge]
    OOD[far_ood_entity_gap_evals]
  end

  DollyG --> Train[CEC_train]
  DollyO --> Train
  Chat --> Train
  Math --> Train
  Code --> Train
  Know --> Train
  OOD --> EvalRefuse[Refuse_eval]
  Train --> Model[cec.pt]
```

---

## 8. Module map (code ownership)

```mermaid
flowchart TB
  cli[cli.py] --> router[router.py]
  cli --> train[train.py]
  cli --> skills[skills_bench.py]
  dash[dash_app.py] --> router
  chat[chat.py] --> router

  router --> train_ask[train.ask_question]

  train_ask --> model[model.py_CEC]
  train_ask --> gate[gate.py]
  train --> model
  train --> data[data.py]
  train --> tok[tokenize.py]
  hf[hf_ingest.py] --> data
```

| File | Responsibility |
| --- | --- |
| `model.py` | LocalStateEncoder, SoftCorrelator, CoverageGate, ExtractiveDecoder, GenerativeAnswerDecoder, CECModel |
| `gate.py` | `decide`, faithfulness, token F1 |
| `train.py` | train loop, `ask_question`, corpus eval |
| `router.py` | intent + CEC-only product entry |
| `hf_ingest.py` | Dolly + skill seeds → JSONL (optional `[ingest]`) |
| `skills_bench.py` | chat/math/code/corpus/refuse suite |

---

## 9. Benchmark surface

```mermaid
flowchart LR
  Skills[bench_skills] --> CEC1[CEC_chat_math_code_refuse]
  All[bench_all] --> Ret[CEC_vs_BM25_embed]
  Eval[cli_eval] --> Metrics[F1_ECE_AURC_refuse_P_R]
```

---

## Reading order

1. This file — full control flow  
2. [ARCHITECTURE.md](ARCHITECTURE.md) — equations + claims  
3. [PRIOR_ART.md](PRIOR_ART.md) — known families  
4. [PATENT_LETTER.md](PATENT_LETTER.md) — invention disclosure draft (**not a granted patent**)
