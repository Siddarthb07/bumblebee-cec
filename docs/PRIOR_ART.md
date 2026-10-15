# Prior art checklist

This project is an engineering / measurement study. It does **not** claim to be the first abstaining QA system.

| Area | Related work | Bumblebee relationship |
| --- | --- | --- |
| Selective prediction / reject option | Chow reject rule; Geifman & El-Yaniv selective classification; risk-coverage / AURC | CoverageGate is in this family |
| RAG + abstain | Dense/sparse retrieval with score thresholds | Explicit BM25 and embedding baselines |
| Memory networks / RETRO | External memory + attention into memories | SoftCorrelator is a small learned binder, not RETRO-scale |
| SSM / Mamba / RWKV | Linear-time sequence models as LMs | LocalStateEncoder is recurrent local state only; not a foundation LM claim |
| Sparsemax attention | Martins & Astudillo sparsemax | Used as the differentiable mass function |
| Extractive QA | Span prediction (SQuAD-style) | Decoder is extractive by design |

## Reading list (starting points)

1. Vaswani et al., Attention Is All You Need (2017) — transformer baseline the PDF contrasts against.
2. Geifman & El-Yaniv, Selective Classification for Deep Neural Networks (NeurIPS 2017).
3. Martins & Astudillo, From Softmax to Sparsemax (ICML 2016).
4. Lewis et al., RAG (NeurIPS 2020) — retrieval + generation family.
5. Sukhbaatar et al., End-To-End Memory Networks (2015).

## Patent posture

See [PATENT_APPENDIX.md](PATENT_APPENDIX.md): **not pursued** for v1 after this checklist. Optional future upside only if counsel finds a narrow gap.
