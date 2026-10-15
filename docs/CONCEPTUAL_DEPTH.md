# Conceptual depth

## What this demonstrates

1. **Closed-world discipline** — answer from frozen evidence or refuse.
2. **Coverage-gated abstention** — confidence is tied to evidence-mass concentration and scored with ECE and AURC. On the sealed pack the answered-subset ECE is about 0.915, so this is not a calibrated probability.
3. **Citation faithfulness** — refuse ungrounded spans.
4. **Fair comparison** — same retrieved protocol vs BM25+abstain and embedding+abstain. No open language-model sidecar.

## What this is for

A public, runnable closed-corpus correlator with the scoreboard in the repo. It does not replace a general chatbot.

## Challenge to typical LLM behavior

Transformers are trained to continue tokens. They are not required to know when evidence is missing. Bumblebee challenges **forced answering** and **ungrounded fluency** on a bounded corpus, not next-token perplexity at web scale.

## How to talk about results

Lead with measurement, not slogans:

- “On this frozen pack, BM25 leads in-domain F1 (0.133 vs 0.131). CEC n is 497 and BM25 n is 2079. Entity-gap recall is 0.767 and far-ood recall is 0.957, precision 1.0 on that small N.”
- “That split is the point — the comparison is published, including the loss to BM25.”

See [RESULTS.md](RESULTS.md) for the table and [PORTFOLIO.md](PORTFOLIO.md) for a short demo script.
