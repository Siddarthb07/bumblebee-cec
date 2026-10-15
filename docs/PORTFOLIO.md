# Demo script

Short path for someone who just cloned the repo. Label the source the CLI prints. Do not describe a pack or tool answer as core CEC.

## 90 seconds, no checkpoint

```bash
python -m bumblebee.cli ask "who are you"
python -m bumblebee.cli ask "what is 17 * 19"
python -m bumblebee.cli ask "What is photosynthesis?"
python -m bumblebee.cli ask "Who won the 2022 FIFA World Cup?"
```

1. `who are you` → `source=pack`.
2. `17 * 19` → `source=tool`.
3. Photosynthesis → `source=pack`. That sentence is a curated card, not a retrieved CEC span.
4. World Cup, with no checkpoint → `reason=checkpoint_missing`. After training, the same question should refuse as outside the pack if the passage is not there.

Then open `docs/RESULTS.md`.

`python -m bumblebee.cli dash` is the same router in a browser at http://127.0.0.1:7860.

## After `train`

```bash
python -m bumblebee.cli ask "What is photosynthesis?" --mode corpus
```

`--mode corpus` still hits the knowledge pack before CEC, because the router checks packs first. A question that is not in the pack is what exercises the correlator. Expect a citation or a refuse, and read `source=`.

## What to say

- “On the sealed pack, BM25 leads in-domain F1, 0.136 to 0.122.”
- “Product skills are 20/20. Core CEC on that suite is 4/17.”
- “Nothing is patented. Prior art is listed in docs/PRIOR_ART.md.”

## What not to say

- “This replaces transformers.”
- “This is patented.”
- “The gate is calibrated.” Answered-subset ECE is about 0.915.
- “OOD recall is high.” Entity-gap and far-ood refuse recall on the bench slice is about 0.10.
