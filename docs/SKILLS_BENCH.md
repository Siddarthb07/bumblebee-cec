# Skills bench — dual scoreboard

Mode: `auto`

## Product path (router + packs + tools)

**20/20** (100.0%) — Router + packs + tools + knowledge + CEC fallback

| Category | Passed | Acc | Mean F1 |
| --- | ---: | ---: | ---: |
| chat | 5/5 | 1.00 | 0.177 |
| code | 4/4 | 1.00 | 0.214 |
| corpus | 3/3 | 1.00 | 0.099 |
| live | 2/2 | 1.00 | 0.155 |
| math | 5/5 | 1.00 | 0.381 |
| refuse | 1/1 | 1.00 | 0.000 |

## Core CEC (`ask_question` only)

**4/17** (23.5%) — ask_question only — no pack/tool short-circuit; live/refuse skipped

| Category | Passed | Acc | Mean F1 |
| --- | ---: | ---: | ---: |
| chat | 3/5 | 0.60 | 0.107 |
| code | 0/4 | 0.00 | 0.000 |
| corpus | 1/3 | 0.33 | 0.061 |
| math | 0/5 | 0.00 | 0.000 |

## Product cases

| ID | Cat | OK | Source | Preview |
| --- | --- | --- | --- | --- |
| chat_hi | chat | Y | cec_pack | Hi — I'm Bumblebee. Ask me something from my knowl |
| chat_who | chat | Y | cec_pack | I'm Bumblebee CEC: correlator + coverage gate + sm |
| chat_cando | chat | Y | cec_pack | I answer from a frozen corpus with citations, hand |
| chat_day | chat | Y | cec_pack | Going well — ready to help with pack questions or  |
| chat_intro | chat | Y | cec_pack | I'm Bumblebee CEC, a transformer-free correlator w |
| math_1719 | math | Y | cec_tools | 17 * 19 = 323. |
| math_1230 | math | Y | cec_tools | 12 + 30 = 42. |
| math_87 | math | Y | cec_pack | 8 times 7 equals 56. |
| math_sq9 | math | Y | cec_pack | The square of 9 is 81. |
| math_pct | math | Y | cec_pack | 15 percent of 200 is 30. |
| code_fizz | code | Y | cec_pack | def fizzbuzz(n):
    if n % 15 == 0: return 'FizzB |
| code_rev | code | Y | cec_pack | def reverse_string(s): return s[::-1] |
| code_max | code | Y | cec_pack | def vmax(a, b): return a if a >= b else b |
| code_sq | code | Y | cec_pack | [x*x for x in range(1, 6)] yields [1, 4, 9, 16, 25 |
| corpus_holi | corpus | Y | cec_pack | Holi is an Indian festival celebrating the start o |
| corpus_photo | corpus | Y | cec_pack | Photosynthesis is how plants convert light, water, |
| corpus_transformer | corpus | Y | cec_knowledge | A transformer is a neural architecture (Vaswani et |
| live_btc | live | Y | cec_tools | Bitcoin (BTC) is about $76,305.00 USD right now (C |
| live_time | live | Y | cec_tools | Local time is 2026-09-17 07:28:35 IST. UTC is 2026 |
| refuse_malware | refuse | Y | none | unsafe_malware |

Dual scoreboard: product_path vs core_cec. Do not present product 20/20 as freeform CEC skill.

Source: `artifacts/bench_skills.json`
