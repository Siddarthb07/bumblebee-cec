# Hearth benchmark

Hearth uses a short list only when the residual bound certifies that the dense winner is inside it.
Real heads score an int8 copy, then add the exact rounding residual on the coordinates where this hidden state is largest. A bound covers the quiet coordinates, and the rows that can still win are dotted in full precision. Each word carries a seal: pulse, sketch, or dense. The id matches the dense head. On 13 GPT-2 steps (France, a code prompt, and punctuation) the pulse seal matched 13/13, with 1 to 11 live rows.
A frozen cap misses on code, punctuation, and longer runs, so that path stays off.
Agreement is against the dense model on the same weights.
Attention and MLP layers still need every output channel. Those matrices are int8, read in place for one token.

## Word table

| case | short list | agree | rows read | dense rows | hearth ms | dense ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| low-rank 32768x384 | True | 1.000 | 40 | 32768 | 8.36 | 4.44 |
| random 32768x384 | False | 1.000 | 32768 | 32768 | 5.34 | 5.34 |

low-rank 32768x384: short list of 40 rows, same winning word, 8.36 ms vs dense 4.44 ms.
random 32768x384: full table, same winning word, 5.34 ms vs dense 5.34 ms.

## MLP-shaped linear

Cosine vs the dense output: **1.000**. full table, every output row kept Bytes 16777216 vs dense 16777216.

## Trained tiny transformer

Pattern `next = (prev + 3) mod vocab` on vocab 256, dim 64, 2 layers, n=24. Dense gold accuracy 1.000. Hearth gold accuracy 1.000. Greedy agreement 1.000. Gold token inside the shortlist 1.000. 8-token generate: dense 23.2 ms, Hearth 34.9 ms.

## End-to-end decoder (random weights)

vocab 8192, dim 256, 2 layers, head cap 256. Dense step bytes 14681088 (head alone 8388608). Cold head bytes for 1 token 0. Warm head bytes across 4 tokens 0. 8-token generate: dense 51.2 ms, Hearth 47.0 ms. Head cache resident 0 bytes. This random model has no safe short list, so the word table is the full matmul.

## Language models

gpt2 head 50257 x 768, sketch rank 256 (52252672 resident bytes), exact cap 50257. First-token match 12/12. Short list trusted: False. Worst sketch rank 1609. Eight new tokens matching dense greedy: 8/8. Stress steps (code, punctuation, a fence) 32/32. Retimed on the int8 table plus an exact check of the live rows: 36/36 teacher-forced steps matched, one head step 10.2 ms vs dense 15.7 ms.

Dense: The capital of France is the capital of the French Republic, and

Hearth: The capital of France is the capital of the French Republic, and

distilgpt2 head 50257 x 768, sketch rank 256 (52252672 resident bytes), exact cap 50257. First-token match 12/12. Short list trusted: False. Worst sketch rank 391. Eight new tokens matching dense greedy: 8/8. Stress steps (code, punctuation, a fence) 32/32. Head-row bytes during that generate 1235116032 versus 154389504 per dense step (8 steps would read 1235116032). One head step 18.91 ms vs dense 18.91 ms.

Dense: The capital of France is the capital of the French Republic. The

Hearth: The capital of France is the capital of the French Republic. The

openai-community/gpt2-medium head 50257 x 1024, sketch rank 256 (52515840 resident bytes), exact cap 50257. First-token match 12/12. Short list trusted: False. Worst sketch rank 1687. Eight new tokens matching dense greedy: 8/8. Stress steps (code, punctuation, a fence) 32/32. Head-row bytes during that generate 1646821376 versus 205852672 per dense step (8 steps would read 1646821376). One head step 21.31 ms vs dense 21.31 ms.

Dense: The capital of France is Paris, and the capital of France is

Hearth: The capital of France is Paris, and the capital of France is

HuggingFaceTB/SmolLM-360M head 49152 x 960, sketch rank 256 (51318528 resident bytes), exact cap 49152. First-token match 12/12. Short list trusted: False. Worst sketch rank 9792. Eight new tokens matching dense greedy: 8/8. Stress steps (code, punctuation, a fence) 32/32. Head-row bytes during that generate 1509949440 versus 188743680 per dense step (8 steps would read 1509949440). One head step 20.10 ms vs dense 20.10 ms.

Dense: The capital of France is Paris.
The capital of France is

Hearth: The capital of France is Paris.
The capital of France is

facebook/opt-350m head 50272 x 512, sketch rank 170 (34535168 resident bytes), exact cap 50272. First-token match 12/12. Short list trusted: False. Worst sketch rank 8481. Eight new tokens matching dense greedy: 8/8. Stress steps (code, punctuation, a fence) 32/32. Head-row bytes during that generate 823656448 versus 102957056 per dense step (8 steps would read 823656448). One head step 6.79 ms vs dense 6.79 ms.

Dense: </s>The capital of France is Paris.
I'm not sure if

Hearth: </s>The capital of France is Paris.
I'm not sure if

## Bumblebee

CEC answer decoder from the measured dim=640 checkpoint (127,106,806 params), vocab 62322, 4 passage-conditioned questions, 12 new tokens. That file is not in this clone. Tokens matching the dense decoder: 48/48. Int8 certified table: True. Generate 679.1 ms with Hearth, 953.1 ms dense.

## Large models

One random MLP at Llama-30B size, scaled by parameter count. Not a Llama run.
Measured matrix 17920 x 6656. Packed cosine vs fp32: **0.99996**. Bytes 124150296 vs fp32 477102080. One fp32 token 45.0 ms. The same token through the int8 table 12.8 ms.

Scaled to 60 layers of that shape plus the word table: fp32 129260584960 bytes, fp16 64630292480 bytes, packed layers plus an fp16 word table 33840149380 bytes. At the measured rate, one fp32 token is about 12100 ms and one packed token is about 3500 ms. A full GPT-2 generate with the blocks packed and the head left exact matched the dense sentence (16 new tokens) and took 1.36 s vs 1.44 s, because those layer matrices are small. Teacher-forced token match on the earlier torch upcast path was 24/24.

