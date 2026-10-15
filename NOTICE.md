# Notice

The [MIT License](LICENSE) covers the source code in this repository.

It does **not** cover passage and question text under `data/`. That text is a
derived slice of public datasets. Keep the upstream license when you
redistribute the pack, and check the dataset card if a row below says to.

| Source | What this repo ships | License to respect |
| --- | --- | --- |
| `databricks/databricks-dolly-15k` | grounded Q&A slice plus local skill seeds | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) for Dolly text |
| `wikitext` / `wikitext-2-raw-v1` | chunks | Wikipedia text, CC BY-SA. See the dataset card. |
| `HuggingFaceTB/smollm-corpus` `cosmopedia-v2` | chunks | See the dataset card. |
| `HuggingFaceTB/smollm-corpus` `fineweb-edu-dedup` | chunks | FineWeb family: [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/). Confirm the card. |
| `HuggingFaceTB/finemath` `finemath-3plus` | chunks | See the dataset card. |
| `HuggingFaceFW/fineweb-edu` | streaming slice, not the full corpus | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) |
| `wikimedia/wikipedia` | encyclopedic chunks | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `hotpot_qa` | multi-hop contexts | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `google-research-datasets/natural_questions` | capped streaming slice | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

Skill-pack replies written for this project are mixed into the same JSONL
files as third-party text. Treat `data/corpus/passages.jsonl` and
`data/evals/questions.jsonl` as upstream-licensed data, not as MIT code.

Counts and checksums: `data/sealed/SEAL.json`. The seal is a reproducibility
lock. It is not a claim of external validity, and it is not a license.
