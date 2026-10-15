# Contributing

Bumblebee is a measured research system. Pull requests that keep the scoreboard
honest are welcome. Pull requests that raise a headline number by giving the
model the gold passage, or by quoting the product skill suite as core CEC, are not.

## Setup

```bash
git clone https://github.com/Siddarthb07/bumblebee-cec.git
cd bumblebee-cec
python -m pip install -e ".[dev]"
python -m pytest -q
```

Python 3.9 or newer. CI runs 3.9, 3.11, and 3.12.

## What to change

- Bug fixes and docs that match `data/sealed/SEAL.json` and `docs/RESULTS.md`.
- Tests in `tests/`. The default suite does not train the full pack.
- New refusal cases belong in `src/bumblebee/safety.py` with a test.

## What not to change in a drive-by PR

- Do not cite `docs/SYSTEMS_BENCH.md`. It is quarantined.
- Do not describe product skills (20/20) as core CEC. Core CEC on that suite is 10/17. See `docs/SKILLS_BENCH.md`.
- Do not call the gate calibrated. Answered-subset ECE on the sealed pack is about 0.915.
- Do not put patent-pending, "replaces transformers," or a 30B result in the README. Nothing is filed. Scale-to-30B is a hypothesis.
- Do not reintroduce an open language-model sidecar.

## Docs

`docs/INDEX.md` says which file is current. If you change a measured number, update the doc the bench writes and the index note in the same PR.

## License

Contributions are under the MIT license in `LICENSE`. Corpus text stays under the upstream licenses in `NOTICE.md`. Do not add third-party passages without naming the source and its license.

## Reporting issues

Use the bug or question template. Security reports go through `SECURITY.md`, not a public issue.
