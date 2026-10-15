from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.table import Table

from bumblebee.baselines import BM25Index, EmbedIndex, retrieval_ask
from bumblebee.data import load_evals, load_passages, verify_manifest, write_manifest
from bumblebee.eval_protocol import summarize_ask_results, train_eval_split
from bumblebee.train import evaluate, train_model

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Bumblebee CEC CLI")


def root_dir() -> Path:
    # package lives in src/bumblebee; project root is parents[2] from this file when editable,
    # but prefer cwd if it looks like the project.
    cwd = Path.cwd()
    if (cwd / "data" / "corpus" / "passages.jsonl").exists():
        return cwd
    return Path(__file__).resolve().parents[2]


@app.command("ingest-hf")
def ingest_hf_cmd(
    dataset: str = typer.Option("dolly", help="HF dataset key: dolly|chat|squad"),
    contexts: int = typer.Option(80, help="SQuAD contexts, or Dolly scale hint (×6 → max grounded)"),
    max_q_per_context: int = typer.Option(3, help="Max train questions kept per SQuAD context"),
    max_open: int = typer.Option(200, help="Dolly open-instruction memories to keep"),
    seed: int = 7,
):
    """Download a Hugging Face dataset and replace data/corpus + data/evals."""
    from bumblebee.hf_ingest import ingest_hf

    root = root_dir()
    meta = ingest_hf(
        dataset,
        root / "data",
        train_contexts=contexts,
        max_q_per_context=max_q_per_context,
        max_open=max_open,
        seed=seed,
    )
    write_manifest(root / "data" / "corpus")
    rprint(meta)
    rprint("[green]Wrote HF corpus + evals under data/[/green]")


@app.command("ingest-assets")
def ingest_assets_cmd(
    dolly_grounded: int = typer.Option(800, help="Max Dolly grounded contexts"),
    dolly_open: int = typer.Option(200, help="Max Dolly open memories"),
    wikitext_chunks: int = typer.Option(800, help="WikiText-2 chunks"),
    cosmopedia_chunks: int = typer.Option(800, help="Cosmopedia-v2 chunks"),
    smollm_chunks: int = typer.Option(600, help="SmolLM-corpus slice chunks"),
    finemath_chunks: int = typer.Option(400, help="FineMath chunks"),
    fineweb_edu_chunks: int = typer.Option(2500, help="FineWeb-Edu streaming chunks"),
    wikipedia_chunks: int = typer.Option(2500, help="Wikipedia EN streaming chunks"),
    hotpot_passages: int = typer.Option(2000, help="HotpotQA context passages"),
    hotpot_evals: int = typer.Option(1500, help="HotpotQA grounded evals"),
    nq_chunks: int = typer.Option(1000, help="Natural Questions document chunks"),
    nq_evals: int = typer.Option(800, help="Natural Questions short-answer evals"),
    seed: int = 7,
    skip: Optional[str] = typer.Option(
        None,
        help="Comma-separated sources to skip: wikitext,cosmopedia,smollm,finemath,fineweb_edu,wikipedia,hotpot,nq",
    ),
):
    """
    Build multi-corpus CEC asset pack (Dolly+WikiText+Cosmopedia+SmolLM+FineMath
    +FineWeb-Edu+Wikipedia+HotpotQA+NQ). Streaming caps — never loads transformer weights.
    """
    from bumblebee.asset_ingest import ingest_assets_to_project

    skip_list = [s.strip() for s in (skip or "").split(",") if s.strip()]
    summary = ingest_assets_to_project(
        root_dir(),
        dolly_grounded=dolly_grounded,
        dolly_open=dolly_open,
        wikitext_chunks=wikitext_chunks,
        cosmopedia_chunks=cosmopedia_chunks,
        smollm_chunks=smollm_chunks,
        finemath_chunks=finemath_chunks,
        fineweb_edu_chunks=fineweb_edu_chunks,
        wikipedia_chunks=wikipedia_chunks,
        hotpot_passages=hotpot_passages,
        hotpot_evals=hotpot_evals,
        nq_chunks=nq_chunks,
        nq_evals=nq_evals,
        seed=seed,
        skip_sources=skip_list or None,
    )
    rprint(summary)
    rprint(
        f"[green]Asset pack[/green] passages={summary.get('n_passages')} "
        f"evals={summary.get('n_evals')} seal={summary.get('seal')}"
    )
    if summary.get("errors"):
        rprint(f"[yellow]Partial source errors:[/yellow] {summary['errors']}")


@app.command()
def ingest():
    """Hash corpus files into MANIFEST.json and verify integrity."""
    root = root_dir()
    path = write_manifest(root / "data" / "corpus")
    ok, problems = verify_manifest(root / "data" / "corpus")
    rprint(f"[green]Wrote[/green] {path}")
    if ok:
        rprint("[green]Manifest verified[/green]")
    else:
        rprint(f"[yellow]Manifest issues:[/yellow] {problems}")


@app.command()
def train(
    epochs: int = 8,
    dim: int = 128,
    threshold: float = 0.35,
    max_passage_len: int = 192,
    max_question_len: int = 64,
    mac_fit: bool = typer.Option(
        False,
        "--mac-fit",
        help="Train embedding-heavy ~500M CEC sized to fit small Mac RAM",
    ),
    target_params: int = typer.Option(
        1_500_000_000,
        help="Target parameter count when using --mac-fit (sparse hash; ~0.5B–1.5B)",
    ),
):
    """Train the CEC model on the closed eval/corpus pack."""
    root = root_dir()
    if mac_fit:
        from bumblebee.train import train_model_mac_fit

        path = train_model_mac_fit(
            root,
            target_params=target_params,
            epochs=epochs,
            threshold=threshold,
        )
    else:
        path = train_model(
            root,
            epochs=epochs,
            dim=dim,
            threshold=threshold,
            max_p=max_passage_len,
            max_q=max_question_len,
        )
    rprint(f"[green]Saved checkpoint[/green] {path}")


@app.command()
def chat(
    mode: str = typer.Option("auto", help="Routing: auto | corpus"),
):
    """Interactive terminal chat (CEC-only)."""
    from bumblebee.chat import run_chat

    run_chat(root_dir(), mode=mode)


@app.command()
def dash(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = typer.Option(False, help="Unused (stdlib dash); kept for CLI compat"),
):
    """Open a local web chat UI in your browser (CEC-only)."""
    from bumblebee.dash_app import run_dash

    run_dash(root_dir(), host=host, port=port, share=share)


@app.command("arch-dash")
def arch_dash(
    host: str = "127.0.0.1",
    port: int = 7861,
    no_browser: bool = typer.Option(False, help="Do not open a browser tab"),
):
    """Architecture visualization dash (NN growth graph) — separate from chat."""
    from bumblebee.architecture_dash import run_architecture_dash

    run_architecture_dash(root_dir(), host=host, port=port, open_browser=not no_browser)


@app.command()
def ask(
    question: str,
    passage_id: Optional[str] = None,
    mode: str = typer.Option("auto", help="Routing: auto | corpus"),
):
    """Ask via CEC router (frozen pack only)."""
    from bumblebee.router import ask_hybrid

    root = root_dir()
    result = ask_hybrid(root, question, passage_id=passage_id, mode=mode)  # type: ignore[arg-type]
    mode_l = result.get("mode", "?")
    model_l = result.get("model_id", "?")
    if result["status"] == "answer":
        src = result.get("source", "?")
        rprint(f"[bold]ANSWER[/bold] source={src} mode={mode_l} model={model_l} ({result['confidence']:.3f})")
        rprint(result["answer"])
        if result.get("citation_span"):
            rprint(f"[dim]citation[/dim] [{result['passage_id']}] {result['citation_span']}")
    else:
        rprint(
            f"[yellow]REFUSE[/yellow] source={result.get('source', 'refuse')} "
            f"mode={mode_l} reason={result['reason']} conf={result['confidence']:.3f}"
        )
        if result.get("hint"):
            rprint(result["hint"])

@app.command()
def calibrate():
    """Fit post-hoc temperature on the calibration split; write artifacts/calibration.json."""
    from bumblebee.calibration import fit_gate_temperature
    from bumblebee.train import clear_runtime_cache

    clear_runtime_cache()
    report = fit_gate_temperature(root_dir())
    rprint(report)
    rprint("[green]Wrote[/green] artifacts/calibration.json (+ config.json temperature)")


@app.command(name="eval")
def eval_cmd():
    """Evaluate CEC on held-out in-domain + OOD (no gold leakage; disjoint OOD)."""
    from bumblebee.train import clear_runtime_cache

    clear_runtime_cache()
    root = root_dir()
    report = evaluate(root)
    table = Table(title="Bumblebee CEC eval (honest holdout)")
    table.add_column("metric")
    table.add_column("value")
    for k, v in report.items():
        if k == "by_split":
            continue
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    rprint(table)
    if report.get("by_split"):
        rprint(report["by_split"])


@app.command()
def bench(
    mode: str = typer.Option(
        "retrieval",
        help="retrieval | all | skills | llm-compare | adversarial | ablation | memory",
    ),
    llm_model: str = typer.Option(
        "gpt2-medium",
        help="Peer LLM for llm-compare: gpt2-medium | smollm-360m | pythia-410m | gpt2 | or HF id",
    ),
    max_in_domain: int = typer.Option(20, help="Held-out in-domain sample size for llm-compare"),
    max_ood: int = typer.Option(20, help="Held-out OOD sample size for llm-compare"),
    ablation_epochs: int = typer.Option(2, help="Epochs per mass_fn for ablation mode"),
    memory_asks: int = typer.Option(8, help="Ask count for memory peak-RSS probe"),
):
    """
    Benchmark on the same held-out eval splits.
    retrieval: BM25+abstain and embedding+abstain (shared refuse family)
    all: retrieval + fold in CEC eval_report
    skills: chat/math/code/corpus/live/refuse suite scoring CEC itself
    llm-compare: CEC vs a real dense transformer LLM peer (optional transformers install)
    adversarial: paraphrase / distractor / conflict suite
    ablation: SoftCorrelator sparsemax vs softmax vs topk (short train)
    memory: CEC disk/RSS/params vs modern LLM weight catalogs
    """
    root = root_dir()
    if mode == "retrieval":
        report = _bench_retrieval(root)
        out = root / "artifacts" / "bench_retrieval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        rprint(report)
        rprint(f"[green]Wrote[/green] {out}")
    elif mode == "all":
        from bumblebee.results import bench_all

        if (root / "artifacts" / "cec.pt").exists():
            evaluate(root)
        report = bench_all(root)
        rprint(report)
        rprint("[green]Wrote[/green] artifacts/bench_all.json and docs/RESULTS.md")
    elif mode in {"slm", "systems"}:
        raise typer.BadParameter(
            f"mode={mode} removed as a product path. Use llm-compare for optional peer benches."
        )
    elif mode == "skills":
        from bumblebee.skills_bench import run_skills_bench

        report = run_skills_bench(root, mode="auto")
        rprint(
            {
                "passed": report["passed"],
                "n": report["n"],
                "accuracy": report["accuracy"],
                "by_category": report["by_category"],
            }
        )
        rprint("[green]Wrote[/green] artifacts/bench_skills.json and docs/SKILLS_BENCH.md")
    elif mode == "adversarial":
        from bumblebee.adversarial import run_adversarial_bench, write_adversarial_file
        from bumblebee.gate import AskResult
        from bumblebee.train import ask_question_fast, clear_runtime_cache

        write_adversarial_file(root)
        clear_runtime_cache()

        def ask_fn(ex):
            d = ask_question_fast(root, ex.question)
            return AskResult(
                d["status"],
                d.get("answer") or "",
                float(d.get("confidence") or 0.0),
                d.get("reason"),
                d.get("citation_span") or "",
                int(d.get("citation_start") or 0),
                int(d.get("citation_end") or 0),
            )

        report = run_adversarial_bench(root, ask_fn)
        rprint(report)
        rprint("[green]Wrote[/green] artifacts/bench_adversarial.json")
    elif mode == "ablation":
        from bumblebee.ablation import run_mass_ablation

        report = run_mass_ablation(root, epochs=ablation_epochs)
        rprint(report)
        rprint("[green]Wrote[/green] artifacts/ablation_mass.json and docs/ABLATION_MASS.md")
    elif mode == "memory":
        from bumblebee.memory_bench import run_memory_bench

        report = run_memory_bench(root, n_asks=memory_asks)
        rprint(
            {
                "peak_rss_gb": report["cec"]["peak_rss_gb"],
                "n_params_m": report["cec"]["n_params_m"],
                "n_passages": report["cec"]["n_passages"],
                "ratio_vs_mistral7b_fp16": report["comparison"]["ratio_vs_mistral7b_fp16"],
                "beats": report["comparison"]["beats_fp16_weight_footprint"][:6],
            }
        )
        rprint("[green]Wrote[/green] artifacts/bench_memory.json and docs/MEMORY_BENCH.md")
    elif mode == "llm-compare":
        try:
            import transformers  # noqa: F401
            from bumblebee.llm_compare import run_llm_compare
        except ImportError as exc:
            rprint(
                "[red]llm-compare needs transformers.[/red] "
                "Install: pip install 'bumblebee[llm-bench]' or pip install transformers accelerate"
            )
            raise typer.Exit(code=1) from exc
        report = run_llm_compare(
            root,
            model=llm_model,
            max_in_domain=max_in_domain,
            max_ood=max_ood,
        )
        rprint(
            {
                "llm_peer": report["llm_peer"],
                "skills": report["skills"],
                "heldout_sample": report["heldout_sample"],
            }
        )
        rprint("[green]Wrote[/green] artifacts/bench_llm_compare.json and docs/LLM_COMPARE.md")
    else:
        raise typer.BadParameter(
            "mode must be retrieval, all, skills, adversarial, ablation, memory, or llm-compare"
        )


@app.command()
def seal():
    """Write / refresh data/sealed/SEAL.json checksums for the frozen pack."""
    from bumblebee.sealed import verify_seal, write_seal

    path = write_seal(root_dir())
    ok, problems = verify_seal(root_dir())
    rprint(f"[green]Wrote[/green] {path}")
    if ok:
        rprint("[green]Seal verified[/green]")
    else:
        rprint(f"[yellow]Seal issues:[/yellow] {problems}")


@app.command("grow-entity-gap")
def grow_entity_gap(target_holdout: int = typer.Option(30, help="Minimum entity-gap holdout N")):
    """Synthesize more entity-gap probes until holdout N >= target."""
    from bumblebee.entity_gap import append_entity_gaps
    from bumblebee.sealed import write_seal

    report = append_entity_gaps(root_dir(), target_holdout=target_holdout)
    write_seal(root_dir())
    rprint(report)
    rprint("[green]Appended entity-gap probes + refreshed seal[/green]")


@app.command("pack-build")
def pack_build(
    src: Path = typer.Argument(..., help="Folder of .txt/.md/.pdf notes"),
    out: Path = typer.Option(
        Path("data/pack_builder"),
        help="Output data root (writes corpus/ + evals/)",
    ),
):
    """Drop notes → passages.jsonl + seed evals (train + ask with refuse next)."""
    from bumblebee.pack_builder import ingest_folder

    meta = ingest_folder(src, out_corpus=out / "corpus")
    rprint(meta)
    rprint(
        "[green]Pack built.[/green] Next:\n"
        f"  # point data/ at the new pack or copy into data/\n"
        f"  python -m bumblebee.cli train --epochs 5 --dim 96\n"
        f"  python -m bumblebee.cli ask \"...\"\n"
    )


@app.command()
def demo():
    """Scripted live path via CEC core ask_question (2 in-domain, 2 entity-gap, 2 far-OOD)."""
    from bumblebee.data import load_evals
    from bumblebee.eval_protocol import is_holdout
    from bumblebee.train import ask_question_fast, clear_runtime_cache

    root = root_dir()
    clear_runtime_cache()
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    # Prefer short answerable questions (avoid abstractive Dolly sprawl)
    in_dom = sorted(
        [
            e
            for e in evals
            if e.split == "in_domain"
            and e.answerable
            and e.answer
            and 20 < len(e.question) < 80
            and e.question.lower().strip() not in {"hi", "hello", "hey"}
        ],
        key=lambda e: len(e.question),
    )[:2]
    gaps = [e for e in evals if e.split == "entity_gap" and is_holdout(e.id)][:2]
    far = [e for e in evals if e.split == "far_ood" and len(e.question) < 60][:2]
    cases = (
        [("IN-DOMAIN", e) for e in in_dom]
        + [("ENTITY-GAP (hard)", e) for e in gaps]
        + [("FAR-OOD (easy)", e) for e in far]
    )
    rprint("[dim]CEC fast path (ask_question_fast) — product packs/tools not used[/dim]")
    for label, ex in cases:
        r = ask_question_fast(root, ex.question)
        status = r["status"]
        src = r.get("source", "?")
        if status == "answer" and "ENTITY" in label:
            rprint(
                f"[red]MISS[/red] {label} answered (should prefer refuse) "
                f"source={src} conf={r['confidence']:.3f}\n  Q: {ex.question}\n  A: {r['answer'][:120]}"
            )
        elif status == "refuse":
            rprint(
                f"[yellow]REFUSE[/yellow] {label} reason={r.get('reason')} source={src} "
                f"conf={r['confidence']:.3f}\n  Q: {ex.question}"
            )
        else:
            cite = r.get("citation_span") or ""
            rprint(
                f"[green]ANSWER[/green] {label} source={src} conf={r['confidence']:.3f} "
                f"multihop={r.get('multihop')}\n  Q: {ex.question}\n  A: {r['answer'][:160]}\n"
                f"  cite[{r.get('passage_id')}]: {cite[:100]}"
            )


def _bench_retrieval(root: Path) -> dict:
    from bumblebee.tokenize import tokenize

    passages = load_passages(root / "data" / "corpus")
    evals = load_evals(root / "data" / "evals" / "questions.jsonl")
    _, test_evals = train_eval_split(evals)
    pairs = [(p.id, p.text) for p in passages]
    corpus_toks = {t for p in passages for t in tokenize(p.text) if len(t) > 2}
    indexes = {"bm25": BM25Index(pairs), "embed": EmbedIndex(pairs)}
    thresholds = {"bm25": 0.45, "embed": 0.55}
    report = {}
    for name, index in indexes.items():
        results = (
            (
                ex,
                retrieval_ask(
                    index,
                    ex.question,
                    thresholds[name],
                    faithfulness=True,
                    lexical_ood=True,
                    corpus_tokens=corpus_toks,
                ),
            )
            for ex in test_evals
        )
        report[name] = summarize_ask_results(results)
        report[name]["threshold"] = thresholds[name]
        report[name]["n"] = len(test_evals)
        report[name]["protocol"] = "held_out; threshold+faithfulness+lexical_ood; no gold"
    return report


if __name__ == "__main__":
    app()
