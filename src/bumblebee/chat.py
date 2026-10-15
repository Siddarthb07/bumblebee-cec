"""Interactive terminal chat for Bumblebee (CEC-only)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from bumblebee.router import ask_hybrid

console = Console()


def run_chat(root: Path, mode: str = "auto") -> None:
    if not (root / "artifacts" / "cec.pt").exists():
        raise SystemExit("Missing artifacts/cec.pt — train first: python -m bumblebee.cli train")

    console.print(
        Panel(
            "[bold]Bumblebee chat[/bold]\n"
            "CEC-only — answers from the frozen pack, refuses when outside it.\n"
            f"Mode: [cyan]{mode}[/cyan]  ·  Commands: /quit /help /clear /reset /mode auto|corpus",
            border_style="green",
        )
    )

    history: List[Tuple[str, str, str]] = []  # (user, answer, passage_id)
    turn = 0
    while True:
        try:
            q = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break
        if not q:
            continue
        low = q.lower()
        if low in {"/quit", "/exit", ":q", "quit", "exit"}:
            console.print("[dim]bye[/dim]")
            break
        if low == "/help":
            console.print(
                "auto/corpus: CEC only — refuse when outside the frozen pack.\n"
                "Open LM sidecar has been removed."
            )
            continue
        if low == "/clear":
            console.clear()
            continue
        if low == "/reset":
            history.clear()
            console.print("[dim]history cleared[/dim]")
            continue
        if low.startswith("/mode"):
            parts = low.split()
            if len(parts) == 2 and parts[1] in {"auto", "corpus"}:
                mode = parts[1]
                console.print(f"[dim]mode={mode}[/dim]")
            else:
                console.print("[yellow]usage: /mode auto|corpus[/yellow]")
            continue

        result = ask_hybrid(root, q, history=history, mode=mode)  # type: ignore[arg-type]

        turn += 1
        mode_label = result.get("mode", "?")
        model_label = result.get("model_id", "?")
        src = result.get("source", "?")
        if result["status"] == "answer":
            history.append((q, result["answer"], result.get("passage_id") or ""))
            unverified = " · unverified" if result.get("unverified") else ""
            label = result.get("source_label") or src
            body = (
                f"**ANSWER** · source=`{label}` · `{mode_label}` · `{model_label}` "
                f"(conf={result['confidence']:.3f}"
                f"{', multihop' if result.get('multihop') else ''}{unverified})\n\n"
                f"{result['answer']}"
            )
            if result.get("hint") and result.get("unverified"):
                body += f"\n\n*{result['hint']}*"
            if result.get("citation_span") and src not in {"web_unverified", "knowledge", "pack"}:
                body += (
                    f"\n\n*citation [{result.get('passage_id')}]:* "
                    f"{result['citation_span']}"
                )
            web = result.get("web")
            if isinstance(web, dict) and web.get("answer") and src != "web_unverified":
                body += (
                    f"\n\n---\n**WEB (unverified)** conf={web.get('confidence', 0):.2f} "
                    f"overlap={web.get('overlap', '?')} · `{web.get('reason', '')}`\n\n"
                    f"{web['answer']}"
                )
            console.print(Panel(Markdown(body), title=f"bumblebee #{turn}", border_style="green"))
        else:
            body = (
                f"**REFUSE** · source=`{src}` · `{mode_label}` "
                f"(conf={result['confidence']:.3f})\n\n"
                f"{result.get('hint') or 'I could not ground an answer in my pack.'}\n\n"
                f"reason: `{result.get('reason')}`"
            )
            web = result.get("web")
            if isinstance(web, dict) and web.get("answer"):
                body += (
                    f"\n\n---\n**WEB (unverified)** conf={web.get('confidence', 0):.2f} "
                    f"overlap={web.get('overlap', '?')}\n\n{web['answer']}"
                )
            console.print(Panel(Markdown(body), title=f"bumblebee #{turn}", border_style="yellow"))
