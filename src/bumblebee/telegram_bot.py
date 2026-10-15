"""Optional Telegram bot over ask_hybrid (personal pack, refuse-first).

Requires: pip install python-telegram-bot
Env: BUMBLEBEE_TELEGRAM_TOKEN

  python -m bumblebee.telegram_bot

WhatsApp is not shipped (Meta Cloud API keys + webhook hosting) — same ask_hybrid
wrapper pattern; see docs/MESSAGING.md.
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    token = os.environ.get("BUMBLEBEE_TELEGRAM_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set BUMBLEBEE_TELEGRAM_TOKEN")

    try:
        from telegram import Update
        from telegram.ext import Application, ContextTypes, MessageHandler, filters
    except ImportError as exc:
        raise SystemExit("pip install python-telegram-bot") from exc

    from bumblebee.router import ask_hybrid

    root = Path.cwd()
    if not (root / "data" / "corpus" / "passages.jsonl").exists():
        root = Path(__file__).resolve().parents[2]

    async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
        hist = context.user_data.setdefault("history", [])
        result = ask_hybrid(root, update.message.text, history=hist[-3:])
        src = result.get("source", "?")
        if result["status"] == "answer":
            hist.append(
                (update.message.text, result["answer"], result.get("passage_id") or "")
            )
            if len(hist) > 6:
                del hist[:-6]
            cite = result.get("citation_span") or ""
            pid = result.get("passage_id") or ""
            text = (
                f"[{src}] {result['answer']}\n"
                f"(conf={result['confidence']:.2f}"
                + (f", pid={pid}" if pid else "")
                + (f", cite={cite[:100]})" if cite else ")")
            )
        else:
            text = (
                f"[refuse/{src}] reason={result.get('reason')} "
                f"conf={result['confidence']:.2f}"
            )
        await update.message.reply_text(text[:3500])

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    print("Telegram bot polling…")
    app.run_polling()


if __name__ == "__main__":
    main()
