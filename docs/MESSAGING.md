# Messaging bots

## Telegram (shipped stub)

```bash
pip install python-telegram-bot
export BUMBLEBEE_TELEGRAM_TOKEN=...
python -m bumblebee.telegram_bot
```

Uses `ask_hybrid` over the frozen pack. Hard-refuses banned topics (`unsafe_*`) before retrieval. Otherwise tries tools/packs, then a multi-passage try-harder cascade (single → stitch → synthesize) before labeling an out-of-pack refuse. History is short (last 3 turns).

## WhatsApp

Not shipped: needs Meta Cloud API app, webhook URL, and verify token. Same pattern as Telegram — wrap `ask_hybrid` and label `source` / refuse reasons. Do not default to an open LM.
