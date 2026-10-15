from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    unk_id: int = 1
    pad_id: int = 0

    @classmethod
    def build(cls, texts: Sequence[str], min_count: int = 1) -> "Vocab":
        counts: Dict[str, int] = {}
        for text in texts:
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
        itos = ["<pad>", "<unk>", "<eos>"]
        for tok, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if c >= min_count and tok not in {"<pad>", "<unk>", "<eos>"}:
                itos.append(tok)
        stoi = {t: i for i, t in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def eos_id(self) -> int:
        return self.stoi.get("<eos>", 2)

    def encode(self, text: str, max_len: int | None = None, add_eos: bool = False) -> List[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize(text)]
        if not ids:
            # Avoid empty tensors crashing the encoder stack
            ids = [self.unk_id]
        if add_eos:
            ids = ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
            if add_eos and ids and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        toks = []
        for i in ids:
            if i == self.pad_id:
                continue
            if i == self.eos_id:
                break
            if 0 <= i < len(self.itos):
                toks.append(self.itos[i])
        return " ".join(toks)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"itos": self.itos}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        itos = json.loads(path.read_text(encoding="utf-8"))["itos"]
        return cls(stoi={t: i for i, t in enumerate(itos)}, itos=itos)
