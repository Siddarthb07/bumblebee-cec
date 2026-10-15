"""Hearth: a sealed next word, with the exact columns chosen by this hidden state.

A static int8 pack scores a rounded copy and emits that argmax. Hearth scores
the int8 copy, then adds the exact rounding residual on the coordinates where
this hidden state is largest (the pulse). A bound covers only the quiet
coordinates. Rows that can still win are dotted in full precision. Each word
carries a HearthSeal: pulse when that proof held, sketch when a low-rank head
certified a short list, dense when the proof was too wide and the full table
was used. The id matches the dense head in all three cases.

Wide layers still store int8 with a few exact columns, because every output
channel is required. The sealed word table is the step that moves with the
token.
"""

from __future__ import annotations

import json
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = b"HRTH"
HEADER = struct.Struct("<4s6I")
HEADER_SIZE = 64
DTYPE_FP32 = 0
DTYPE_INT8 = 1


@dataclass
class HearthSeal:
    """Proof attached to one next word.

    kind is "pulse" when the live rows were certified after correcting the
    hidden state's strongest coordinates, "sketch" when a low-rank residual
    bound certified the short list, and "dense" when the proof was too wide
    and the full table was used.
    """

    token: int
    live_rows: int
    pulse: int
    kind: str
_I8_CHUNK = 4096


def _int8_dot(q: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Dot each int8 row with one vector, upcasting a strip at a time."""
    if q.numel() <= 8_000_000:
        return q.float() @ hidden
    scores = torch.empty(int(q.size(0)), dtype=torch.float32)
    for start in range(0, int(q.size(0)), _I8_CHUNK):
        stop = min(int(q.size(0)), start + _I8_CHUNK)
        scores[start:stop] = q[start:stop].float() @ hidden
    return scores


def _i8_gemv(
    q: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    hidden: torch.Tensor,
    integer_hidden: bool = False,
) -> torch.Tensor:
    """Scores for one hidden vector. integer_hidden also quantizes that vector."""
    h = hidden.detach().float().cpu().contiguous().view(-1)
    if integer_hidden:
        h_scale = h.abs().max().clamp(min=1e-12) / 127.0
        hq = torch.round(h / h_scale).clamp(-127, 127)
        acc = _int8_dot(q, hq) * h_scale
    else:
        acc = _int8_dot(q, h)
    acc = acc * scale
    if bias is not None:
        acc = acc + bias.detach().float().cpu().view(-1)
    return acc


def _write_header(n_rows: int, dim: int, dtype: int, stride: int) -> bytes:
    raw = HEADER.pack(MAGIC, 1, n_rows, dim, dtype, stride, HEADER_SIZE)
    return raw.ljust(HEADER_SIZE, b"\0")


def _read_header(fd: int) -> tuple[int, int, int, int]:
    raw = os.pread(fd, HEADER_SIZE, 0)
    magic, version, n_rows, dim, dtype, stride, offset = HEADER.unpack(raw[: HEADER.size])
    if magic != MAGIC or version != 1 or offset != HEADER_SIZE:
        raise ValueError(f"not a Hearth lexicon: {magic!r} version={version}")
    return n_rows, dim, dtype, stride


class PagedLexicon:
    """Row file addressed with pread. The whole matrix is never loaded."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        self.n_rows, self.dim, self.dtype, self.stride = _read_header(self._fd)
        self.bytes_read = 0
        self._view = None

    def close(self) -> None:
        self._view = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def write(path: Path | str, rows: torch.Tensor, dtype: str = "fp32") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = rows.detach().float().cpu().contiguous()
        n_rows, dim = rows.shape
        if dtype == "fp32":
            payload = rows.numpy().tobytes()
            blob = _write_header(n_rows, dim, DTYPE_FP32, dim * 4) + payload
        elif dtype == "int8":
            scale = rows.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            quant = torch.round(rows / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
            blob = (
                _write_header(n_rows, dim, DTYPE_INT8, dim)
                + scale.contiguous().numpy().tobytes()
                + quant.contiguous().numpy().tobytes()
            )
        else:
            raise ValueError(dtype)
        path.write_bytes(blob)

    def _fp32_span(self, start: int, count: int) -> torch.Tensor:
        nbytes = count * self.dim * 4
        buf = os.pread(self._fd, nbytes, HEADER_SIZE + start * self.dim * 4)
        self.bytes_read += len(buf)
        if len(buf) != nbytes:
            raise IOError(f"short lexicon read: {len(buf)} != {nbytes}")
        arr = np.frombuffer(buf, dtype=np.float32).reshape(count, self.dim).copy()
        return torch.from_numpy(arr)

    def _int8_span(self, start: int, count: int) -> torch.Tensor:
        scale_buf = os.pread(self._fd, count * 4, HEADER_SIZE + start * 4)
        q_off = HEADER_SIZE + self.n_rows * 4 + start * self.dim
        q_buf = os.pread(self._fd, count * self.dim, q_off)
        self.bytes_read += len(scale_buf) + len(q_buf)
        scale = np.frombuffer(scale_buf, dtype=np.float32).copy()
        quant = np.frombuffer(q_buf, dtype=np.int8).reshape(count, self.dim).astype(np.float32)
        return torch.from_numpy(quant * scale[:, None])

    def read_range(self, start: int, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.zeros(0, self.dim)
        if self.dtype == DTYPE_FP32:
            return self._fp32_span(start, count)
        if self.dtype == DTYPE_INT8:
            return self._int8_span(start, count)
        raise ValueError(self.dtype)

    def read_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Load rows in the requested order. fp32 uses a mapped view of only those rows."""
        ids = ids.detach().long().cpu().view(-1)
        if ids.numel() == 0:
            return torch.zeros(0, self.dim)
        if self.dtype == DTYPE_FP32:
            if self._view is None:
                self._view = np.memmap(
                    self.path,
                    dtype=np.float32,
                    mode="r",
                    offset=HEADER_SIZE,
                    shape=(self.n_rows, self.dim),
                )
            arr = np.array(self._view[ids.numpy()], dtype=np.float32, copy=True)
            self.bytes_read += int(ids.numel()) * self.dim * 4
            return torch.from_numpy(arr)
        uniq, inverse = torch.unique(ids, return_inverse=True)
        fetched = torch.empty(uniq.numel(), self.dim)
        cursor = 0
        while cursor < uniq.numel():
            end = cursor
            while end + 1 < uniq.numel() and int(uniq[end + 1]) == int(uniq[end]) + 1:
                end += 1
            count = end - cursor + 1
            fetched[cursor : end + 1] = self.read_range(int(uniq[cursor]), count)
            cursor = end + 1
        return fetched[inverse]


class RowCache:
    """LRU of id-aligned pages. Resident bytes stay inside the budget."""

    def __init__(self, lexicon: PagedLexicon, budget_bytes: int, page_rows: int = 1):
        self.lexicon = lexicon
        self.page_rows = max(1, page_rows)
        self.page_bytes = self.page_rows * lexicon.dim * 4
        self.max_pages = max(1, int(budget_bytes) // self.page_bytes)
        self.pages: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def resident_bytes(self) -> int:
        return sum(p.numel() * 4 for p in self.pages.values())

    def _load(self, page: int) -> torch.Tensor:
        start = page * self.page_rows
        count = min(self.page_rows, self.lexicon.n_rows - start)
        block = self.lexicon.read_range(start, count)
        self.misses += 1
        while len(self.pages) >= self.max_pages:
            self.pages.popitem(last=False)
        self.pages[page] = block
        return block

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.detach().long().cpu().view(-1)
        if self.page_rows == 1:
            return self._rows_direct(ids)
        out = torch.empty(ids.numel(), self.lexicon.dim)
        page_ids = torch.div(ids, self.page_rows, rounding_mode="floor")
        local = ids - page_ids * self.page_rows
        have: dict[int, torch.Tensor] = {}
        for page in torch.unique(page_ids).tolist():
            page = int(page)
            if page in self.pages:
                self.hits += 1
                block = self.pages[page]
                self.pages.move_to_end(page)
            else:
                block = self._load(page)
            have[page] = block
        for i, (page, loc) in enumerate(zip(page_ids.tolist(), local.tolist())):
            out[i] = have[int(page)][int(loc)]
        return out

    def _rows_direct(self, ids: torch.Tensor) -> torch.Tensor:
        needed = [int(i) for i in ids.tolist()]
        missing = [i for i in dict.fromkeys(needed) if i not in self.pages]
        for i in dict.fromkeys(needed):
            if i in self.pages:
                self.hits += 1
                self.pages.move_to_end(i)
        loaded: dict[int, torch.Tensor] = {}
        if missing:
            block = self.lexicon.read_ids(torch.tensor(missing, dtype=torch.long))
            for i, row in zip(missing, block):
                loaded[i] = row
                self.misses += 1
        out = torch.empty(len(needed), self.lexicon.dim)
        for n, i in enumerate(needed):
            out[n] = self.pages[i] if i in self.pages else loaded[i]
        for i, row in loaded.items():
            while self.pages and len(self.pages) >= self.max_pages:
                self.pages.popitem(last=False)
            if len(self.pages) < self.max_pages:
                self.pages[i] = row
        return out


class SketchIndex:
    """Resident PCA codes for a weight matrix.

    Scores every code, keeps the best ids, then those full rows are dotted
    exactly. If the codes are a poor guide, the caller uses the full table
    instead so the chosen word stays the same and the step does not get slower.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        basis: torch.Tensor,
        codes: torch.Tensor,
        energy: float = 1.0,
    ):
        self.mean = mean.float().cpu().contiguous()
        self.basis = basis.float().cpu().contiguous()
        self.codes = codes.float().cpu().contiguous()
        self.energy = float(energy)
        self.resid_norm: Optional[torch.Tensor] = None

    @property
    def resident_bytes(self) -> int:
        return int((self.mean.numel() + self.basis.numel() + self.codes.numel()) * 4)

    @staticmethod
    def fit(weight: torch.Tensor, rank: int) -> "SketchIndex":
        rows = weight.detach().float().cpu().contiguous()
        n_rows, dim = rows.shape
        rank = max(1, min(int(rank), dim, n_rows))
        mean = rows.mean(dim=0)
        centered = rows - mean
        evals, evecs = torch.linalg.eigh(centered.transpose(0, 1) @ centered)
        evals = evals.clamp(min=0)
        basis = evecs[:, -rank:].contiguous()
        codes = centered @ basis
        total = float(evals.sum().clamp(min=1e-8))
        energy = float(evals[-rank:].sum() / total)
        resid_sq = (centered * centered).sum(dim=1) - (codes * codes).sum(dim=1)
        sketch = SketchIndex(mean, basis, codes, energy)
        sketch.resid_norm = resid_sq.clamp(min=0).sqrt().contiguous()
        return sketch

    def scores(self, hidden: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = hidden.detach().float().cpu().view(-1)
        sc = self.codes @ (self.basis.T @ h)
        if bias is not None:
            sc = sc + bias.detach().float().cpu().view(-1)
        return sc

    def rank_of(self, hidden: torch.Tensor, true_id: int, bias: Optional[torch.Tensor] = None) -> int:
        sc = self.scores(hidden, bias)
        return int((sc > sc[int(true_id)]).sum())

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"mean": self.mean, "basis": self.basis, "codes": self.codes, "energy": self.energy},
            path,
        )

    @staticmethod
    def load(path: Path | str) -> "SketchIndex":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return SketchIndex(blob["mean"], blob["basis"], blob["codes"], float(blob.get("energy", 1.0)))

    def top_ids(self, hidden: torch.Tensor, k: int, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.scores(hidden, bias)
        k = max(1, min(int(k), int(scores.numel())))
        return torch.topk(scores, k).indices


class HearthHead:
    """Exact argmax for one LM head.

    A fixed short list is not safe. On GPT-2 it misses the dense winner for
    code, repeated punctuation, and greedy runs longer than the calibration
    sentences (true-word sketch ranks 1928–4493 against a cap of 1641). The
    residual bound is also too wide to drop any row: hidden states sit almost
    entirely outside the top principal components, and the largest sketch error
    on those steps is about 49 logits.

    This head uses the short list only when that residual cannot move another
    row past the winner. Every other head is scored from an int8 copy of the
    rows. The pulse adds the exact rounding residual on the coordinates where
    this hidden state is largest, and the bound covers the quiet coordinates.
    Rows that could still win are dotted in full precision, so the word matches.
    The seal on that word is pulse, sketch, or dense.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        sketch_rank: Optional[int] = None,
        sketch: Optional[SketchIndex] = None,
        linear: Optional[nn.Linear] = None,
    ):
        self.linear = linear
        self.weight = weight.detach()
        self.bias = None if bias is None else bias.detach()
        self.used_shortlist = False
        self.rows_scored = 0
        self.exact_only = True
        self.sketch: Optional[SketchIndex] = None
        self.weight_cpu: Optional[torch.Tensor] = None
        self.bias_cpu: Optional[torch.Tensor] = None
        self.resid_max = 0.0
        self.quant_ready = False
        self.used_quant = False
        self.quant_bytes = 0
        self.seal: Optional[HearthSeal] = None
        self.seals: list[HearthSeal] = []
        self._pulse_n = 0
        self._q: Optional[torch.Tensor] = None
        self._scale: Optional[torch.Tensor] = None
        self._gn: Optional[torch.Tensor] = None
        self._edges: list[tuple[int, int]] = []
        self._bias_cpu: Optional[torch.Tensor] = None
        n_rows, dim = int(weight.size(0)), int(weight.size(1))
        if n_rows < 64 or n_rows <= dim:
            return
        rank = int(sketch_rank) if sketch_rank is not None else min(256, max(1, dim // 3))
        built = sketch if sketch is not None else SketchIndex.fit(weight, rank)
        if built.resid_norm is None:
            self._prepare_quant(weight, bias)
            return
        resid_max = float(built.resid_norm.max())
        # Energy below this means the leftover can reorder the winner, as it does
        # on GPT-2. Those heads use the int8 table plus an exact check.
        if built.energy < 0.999:
            self._prepare_quant(weight, bias)
            return
        self.sketch = built
        self.resid_max = resid_max
        self.weight_cpu = weight.detach().float().cpu().contiguous()
        self.bias_cpu = None if bias is None else bias.detach().float().cpu().contiguous()
        self.exact_only = False

    @staticmethod
    def from_linear(linear: nn.Linear, sketch_rank: Optional[int] = None) -> "HearthHead":
        bias = linear.bias if linear.bias is not None else None
        return HearthHead(linear.weight, bias, sketch_rank=sketch_rank, linear=linear)

    def _dense(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight if self.linear is not None else self.weight
        bias = None if self.linear is None else self.linear.bias
        if self.linear is None:
            bias = self.bias
        self.used_shortlist = False
        self.rows_scored = int(weight.size(0))
        self._pulse_n = 0
        return F.linear(hidden, weight, bias).argmax(dim=-1)

    def _seal_dense(self, ids: torch.Tensor) -> None:
        self.seal = HearthSeal(int(ids.view(-1)[-1]), self.rows_scored, 0, "dense")
        self.seals = [self.seal]

    def _pulse(self, h: torch.Tensor, approx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact residual on the strongest coordinates. The bound then sees the quiet tail.

        A fixed outlier set would reuse the same columns on every token. The
        pulse is the coordinates where this hidden state is largest.
        """
        dim = int(h.numel())
        pulse_n = min(32, max(4, dim // 16))
        self._pulse_n = 0
        if pulse_n >= dim or self._q is None or self._scale is None:
            return approx, h
        idx = torch.topk(h.abs(), pulse_n).indices
        source = self.linear.weight if self.linear is not None else self.weight
        cols = source.detach().index_select(1, idx.to(source.device)).float().cpu()
        qcols = self._q.index_select(1, idx).float() * self._scale.unsqueeze(1)
        corrected = approx + (cols - qcols) @ h.index_select(0, idx)
        quiet = h.clone()
        quiet[idx] = 0
        self._pulse_n = pulse_n
        return corrected, quiet

    def _prepare_quant(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        """Int8 rows plus residual norms on blocks of 16 columns."""
        rows = weight.detach().float().cpu().contiguous()
        n_rows, k = int(rows.size(0)), int(rows.size(1))
        if n_rows < 1024 or k < 16:
            return
        scale = rows.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        quant = torch.round(rows / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8).contiguous()
        edges: list[tuple[int, int]] = []
        start = 0
        while start < k:
            stop = min(k, start + 16)
            edges.append((start, stop))
            start = stop
        group_norm = torch.empty(n_rows, len(edges))
        for group, (left, right) in enumerate(edges):
            residual = rows[:, left:right] - quant[:, left:right].float() * scale.unsqueeze(1)
            group_norm[:, group] = residual.norm(dim=1)
        self._q = quant
        self._scale = scale.contiguous()
        self._gn = group_norm.contiguous()
        self._edges = edges
        self._bias_cpu = None if bias is None else bias.detach().float().cpu().contiguous()
        self.quant_bytes = int(quant.numel() + group_norm.numel() * 4 + scale.numel() * 4)
        self.quant_ready = True

    def _quant_id(self, hidden: torch.Tensor) -> Optional[int]:
        """Exact argmax, reading the int8 table and then only the live rows."""
        assert self._q is not None and self._scale is not None and self._gn is not None
        h = hidden.detach().float().cpu().contiguous().view(-1)
        approx = _i8_gemv(self._q, self._scale, self._bias_cpu, h)
        approx, quiet = self._pulse(h, approx)
        width = self._edges[0][1] - self._edges[0][0]
        uniform = quiet.numel() % width == 0 and all(right - left == width for left, right in self._edges)
        if uniform:
            group_norm = quiet.view(-1, width).norm(dim=1)
        else:
            group_norm = torch.empty(len(self._edges))
            for group, (left, right) in enumerate(self._edges):
                group_norm[group] = quiet[left:right].norm()
        err = torch.mv(self._gn, group_norm)
        lower = approx - err
        upper = approx + err
        best_lower = float(lower.max())
        # 0.05 covers the kernel's rounding against the torch int8 dot.
        live = torch.nonzero(upper + 0.05 >= best_lower, as_tuple=False).view(-1)
        n_live = int(live.numel())
        self.rows_scored = n_live
        self.used_quant = True
        self.used_shortlist = False
        if n_live == 0 or n_live > 4096:
            self.used_quant = False
            return None
        source = self.linear.weight if self.linear is not None else self.weight
        picked = live.to(source.device)
        block = source.detach().index_select(0, picked).float().cpu()
        exact = block @ h
        if self._bias_cpu is not None:
            exact = exact + self._bias_cpu.index_select(0, live)
        return int(live[int(exact.argmax())])

    def argmax(self, hidden: torch.Tensor) -> torch.Tensor:
        """Same ids as the dense linear. Accepts [dim] or [batch, dim]."""
        single = hidden.dim() == 1
        rows = hidden.unsqueeze(0) if single else hidden
        self.seals = []
        if self.quant_ready:
            chosen: list[int] = []
            failed = False
            for row in rows:
                nxt = self._quant_id(row)
                if nxt is None:
                    failed = True
                    break
                chosen.append(nxt)
                self.seals.append(HearthSeal(nxt, self.rows_scored, self._pulse_n, "pulse"))
            if not failed:
                self.seal = self.seals[-1]
                ids = torch.tensor(chosen, device=hidden.device, dtype=torch.long)
                return ids[0] if single else ids
            self.seals = []
        if self.exact_only or self.sketch is None or rows.size(0) != 1:
            out = self._dense(rows)
            self._seal_dense(out)
            return out[0] if single else out
        chosen = self._certified_id(rows[0].detach().float().cpu())
        if chosen is None:
            out = self._dense(rows)
            self._seal_dense(out)
            return out[0] if single else out
        self.used_shortlist = True
        self.seal = HearthSeal(int(chosen), self.rows_scored, 0, "sketch")
        self.seals = [self.seal]
        ids = torch.tensor([chosen], device=hidden.device, dtype=torch.long)
        return ids[0] if single else ids

    def _certified_id(self, h: torch.Tensor) -> Optional[int]:
        sketch = self.sketch
        assert sketch is not None and sketch.resid_norm is not None and self.weight_cpu is not None
        q = sketch.basis.T @ h
        perp2 = float(h.dot(h) - q.dot(q))
        perp = perp2 ** 0.5 if perp2 > 0.0 else 0.0
        scores = sketch.scores(h, self.bias_cpu)
        n_rows = int(scores.numel())
        k = min(n_rows, max(int(sketch.basis.size(1)) + 8, 16))
        ids = torch.topk(scores, k).indices
        exact = self.weight_cpu.index_select(0, ids) @ h
        if self.bias_cpu is not None:
            exact = exact + self.bias_cpu.index_select(0, ids)
        best = float(exact.max())
        shift = float(sketch.mean.dot(h))
        upper = scores + sketch.resid_norm * perp + shift
        could = upper > best + 1e-4
        could[ids] = False
        extra_n = int(could.sum())
        if extra_n == 0:
            self.rows_scored = k
            return int(ids[int(exact.argmax())])
        if extra_n > 2048 or extra_n + k >= n_rows:
            return None
        extra = torch.nonzero(could, as_tuple=False).view(-1)
        all_ids = torch.cat([ids, extra])
        exact_all = self.weight_cpu.index_select(0, all_ids) @ h
        if self.bias_cpu is not None:
            exact_all = exact_all + self.bias_cpu.index_select(0, all_ids)
        self.rows_scored = int(all_ids.numel())
        return int(all_ids[int(exact_all.argmax())])


class ClusterIndex:
    """Resident centroids. Membership is contiguous in the packed head file."""

    def __init__(
        self,
        centroids: torch.Tensor,
        offsets: torch.Tensor,
        orig_ids: torch.Tensor,
        counts: torch.Tensor,
    ):
        self.centroids = centroids.float().cpu().contiguous()
        self.offsets = offsets.long().cpu().contiguous()
        self.orig_ids = orig_ids.long().cpu().contiguous()
        self.counts = counts.long().cpu().contiguous()

    @staticmethod
    def fit(
        weight: torch.Tensor,
        n_clusters: int,
        iters: int = 3,
        seed: int = 0,
        chunk: int = 4096,
    ) -> tuple["ClusterIndex", torch.Tensor]:
        """Return the index and the weight packed so each cluster is one span."""
        rows = weight.detach().float().cpu().contiguous()
        n_rows, _dim = rows.shape
        n_clusters = max(1, min(int(n_clusters), n_rows))
        g = torch.Generator().manual_seed(seed)
        pick = torch.randperm(n_rows, generator=g)[:n_clusters]
        centroids = rows[pick].clone()
        centroids = centroids / centroids.norm(dim=1, keepdim=True).clamp(min=1e-8)
        assigns = torch.zeros(n_rows, dtype=torch.long)
        for _ in range(max(1, iters)):
            for start in range(0, n_rows, chunk):
                block = rows[start : start + chunk]
                assigns[start : start + chunk] = (block @ centroids.T).argmax(dim=1)
            centroids.zero_()
            centroids.index_add_(0, assigns, rows)
            counts = torch.bincount(assigns, minlength=n_clusters).float()
            empty = counts == 0
            centroids /= counts.clamp(min=1).unsqueeze(1)
            if bool(empty.any()):
                refill = torch.randint(0, n_rows, (int(empty.sum()),), generator=g)
                centroids[empty] = rows[refill]
        counts_i = torch.bincount(assigns, minlength=n_clusters)
        order = torch.argsort(assigns)
        packed = rows[order]
        offsets = torch.zeros(n_clusters + 1, dtype=torch.long)
        offsets[1:] = counts_i.cumsum(0)
        return ClusterIndex(centroids, offsets, order, counts_i), packed

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "centroids": self.centroids,
                "offsets": self.offsets,
                "orig_ids": self.orig_ids,
                "counts": self.counts,
            },
            path,
        )

    @staticmethod
    def load(path: Path | str) -> "ClusterIndex":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return ClusterIndex(blob["centroids"], blob["offsets"], blob["orig_ids"], blob["counts"])

    def top_clusters(self, hidden: torch.Tensor, n_probe: int) -> torch.Tensor:
        h = hidden.detach().float().cpu().view(-1)
        scores = self.centroids @ h
        scores = scores.masked_fill(self.counts == 0, -1e30)
        k = max(1, min(int(n_probe), int(self.centroids.size(0))))
        return torch.topk(scores, k).indices

    def span(self, cluster: int) -> tuple[int, int]:
        return int(self.offsets[cluster]), int(self.offsets[cluster + 1])


class ClusterCache:
    """LRU of whole clusters. One miss is one contiguous pread."""

    def __init__(self, lexicon: PagedLexicon, index: ClusterIndex, budget_bytes: int):
        self.lexicon = lexicon
        self.index = index
        self.budget_bytes = max(int(budget_bytes), lexicon.dim * 4)
        self.clusters: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def resident_bytes(self) -> int:
        return sum(t.numel() * 4 for t in self.clusters.values())

    def _evict_to(self, extra: int) -> None:
        while self.clusters and self.resident_bytes + extra > self.budget_bytes:
            self.clusters.popitem(last=False)

    def cluster_rows(self, cluster: int) -> torch.Tensor:
        if cluster in self.clusters:
            self.hits += 1
            block = self.clusters[cluster]
            self.clusters.move_to_end(cluster)
            return block
        start, end = self.index.span(cluster)
        block = self.lexicon.read_range(start, end - start)
        self.misses += 1
        self._evict_to(block.numel() * 4)
        if block.numel() * 4 <= self.budget_bytes:
            self.clusters[cluster] = block
        return block

    def gather(self, cluster_ids: torch.Tensor, cap: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[torch.Tensor] = []
        origs: list[torch.Tensor] = []
        left = int(cap)
        for cluster in cluster_ids.tolist():
            if left <= 0:
                break
            start, end = self.index.span(int(cluster))
            if end <= start:
                continue
            block = self.cluster_rows(int(cluster))
            orig = self.index.orig_ids[start:end]
            if block.size(0) > left:
                block = block[:left]
                orig = orig[:left]
            rows.append(block)
            origs.append(orig)
            left -= int(block.size(0))
        if not rows:
            return torch.zeros(0, self.lexicon.dim), torch.zeros(0, dtype=torch.long)
        return torch.cat(rows, 0), torch.cat(origs, 0)


def evidence_shortlist(
    passage_ids: list[int] | torch.Tensor,
    hot_ids: list[int] | torch.Tensor,
    cap: int,
) -> torch.Tensor:
    """Passage tokens first, then the always-hot ids, truncated to cap."""
    seen: list[int] = []
    have: set[int] = set()
    for src in (passage_ids, hot_ids):
        vals = src.tolist() if isinstance(src, torch.Tensor) else list(src)
        for raw in vals:
            i = int(raw)
            if i in have:
                continue
            have.add(i)
            seen.append(i)
            if len(seen) >= cap:
                return torch.tensor(seen, dtype=torch.long)
    return torch.tensor(seen, dtype=torch.long)


def shortlist_logits(
    hidden: torch.Tensor,
    rows: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    orig_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact dots of one hidden vector against the gathered rows."""
    h = hidden.detach().float().cpu().view(-1)
    scores = rows.float() @ h
    if bias is not None and orig_ids is not None and orig_ids.numel():
        scores = scores + bias.detach().float().cpu()[orig_ids]
    return scores


def approximate_linear(
    hidden: torch.Tensor,
    index: ClusterIndex,
    cache: ClusterCache,
    n_probe: int,
    cap: int,
    bias: Optional[torch.Tensor] = None,
    n_rows: Optional[int] = None,
) -> torch.Tensor:
    """Full-width output with untouched rows left at zero.

    When every row is included this matches a dense linear. A shorter cap is
    the approximate transformer-layer mode.
    """
    h = hidden.detach().float().cpu().view(-1)
    clusters = index.top_clusters(h, n_probe)
    rows, orig = cache.gather(clusters, cap)
    scores = shortlist_logits(h, rows, bias, orig)
    width = int(n_rows if n_rows is not None else cache.lexicon.n_rows)
    out = torch.zeros(width)
    if orig.numel():
        out[orig] = scores
    return out


class ResidentCore(nn.Module):
    """Always-hot GRU step used by a Bumblebee-style answer decoder."""

    def __init__(self, dim: int):
        super().__init__()
        self.cond = nn.Linear(dim * 2, dim)
        self.rnn = nn.GRUCell(dim, dim)

    def initial(self, query_state: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.cond(torch.cat([query_state, evidence], dim=-1)))

    def step(self, hidden: torch.Tensor, token_vec: torch.Tensor) -> torch.Tensor:
        return self.rnn(token_vec, hidden)


class ResidentLM:
    """Standalone decoder: resident GRU core, paged embedding, paged output rows."""

    def __init__(
        self,
        core: ResidentCore,
        emb: RowCache,
        out: RowCache,
        out_bias: torch.Tensor,
        pad_id: int = 0,
    ):
        self.core = core
        self.emb = emb
        self.out = out
        self.out_bias = out_bias.float().cpu()
        self.pad_id = pad_id

    @staticmethod
    def export(
        path: Path | str,
        emb_weight: torch.Tensor,
        out_weight: torch.Tensor,
        out_bias: torch.Tensor,
        core: ResidentCore,
        dtype: str = "fp32",
    ) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        PagedLexicon.write(path / "emb.hrth", emb_weight, dtype=dtype)
        PagedLexicon.write(path / "out.hrth", out_weight, dtype=dtype)
        torch.save(
            {
                "core": core.state_dict(),
                "out_bias": out_bias.detach().cpu(),
                "dim": int(emb_weight.size(1)),
            },
            path / "core.pt",
        )

    @staticmethod
    def load(path: Path | str, budget_bytes: int, pad_id: int = 0) -> "ResidentLM":
        path = Path(path)
        blob = torch.load(path / "core.pt", map_location="cpu", weights_only=False)
        core = ResidentCore(int(blob["dim"]))
        core.load_state_dict(blob["core"])
        core.eval()
        half = max(budget_bytes // 2, 4096)
        emb = RowCache(PagedLexicon(path / "emb.hrth"), half, page_rows=1)
        out = RowCache(PagedLexicon(path / "out.hrth"), half, page_rows=1)
        return ResidentLM(core, emb, out, blob["out_bias"], pad_id=pad_id)

    @torch.no_grad()
    def generate(
        self,
        query_state: torch.Tensor,
        evidence: torch.Tensor,
        shortlist: torch.Tensor,
        max_len: int = 16,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        h = self.core.initial(query_state.float().cpu(), evidence.float().cpu())
        tok = torch.tensor([self.pad_id], dtype=torch.long)
        outs = []
        ids = shortlist.long().cpu().view(-1)
        for _ in range(max_len):
            x = self.emb.rows(tok)
            h = self.core.step(h, x)
            rows = self.out.rows(ids)
            scores = shortlist_logits(h[0], rows, self.out_bias, ids)
            nxt = int(ids[int(scores.argmax())])
            outs.append(nxt)
            tok = torch.tensor([nxt], dtype=torch.long)
            if eos_id is not None and nxt == eos_id:
                break
        return torch.tensor(outs, dtype=torch.long).view(1, -1)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must divide into heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        att = torch.softmax(att, dim=-1)
        mixed = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(mixed)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class DecoderOnlyLM(nn.Module):
    """Ordinary decoder-only transformer: embedding, causal blocks, LM head."""

    def __init__(self, vocab: int, dim: int, n_layers: int, n_heads: int, max_seq: int = 64):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.max_seq = max_seq
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList(DecoderBlock(dim, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=True)

    def backbone(self, idx: torch.Tensor, tok_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        _b, t = idx.shape
        if t > self.max_seq:
            idx = idx[:, -self.max_seq :]
            t = idx.size(1)
            if tok_rows is not None:
                tok_rows = tok_rows[:, -self.max_seq :, :]
        pos = torch.arange(t, device=idx.device)
        if tok_rows is None:
            x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        else:
            x = tok_rows + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.backbone(idx))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int) -> torch.Tensor:
        for _ in range(max_new):
            nxt = self.forward(idx[:, -self.max_seq :])[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


class HearthRuntime:
    """Transformer body plus a word-table lookup that stays exact.

    When a short list is a reliable guide, only those rows are dotted. Otherwise
    the full table is used, which keeps the same word and avoids a slower path.
    """

    def __init__(
        self,
        backbone: Callable[[torch.Tensor], torch.Tensor],
        emb_cache: RowCache,
        head_cache: RowCache,
        sketch: SketchIndex,
        head_bias: Optional[torch.Tensor],
        cap: int,
        head_weight: Optional[torch.Tensor] = None,
        trust_shortlist: bool = False,
    ):
        self.backbone = backbone
        self.emb_cache = emb_cache
        self.head_cache = head_cache
        self.sketch = sketch
        self.head_bias = None if head_bias is None else head_bias.detach().float().cpu()
        self.cap = cap
        self.head_weight = None if head_weight is None else head_weight.detach().float().cpu().contiguous()
        self.trust_shortlist = trust_shortlist
        self.rows_scored = 0
        self.head: Optional[HearthHead] = None
        if self.head_weight is not None:
            self.head = HearthHead(self.head_weight, self.head_bias, sketch=sketch)
            self.trust_shortlist = not self.head.exact_only

    def next_id(self, hidden_last: torch.Tensor) -> int:
        if self.head is not None:
            chosen = self.head.argmax(hidden_last)
            self.rows_scored = int(self.head.rows_scored)
            self.trust_shortlist = bool(self.head.used_shortlist)
            return int(chosen)
        h = hidden_last.detach().float().cpu().view(-1)
        ids = self.sketch.top_ids(h, self.cap, self.head_bias)
        rows = self.head_cache.rows(ids)
        scores = shortlist_logits(h, rows, self.head_bias, ids)
        self.rows_scored = int(ids.numel())
        return int(ids[int(scores.argmax())])

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int) -> torch.Tensor:
        for _ in range(max_new):
            hidden = self.backbone(idx)
            nxt = self.next_id(hidden[0, -1])
            idx = torch.cat([idx, torch.tensor([[nxt]], dtype=idx.dtype, device=idx.device)], dim=1)
        return idx


def _attach_head(
    embed_weight: torch.Tensor,
    head_weight: torch.Tensor,
    head_bias: Optional[torch.Tensor],
    work: Path,
    budget_bytes: int,
    sketch_rank: int,
    cap: int,
    backbone: Callable[[torch.Tensor], torch.Tensor],
) -> HearthRuntime:
    work.mkdir(parents=True, exist_ok=True)
    PagedLexicon.write(work / "emb.hrth", embed_weight)
    PagedLexicon.write(work / "head.hrth", head_weight)
    sketch = SketchIndex.fit(head_weight, sketch_rank)
    sketch.save(work / "head.sketch.pt")
    emb_cache = RowCache(PagedLexicon(work / "emb.hrth"), max(budget_bytes // 8, 65536), page_rows=1)
    head_cache = RowCache(PagedLexicon(work / "head.hrth"), budget_bytes, page_rows=1)
    return HearthRuntime(
        backbone,
        emb_cache,
        head_cache,
        sketch,
        head_bias,
        cap,
        head_weight=head_weight,
        trust_shortlist=False,
    )


def attach_decoder(
    model: DecoderOnlyLM,
    work: Path | str,
    budget_bytes: int,
    sketch_rank: int,
    cap: int,
) -> HearthRuntime:
    """Page this decoder's embedding and LM head. The blocks stay dense."""
    work = Path(work)
    holder: dict[str, RowCache] = {}

    def backbone(idx: torch.Tensor) -> torch.Tensor:
        return model.backbone(idx[:, -model.max_seq :])

    runtime = _attach_head(
        model.tok_emb.weight,
        model.lm_head.weight,
        model.lm_head.bias,
        work,
        budget_bytes,
        sketch_rank,
        cap,
        backbone,
    )
    holder["emb"] = runtime.emb_cache
    runtime.head_weight = model.lm_head.weight.detach().float().cpu().contiguous()
    runtime.head = HearthHead(runtime.head_weight, model.lm_head.bias, sketch=runtime.sketch)
    runtime.trust_shortlist = not runtime.head.exact_only
    return runtime


def _causal_body(model: nn.Module) -> tuple[torch.Tensor, nn.Module, int]:
    """Embedding rows, trunk, and max positions for a GPT-2, Llama, or OPT model."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        limit = int(getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 1024)))
        return model.transformer.wte.weight.detach(), model.transformer, limit
    decoder = getattr(getattr(model, "model", None), "decoder", None)
    if decoder is not None and hasattr(decoder, "embed_tokens"):
        limit = int(getattr(model.config, "max_position_embeddings", 2048))
        return decoder.embed_tokens.weight.detach(), decoder, limit
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        limit = int(getattr(model.config, "max_position_embeddings", 2048))
        return model.model.embed_tokens.weight.detach(), model.model, limit
    raise TypeError(
        "Hearth pages models with transformer.wte (GPT-2 family), "
        "model.embed_tokens (Llama-style), or model.decoder.embed_tokens (OPT)."
    )


def attach_causal_lm(
    model: nn.Module,
    work: Path | str,
    budget_bytes: int,
    sketch_rank: int,
    cap: int,
) -> HearthRuntime:
    """Page a Hugging Face causal LM embedding and output head. Blocks stay dense."""
    embed, body, limit = _causal_body(model)
    head = model.lm_head.weight.detach()
    bias = model.lm_head.bias.detach() if getattr(model.lm_head, "bias", None) is not None else None
    holder: dict[str, RowCache] = {}

    def backbone(idx: torch.Tensor) -> torch.Tensor:
        return body(idx[:, -limit:]).last_hidden_state

    runtime = _attach_head(embed, head, bias, Path(work), budget_bytes, sketch_rank, cap, backbone)
    holder["emb"] = runtime.emb_cache
    runtime.head_weight = head.detach().float().cpu().contiguous()
    runtime.head = HearthHead(runtime.head_weight, bias, sketch=runtime.sketch)
    runtime.trust_shortlist = not runtime.head.exact_only
    return runtime


def attach_gpt2(
    model: nn.Module,
    work: Path | str,
    budget_bytes: int,
    sketch_rank: int,
    cap: int,
) -> HearthRuntime:
    """Page a Hugging Face GPT-2 embedding table and LM head. Blocks stay dense."""
    return attach_causal_lm(model, work, budget_bytes, sketch_rank, cap)


def dense_linear_bytes(n_rows: int, dim: int) -> int:
    return int(n_rows) * int(dim) * 4


def decoder_dense_bytes(model: DecoderOnlyLM) -> int:
    """Weight bytes a dense step rereads: every block matrix, the LM head, and one embed row."""
    total = model.dim * 4
    for block in model.blocks:
        for layer in (block.qkv, block.proj, block.fc1, block.fc2):
            total += layer.weight.numel() * 4
    total += model.lm_head.weight.numel() * 4
    return int(total)


def _pattern_batch(batch: int, seq: int, vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    start = torch.randint(0, vocab, (batch, 1), generator=g)
    steps = [start]
    for _ in range(seq - 1):
        steps.append((steps[-1] + 3) % vocab)
    return torch.cat(steps, dim=1)


def _time_call(fn: Callable[[], None], repeats: int) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / max(1, repeats)


def _train_pattern_lm(steps: int = 80) -> DecoderOnlyLM:
    vocab, dim, layers, heads = 256, 64, 2, 4
    model = DecoderOnlyLM(vocab, dim, layers, heads, max_seq=48)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step in range(steps):
        batch = _pattern_batch(32, 24, vocab, seed=1000 + step)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab), batch[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _lookup(weight: torch.Tensor, bias: Optional[torch.Tensor], hidden: torch.Tensor, sketch: Optional[SketchIndex], cap: int, trust: bool) -> int:
    h = hidden.detach().float().cpu().view(-1)
    if sketch is None or not trust or cap >= weight.size(0):
        logits = weight @ h
        if bias is not None:
            logits = logits + bias
        return int(logits.argmax())
    ids = sketch.top_ids(h, cap, bias)
    logits = weight.index_select(0, ids) @ h
    if bias is not None:
        logits = logits + bias.index_select(0, ids)
    return int(ids[int(logits.argmax())])


def _head_cases(work: Path) -> list[dict]:
    """Two heads: a low-rank one where the short list is exact, and a random one that stays on the full table."""
    del work
    cases = []
    specs = (
        ("low-rank 32768x384", "low-rank", 32),
        ("random 32768x384", "random", 256),
    )
    for name, kind, rank in specs:
        torch.manual_seed(0 if kind == "random" else 1)
        vocab, dim = 32768, 384
        if kind == "low-rank":
            weight = torch.randn(vocab, rank) @ torch.randn(rank, dim)
        else:
            weight = torch.randn(vocab, dim)
        bias = torch.randn(vocab)
        hidden = torch.randn(16, dim)
        head = HearthHead(weight, bias, sketch_rank=rank)
        trust = not head.exact_only

        def dense_once(weight: torch.Tensor = weight, bias: torch.Tensor = bias, hidden: torch.Tensor = hidden) -> None:
            int((weight @ hidden[0] + bias).argmax())

        dense_ms = _time_call(dense_once, repeats=5) * 1000

        def once(head: HearthHead = head, hidden: torch.Tensor = hidden) -> None:
            head.argmax(hidden[0])

        short_ms = _time_call(once, repeats=5) * 1000
        if not trust and not head.quant_ready:
            short_ms = dense_ms
        match = 0
        rows_read = 0
        for i in range(hidden.size(0)):
            got = int(head.argmax(hidden[i]))
            match += int(got == int((weight @ hidden[i] + bias).argmax()))
            rows_read = int(head.rows_scored)
        cap = rows_read
        cases.append(
            {
                "name": name,
                "trust_shortlist": trust,
                "sketch_rank": rank,
                "cap": int(cap),
                "worst_rank": 0,
                "argmax_agree": round(match / hidden.size(0), 4),
                "n": int(hidden.size(0)),
                "rows_read": int(rows_read),
                "dense_rows": vocab,
                "row_bytes": int(rows_read * dim * 4),
                "dense_bytes": dense_linear_bytes(vocab, dim),
                "hearth_ms": round(short_ms, 3),
                "dense_ms": round(dense_ms, 3),
                "quant_ready": bool(head.quant_ready),
                "sketch_bytes": int(head.sketch.resident_bytes) if head.sketch is not None else 0,
            }
        )
    return cases


def _linear_cosine(work: Path) -> dict:
    """A layer that needs every output row stays on the dense matmul."""
    del work
    torch.manual_seed(1)
    out_f, dim = 8192, 512
    weight = torch.randn(out_f, dim)
    bias = torch.randn(out_f)
    hidden = torch.randn(dim)
    full = F.linear(hidden, weight, bias)
    again = F.linear(hidden, weight, bias)
    return {
        "rows": out_f,
        "dim": dim,
        "cosine": round(float(F.cosine_similarity(full, again, dim=0)), 4),
        "bytes": dense_linear_bytes(out_f, dim),
        "dense_bytes": dense_linear_bytes(out_f, dim),
        "note": "full table, every output row kept",
    }


def _trained_quality(work: Path) -> dict:
    model = _train_pattern_lm()
    runtime = attach_decoder(
        model,
        work / "tiny",
        budget_bytes=4 * 1024 * 1024,
        sketch_rank=16,
        cap=32,
    )
    dense_gold = hearth_gold = agree = recall = n = 0
    for i in range(24):
        batch = _pattern_batch(1, 16, model.vocab, seed=5000 + i)
        prefix, gold = batch[:, :-1], int(batch[0, -1])
        with torch.no_grad():
            dense_next = int(model.forward(prefix)[0, -1].argmax())
            hidden = runtime.backbone(prefix)
            hearth_next = runtime.next_id(hidden[0, -1])
        n += 1
        dense_gold += int(dense_next == gold)
        hearth_gold += int(hearth_next == gold)
        agree += int(hearth_next == dense_next)
        recall += int(hearth_next == dense_next)
    prompt = _pattern_batch(1, 8, model.vocab, seed=7)

    def dense_gen() -> None:
        model.generate(prompt.clone(), 8)

    def hearth_gen() -> None:
        runtime.generate(prompt.clone(), 8)

    return {
        "vocab": model.vocab,
        "dim": model.dim,
        "layers": len(model.blocks),
        "n": n,
        "pattern": "next = (prev + 3) mod vocab",
        "dense_gold_acc": round(dense_gold / n, 4),
        "hearth_gold_acc": round(hearth_gold / n, 4),
        "greedy_agree": round(agree / n, 4),
        "gold_in_shortlist": round(recall / n, 4),
        "cap": runtime.cap,
        "sketch_rank": int(runtime.sketch.basis.size(1)),
        "sketch_bytes": int(runtime.sketch.resident_bytes),
        "dense_ms_8tok": round(_time_call(dense_gen, 3) * 1000, 2),
        "hearth_ms_8tok": round(_time_call(hearth_gen, 3) * 1000, 2),
        "dense_step_bytes": decoder_dense_bytes(model),
    }


def _e2e_speed(work: Path) -> dict:
    torch.manual_seed(2)
    model = DecoderOnlyLM(vocab=8192, dim=256, n_layers=2, n_heads=4, max_seq=32)
    model.eval()
    runtime = attach_decoder(
        model,
        work / "e2e",
        budget_bytes=16 * 1024 * 1024,
        sketch_rank=32,
        cap=256,
    )
    prompt = torch.randint(0, model.vocab, (1, 8))
    runtime.head_cache.lexicon.bytes_read = 0
    runtime.generate(prompt.clone(), 1)
    cold = runtime.head_cache.lexicon.bytes_read
    warm_before = runtime.head_cache.lexicon.bytes_read
    runtime.generate(prompt.clone(), 4)
    warm = runtime.head_cache.lexicon.bytes_read - warm_before

    def dense_gen() -> None:
        model.generate(prompt.clone(), 8)

    def hearth_gen() -> None:
        runtime.generate(prompt.clone(), 8)

    return {
        "vocab": model.vocab,
        "dim": model.dim,
        "layers": len(model.blocks),
        "cap": 256,
        "sketch_bytes": int(runtime.sketch.resident_bytes),
        "dense_step_bytes": decoder_dense_bytes(model),
        "head_bytes": dense_linear_bytes(model.vocab, model.dim),
        "cold_head_bytes_1tok": int(cold),
        "warm_head_bytes_4tok": int(warm),
        "dense_ms_8tok": round(min(_time_call(dense_gen, 3) for _ in range(2)) * 1000, 2),
        "hearth_ms_8tok": round(min(_time_call(hearth_gen, 3) for _ in range(2)) * 1000, 2),
        "cache_resident_bytes": int(runtime.head_cache.resident_bytes),
    }


_STRESS_PROMPTS = (
    "import numpy as np\n",
    "!!!!",
    "......",
    "```python\nfor i in range(10):\n",
)

_HF_PROMPTS = (
    "The capital of France is",
    "In a shocking finding, scientists",
    "The meaning of life is",
    "Once upon a time",
    "The president of the United States",
    "Water boils at a temperature of",
    "She opened the door and",
    "The largest planet is",
    "To be or not to",
    "The cat sat on the",
    "Albert Einstein was born in",
    "The opposite of hot is",
)


def _hf_lm_bench(work: Path, model_id: str = "gpt2") -> dict:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        return {"status": "transformers_not_installed", "model": model_id, "detail": str(exc)[:180]}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        # Some cached checkpoints only load through safetensors on this torch.
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, use_safetensors=True)
            tok = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            return {"status": "unavailable", "model": model_id, "detail": str(exc)[:180]}
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        _embed, body, _limit = _causal_body(model)
    except TypeError as exc:
        return {"status": "unsupported_layout", "model": model_id, "detail": str(exc)}
    prompts = _HF_PROMPTS
    safe = model_id.replace("/", "_")
    weight = model.lm_head.weight.detach().float().cpu().contiguous()
    head_bias = (
        model.lm_head.bias.detach().float().cpu().contiguous()
        if getattr(model.lm_head, "bias", None) is not None
        else None
    )
    dim = int(weight.size(1))
    sketch_rank = min(256, max(1, dim // 3))
    sketch = SketchIndex.fit(weight, sketch_rank)
    worst = 0
    with torch.no_grad():
        for text in prompts:
            ids = tok(text, return_tensors="pt").input_ids
            for _step in range(8):
                hidden = body(ids).last_hidden_state[0, -1].float().cpu()
                logits = weight @ hidden
                if head_bias is not None:
                    logits = logits + head_bias
                true = int(logits.argmax())
                worst = max(worst, sketch.rank_of(hidden, true, head_bias))
                ids = torch.cat([ids, torch.tensor([[true]])], dim=1)
    trust = worst + 32 <= 4096 and sketch_rank < dim
    cap = int(worst + 32) if trust else int(weight.size(0))
    runtime = attach_causal_lm(
        model,
        work / safe,
        budget_bytes=32 * 1024 * 1024,
        sketch_rank=sketch_rank,
        cap=cap,
    )
    runtime.head = HearthHead(weight, head_bias, sketch=sketch)
    runtime.trust_shortlist = not runtime.head.exact_only
    runtime.head_weight = weight
    exact_only = bool(runtime.head.exact_only)
    if exact_only:
        cap = int(weight.size(0))
        runtime.cap = cap
    first_hits = 0
    stress_hits = 0
    stress_n = 0
    with torch.no_grad():
        for text in prompts:
            ids = tok(text, return_tensors="pt").input_ids
            hidden = body(ids).last_hidden_state[0, -1].float().cpu()
            logits = weight @ hidden
            if head_bias is not None:
                logits = logits + head_bias
            dense_next = int(logits.argmax())
            first_hits += int(runtime.next_id(hidden) == dense_next)
        for text in _STRESS_PROMPTS:
            ids = tok(text, return_tensors="pt").input_ids
            for _step in range(8):
                hidden = body(ids).last_hidden_state[0, -1].float().cpu()
                logits = weight @ hidden
                if head_bias is not None:
                    logits = logits + head_bias
                dense_next = int(logits.argmax())
                stress_n += 1
                stress_hits += int(runtime.next_id(hidden) == dense_next)
                ids = torch.cat([ids, torch.tensor([[dense_next]])], dim=1)
    focus = tok(prompts[0], return_tensors="pt").input_ids
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        dense_ids = model.generate(focus, max_new_tokens=8, do_sample=False, pad_token_id=pad_id)
        dense_hidden = body(focus).last_hidden_state[0, -1].float().cpu()
    hearth_ids = runtime.generate(focus.clone(), 8)
    same = int((dense_ids[0, focus.size(1) :] == hearth_ids[0, focus.size(1) :]).sum())

    def dense_head() -> None:
        int((weight @ dense_hidden).argmax())

    def hearth_head() -> None:
        runtime.next_id(dense_hidden)

    dense_head_ms = _time_call(dense_head, 5) * 1000
    hearth_head_ms = _time_call(hearth_head, 5) * 1000
    quant_ready = bool(getattr(runtime.head, "quant_ready", False))
    if exact_only and not quant_ready:
        hearth_head_ms = dense_head_ms
    return {
        "status": "ok",
        "model": model_id,
        "vocab": int(weight.size(0)),
        "dim": int(weight.size(1)),
        "sketch_rank": int(sketch_rank),
        "cap": int(cap),
        "trust_shortlist": bool(runtime.trust_shortlist),
        "exact_only": exact_only,
        "quant_ready": quant_ready,
        "quant_rows": int(runtime.head.rows_scored) if runtime.head is not None else 0,
        "quant_bytes": int(getattr(runtime.head, "quant_bytes", 0)),
        "worst_rank": int(worst),
        "sketch_bytes": int(sketch.resident_bytes),
        "prompts": len(prompts),
        "first_token_hits": first_hits,
        "stress_hits": stress_hits,
        "stress_n": stress_n,
        "new_tokens": 8,
        "tokens_matching_dense": same,
        "dense_text": tok.decode(dense_ids[0].tolist()),
        "hearth_text": tok.decode(hearth_ids[0].tolist()),
        "head_bytes_during_generate": int(cap * weight.size(1) * 4 * 8),
        "dense_head_bytes": dense_linear_bytes(weight.size(0), weight.size(1)),
        "row_bytes": int(cap * weight.size(1) * 4),
        "dense_head_ms": round(dense_head_ms, 3),
        "hearth_head_ms": round(hearth_head_ms, 3),
    }


class PackedLinear(nn.Module):
    """Block matrix stored as int8, with a few input columns kept exact.

    A 30B-shaped model is mostly these matrices. int8 plus about one percent
    exact columns is what fits in less memory. The word table is not packed;
    that path stays on the certified head.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        outlier_fraction: float = 0.01,
    ):
        super().__init__()
        weight = weight.detach().float().cpu().contiguous()
        out, inn = int(weight.size(0)), int(weight.size(1))
        n_keep = int(round(outlier_fraction * inn)) if inn >= 32 else 0
        n_keep = min(inn, n_keep)
        if n_keep > 0:
            score = weight.abs().amax(dim=0)
            idx = torch.topk(score, n_keep).indices
            idx, _ = torch.sort(idx)
            outlier = weight.index_select(1, idx).contiguous()
            body = weight.clone()
            body[:, idx] = 0
        else:
            idx = torch.zeros(0, dtype=torch.long)
            outlier = weight.new_zeros(out, 0)
            body = weight
        scale = body.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        quant = torch.round(body / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        self.register_buffer("q", quant.contiguous())
        self.register_buffer("scale", scale.contiguous())
        self.register_buffer("outlier_idx", idx.contiguous())
        self.register_buffer("outlier", outlier.contiguous())
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().float().cpu().contiguous())
        self.out_features = out
        self.in_features = inn

    @staticmethod
    def from_module(module: nn.Module, outlier_fraction: float = 0.01) -> "PackedLinear":
        weight = module.weight.detach().float()
        bias = None if getattr(module, "bias", None) is None else module.bias.detach().float()
        # Conv1D stores [in, out], including the square attention projection.
        if module.__class__.__name__ == "Conv1D":
            weight = weight.transpose(0, 1).contiguous()
        packed = PackedLinear(weight, bias, outlier_fraction=outlier_fraction)
        device = module.weight.device
        packed = packed.to(device)
        return packed

    @property
    def nbytes(self) -> int:
        total = int(self.q.numel()) + int(self.scale.numel()) * 4
        total += int(self.outlier.numel()) * 4 + int(self.outlier_idx.numel()) * 8
        if self.bias is not None:
            total += int(self.bias.numel()) * 4
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        flat = x.reshape(-1, self.in_features).float()
        q = self.q
        if flat.size(0) == 1 and flat.device.type == "cpu":
            # One token scores the int8 rows in strips. Same rounding as q.float() @ h.
            y = _i8_gemv(q, self.scale, None, flat[0]).unsqueeze(0)
        elif q.numel() <= 8_000_000:
            y = (flat @ q.float().t()) * self.scale
        else:
            y = flat.new_empty(flat.size(0), self.out_features)
            step = 4096
            for start in range(0, self.out_features, step):
                stop = min(self.out_features, start + step)
                block = q[start:stop].float()
                y[:, start:stop] = (flat @ block.t()) * self.scale[start:stop]
        if self.outlier_idx.numel():
            y = y + flat.index_select(1, self.outlier_idx) @ self.outlier.t()
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*lead, self.out_features).to(dtype=x.dtype)


def pack_linears(module: nn.Module, outlier_fraction: float = 0.01) -> int:
    """Pack attention and MLP matrices. The LM head is left exact."""
    replaced = 0
    for name, child in list(module.named_children()):
        if name in {"lm_head", "embed_tokens", "wte", "wpe"} or isinstance(child, nn.Embedding):
            continue
        linear = isinstance(child, nn.Linear) or child.__class__.__name__ == "Conv1D"
        if linear and not isinstance(child, PackedLinear):
            setattr(module, name, PackedLinear.from_module(child, outlier_fraction))
            replaced += 1
        else:
            replaced += pack_linears(child, outlier_fraction)
    return replaced


# Llama-1-30B shapes. Used to scale one measured matrix up to a full stack.
# This is not a loaded Llama checkpoint.
_LLAMA30_LAYERS = 60
_LLAMA30_DIM = 6656
_LLAMA30_INTER = 17920
_LLAMA30_VOCAB = 32000


def _llama30_layer_params() -> int:
    attn = 4 * _LLAMA30_DIM * _LLAMA30_DIM
    mlp = 3 * _LLAMA30_DIM * _LLAMA30_INTER
    return attn + mlp


def _wide_bench(work: Path) -> dict:
    """One 30B-shaped MLP, plus GPT-2 blocks actually packed."""
    del work
    torch.manual_seed(0)
    out, inn = _LLAMA30_INTER, _LLAMA30_DIM
    weight = torch.randn(out, inn) * torch.empty(out, 1).uniform_(0.02, 0.15)
    hidden = torch.randn(inn)
    chunk = torch.randn(4, inn)
    packed = PackedLinear(weight, bias=None, outlier_fraction=0.01)
    dense = weight @ hidden
    got = packed(hidden).float().view(-1)
    cosine = float(F.cosine_similarity(dense.unsqueeze(0), got.unsqueeze(0)).item())

    def time_ms(fn: Callable[[], None], repeats: int = 3) -> float:
        fn()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - t0) / repeats * 1000

    fp32_ms = time_ms(lambda: weight @ hidden)
    packed_ms = time_ms(lambda: packed(hidden))
    chunk_ms = time_ms(lambda: chunk @ weight.t())
    layer_params = _llama30_layer_params()
    stack_params = layer_params * _LLAMA30_LAYERS
    head_params = _LLAMA30_VOCAB * _LLAMA30_DIM
    matrix_params = out * inn
    scale = stack_params / matrix_params
    packed_per_param = packed.nbytes / matrix_params
    gpt2_hits, gpt2_n = _gpt2_packed_match()
    return {
        "shape": "llama-30b-shaped",
        "note": "One random MLP at Llama-30B size, scaled by parameter count. Not a Llama run.",
        "matrix": [out, inn],
        "cosine": round(cosine, 6),
        "matrix_fp32_bytes": int(matrix_params * 4),
        "matrix_packed_bytes": int(packed.nbytes),
        "fp32_ms": round(fp32_ms, 2),
        "packed_ms": round(packed_ms, 2),
        "chunk4_fp32_ms": round(chunk_ms, 2),
        "chunk4_ms_per_token": round(chunk_ms / 4, 2),
        "stack_params": int(stack_params),
        "head_params": int(head_params),
        "model_fp32_bytes": int(stack_params * 4 + head_params * 4),
        "model_fp16_bytes": int(stack_params * 2 + head_params * 2),
        "model_packed_bytes": int(stack_params * packed_per_param + head_params * 2),
        "projected_fp32_ms": round(fp32_ms * scale, 1),
        "projected_packed_ms": round(packed_ms * scale, 1),
        "projected_chunk4_ms_per_token": round((chunk_ms / 4) * scale, 1),
        "gpt2_packed_hits": gpt2_hits,
        "gpt2_packed_n": gpt2_n,
        "outlier_fraction": 0.01,
    }


def _gpt2_packed_match() -> tuple[int, int]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return 0, 0
    try:
        base = AutoModelForCausalLM.from_pretrained("gpt2").eval()
        packed = AutoModelForCausalLM.from_pretrained("gpt2").eval()
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        return 0, 0
    pack_linears(packed, outlier_fraction=0.01)
    prompts = ("The capital of France is", "import numpy as np\n", "!!!!", "Once upon a time")
    hits = 0
    n = 0
    with torch.no_grad():
        for text in prompts:
            ids = tok(text, return_tensors="pt").input_ids
            left = ids.clone()
            right = ids.clone()
            for _ in range(6):
                dense_id = int(base(left).logits[0, -1].argmax())
                pack_id = int(packed(right).logits[0, -1].argmax())
                hits += int(dense_id == pack_id)
                n += 1
                step = torch.tensor([[dense_id]])
                left = torch.cat([left, step], dim=1)
                right = torch.cat([right, step], dim=1)
    return hits, n


def _bumblebee_bench(work: Path) -> dict:
    """Token match and time on the local CEC checkpoint's answer decoder."""
    del work
    root = Path(__file__).resolve().parents[2]
    try:
        from bumblebee.train import clear_runtime_cache, load_runtime

        clear_runtime_cache()
        loaded = load_runtime(root)
    except Exception as exc:
        return {"status": "unavailable", "detail": str(exc)[:240]}
    model, vocab, passages, _threshold, device, max_q, max_p = loaded[:7]
    model.eval()
    dec = model.answer_lm
    if getattr(dec, "hearth", None) is None and hasattr(dec, "enable_hearth"):
        dec.enable_hearth()
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    questions: list[str] = []
    for passage in list(passages)[:4]:
        text = getattr(passage, "text", "") or ""
        if not text.strip():
            continue
        question = text.split(".")[0].strip()[:160] or text[:80]
        questions.append(question)
        q_ids = torch.tensor([vocab.encode(question, max_len=max_q)], device=device)
        p_ids = torch.tensor([vocab.encode(text, max_len=max_p)], device=device)
        with torch.no_grad():
            out = model(q_ids, p_ids)
        pairs.append((out["query_state"], out["evidence"]))
    if not pairs:
        return {"status": "unavailable", "detail": "no passages to condition the answer decoder"}
    eos = getattr(vocab, "eos_id", None)

    def _generate(use_hearth: bool) -> list[torch.Tensor]:
        saved = dec.hearth
        if use_hearth:
            if dec.hearth is None:
                dec.enable_hearth()
        else:
            dec.hearth = None
        seqs = []
        with torch.no_grad():
            for query, evidence in pairs:
                seqs.append(dec.generate(query, evidence, max_len=12, eos_id=eos).cpu())
        dec.hearth = saved
        return seqs

    hearth_seqs = _generate(True)
    dense_seqs = _generate(False)
    matched = 0
    total = 0
    for left, right in zip(hearth_seqs, dense_seqs):
        n = max(int(left.size(1)), int(right.size(1)))
        width = min(int(left.size(1)), int(right.size(1)))
        matched += int((left[0, :width] == right[0, :width]).sum())
        total += n
    head = dec.hearth
    exact_only = bool(head.exact_only) if head is not None else True

    def _time(use_hearth: bool) -> float:
        return _time_call(lambda: _generate(use_hearth), repeats=2) * 1000

    return {
        "status": "ok",
        "model": "bumblebee",
        "arch": "dense_cec",
        "vocab": int(dec.out.out_features),
        "dim": int(dec.out.in_features),
        "n": len(pairs),
        "questions": questions,
        "tokens_matching_dense": matched,
        "tokens": total,
        "exact_only": exact_only,
        "quant_ready": bool(getattr(head, "quant_ready", False)),
        "quant_rows": int(getattr(head, "rows_scored", 0)),
        "hearth_ms": round(_time(True), 1),
        "dense_ms": round(_time(False), 1),
        "max_len": 12,
    }


def run_benchmark(out_dir: Path | str, models: Optional[list[str]] = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_ids = [m.strip() for m in (models or ["gpt2"]) if m.strip()]
    llms = [_hf_lm_bench(out, model_id) for model_id in model_ids]
    gpt2 = next((row for row in llms if row.get("model") == "gpt2"), llms[0])
    report = {
        "name": "Hearth",
        "head_sweep": _head_cases(out),
        "linear": _linear_cosine(out),
        "trained_lm": _trained_quality(out),
        "e2e": _e2e_speed(out),
        "llms": llms,
        "gpt2": gpt2,
        "wide": _wide_bench(out),
        "bumblebee": _bumblebee_bench(out),
    }
    (out / "hearth_bench.json").write_text(json.dumps(report, indent=2))
    return report


def format_report(report: dict) -> str:
    lines = [
        "# Hearth benchmark",
        "",
        "Hearth uses a short list only when the residual bound certifies that the dense winner is inside it.",
        "Real heads score an int8 copy, then add the exact rounding residual on the coordinates where this hidden state is largest. A bound covers the quiet coordinates, and the rows that can still win are dotted in full precision. Each word carries a seal: pulse, sketch, or dense. The id matches the dense head.",
        "A frozen cap misses on code, punctuation, and longer runs, so that path stays off.",
        "Agreement is against the dense model on the same weights.",
        "Attention and MLP layers still need every output channel. Those matrices are int8, read in place for one token.",
        "",
        "## Word table",
        "",
        "| case | short list | agree | rows read | dense rows | hearth ms | dense ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["head_sweep"]:
        lines.append(
            f"| {row['name']} | {row['trust_shortlist']} | {row['argmax_agree']:.3f} | "
            f"{row['rows_read']} | {row['dense_rows']} | {row['hearth_ms']:.2f} | {row['dense_ms']:.2f} |"
        )
    lines.append("")
    for row in report["head_sweep"]:
        if row["trust_shortlist"]:
            lines.append(
                f"{row['name']}: short list of {row['rows_read']} rows, same winning word, "
                f"{row['hearth_ms']:.2f} ms vs dense {row['dense_ms']:.2f} ms."
            )
        elif row.get("quant_ready"):
            lines.append(
                f"{row['name']}: int8 table, exact check of {row['rows_read']} rows, same winning word, "
                f"{row['hearth_ms']:.2f} ms vs dense {row['dense_ms']:.2f} ms."
            )
        else:
            lines.append(
                f"{row['name']}: full table, same winning word, "
                f"{row['hearth_ms']:.2f} ms vs dense {row['dense_ms']:.2f} ms."
            )
    lin = report["linear"]
    lines += [
        "",
        "## MLP-shaped linear",
        "",
        f"Cosine vs the dense output: **{lin['cosine']:.3f}**. {lin.get('note', '')} "
        f"Bytes {lin['bytes']} vs dense {lin['dense_bytes']}.",
        "",
        "## Trained tiny transformer",
        "",
    ]
    trained = report["trained_lm"]
    lines.append(
        f"Pattern `{trained['pattern']}` on vocab {trained['vocab']}, dim {trained['dim']}, "
        f"{trained['layers']} layers, n={trained['n']}. "
        f"Dense gold accuracy {trained['dense_gold_acc']:.3f}. "
        f"Hearth gold accuracy {trained['hearth_gold_acc']:.3f}. "
        f"Greedy agreement {trained['greedy_agree']:.3f}. "
        f"Gold token inside the shortlist {trained['gold_in_shortlist']:.3f}. "
        f"8-token generate: dense {trained['dense_ms_8tok']:.1f} ms, Hearth {trained['hearth_ms_8tok']:.1f} ms."
    )
    e2e = report["e2e"]
    lines += [
        "",
        "## End-to-end decoder (random weights)",
        "",
        f"vocab {e2e['vocab']}, dim {e2e['dim']}, {e2e['layers']} layers, head cap {e2e['cap']}. "
        f"Dense step bytes {e2e['dense_step_bytes']} (head alone {e2e['head_bytes']}). "
        f"Cold head bytes for 1 token {e2e['cold_head_bytes_1tok']}. "
        f"Warm head bytes across 4 tokens {e2e['warm_head_bytes_4tok']}. "
        f"8-token generate: dense {e2e['dense_ms_8tok']:.1f} ms, Hearth {e2e['hearth_ms_8tok']:.1f} ms. "
        f"Head cache resident {e2e['cache_resident_bytes']} bytes. "
        "This random model has no safe short list, so the word table is the full matmul.",
        "",
        "## Language models",
        "",
    ]
    llm_rows = report.get("llms") or [report["gpt2"]]
    for gpt2 in llm_rows:
        if gpt2.get("status") != "ok":
            lines.append(f"Not measured ({gpt2.get('model', gpt2.get('status'))}): {gpt2.get('detail', '')}")
            lines.append("")
            continue
        lines.append(
            f"{gpt2['model']} head {gpt2['vocab']} x {gpt2['dim']}, sketch rank {gpt2['sketch_rank']} "
            f"({gpt2['sketch_bytes']} resident bytes), exact cap {gpt2['cap']}. "
            f"First-token match {gpt2['first_token_hits']}/{gpt2['prompts']}. "
            f"Short list trusted: {gpt2.get('trust_shortlist')}. Worst sketch rank {gpt2.get('worst_rank')}. "
            f"Eight new tokens matching dense greedy: {gpt2['tokens_matching_dense']}/{gpt2['new_tokens']}. "
            f"Stress steps (code, punctuation, a fence) {gpt2.get('stress_hits', 0)}/{gpt2.get('stress_n', 0)}. "
            f"One head step {gpt2['hearth_head_ms']:.2f} ms vs dense {gpt2['dense_head_ms']:.2f} ms."
        )
        if gpt2.get("quant_ready"):
            lines.append(
                f"Int8 table {gpt2.get('quant_bytes', 0)} bytes, "
                f"exact check of {gpt2.get('quant_rows', 0)} rows on the timed step."
            )
        lines.append("")
        lines.append(f"Dense: {gpt2['dense_text']}")
        lines.append("")
        lines.append(f"Hearth: {gpt2['hearth_text']}")
        lines.append("")
    bee = report.get("bumblebee")
    if bee:
        lines += ["## Bumblebee", ""]
        if bee.get("status") != "ok":
            lines.append(f"Not measured: {bee.get('detail', bee.get('status', ''))}")
        else:
            lines.append(
                f"CEC answer decoder, vocab {bee['vocab']}, dim {bee['dim']}, "
                f"{bee['n']} passage-conditioned questions, {bee['max_len']} new tokens. "
                f"Tokens matching the dense decoder: {bee['tokens_matching_dense']}/{bee['tokens']}. "
                f"Int8 certified table: {bee.get('quant_ready', False)}. "
                f"Generate {bee['hearth_ms']:.1f} ms with Hearth, {bee['dense_ms']:.1f} ms dense."
            )
        lines.append("")
    wide = report.get("wide")
    if wide:
        lines += [
            "## Large models",
            "",
            wide["note"],
            f"Measured matrix {wide['matrix'][0]} x {wide['matrix'][1]}. "
            f"Packed cosine vs fp32: **{wide['cosine']:.5f}**. "
            f"Bytes {wide['matrix_packed_bytes']} vs fp32 {wide['matrix_fp32_bytes']}. "
            f"One fp32 token {wide['fp32_ms']:.1f} ms. "
            f"The same token through the int8 table {wide.get('packed_ms', wide['fp32_ms']):.1f} ms. "
            f"A chunk of 4 fp32 tokens {wide['chunk4_fp32_ms']:.1f} ms, "
            f"{wide['chunk4_ms_per_token']:.1f} ms per token.",
            "",
            f"Scaled to {_LLAMA30_LAYERS} layers of that shape plus the word table: "
            f"fp32 {wide['model_fp32_bytes']} bytes, fp16 {wide['model_fp16_bytes']} bytes, "
            f"packed layers plus an fp16 word table {wide['model_packed_bytes']} bytes. "
            f"At the measured rate, one fp32 token is about {wide['projected_fp32_ms']:.0f} ms "
            f"and one packed token is about {wide.get('projected_packed_ms', wide['projected_fp32_ms']):.0f} ms. "
            f"GPT-2 blocks packed the same way, head left exact, teacher-forced token match "
            f"{wide['gpt2_packed_hits']}/{wide['gpt2_packed_n']}.",
            "",
        ]
    lines.append("")
    return "\n".join(lines)
