"""Parameter scaling helpers for Mac-fit CEC training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ScaleConfig:
    """Embedding-heavy CEC so most of ~500M–1.5B params sit in a sparse hash table."""

    target_params: int
    vocab_buckets: int
    emb_dim: int
    dim: int
    max_p: int
    max_q: int
    max_a: int
    batch_size: int
    grad_accum: int
    estimated_params: int
    estimated_weight_gb_fp32: float
    estimated_train_gb: float
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_params(
    vocab_buckets: int,
    emb_dim: int,
    dim: int,
    answer_vocab: int = 8000,
) -> int:
    """Sparse hash encoder emb + small core + small answer head."""
    emb = vocab_buckets * emb_dim
    proj = emb_dim * dim + dim
    gru = 2 * (6 * dim * dim + 6 * dim)  # encoder + answer GRUCell
    misc = (
        2 * (dim * dim + dim)  # correlator
        + (2 * dim * dim + dim)
        + (dim + 1)
        + (dim + 1)
        + (2 * dim * dim + dim)  # cond
        + 4 * dim
    )
    ans = answer_vocab * dim + dim * answer_vocab + answer_vocab
    return int(emb + proj + gru + misc + ans)


def choose_mac_fit_config(
    target_params: int = 500_000_000,
    ram_gb: float = 8.0,
    *,
    dim: Optional[int] = None,
) -> ScaleConfig:
    """
    Pick an embedding-dominant layout for Mac training.

    Sparse hash emb tables are mostly untouched zeros (SparseAdam); cold pages
    compress, so ~1.5B headline params can train on 8GB-class machines.
    """
    usable = max(2.5, ram_gb - 3.5)
    if dim is not None:
        core_dim = dim
    elif target_params >= 1_200_000_000:
        core_dim = 256
    elif ram_gb >= 8:
        core_dim = 192
    else:
        core_dim = 128

    def fits(weight_gb: float, dense: int) -> bool:
        train_gb = weight_gb + dense * 8 / (1024**3) + 0.8
        if target_params >= 1_200_000_000:
            return dense * 8 / (1024**3) + 1.5 < usable
        if target_params >= 800_000_000:
            return train_gb * 0.55 <= usable + 1.0
        return train_gb <= usable + 0.5

    bucket_exps = (18, 19, 20, 21) if target_params >= 800_000_000 else (18, 19, 20)
    emb_dims = (
        (512, 640, 768, 896, 1024, 1280, 1408, 1536)
        if target_params >= 800_000_000
        else (512, 640, 768, 896, 1024)
    )
    large = target_params >= 1_200_000_000
    candidates = []
    for buckets_exp in bucket_exps:
        buckets = 2**buckets_exp
        for emb_dim in emb_dims:
            est = estimate_params(buckets, emb_dim, core_dim)
            weight_gb = est * 4 / (1024**3)
            dense = est - buckets * emb_dim
            train_gb = weight_gb + dense * 8 / (1024**3) + 0.8
            if not fits(weight_gb, dense):
                continue
            candidates.append(
                ScaleConfig(
                    target_params=target_params,
                    vocab_buckets=buckets,
                    emb_dim=emb_dim,
                    dim=core_dim,
                    max_p=80 if large else 96,
                    max_q=40 if large else 48,
                    max_a=28 if large else 32,
                    batch_size=1,
                    grad_accum=12 if large else 8,
                    estimated_params=est,
                    estimated_weight_gb_fp32=round(weight_gb, 2),
                    estimated_train_gb=round(train_gb, 2),
                    notes=(
                        f"hash-emb {buckets}x{emb_dim}, core dim={core_dim}; "
                        f"sparse Adam on emb; small answer head; "
                        f"bs=1 accum={12 if large else 8}"
                    ),
                )
            )

    if not candidates:
        if target_params >= 1_200_000_000:
            buckets, emb_dim, core_dim = 2**21, 704, 256
        else:
            buckets, emb_dim, core_dim = 2**20, 1024, 192
        est = estimate_params(buckets, emb_dim, core_dim)
        return ScaleConfig(
            target_params=target_params,
            vocab_buckets=buckets,
            emb_dim=emb_dim,
            dim=core_dim,
            max_p=64,
            max_q=32,
            max_a=24,
            batch_size=1,
            grad_accum=16,
            estimated_params=est,
            estimated_weight_gb_fp32=round(est * 4 / (1024**3), 2),
            estimated_train_gb=round(est * 4 / (1024**3) + 1.0, 2),
            notes="emergency layout (still embedding-heavy sparse hash)",
        )

    return min(candidates, key=lambda c: abs(c.estimated_params - target_params))


def nearest_config_report(cfg: ScaleConfig) -> str:
    pct = 100.0 * cfg.estimated_params / max(cfg.target_params, 1)
    return (
        f"target={cfg.target_params:,} got={cfg.estimated_params:,} ({pct:.1f}%) | "
        f"weights≈{cfg.estimated_weight_gb_fp32:.2f}GiB fp32 | "
        f"train≈{cfg.estimated_train_gb:.2f}GiB | {cfg.notes}"
    )
