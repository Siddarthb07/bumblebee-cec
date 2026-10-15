"""Live CEC neural monitor — correlational evidence pipeline + training.

  PYTHONPATH=src python3 -m bumblebee.cli arch-dash
  → http://127.0.0.1:7861

The visualization is CEC-native: QUERY tokens → GRU ENCODER → PASSAGE tokens →
SoftCorrelator (sparsemax mass binding) → EVIDENCE → CoverageGate → EXTRACT/REFUSE.
There are no transformer layers or attention heads in this model.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bumblebee.telemetry import read_live_bundle

try:  # read_events landed with the event log; stay importable without it
    from bumblebee.telemetry import read_events as _read_events
except Exception:  # pragma: no cover - defensive
    _read_events = None  # type: ignore[assignment]


_CKPT_CACHE: dict = {"ts": 0.0, "attempt": 0.0, "mtime": None, "data": {}, "busy": False}
_CKPT_LOCK = threading.Lock()
_CKPT_RETRY_S = 60.0


def _probe_worker(path: Path, mtime: float) -> None:
    out: dict = {}
    try:
        import torch

        blob = torch.load(path, map_location="cpu")
        if blob.get("dim") is not None:
            out["ckpt_dim"] = int(blob["dim"])
        if blob.get("n_params"):
            out["ckpt_n_params"] = int(blob["n_params"])
        elif isinstance(blob.get("model"), dict):
            n = sum(int(v.numel()) for v in blob["model"].values() if hasattr(v, "numel"))
            if n:
                out["ckpt_n_params"] = n
        if blob.get("arch"):
            out["ckpt_arch"] = blob["arch"]
        if blob.get("vocab_size"):
            out["ckpt_vocab_size"] = int(blob["vocab_size"])
        del blob
    except Exception:
        out = {}
    with _CKPT_LOCK:
        if out:
            _CKPT_CACHE["data"] = out
            _CKPT_CACHE["ts"] = time.time()
            _CKPT_CACHE["mtime"] = mtime
        _CKPT_CACHE["busy"] = False


def _probe_checkpoint(root: Path) -> dict:
    """Best-effort dim / n_params from cec.pt.

    The checkpoint can be several GB, so the read runs on a background thread
    and callers always get whatever is cached right now. This keeps every HTTP
    handler non-blocking and keeps the dash from fighting a live trainer for RAM.
    """
    path = root / "artifacts" / "cec.pt"
    if not path.exists():
        return {}
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    now = time.time()
    with _CKPT_LOCK:
        cached = dict(_CKPT_CACHE["data"])
        current = cached and _CKPT_CACHE.get("mtime") == mtime
        if current and now - float(_CKPT_CACHE["ts"]) < _CKPT_RETRY_S:
            return cached
        if _CKPT_CACHE["busy"] or now - float(_CKPT_CACHE["attempt"]) < _CKPT_RETRY_S:
            return cached
        _CKPT_CACHE["busy"] = True
        _CKPT_CACHE["attempt"] = now
    threading.Thread(target=_probe_worker, args=(path, mtime), daemon=True).start()
    return cached


def _train_snapshot(root: Path) -> dict:
    path = root / "artifacts" / "live" / "train.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_config(root: Path, *, probe_ckpt: bool = True) -> dict:
    path = root / "artifacts" / "config.json"
    cfg = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {
            "dim": 128,
            "vocab_size": 0,
            "n_params": 0,
            "mass_fn": "sparsemax",
            "threshold": 0.35,
            "max_q": 64,
            "max_p": 192,
            "arch": "cec",
        }
    )
    t = _train_snapshot(root)
    # A live trainer already reports the authoritative dim / n_params, so don't
    # pull a multi-GB checkpoint into memory alongside it.
    training = t.get("status") == "running" and (time.time() - float(t.get("ts") or 0)) < 15.0
    ckpt = (
        _probe_checkpoint(root)
        if (probe_ckpt and not training)
        else dict(_CKPT_CACHE.get("data") or {})
    )
    cfg.update(ckpt)
    if ckpt.get("ckpt_n_params"):
        cfg["n_params"] = ckpt["ckpt_n_params"]
    if ckpt.get("ckpt_dim"):
        cfg["dim"] = ckpt["ckpt_dim"]
    if ckpt.get("ckpt_vocab_size"):
        cfg["vocab_size"] = ckpt["ckpt_vocab_size"]
    if ckpt.get("ckpt_arch"):
        cfg["arch"] = ckpt["ckpt_arch"]
    # Live train overrides while running
    if t.get("status") == "running":
        if t.get("dim"):
            cfg["live_dim"] = int(t["dim"])
            cfg["dim"] = int(t["dim"])
        if t.get("n_params"):
            cfg["live_n_params"] = int(t["n_params"])
            cfg["n_params"] = int(t["n_params"])
        cfg["live_arch"] = t.get("arch")
    return cfg


def _topology(cfg: dict) -> dict:
    dim = int(cfg.get("dim") or 128)
    mass_fn = cfg.get("mass_fn") or "sparsemax"
    thr = float(cfg.get("threshold") or 0.35)
    v = int(cfg.get("vocab_size") or 0)
    return {
        "model": "CECModel",
        "dim": dim,
        "vocab_size": v,
        "mass_fn": mass_fn,
        "threshold": thr,
        "layers": [
            {"id": "q_tok", "name": "Question", "kind": "tokens", "role": "input"},
            {"id": "encoder", "name": "GRU Encoder", "kind": f"D={dim}", "role": "hidden", "units": dim},
            {"id": "p_tok", "name": "Passage", "kind": "tokens", "role": "input"},
            {"id": "correlator", "name": "SoftCorrelator", "kind": mass_fn, "role": "bind", "units": dim},
            {"id": "evidence", "name": "Evidence", "kind": f"D={dim}", "role": "hidden", "units": dim},
            {"id": "gate", "name": "CoverageGate", "kind": f"{2*dim}→{dim}→1", "role": "hidden", "units": dim},
            {"id": "extract", "name": "Extractive", "kind": "span", "role": "out"},
            {"id": "refuse", "name": "Refuse", "kind": "out", "role": "out"},
        ],
        "equations": {
            "correlator": [f"mass = {mass_fn}(q·k/τ)", "e = Σ massᵢ hᵢ"],
            "gate": ["conf = σ(MLP[q;e]) × conc", f"answer if conf ≥ {thr:.2f}"],
        },
    }


def _idle_forward(cfg: dict) -> dict:
    dim = int(cfg.get("dim") or 128)
    return {
        "mode": "idle",
        "phase": "idle",
        "mass": [],
        "mass_tokens": [],
        "confidence": 0.0,
        "concentration": 0.0,
        "decision": "idle",
        "query_units": [0.08] * min(dim, 640),
        "evidence_units": [0.08] * min(dim, 640),
        "activation": {
            "encoder": 0.1,
            "correlator": 0.1,
            "evidence": 0.1,
            "gate": 0.1,
            "extractive": 0.05,
            "refuse": 0.05,
            "generative": 0.05,
        },
    }


def _events(root: Path, limit: int = 40) -> list:
    """Last N lines of artifacts/live/events.jsonl. Soft-fail to []."""
    if _read_events is not None:
        try:
            return list(_read_events(root, limit=limit))
        except Exception:
            return []
    path = Path(root) / "artifacts" / "live" / "events.jsonl"
    if not path.exists():
        return []
    try:
        out = []
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            if line.strip():
                out.append(json.loads(line))
        return out
    except Exception:
        return []


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Bumblebee · Live CEC Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#04070a; --panel:#080d11; --raised:#0b1318; --well:#04080a; --deep:#020507;
  --line:#152128; --line-hi:#22333c;
  --text:#e9f3f2; --dim:#9db0b8; --muted:#5e7480;
  --bee:#f0c14a; --bee-hi:#ffdc82; --cyan:#3fe0d4; --teal:#39d6b0;
  --green:#5ed48e; --lime:#a9dc6a; --olive:#d8d45f; --blue:#5b9fd4; --red:#ef5f5f;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --sidew:352px; --logh:216px;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;overflow:hidden}
body{
  background:
    radial-gradient(1200px 520px at 18% -10%, rgba(63,224,212,.055), transparent 70%),
    radial-gradient(900px 480px at 88% 6%, rgba(240,193,74,.05), transparent 70%),
    var(--bg);
  color:var(--text);
  font-family:"Space Grotesk",system-ui,-apple-system,sans-serif;
  display:grid;grid-template-rows:50px 3px minmax(0,1fr);
  -webkit-font-smoothing:antialiased;
}

/* ───────── header ───────── */
.top{
  display:flex;align-items:center;gap:1rem;
  padding:0 .9rem;border-bottom:1px solid var(--line);
  background:linear-gradient(180deg,#0a1116,#06090c);
}
.brand{display:flex;align-items:center;gap:.6rem;min-width:0}
.bee{width:26px;height:26px;flex:none;filter:drop-shadow(0 0 6px rgba(240,193,74,.45))}
.brand .name{font-weight:700;letter-spacing:.09em;font-size:1.04rem;white-space:nowrap}
.brand .name em{font-style:normal;color:var(--bee)}
.brand .tag{
  padding-left:.7rem;border-left:1px solid var(--line-hi);
  font:500 .58rem/1.35 var(--mono);color:var(--muted);letter-spacing:.15em;white-space:nowrap;
}
.brand .tag b{display:block;color:#40525b;font-weight:500;letter-spacing:.13em;font-size:.53rem}
.sp{flex:1}
.top-right{display:flex;align-items:center;gap:.42rem}
.runpill{
  display:inline-flex;align-items:center;gap:.55rem;white-space:nowrap;
  font:600 .6rem/1 var(--mono);letter-spacing:.1em;padding:.44rem .7rem;
  border:1px solid var(--line-hi);border-radius:3px;background:#070d11;color:var(--dim);
}
.runpill .arch{color:var(--cyan)}
.runpill .sep{color:#2b3b44}
.runpill .par{color:var(--bee)}
.pill{
  font:600 .61rem/1.25 var(--mono);letter-spacing:.09em;padding:.32rem .65rem;
  border:1px solid var(--line);border-radius:3px;background:#070d11;
  display:inline-flex;gap:.5rem;align-items:center;color:var(--dim);white-space:nowrap;
}
.pill .d{width:7px;height:7px;border-radius:50%;background:#3a4a52;flex:none}
.pill .stack{display:flex;flex-direction:column;gap:1px}
.pill .stack b{font-weight:600;letter-spacing:.11em}
.pill .stack i{font-style:normal;font-size:.53rem;color:var(--muted);letter-spacing:.07em}
.pill.on{border-color:rgba(94,212,142,.42);color:#d3ecdd}
.pill.on .d{background:var(--green);box-shadow:0 0 9px var(--green)}
.pill.tr{border-color:rgba(240,193,74,.5);color:#f8e9bf;background:rgba(240,193,74,.05)}
.pill.tr .d{background:var(--bee);box-shadow:0 0 10px var(--bee);animation:blink .85s infinite}
@keyframes blink{50%{opacity:.3}}
#hdrSpark{display:block;width:52px;height:18px;opacity:.92}
.btn{
  font:600 .59rem/1 var(--mono);letter-spacing:.1em;background:#070d11;color:var(--bee);
  border:1px solid rgba(240,193,74,.3);padding:.42rem .6rem;cursor:pointer;border-radius:3px;
  transition:background .15s,border-color .15s,color .15s;
}
.btn:hover{background:rgba(240,193,74,.11);border-color:rgba(240,193,74,.7)}
.btn.active{background:rgba(240,193,74,.19);border-color:var(--bee);color:#ffe9a8}
.btn.ghost{color:var(--muted);border-color:var(--line)}
.btn.ghost:hover{color:var(--dim);background:#0d161b;border-color:var(--line-hi)}

/* ───────── progress rail ───────── */
.rail{position:relative;background:#0d161b;border-bottom:1px solid var(--line)}
.rail>i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--green),var(--bee));box-shadow:0 0 12px rgba(240,193,74,.5);transition:width .45s ease}

/* ───────── app shell ───────── */
.app{
  min-height:0;display:grid;
  grid-template-columns:minmax(0,1fr) var(--sidew);
  grid-template-rows:auto minmax(0,1fr) var(--logh);
  gap:6px;padding:6px;
}
.metrics{grid-column:1/-1;grid-row:1}
.viz{grid-column:1;grid-row:2}
.side{grid-column:2;grid-row:2}
.logwrap{grid-column:1/-1;grid-row:3}

body.logtall{--logh:330px}
body.logslim{--logh:126px}
body.present{--logh:138px}
body.present.logtall{--logh:232px}
body.present .app{grid-template-columns:minmax(0,1fr)}
body.present .side{display:none}
body.present .evbox{display:none}
body.present .logwrap{grid-template-columns:minmax(0,1fr)}
body.present #log{font-size:.62rem;line-height:1.42}

/* ───────── metrics strip ───────── */
.metrics{
  display:grid;grid-template-columns:1.85fr repeat(8,minmax(0,1fr));
  gap:4px;
}
.m{
  background:linear-gradient(180deg,#0a1318,#070c10);
  border:1px solid var(--line);border-radius:4px;
  padding:.28rem .48rem .32rem;font-family:var(--mono);min-width:0;position:relative;overflow:hidden;
}
.m::before{content:"";position:absolute;left:0;right:0;top:0;height:1px;background:linear-gradient(90deg,transparent,rgba(63,224,212,.22),transparent)}
.m .k{display:flex;align-items:center;gap:.28rem;font-size:.48rem;color:var(--muted);letter-spacing:.12em;font-weight:500;white-space:nowrap;overflow:hidden}
.m .k s{text-decoration:none;color:#33454e;font-size:.58rem;line-height:1}
.m .v{font-size:.95rem;line-height:1.08;margin-top:.1rem;font-weight:500;letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.m .s{font-size:.48rem;color:var(--muted);margin-top:.08rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-height:.65rem;letter-spacing:.04em}
.v.y{color:var(--bee)}.v.g{color:var(--green)}.v.c{color:var(--cyan)}.v.r{color:var(--red)}.v.b{color:var(--blue)}.v.o{color:var(--olive)}
.s.up{color:var(--red)}.s.dn{color:var(--green)}
.m.hero{
  background:linear-gradient(135deg,#0d181e,#070d11 62%);
  border-color:rgba(240,193,74,.18);
  display:grid;grid-template-columns:minmax(0,1.02fr) minmax(0,1.25fr);gap:.4rem;align-items:stretch;
  padding:.28rem .55rem .3rem;
}
.hero .left{min-width:0;display:flex;flex-direction:column}
.hero .big{font-size:1.45rem;line-height:1.02;color:var(--bee);font-weight:500;letter-spacing:-.035em;text-shadow:0 0 12px rgba(240,193,74,.18)}
.hero .steps{font-size:.48rem;color:var(--muted);margin-top:.12rem;letter-spacing:.05em}
.hero .right{display:flex;flex-direction:column;min-width:0}
.hero .etarow{display:flex;gap:.7rem}
.hero .eta{min-width:0}
.hero .eta u{display:block;text-decoration:none;font-size:.45rem;color:#3c4f58;letter-spacing:.14em}
.hero .eta b{display:block;font-size:.68rem;color:#d9e6e4;font-weight:500;letter-spacing:-.01em}
.hero .eta i{display:block;font-style:normal;font-size:.45rem;color:var(--muted)}
#progChart{width:100%;height:100%;min-height:22px;display:block;margin-top:.08rem;flex:1}
#tpChart{width:100%;height:14px;display:block;margin-top:.08rem}

/* ───────── viz ───────── */
.viz{
  position:relative;min-height:0;min-width:0;overflow:hidden;
  background:linear-gradient(180deg,#060b0e,#03070a);
  border:1px solid var(--line);border-radius:6px;
  display:grid;grid-template-rows:29px minmax(0,1fr);
}
.vizhead{
  display:flex;align-items:center;gap:.6rem;padding:0 .7rem;
  border-bottom:1px solid var(--line);background:linear-gradient(180deg,#0a1216,#070d10);
}
.vizhead h3{margin:0;font:600 .6rem/1 var(--mono);letter-spacing:.2em;color:#cddbdd;display:flex;align-items:center;gap:.45rem}
.vizhead h3::before{content:"";width:3px;height:12px;background:linear-gradient(180deg,var(--cyan),var(--bee));border-radius:1px}
.vizhead .vsub{font:500 .53rem/1 var(--mono);letter-spacing:.17em;color:#3f545e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stage{position:relative;min-height:0;min-width:0;overflow:hidden}
#cv{display:block;cursor:grab}
#cv:active{cursor:grabbing}
.chrome{position:absolute;inset:0;pointer-events:none;z-index:2}
.chrome .hint{
  position:absolute;left:12px;bottom:10px;max-width:62%;
  font:500 .55rem/1.5 var(--mono);color:#43585f;letter-spacing:.08em;
}
.zbar{display:flex;gap:4px;pointer-events:auto}
.zbar button,.zbar i{
  font:500 .58rem/1 var(--mono);background:rgba(10,17,21,.9);color:var(--bee);
  border:1px solid rgba(240,193,74,.3);padding:.3rem .48rem;cursor:pointer;
  min-width:2rem;text-align:center;font-style:normal;border-radius:3px;letter-spacing:.06em;
}
.zbar button:hover{background:rgba(240,193,74,.14);border-color:var(--bee)}
.zbar i{color:var(--muted);border-color:var(--line);cursor:default;background:rgba(10,17,21,.75)}
.legend{
  position:absolute;right:12px;bottom:10px;display:flex;gap:.65rem;flex-wrap:wrap;justify-content:flex-end;
  font:500 .53rem/1 var(--mono);color:#4e646d;letter-spacing:.09em;
}
.legend span{display:inline-flex;align-items:center;gap:.28rem}
.legend b{width:7px;height:7px;border-radius:50%;display:inline-block}

/* ───────── side panel ───────── */
.side{
  min-height:0;min-width:0;overflow-y:auto;overscroll-behavior:contain;
  display:flex;flex-direction:column;gap:6px;
}
.side::-webkit-scrollbar{width:7px}
.side::-webkit-scrollbar-track{background:transparent}
.side::-webkit-scrollbar-thumb{background:#1a262c;border-radius:4px}
.box{
  border:1px solid var(--line);border-radius:5px;
  background:linear-gradient(180deg,#0a1318,#070c10);
  padding:.5rem .62rem .6rem;
}
.box h3{
  margin:0 0 .45rem;font:600 .55rem/1 var(--mono);letter-spacing:.17em;color:#8ba0a8;
  display:flex;align-items:center;gap:.4rem;
}
.box h3::before{content:"";width:5px;height:5px;border-radius:50%;background:var(--bee);box-shadow:0 0 6px var(--bee);flex:none}
.box h3 b{margin-left:auto;font:500 .53rem/1 var(--mono);letter-spacing:.07em;color:var(--muted)}
.qtext{
  font:500 .76rem/1.45 var(--mono);color:#dcebe9;word-break:break-word;min-height:2.5em;
  background:var(--deep);border:1px solid #101b21;border-radius:4px;padding:.4rem .5rem;
}
.atext{margin-top:.34rem;font:400 .63rem/1.45 var(--mono);color:var(--green);word-break:break-word;min-height:1em}
.atext.ref{color:var(--red)}
.bar{display:grid;grid-template-columns:5.2rem 1fr 2.2rem;gap:.42rem;align-items:center;margin:.26rem 0;font:400 .55rem/1 var(--mono);color:var(--muted);letter-spacing:.06em}
.bar>span:last-child{text-align:right;color:var(--dim)}
.bt{height:6px;background:#050a0c;border:1px solid #141f25;border-radius:3px;overflow:hidden}
.bf{height:100%;width:0;transition:width .3s ease;background:var(--cyan);box-shadow:0 0 8px rgba(63,224,212,.5)}
.bf.y{background:var(--bee);box-shadow:0 0 8px rgba(240,193,74,.5)}
.bf.b{background:var(--blue);box-shadow:0 0 8px rgba(91,159,212,.5)}
.bf.r{background:var(--red);box-shadow:0 0 8px rgba(239,95,95,.5)}
.bf.g{background:var(--green);box-shadow:0 0 8px rgba(94,212,142,.5)}
.bf.l{background:var(--lime);box-shadow:0 0 8px rgba(169,220,106,.5)}
#loss{width:100%;height:100px;display:block;background:var(--deep);border-radius:4px;border:1px solid #101b21}
.mtab{width:100%;border-collapse:collapse;font:400 .55rem/1.5 var(--mono)}
.mtab th{text-align:left;color:#3f545e;font-weight:500;letter-spacing:.13em;padding:0 0 .18rem;border-bottom:1px solid var(--line)}
.mtab th:not(:first-child),.mtab td:not(:first-child){text-align:right}
.mtab td{padding:.1rem 0;color:var(--dim);border-bottom:1px solid #0b1216;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mtab td.t{color:#cbdad8;max-width:9rem}
.mtab tr.hot td.t{color:var(--bee);font-weight:600}
.mtab tr.hot td{color:#e2d09a}
.chips{display:flex;flex-wrap:wrap;gap:3px;margin-top:.42rem;max-height:76px;overflow-y:auto}
.chips::-webkit-scrollbar{width:6px}
.chips::-webkit-scrollbar-thumb{background:#1a262c;border-radius:3px}
.tk{font:400 .54rem/1.3 var(--mono);padding:2px 4px;border:1px solid #16232a;color:#4e646d;border-radius:2px;max-width:6.4rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tk.w{color:var(--bee);border-color:rgba(240,193,74,.35)}
.tk.h{background:var(--bee);color:#100d05;border-color:var(--bee);font-weight:600;box-shadow:0 0 9px rgba(240,193,74,.4)}
.sidefoot{font:400 .53rem/1.55 var(--mono);color:#3f545e;padding:.1rem .2rem .2rem;letter-spacing:.05em}

/* ───────── bottom log ───────── */
.logwrap{
  position:relative;z-index:6;min-height:0;min-width:0;
  display:grid;grid-template-columns:minmax(0,1fr) var(--sidew);gap:6px;
}
.logbox{
  min-width:0;min-height:0;display:grid;grid-template-rows:26px 18px minmax(0,1fr);
  border:1px solid var(--line-hi);border-radius:5px;overflow:hidden;
  background:linear-gradient(180deg,#070d10,#04080a);
}
.loghead{
  display:flex;align-items:center;gap:.55rem;padding:0 .7rem;
  background:linear-gradient(180deg,#0b1317,#080e11);border-bottom:1px solid var(--line);
}
.loghead h3{margin:0;font:600 .56rem/1 var(--mono);letter-spacing:.19em;color:#a9bcc2;display:flex;align-items:center;gap:.42rem}
.loghead h3::before{content:"";width:3px;height:11px;background:var(--bee);border-radius:1px}
.loghead .meta{font:400 .54rem/1 var(--mono);color:var(--muted);letter-spacing:.06em}
.logbtn{
  font:500 .53rem/1 var(--mono);letter-spacing:.11em;background:transparent;color:var(--muted);
  border:1px solid var(--line);border-radius:2px;padding:.26rem .44rem;cursor:pointer;
}
.logbtn:hover{color:var(--dim);border-color:var(--line-hi);background:#0d161b}
.logbtn.on{color:var(--bee);border-color:rgba(240,193,74,.42);background:rgba(240,193,74,.08)}
.lhdr,.lrow{
  display:grid;
  grid-template-columns:5.6rem 4.2rem 6.4rem 7.6rem 7.6rem 7.6rem 6.2rem 4.6rem minmax(0,1fr);
  gap:.5rem;align-items:center;
}
.lhdr{
  padding:0 .7rem;font:500 .52rem/1 var(--mono);letter-spacing:.15em;color:#35474f;
  background:#060b0e;border-bottom:1px solid var(--line);
}
#log{
  min-height:0;overflow-y:auto;overflow-x:hidden;overscroll-behavior:contain;
  padding:.24rem .2rem .4rem;font:400 .64rem/1.55 var(--mono);color:var(--dim);
  background:var(--well);
}
#log::-webkit-scrollbar{width:9px;height:9px}
#log::-webkit-scrollbar-track{background:#04080a}
#log::-webkit-scrollbar-thumb{background:#1b272d;border-radius:5px}
#log::-webkit-scrollbar-thumb:hover{background:#2a3c45}
.lrow{padding:0 .7rem 0 calc(.7rem - 2px);border-left:2px solid transparent;white-space:nowrap}
.lrow:hover{background:#080f13}
.lrow.fresh{border-left-color:var(--bee);background:rgba(240,193,74,.055)}
.lrow>span{overflow:hidden;text-overflow:ellipsis}
.lrow .tm{color:#4f656e}
.lrow .ty{color:var(--cyan)}
.lrow .st{color:#b9cad1}
.lrow .ls{color:var(--bee);font-weight:600}
.lrow .ax{color:#526871}
.lrow .msg{color:#6e858e;white-space:nowrap}
.lrow.sys{color:var(--muted)}
.lrow.sys .msg{color:var(--muted)}
.lrow.ask{border-left-color:var(--cyan);background:rgba(63,224,212,.05)}
.lrow.ask .ty{color:var(--green)}
.jump{
  position:absolute;right:calc(var(--sidew) + 20px);bottom:11px;z-index:8;display:none;
  font:600 .54rem/1 var(--mono);letter-spacing:.1em;
  background:rgba(240,193,74,.94);color:#100d05;border:0;border-radius:3px;
  padding:.38rem .58rem;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.6);
}
body.present .jump{right:20px}
.jump.show{display:block}
.evbox{
  min-width:0;min-height:0;display:grid;grid-template-rows:26px minmax(0,1fr);
  border:1px solid var(--line);border-radius:5px;overflow:hidden;
  background:linear-gradient(180deg,#0a1215,#04080a);
}
#events{
  min-height:0;overflow-y:auto;overscroll-behavior:contain;padding:.3rem .6rem .4rem;
  font:400 .57rem/1.6 var(--mono);color:#6e858e;
}
#events::-webkit-scrollbar{width:7px}
#events::-webkit-scrollbar-thumb{background:#1a262c;border-radius:4px}
.ev{display:flex;gap:.5rem;padding:.06rem 0}
.ev b{color:var(--green);font-weight:500;flex:none}
.ev u{text-decoration:none;color:#3f545e;flex:none}
.ev span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* ───────── responsive ───────── */
@media(max-width:1640px){ .metrics{grid-template-columns:2.35fr repeat(4,minmax(0,1fr))} }
@media(max-width:1280px){
  :root{--sidew:300px}
  .metrics{grid-template-columns:repeat(3,minmax(0,1fr))}
  .m.hero{grid-column:1/-1}
  .lhdr,.lrow{grid-template-columns:5rem 3.6rem 5.6rem 6.8rem 6.8rem 0 0 4.2rem minmax(0,1fr)}
}
@media(max-width:980px){
  .app{grid-template-columns:minmax(0,1fr);grid-template-rows:auto minmax(0,1fr) auto var(--logh)}
  .metrics{grid-template-columns:repeat(2,minmax(0,1fr))}
  .viz{grid-column:1;grid-row:2}
  .side{grid-column:1;grid-row:3;max-height:36vh}
  .logwrap{grid-column:1;grid-row:4;grid-template-columns:minmax(0,1fr)}
  .evbox{display:none}
}
</style>
</head>
<body>

<header class="top">
  <div class="brand">
    <svg class="bee" viewBox="0 0 40 40" aria-hidden="true">
      <g fill="none" stroke="#f0c14a" stroke-width="2.1" stroke-linecap="round">
        <path d="M13 9 C9 4 4 4 3 7"/><path d="M27 9 C31 4 36 4 37 7"/>
      </g>
      <ellipse cx="20" cy="23" rx="9.5" ry="12" fill="#f0c14a"/>
      <path d="M11 18 h18 M10.6 24 h18.8 M13 30.5 h14" stroke="#04070a" stroke-width="3.1" stroke-linecap="round"/>
      <ellipse cx="10" cy="17" rx="7" ry="4.6" fill="#3fe0d4" opacity=".4" transform="rotate(-28 10 17)"/>
      <ellipse cx="30" cy="17" rx="7" ry="4.6" fill="#3fe0d4" opacity=".4" transform="rotate(28 30 17)"/>
    </svg>
    <div class="name">BUMBLE<em>BEE</em></div>
    <div class="tag">LIVE CEC MONITOR<b>SMALL MODELS. BIGGER IDEAS.</b></div>
  </div>
  <div class="sp"></div>
  <div class="top-right">
    <div class="runpill">
      <span class="arch" id="rArch">CEC</span><span class="sep">·</span>
      <span id="rDim">D=—</span><span class="sep">·</span>
      <span class="par" id="rPar">— params</span>
    </div>
    <button class="btn ghost" type="button" id="logBtn" title="Cycle train-log height (L)">LOG ·</button>
    <button class="btn ghost" type="button" id="fitBtn" title="Reset view (F)">FIT</button>
    <button class="btn" type="button" id="presentBtn" title="Presentation mode (P)">PRESENT</button>
    <div class="pill" id="pill">
      <span class="d"></span>
      <span class="stack"><b id="plab">IDLE</b><i id="psub">waiting for telemetry</i></span>
      <canvas id="hdrSpark" width="104" height="36"></canvas>
    </div>
  </div>
</header>

<div class="rail" title="Training progress"><i id="prog"></i></div>

<div class="app">

  <div class="metrics" id="metrics"></div>

  <section class="viz">
    <div class="vizhead">
      <h3>CEC NEURAL PIPELINE</h3>
      <span class="vsub">QUERY → GRU ENCODER → PASSAGE MASS → SOFTCORRELATOR → EVIDENCE → COVERAGE GATE → EXTRACT / REFUSE</span>
      <span class="sp"></span>
      <div class="zbar">
        <button type="button" id="zOut" title="Zoom out">−</button>
        <i id="zPct">100%</i>
        <button type="button" id="zIn" title="Zoom in">+</button>
        <button type="button" id="zFit" title="Reset view (F)">FIT</button>
      </div>
    </div>
    <div class="stage">
      <canvas id="cv"></canvas>
      <div class="chrome">
        <div class="hint" id="hint">Correlational Evidence Computation · transformer-free</div>
        <div class="legend">
          <span><b style="background:#3fe0d4"></b>query</span>
          <span><b style="background:#5ed48e"></b>passage</span>
          <span><b style="background:#f0c14a"></b>mass / flow</span>
          <span><b style="background:#d8d45f"></b>evidence</span>
          <span><b style="background:#8ee06a"></b>extract</span>
          <span><b style="background:#ef5f5f"></b>refuse</span>
        </div>
      </div>
    </div>
  </section>

  <aside class="side">
    <div class="box">
      <h3>LIVE BATCH <b id="sqmeta"></b></h3>
      <div class="qtext" id="sq">waiting for a live question…</div>
      <div class="atext" id="sa"></div>
    </div>
    <div class="box">
      <h3>MODULE ACTIVATION <b id="actmeta"></b></h3>
      <div id="bars"></div>
    </div>
    <div class="box">
      <h3>MASS × TOKENS <b id="tokmeta"></b></h3>
      <table class="mtab"><thead><tr><th>TOKEN</th><th>MASS</th><th>IDX</th></tr></thead><tbody id="mrows"></tbody></table>
      <div class="chips" id="chips"></div>
    </div>
    <div class="box">
      <h3>LOSS CURVE <b id="lossmeta"></b></h3>
      <canvas id="loss" height="100"></canvas>
    </div>
    <div class="sidefoot" id="foot"></div>
  </aside>

  <div class="logwrap">
    <div class="logbox">
      <div class="loghead">
        <h3>TRAIN LOG</h3>
        <span class="meta" id="logmeta">waiting for steps…</span>
        <span class="sp"></span>
        <button class="logbtn on" type="button" id="stickBtn" title="Follow newest line">FOLLOW</button>
        <button class="logbtn" type="button" id="clearBtn" title="Clear log view">CLEAR</button>
      </div>
      <div class="lhdr">
        <span>TIME</span><span>TYPE</span><span>STEP</span><span>LOSS</span>
        <span>GATE</span><span>SPAN</span><span>LR</span><span>GRAD</span><span>MESSAGE</span>
      </div>
      <div id="log" tabindex="0"></div>
    </div>
    <div class="evbox">
      <div class="loghead"><h3>RECENT EVENTS</h3><span class="sp"></span><span class="meta" id="evmeta">—</span></div>
      <div id="events"></div>
    </div>
    <button class="jump" type="button" id="jumpBtn">LATEST ↓</button>
  </div>

</div>

<script>
/* ═══════════════ CEC pipeline stages (NOT transformer layers) ═══════════════ */
const STAGES=[
  {id:'query',      title:'QUERY',          sub:'tokens',        color:'#3fe0d4', act:'encoder',     w:1.32},
  {id:'encoder',    title:'GRU ENCODER',    sub:'query state',   color:'#39d6b0', act:'encoder',     w:1.00},
  {id:'passage',    title:'PASSAGE',        sub:'mass tokens',   color:'#5ed48e', act:'correlator',  w:1.28},
  {id:'correlator', title:'SOFTCORRELATOR', sub:'sparsemax',     color:'#a9dc6a', act:'correlator',  w:1.00},
  {id:'evidence',   title:'EVIDENCE',       sub:'e = Σ mᵢ·hᵢ',   color:'#d8d45f', act:'evidence',    w:1.00},
  {id:'gate',       title:'COVERAGE GATE',  sub:'σ(MLP[q;e])×c', color:'#f0c14a', act:'gate',        w:0.92},
  {id:'decision',   title:'EXTRACT / REFUSE', sub:'span | abstain', color:'#c8b56a', act:'extractive', w:0.88},
];
const CC={mass:'#f0c14a', extract:'#8ee06a', refuse:'#ef5f5f', dust:'#2fbfb6', white:'#ffffff'};
const TAU=Math.PI*2;

let CFG={},TOPO={},LIVE={},F={};
let view={s:1,x:0,y:0},drag=false,lx=0,ly=0;
let lastStep=-1,lastQ='',lastPhase='',logCount=0,logSeeded=false,stick=true;
let stepFlash=0,askFlash=0,t0=performance.now();
let particles=[],dust=[],layout=null,layoutKey='',edges=[],spine=[];
let smooth={},tpHist=[],evSeen=0;
const LOG_MAX=420, LOG_MODES=['','logtall','logslim'];
let logMode=0;

const $=id=>document.getElementById(id);
const clamp=(v,a,b)=>v<a?a:v>b?b:v;
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');
const fmtP=n=>!n?'—':n>=1e9?(n/1e9).toFixed(2)+'B':n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':''+n;
const fmt=(n,d=3)=>n==null||Number.isNaN(+n)?'—':(+n).toFixed(d);
const grp=n=>n==null?'—':(+n).toLocaleString('en-US');
const trainActive=()=>!!(LIVE.train_live&&LIVE.train&&LIVE.train.status==='running');
const dimOf=()=>+(trainActive()&&LIVE.train?.dim?LIVE.train.dim:(CFG.live_dim||CFG.dim||F.query_units?.length||128));
const paramsOf=()=>+(trainActive()&&LIVE.train?.n_params?LIVE.train.n_params:(CFG.live_n_params||CFG.n_params||0));
const dur=s=>{ if(s==null||!isFinite(s)||s<0) return '—'; const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=Math.floor(s%60);
  return h?h+'h '+String(m).padStart(2,'0')+'m':m?m+'m '+String(x).padStart(2,'0')+'s':x+'s'; };
const hhmmss=ts=>{ const d=ts?new Date(ts*1000):new Date(); return String(d.getHours()).padStart(2,'0')+':'+String(d.getMinutes()).padStart(2,'0')+':'+String(d.getSeconds()).padStart(2,'0'); };
function rgba(hex,a){ const h=hex.replace('#',''); const f=h.length===3?h.split('').map(c=>c+c).join(''):h;
  return 'rgba('+parseInt(f.slice(0,2),16)+','+parseInt(f.slice(2,4),16)+','+parseInt(f.slice(4,6),16)+','+clamp(a,0,1)+')'; }
function mix(h1,h2,t){ const p=x=>{const h=x.replace('#','');return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];};
  const a=p(h1),b=p(h2); const c=a.map((v,i)=>Math.round(v+(b[i]-v)*t));
  return '#'+c.map(v=>v.toString(16).padStart(2,'0')).join(''); }
function sm(key,target,rate){ if(smooth[key]==null) smooth[key]=target; smooth[key]+=(target-smooth[key])*(rate||.2); return smooth[key]; }

/* ═══════════════ metrics strip ═══════════════ */
const CELLS=[
  {id:'step', k:'STEP',       g:'▸', c:''},
  {id:'epoch',k:'EPOCH',      g:'◷', c:'c'},
  {id:'speed',k:'THROUGHPUT', g:'⚡', c:'c', chart:'tpChart'},
  {id:'loss', k:'LOSS',       g:'◉', c:'y'},
  {id:'grad', k:'GRAD NORM',  g:'∿', c:''},
  {id:'params',k:'PARAMS',    g:'▦', c:'y'},
  {id:'conf', k:'CONFIDENCE', g:'◎', c:'g'},
  {id:'dim',  k:'DIM / MASS', g:'◫', c:'o'},
];
function buildMetrics(){
  const hero=
    '<div class="m hero">'+
      '<div class="left">'+
        '<div class="k"><s>◐</s>TRAINING PROGRESS</div>'+
        '<div class="big" id="pctBig">—</div>'+
        '<div class="steps" id="pctSteps">waiting for telemetry</div>'+
      '</div>'+
      '<div class="right">'+
        '<div class="etarow">'+
          '<div class="eta"><u>ETA</u><b id="etaV">—</b><i id="etaR"></i></div>'+
          '<div class="eta"><u>ELAPSED</u><b id="elapV">—</b><i id="elapR"></i></div>'+
        '</div>'+
        '<canvas id="progChart"></canvas>'+
      '</div>'+
    '</div>';
  $('metrics').innerHTML=hero+CELLS.map(c=>
    '<div class="m"><div class="k"><s>'+c.g+'</s>'+c.k+'</div>'+
    '<div class="v '+c.c+'" id="v_'+c.id+'">—</div>'+
    '<div class="s" id="s_'+c.id+'"></div>'+
    (c.chart?'<canvas id="'+c.chart+'"></canvas>':'')+'</div>').join('');
}
function setCell(id,val,sub,subcls){
  const v=$('v_'+id); if(v&&v.textContent!==String(val)) v.textContent=val;
  const s=$('s_'+id); if(s){ const nx=sub==null?'':String(sub); if(s.textContent!==nx) s.textContent=nx; s.className='s '+(subcls||''); }
}

/* ═══════════════ boot / poll ═══════════════ */
function applyConfig(c){
  if(!c||c.error) return;
  const t=LIVE.train||{};
  CFG=Object.assign({},c);
  // Only a live trainer may override product checkpoint dim/params
  if(trainActive()){
    if(t.dim) CFG.dim=t.dim;
    if(t.n_params) CFG.n_params=t.n_params;
  }
  renderFoot();
  layout=null;
}
function renderFoot(){
  $('foot').textContent='Transformer-free CEC · '+(CFG.mass_fn||'sparsemax')+' mass · thr '+
    (CFG.threshold??'—')+' · max_q '+(CFG.max_q??'—')+' / max_p '+(CFG.max_p??'—');
}
/* config/topology can sit behind a multi-GB checkpoint probe — never block the UI on them */
function loadStatic(){
  fetch('/api/config').then(r=>r.json()).then(applyConfig).catch(()=>{});
  fetch('/api/topology').then(r=>r.json()).then(t=>{ if(t&&!t.error) TOPO=t; }).catch(()=>{});
}
async function boot(){
  buildMetrics();
  $('bars').innerHTML=['encoder','correlator','evidence','gate','extractive','refuse'].map(id=>
    '<div class="bar"><span>'+id+'</span><div class="bt"><div class="bf" id="b_'+id+'"></div></div><span id="bv_'+id+'">0.00</span></div>').join('');
  renderFoot();
  bindChrome(); bindLog(); seedDust();
  requestAnimationFrame(frame);
  await tick();
  setInterval(tick,600);
  loadStatic();
  setInterval(loadStatic,60000);
}

async function tick(){
  let b; try{ b=await fetch('/api/live').then(r=>r.json()); }catch(e){ return; }
  LIVE=b||{};
  const t=LIVE.train||{}, f=LIVE.forward||{};
  if(f&&Object.keys(f).length) F=f;
  const bc=LIVE.config||{};
  if(bc.dim) CFG.dim=bc.dim;
  if(bc.n_params) CFG.n_params=bc.n_params;
  if(bc.arch) CFG.arch=bc.arch;
  if(trainActive()){
    if(t.dim) CFG.dim=t.dim;
    if(t.n_params) CFG.n_params=t.n_params;
  }
  const hist=trainActive()?(LIVE.history||[]):[];

  const q=(t.sample_question||F.question||'').trim();
  const ph=String(F.phase||'');
  if((q&&q!==lastQ)||(ph&&ph!==lastPhase)){
    if(lastQ||lastPhase){ askFlash=1; spawnTokenRun(); }
    lastQ=q; lastPhase=ph; layout=null;
  }
  if(!logSeeded) seedLog(hist);
  header(t,f); metrics(t,f,hist); sample(t,f); acts(f); massPanel(f); lossChart(hist);
  renderEvents(LIVE.events||[]);
  if(t.step!=null&&t.step!==lastStep){
    lastStep=t.step; pushLog(t); stepFlash=1; spawnBurst(); layout=null;
  }
}

/* ═══════════════ header ═══════════════ */
function header(t,f){
  $('rArch').textContent=String(t.arch||CFG.live_arch||CFG.arch||'cec').toUpperCase();
  $('rDim').textContent='D='+dimOf();
  $('rPar').textContent=fmtP(paramsOf())+' params';
  const p=$('pill'), lab=$('plab'), sub=$('psub');
  p.className='pill';
  if(LIVE.train_live&&t.status==='running'){
    p.classList.add('tr');
    const spe=+(t.steps_per_epoch||1), ep=+(t.epoch||1);
    lab.textContent='TRAINING';
    sub.textContent='ep '+ep+'/'+(t.epochs??'—')+' · '+Math.max(0,(t.step||0)-(ep-1)*spe)+'/'+spe;
  } else if(LIVE.forward_live){
    p.classList.add('on'); lab.textContent='LIVE FORWARD';
    sub.textContent=(f.decision||'—')+' · conf '+fmt(f.confidence,2);
  } else { lab.textContent='IDLE'; sub.textContent='product ckpt · D='+dimOf(); }
  headerSpark(trainActive()?(LIVE.history||[]):[]);
}
function headerSpark(hist){
  const c=$('hdrSpark'); if(!c) return;
  const ctx=c.getContext('2d'), W=c.width, H=c.height;
  ctx.clearRect(0,0,W,H);
  const pts=hist.filter(h=>h.loss!=null).slice(-44).map(h=>h.loss);
  if(pts.length<2) return;
  const lo=Math.min(...pts), hi=Math.max(...pts), sp=(hi-lo)||1e-6;
  ctx.beginPath();
  pts.forEach((v,i)=>{ const x=i/(pts.length-1)*(W-2)+1, y=H-3-(v-lo)/sp*(H-8); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.strokeStyle='rgba(240,193,74,.95)'; ctx.lineWidth=2; ctx.lineJoin='round'; ctx.stroke();
}

/* ═══════════════ metrics ═══════════════ */
function progressOf(t){
  const ep=+(t.epoch||0), epochs=+(t.epochs||1), step=+(t.step||0), spe=+(t.steps_per_epoch||0);
  const total=epochs&&spe?epochs*spe:0;
  const done=spe?clamp(step,0,total||step):step;
  const pct=total?clamp(100*done/total,0,100):0;
  return {done,total,pct,spe,epochs};
}
function rate(hist){
  const pts=hist.filter(h=>h.ts!=null&&h.step!=null).slice(-40);
  if(pts.length<2) return null;
  const dt=pts[pts.length-1].ts-pts[0].ts, ds=pts[pts.length-1].step-pts[0].step;
  if(dt<=0||ds<=0) return null;
  return ds/dt;
}
function metrics(t,f,hist){
  const pr=progressOf(t);
  $('prog').style.width=pr.pct.toFixed(2)+'%';
  $('pctBig').textContent=pr.total?pr.pct.toFixed(1)+'%':'—';
  $('pctSteps').textContent=pr.total?grp(pr.done)+' / '+grp(pr.total)+' steps':(t.step!=null?'step '+grp(t.step):'waiting for telemetry');

  const r=rate(hist);
  if(r) { tpHist.push(r); if(tpHist.length>80) tpHist.shift(); }
  const remain=r&&pr.total?(pr.total-pr.done)/r:null;
  $('etaV').textContent=remain!=null&&LIVE.train_live?dur(remain):'—';
  $('etaR').textContent=r?'≈ '+r.toFixed(2)+' st/s':'';
  const span=hist.length>1?(hist[hist.length-1].ts-hist[0].ts):null;
  $('elapV').textContent=span?dur(span):'—';
  $('elapR').textContent=hist.length?'n='+hist.length+' pts':'';
  progChart(hist,pr);
  tpChart();

  const losses=hist.map(h=>h.loss).filter(v=>v!=null);
  const lo=losses.length?Math.min(...losses):null;
  const prev=losses.length>1?losses[losses.length-2]:null;
  const cur=trainActive()&&t.loss!=null?+t.loss:(losses.length?losses[losses.length-1]:null);
  let dsub='',dcls='';
  if(cur!=null&&prev!=null){ const d=cur-prev; dcls=d>0?'up':'dn'; dsub=(d>=0?'▲ ':'▼ ')+Math.abs(d).toFixed(4)+' · min '+fmt(lo,4); }
  else if(lo!=null) dsub='min '+fmt(lo,4);
  setCell('loss',cur!=null?fmt(cur,4):'—',dsub,dcls);

  setCell('step',grp(t.step),pr.total?'of '+grp(pr.total)+' · '+pr.pct.toFixed(1)+'%':(pr.spe?grp(pr.spe)+'/epoch':''));
  setCell('epoch',t.epoch!=null?t.epoch+' / '+(t.epochs??'—'):'—',pr.spe?grp(pr.spe)+' steps/epoch':'');
  setCell('speed',r?r.toFixed(2)+' st/s':'—',remain!=null&&LIVE.train_live?'eta '+dur(remain):(t.rss_gb!=null?'rss '+fmt(t.rss_gb,2)+' GB':''));
  setCell('grad',t.grad_norm!=null?fmt(t.grad_norm,2):'—',t.lr!=null?'lr '+(+t.lr).toExponential(1):'');
  setCell('params',fmtP(paramsOf()),t.batch_size?'batch '+t.batch_size:(CFG.vocab_size?'vocab '+fmtP(CFG.vocab_size):''));
  const conf=f.confidence;
  setCell('conf',conf!=null?fmt(conf,3):'—',f.decision?String(f.decision).toUpperCase()+' · conc '+fmt(f.concentration,2):'');
  const ce=$('v_conf'); if(ce) ce.className='v '+(f.decision==='refuse'?'r':'g');
  setCell('dim',dimOf(),String(CFG.mass_fn||'sparsemax'));
}
function progChart(hist,pr){
  const c=$('progChart'); if(!c) return;
  const W=Math.max(80,c.clientWidth), H=Math.max(30,c.clientHeight);
  const dpr=Math.min(2,window.devicePixelRatio||1);
  if(c.width!==(W*dpr|0)||c.height!==(H*dpr|0)){ c.width=W*dpr|0; c.height=H*dpr|0; }
  const ctx=c.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='rgba(30,45,52,.85)'; ctx.lineWidth=1;
  for(let i=0;i<=3;i++){ const y=1+i*(H-3)/3; ctx.beginPath(); ctx.moveTo(14,y); ctx.lineTo(W,y); ctx.stroke(); }
  ctx.fillStyle='#31434b'; ctx.font='7px IBM Plex Mono,monospace'; ctx.textAlign='left';
  ['100%','66%','33%','0'].forEach((s,i)=>ctx.fillText(s,0,4+i*(H-3)/3));
  const pts=hist.filter(h=>h.step!=null).slice(-160);
  if(pts.length<2||!pr.total) return;
  const X=i=>14+i/(pts.length-1)*(W-15), Y=v=>H-2-clamp(v,0,1)*(H-5);
  ctx.beginPath(); ctx.moveTo(X(0),H-2);
  pts.forEach((p,i)=>ctx.lineTo(X(i),Y(p.step/pr.total)));
  ctx.lineTo(X(pts.length-1),H-2); ctx.closePath();
  const g=ctx.createLinearGradient(0,0,0,H);
  g.addColorStop(0,'rgba(240,193,74,.3)'); g.addColorStop(1,'rgba(240,193,74,0)');
  ctx.fillStyle=g; ctx.fill();
  ctx.beginPath(); pts.forEach((p,i)=>{ const x=X(i),y=Y(p.step/pr.total); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.strokeStyle='rgba(240,193,74,.95)'; ctx.lineWidth=1.4; ctx.stroke();
}
function tpChart(){
  const c=$('tpChart'); if(!c) return;
  const W=Math.max(60,c.clientWidth), H=20, dpr=Math.min(2,window.devicePixelRatio||1);
  if(c.width!==(W*dpr|0)||c.height!==(H*dpr|0)){ c.width=W*dpr|0; c.height=H*dpr|0; c.style.height=H+'px'; }
  const ctx=c.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
  if(tpHist.length<2) return;
  const lo=Math.min(...tpHist), hi=Math.max(...tpHist), sp=(hi-lo)||1e-6;
  ctx.beginPath();
  tpHist.forEach((v,i)=>{ const x=i/(tpHist.length-1)*W, y=H-2-(v-lo)/sp*(H-5); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.strokeStyle='rgba(94,212,142,.85)'; ctx.lineWidth=1.2; ctx.stroke();
}

/* ═══════════════ side panels ═══════════════ */
function sample(t,f){
  $('sq').textContent=t.sample_question||f.question||'waiting for a live question…';
  const bits=[];
  if(t.sample_answer) bits.push('▸ '+t.sample_answer);
  else if(f.answer_preview) bits.push('▸ '+f.answer_preview);
  if(f.decision) bits.push(String(f.decision).toUpperCase());
  if(t.sample_answerable!=null) bits.push('answerable='+t.sample_answerable);
  const el=$('sa'); el.textContent=bits.join('   ·   ');
  el.className='atext'+(f.decision==='refuse'?' ref':'');
  $('sqmeta').textContent=(f.source?String(f.source)+' · ':'')+(t.step!=null?'step '+grp(t.step):(f.phase||''));
}
function acts(f){
  const a=f.activation||{}, cls={encoder:'',correlator:'l',evidence:'y',gate:'y',extractive:'g',refuse:'r'};
  let top='',tv=-1;
  for(const id of ['encoder','correlator','evidence','gate','extractive','refuse']){
    const v=clamp(+(a[id]||0),0,1);
    if(v>tv){ tv=v; top=id; }
    const el=$('b_'+id); if(el){ el.style.width=(v*100).toFixed(1)+'%'; el.className='bf '+(cls[id]||''); }
    const t=$('bv_'+id); if(t) t.textContent=v.toFixed(2);
  }
  $('actmeta').textContent=top?'peak '+top:'';
}
function topMass(mass,k){
  const idx=mass.map((m,i)=>[m,i]).filter(p=>p[0]>0).sort((a,b)=>b[0]-a[0]).slice(0,k).map(p=>p[1]);
  if(idx.length>=Math.min(k,4)) return idx.sort((a,b)=>a-b);
  const extra=[]; for(let i=0;i<mass.length&&idx.length+extra.length<k;i++) if(!idx.includes(i)) extra.push(i);
  return idx.concat(extra).sort((a,b)=>a-b);
}
function massPanel(f){
  const mass=f.mass||[], toks=f.mass_tokens||[];
  $('tokmeta').textContent=mass.length?mass.filter(m=>m>0.01).length+' nnz / '+mass.length:'';
  const rows=$('mrows'), chips=$('chips');
  if(!mass.length){ rows.innerHTML='<tr><td class="t">—</td><td>—</td><td>—</td></tr>'; chips.innerHTML='<span class="tk">no mass yet</span>'; return; }
  const peak=Math.max(...mass,1e-9);
  const order=mass.map((m,i)=>[m,i]).sort((a,b)=>b[0]-a[0]).slice(0,8);
  rows.innerHTML=order.map(([m,i])=>{
    const tk=toks[i]!=null?String(toks[i]).slice(0,18):'#'+i;
    return '<tr class="'+(m>0.2*peak?'hot':'')+'"><td class="t">'+esc(tk)+'</td><td>'+m.toFixed(3)+'</td><td>'+i+'</td></tr>';
  }).join('');
  chips.innerHTML=mass.slice(0,72).map((m,i)=>{
    const tk=toks[i]!=null?String(toks[i]).slice(0,12):'#'+i;
    const c=m>0.25*peak?'h':m>0.05*peak?'w':'';
    return '<span class="tk '+c+'" title="'+m.toFixed(4)+'">'+esc(tk)+'</span>';
  }).join('');
}
function lossChart(hist){
  const c=$('loss'), W=Math.max(200,c.parentElement.clientWidth-22), H=100;
  const dpr=Math.min(2,window.devicePixelRatio||1);
  if(c.width!==(W*dpr|0)||c.height!==(H*dpr|0)){ c.width=W*dpr|0; c.height=H*dpr|0; c.style.height=H+'px'; }
  const ctx=c.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle='#04080a'; ctx.fillRect(0,0,W,H);
  const pts=hist.filter(h=>h.loss!=null);
  $('lossmeta').textContent=pts.length?'n='+pts.length:'';
  if(pts.length<2){ ctx.fillStyle='#3f545e'; ctx.font='10px IBM Plex Mono,monospace'; ctx.fillText('waiting for steps…',10,H/2); return; }
  const pl=32,pr=6,pt=10,pb=13;
  const ys=pts.map(p=>p.loss), lo=Math.min(...ys), hi=Math.max(...ys), sp=(hi-lo)||1e-6;
  const X=i=>pl+i/(pts.length-1)*(W-pl-pr), Y=v=>pt+(1-(v-lo)/sp)*(H-pt-pb);
  ctx.strokeStyle='#101c22'; ctx.lineWidth=1;
  for(let i=0;i<4;i++){ const y=pt+i*(H-pt-pb)/3; ctx.beginPath(); ctx.moveTo(pl,y); ctx.lineTo(W-pr,y); ctx.stroke(); }
  ctx.beginPath(); ctx.moveTo(X(0),H-pb);
  pts.forEach((p,i)=>ctx.lineTo(X(i),Y(p.loss)));
  ctx.lineTo(X(pts.length-1),H-pb); ctx.closePath();
  const g=ctx.createLinearGradient(0,pt,0,H-pb);
  g.addColorStop(0,'rgba(240,193,74,.24)'); g.addColorStop(1,'rgba(240,193,74,0)');
  ctx.fillStyle=g; ctx.fill();
  ctx.beginPath(); pts.forEach((p,i)=>{ const x=X(i),y=Y(p.loss); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.strokeStyle='rgba(240,193,74,.9)'; ctx.lineWidth=1.4; ctx.stroke();
  let e=null; const ema=pts.map(p=>{ e=e==null?p.loss:e+(p.loss-e)*.16; return e; });
  ctx.beginPath(); ema.forEach((v,i)=>{ const x=X(i),y=Y(v); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.strokeStyle='rgba(63,224,212,.9)'; ctx.lineWidth=1.3; ctx.stroke();
  ctx.beginPath(); ctx.arc(X(pts.length-1),Y(pts[pts.length-1].loss),2.7,0,TAU);
  ctx.fillStyle='#f0c14a'; ctx.fill();
  ctx.fillStyle='#3f545e'; ctx.font='8px IBM Plex Mono,monospace';
  ctx.fillText(hi.toFixed(2),1,pt+3); ctx.fillText(lo.toFixed(2),1,H-4);
  ctx.fillText('ema',W-pr-22,pt+8);
}
function renderEvents(evs){
  if(evs.length===evSeen) return;
  evSeen=evs.length;
  const box=$('events');
  if(!evs.length){ box.innerHTML='<div class="ev"><span>no events yet</span></div>'; $('evmeta').textContent='0'; return; }
  box.innerHTML=evs.slice(-40).reverse().map(e=>
    '<div class="ev"><u>'+hhmmss(e.ts)+'</u><b>▪</b><span>'+esc(e.message||JSON.stringify(e))+'</span></div>').join('');
  $('evmeta').textContent=evs.length+' lines';
}

/* ═══════════════ train log ═══════════════ */
function logRow(o,cls){
  const d=document.createElement('div');
  d.className='lrow '+(cls||'');
  d.innerHTML=
    '<span class="tm">'+hhmmss(o.ts)+'</span>'+
    '<span class="ty">'+esc(o.epoch!=null?'ep'+o.epoch+(o.epochs?'/'+o.epochs:''):(o.type||'—'))+'</span>'+
    '<span class="st">step '+esc(o.step??'—')+'</span>'+
    '<span class="ls">loss '+fmt(o.loss,4)+'</span>'+
    '<span class="ax">gate '+fmt(o.gate_loss,4)+'</span>'+
    '<span class="ax">span '+fmt(o.span_loss,4)+'</span>'+
    '<span class="ax">'+(o.lr!=null?'lr '+(+o.lr).toExponential(1):'—')+'</span>'+
    '<span class="ax">'+(o.grad_norm!=null?'g '+fmt(o.grad_norm,1):'—')+'</span>'+
    '<span class="msg">'+esc(o.q||'')+'</span>';
  return d;
}
function appendRow(el){
  const box=$('log');
  const prev=box.querySelector('.lrow.fresh'); if(prev) prev.classList.remove('fresh');
  box.appendChild(el); logCount++;
  while(box.childElementCount>LOG_MAX) box.removeChild(box.firstElementChild);
  if(stick) box.scrollTop=box.scrollHeight;
  $('logmeta').textContent=logCount+' lines · step '+(lastStep>=0?grp(lastStep):'—');
}
function seedLog(hist){
  const pts=hist.filter(h=>h.loss!=null).slice(-140);
  if(!pts.length){ $('logmeta').textContent='waiting for steps…'; return; }
  logSeeded=true;
  const head=document.createElement('div');
  head.className='lrow sys';
  head.innerHTML='<span class="tm">'+hhmmss()+'</span><span class="ty">replay</span>'+
    '<span class="st">—</span><span class="ls">—</span><span class="ax"></span><span class="ax"></span>'+
    '<span class="ax"></span><span class="ax"></span>'+
    '<span class="msg">replaying last '+pts.length+' steps from artifacts/live/history.jsonl</span>';
  $('log').appendChild(head); logCount++;
  for(const p of pts) appendRow(logRow(p,''));
  $('log').scrollTop=$('log').scrollHeight;
}
function pushLog(t){
  appendRow(logRow({
    ts:t.ts, epoch:t.epoch, epochs:t.epochs, step:t.step, loss:t.loss,
    gate_loss:t.gate_loss, span_loss:t.span_loss, grad_norm:t.grad_norm,
    lr:t.lr, q:(t.sample_question||'').slice(0,160),
  },'fresh'));
}
function bindLog(){
  const box=$('log'), jump=$('jumpBtn');
  box.addEventListener('scroll',()=>{
    const atEnd=box.scrollHeight-box.scrollTop-box.clientHeight<24;
    stick=atEnd;
    $('stickBtn').classList.toggle('on',atEnd);
    jump.classList.toggle('show',!atEnd);
  });
  jump.onclick=()=>{ box.scrollTop=box.scrollHeight; stick=true; $('stickBtn').classList.add('on'); jump.classList.remove('show'); };
  $('stickBtn').onclick=()=>{ stick=!stick; $('stickBtn').classList.toggle('on',stick); if(stick){ box.scrollTop=box.scrollHeight; jump.classList.remove('show'); } };
  $('clearBtn').onclick=()=>{ box.innerHTML=''; logCount=0; $('logmeta').textContent='cleared · waiting for steps…'; };
  $('logBtn').onclick=()=>cycleLog(1);
  cycleLog(0);
}
function cycleLog(step){
  if(step) logMode=(logMode+1)%LOG_MODES.length;
  document.body.classList.remove('logtall','logslim');
  if(LOG_MODES[logMode]) document.body.classList.add(LOG_MODES[logMode]);
  $('logBtn').textContent='LOG '+['·','▲','▼'][logMode];
  layout=null;
  const box=$('log'); if(stick) requestAnimationFrame(()=>{ box.scrollTop=box.scrollHeight; });
}

/* ═══════════════ canvas: CEC pipeline ═══════════════ */
function pool(arr,n,fill){
  const out=[];
  if(!arr||!arr.length){ for(let i=0;i<n;i++) out.push((fill??.18)*(.5+.5*Math.abs(Math.sin(i*1.7)))); return out; }
  for(let i=0;i<n;i++){
    const a=Math.floor(i*arr.length/n), b=Math.max(a+1,Math.floor((i+1)*arr.length/n));
    let s=0,c=0; for(let j=a;j<b;j++){ s+=+arr[j]||0; c++; }
    out.push(clamp(c?s/c:(fill??.18),0,1));
  }
  return out;
}
function seedDust(){
  dust=[];
  for(let i=0;i<86;i++) dust.push({x:Math.random(),y:Math.random(),r:.5+Math.random()*1.5,
    vx:(Math.random()-.5)*.00022,vy:(Math.random()-.5)*.00016,ph:Math.random()*TAU,sp:.4+Math.random()*1.1});
}
function spawnBurst(){
  const strong=edges.filter(e=>e.base>.22);
  for(let i=0;i<16&&strong.length;i++){
    const e=strong[(Math.random()*strong.length)|0];
    particles.push({e,t:Math.random()*.25,v:.013+Math.random()*.018,life:1,r:2.4,col:e.pcol});
  }
}
function spawnTokenRun(){
  for(let i=0;i<10;i++) particles.push({spine:1,t:-i*.055,v:.0075,life:1.4,r:3.1,col:CC.mass});
}

function buildLayout(W,H){
  const D=dimOf();
  const mass=F.mass||[], massToks=F.mass_tokens||[], qU=F.query_units||[], eU=F.evidence_units||[], sU=F.start_units||[];
  const a=F.activation||{};
  const qRaw=(LIVE.train?.sample_question||F.question||'').trim();
  const qToks=qRaw?qRaw.split(/\s+/).slice(0,14):[];
  const padL=16,padR=16,padT=50,padB=40;
  const innerH=Math.max(150,H-padT-padB);
  const maxVis=clamp(Math.floor(innerH/23),7,20);
  const R=clamp(innerH/(maxVis*3.9),3.2,7.2);

  const totW=STAGES.reduce((s,x)=>s+x.w,0);
  const avail=W-padL-padR;
  let cursor=padL;
  const cols=[];
  for(const sp of STAGES){
    const slot=avail*sp.w/totW;
    const cx=cursor+slot/2;
    cursor+=slot;
    let n,vals,full,labels=null;
    if(sp.id==='query'){
      full=qToks.length||8; n=Math.min(full,maxVis);
      vals=qToks.length?pool(qU,n,a.encoder):pool(null,n,.14);
      labels=qToks.slice(0,n).map(s=>s.slice(0,11));
    } else if(sp.id==='passage'){
      full=mass.length||massToks.length||16;
      n=Math.min(maxVis,Math.max(4,full));
      const pick=mass.length?topMass(mass,n):[];
      const peak=mass.length?Math.max(...mass,1e-9):1;
      vals=[]; labels=[];
      for(let i=0;i<n;i++){
        const idx=pick.length?pick[Math.min(pick.length-1,i)]:i;
        const m=mass.length?(mass[idx]||0):0;
        vals.push(clamp(.1+Math.sqrt(m/peak)*.9,0,1));
        labels.push(String(massToks[idx]!=null?massToks[idx]:'#'+idx).slice(0,11));
      }
    } else if(sp.id==='decision'){
      n=6; full=6;
      vals=pool(sU,5,a.extractive).concat([F.decision==='refuse'?1:.1]);
    } else if(sp.id==='gate'){
      full=D; n=Math.min(10,maxVis);
      const gv=clamp(+(a.gate||F.confidence||0),0,1);
      vals=pool(eU,n,a.gate).map(v=>clamp(v*.45+gv*.55,0,1));
    } else if(sp.id==='correlator'){
      full=D; n=maxVis;
      const peak=mass.length?Math.max(...mass,1e-9):1;
      const norm=mass.length?mass.map(m=>Math.sqrt(m/peak)):null;
      vals=pool(norm,n,a.correlator);
    } else if(sp.id==='evidence'){
      full=D; n=maxVis; vals=pool(eU,n,a.evidence);
    } else { /* encoder */
      full=D; n=maxVis; vals=pool(qU,n,a.encoder);
    }
    const nodes=[];
    if(sp.id==='decision'){
      const topY=padT+innerH*.10, botY=padT+innerH*.50;
      for(let i=0;i<5;i++){
        nodes.push({x:cx,y:topY+i*(botY-topY)/4,v:sm(sp.id+':'+i,clamp(vals[i],0,1),.22),g:'extract',color:CC.extract});
      }
      nodes.push({x:cx,y:padT+innerH*.85,v:sm(sp.id+':r',clamp(vals[5],0,1),.22),g:'refuse',color:CC.refuse});
    } else {
      const spread=innerH-R*3;
      for(let i=0;i<n;i++){
        const ty=n===1?.5:i/(n-1);
        nodes.push({x:cx,y:padT+R*1.5+ty*spread,v:sm(sp.id+':'+i,clamp(vals[i],0,1),.22),color:sp.color});
      }
    }
    cols.push({...sp,cx,x0:cursor-slot+slot*.12,x1:cursor-slot*.12,n,full,nodes,labels,slot});
  }

  const by=Object.fromEntries(cols.map(c=>[c.id,c]));
  const E=[];
  const peakM=mass.length?Math.max(...mass,1e-9):1;
  function fan(A,B,spreadN,color,boost,isMass){
    for(let i=0;i<A.nodes.length;i++){
      const na=A.nodes[i];
      const j0=Math.round(i*(B.nodes.length-1)/Math.max(1,A.nodes.length-1));
      for(let d=-spreadN;d<=spreadN;d++){
        const j=j0+d; if(j<0||j>=B.nodes.length) continue;
        const nb=B.nodes[j];
        const w=clamp((na.v*.6+nb.v*.4)*(1-Math.abs(d)/(spreadN+1.6)),0,1);
        if(w<.035) continue;
        E.push({a:na,b:nb,base:w*(boost||1),col:mix(color||A.color,B.color,.5),pcol:isMass?CC.mass:B.color,
          lw:isMass?(.7+w*2.9):(.45+w*1.5),ph:Math.random()*TAU,sp:.6+Math.random()*1.5,
          wob:(Math.random()-.5)*26,mass:!!isMass});
      }
    }
  }
  fan(by.query,by.encoder,2,null,1.0,false);
  fan(by.encoder,by.correlator,2,null,.95,false);
  // mass binding: passage tokens fan into the correlator with sparsemax weight
  if(by.passage&&by.correlator){
    const P=by.passage,Cr=by.correlator;
    for(let i=0;i<P.nodes.length;i++){
      const na=P.nodes[i], m=na.v;
      const j0=Math.round(i*(Cr.nodes.length-1)/Math.max(1,P.nodes.length-1));
      for(let d=-3;d<=3;d++){
        const j=j0+d; if(j<0||j>=Cr.nodes.length) continue;
        const w=clamp(m*(1-Math.abs(d)/4.6),0,1);
        if(w<.05) continue;
        E.push({a:na,b:Cr.nodes[j],base:w,col:mix(P.color,CC.mass,.55),pcol:CC.mass,
          lw:.75+w*3.1,ph:Math.random()*TAU,sp:.5+Math.random()*1.3,wob:(Math.random()-.5)*20,mass:true});
      }
    }
  }
  fan(by.correlator,by.evidence,3,CC.mass,1.05,true);
  fan(by.evidence,by.gate,3,null,1.0,false);
  if(by.gate&&by.decision){
    const G=by.gate, Dn=by.decision;
    const ex=F.decision!=='refuse', conf=clamp(+(F.confidence||0),0,1);
    for(const na of G.nodes){
      for(const nb of Dn.nodes){
        const on=(nb.g==='extract')===ex;
        const w=clamp((on?.35+conf*.65:.06)*(na.v*.5+.5),0,1);
        E.push({a:na,b:nb,base:w,col:nb.g==='extract'?CC.extract:CC.refuse,pcol:nb.color,
          lw:on?(.8+w*2.6):.5,ph:Math.random()*TAU,sp:.7+Math.random()*1.4,wob:(Math.random()-.5)*14,mass:false});
      }
    }
  }
  E.sort((x,y)=>y.base-x.base);
  edges=E.slice(0,560);
  spine=cols.map(c=>c.nodes[(c.nodes.length/2)|0]).filter(Boolean);
  $('hint').textContent='CEC · D='+D+' · '+fmtP(paramsOf())+' params · '+(CFG.mass_fn||'sparsemax')+' mass · '+edges.length+' synapses';
  return {cols,by,R,W,H,padT,padB,innerH};
}

function bez(a,b,t,wob){
  const mx=(a.x+b.x)/2,u=1-t;
  const c1y=a.y+(wob||0), c2y=b.y-(wob||0);
  return {x:u*u*u*a.x+3*u*u*t*mx+3*u*t*t*mx+t*t*t*b.x,
          y:u*u*u*a.y+3*u*u*t*c1y+3*u*t*t*c2y+t*t*t*b.y};
}
function edgePoint(e,t,T){ return bez(e.a,e.b,t,Math.sin(T*.9+e.ph)*e.wob*.5); }

function frame(){ draw(); requestAnimationFrame(frame); }

function draw(){
  const cv=$('cv'), stage=cv.parentElement;
  const W=stage.clientWidth, H=stage.clientHeight;
  if(!W||!H) return;
  const dpr=Math.min(2,window.devicePixelRatio||1);
  if(cv.width!==(W*dpr|0)||cv.height!==(H*dpr|0)){
    cv.width=W*dpr|0; cv.height=H*dpr|0; cv.style.width=W+'px'; cv.style.height=H+'px'; layout=null;
  }
  const ctx=cv.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const T=(performance.now()-t0)/1000;

  /* backdrop */
  ctx.fillStyle='#03070a'; ctx.fillRect(0,0,W,H);
  const vg=ctx.createRadialGradient(W*.42,H*.5,10,W*.42,H*.5,Math.max(W,H)*.72);
  vg.addColorStop(0,'rgba(12,30,34,.58)'); vg.addColorStop(1,'rgba(2,5,8,0)');
  ctx.fillStyle=vg; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='rgba(20,38,44,.4)'; ctx.lineWidth=1;
  for(let x=0;x<W;x+=48){ ctx.beginPath(); ctx.moveTo(x+.5,0); ctx.lineTo(x+.5,H); ctx.stroke(); }
  for(let y=0;y<H;y+=48){ ctx.beginPath(); ctx.moveTo(0,y+.5); ctx.lineTo(W,y+.5); ctx.stroke(); }

  /* ambient dust */
  ctx.globalCompositeOperation='lighter';
  for(const d of dust){
    d.x+=d.vx; d.y+=d.vy;
    if(d.x<0)d.x=1; if(d.x>1)d.x=0; if(d.y<0)d.y=1; if(d.y>1)d.y=0;
    const tw=.18+.22*(.5+.5*Math.sin(T*d.sp+d.ph));
    ctx.beginPath(); ctx.arc(d.x*W,d.y*H,d.r,0,TAU);
    ctx.fillStyle=rgba(CC.dust,tw); ctx.fill();
  }
  ctx.globalCompositeOperation='source-over';

  ctx.save();
  ctx.translate(view.x,view.y); ctx.scale(view.s,view.s);

  const key=W+'x'+H+':'+lastStep+':'+dimOf()+':'+lastQ.length;
  if(!layout||layoutKey!==key){ layout=buildLayout(W,H); layoutKey=key; }
  const L=layout, R=L.R, a=F.activation||{};
  const training=!!(LIVE.train_live&&LIVE.train?.status==='running');
  const hot=training||!!LIVE.forward_live;
  if(stepFlash>0) stepFlash*=.92;
  if(askFlash>0) askFlash*=.965;
  const flash=clamp(stepFlash*.6+askFlash,0,1.4);

  /* column shells — quieter on decision column */
  for(const col of L.cols){
    const act=sm('act:'+col.id,clamp(+(a[col.act]||.12),0,1),.1);
    const y0=L.padT-16, hh=L.H-L.padT-L.padB+28;
    const isDec=col.id==='decision';
    ctx.beginPath(); roundRect(ctx,col.x0,y0,col.x1-col.x0,hh,9);
    const cg=ctx.createLinearGradient(0,y0,0,y0+hh);
    const shell=isDec?(.035+act*.04):(.075+act*.075+flash*.05);
    cg.addColorStop(0,rgba(col.color,shell));
    cg.addColorStop(1,rgba(col.color,.01));
    ctx.fillStyle=cg; ctx.fill();
    ctx.strokeStyle=rgba(col.color,isDec?(.12+act*.18):(.16+act*.34+flash*.18));
    ctx.lineWidth=1; ctx.stroke();
  }

  /* synapses — glow pass + core pass */
  ctx.globalCompositeOperation='lighter';
  ctx.lineCap='round';
  for(const e of edges){
    const sh=.5+.5*Math.sin(T*e.sp+e.ph);
    const al=e.base*(.38+.62*sh)*(1+flash*.75)*(hot?1:.6);
    if(al<.02) continue;
    const wob=Math.sin(T*.9+e.ph)*e.wob*.5;
    const mx=(e.a.x+e.b.x)/2;
    ctx.beginPath();
    ctx.moveTo(e.a.x,e.a.y);
    ctx.bezierCurveTo(mx,e.a.y+wob,mx,e.b.y-wob,e.b.x,e.b.y);
    ctx.strokeStyle=rgba(e.col,Math.min(.24,al*.3));
    ctx.lineWidth=e.lw*3.4; ctx.stroke();
    ctx.strokeStyle=rgba(e.col,Math.min(.92,al));
    ctx.lineWidth=e.lw; ctx.stroke();
  }
  ctx.globalCompositeOperation='source-over';

  /* spine pulse across the whole pipeline */
  if(spine.length>=2){
    ctx.beginPath(); ctx.moveTo(spine[0].x,spine[0].y);
    for(let i=1;i<spine.length;i++){ const A=spine[i-1],B=spine[i]; ctx.bezierCurveTo((A.x+B.x)/2,A.y,(A.x+B.x)/2,B.y,B.x,B.y); }
    ctx.strokeStyle=rgba(CC.mass,(hot?.16:.07)+flash*.22); ctx.lineWidth=hot?4.4:2.2; ctx.stroke();
    ctx.setLineDash([11,13]); ctx.lineDashOffset=-((T*46)%24);
    ctx.strokeStyle=rgba(CC.mass,(hot?.62:.24)+flash*.3); ctx.lineWidth=hot?1.9:1.1; ctx.stroke();
    ctx.setLineDash([]);
  }

  /* particles */
  if(hot&&edges.length&&particles.length<130&&Math.random()<.85){
    const n=1+(Math.random()*3|0);
    for(let i=0;i<n;i++){
      const pick=edges[(Math.pow(Math.random(),1.7)*edges.length)|0];
      if(pick&&pick.base>.1) particles.push({e:pick,t:0,v:.01+Math.random()*.02,life:1,r:pick.mass?2.3:1.7,col:pick.pcol});
    }
  }
  ctx.globalCompositeOperation='lighter';
  const keep=[];
  for(const p of particles){
    p.t+=p.v; p.life-=.0085;
    if(p.t>1||p.life<=0) continue;
    let pos;
    if(p.spine){
      if(p.t<0){ keep.push(p); continue; }
      const n=spine.length-1, f=p.t*n, i=Math.min(n-1,f|0);
      if(n<1) continue;
      pos=bez(spine[i],spine[i+1],f-i,0);
    } else if(p.e){ pos=edgePoint(p.e,p.t,T); } else continue;
    const al=clamp(p.life,0,1);
    ctx.beginPath(); ctx.arc(pos.x,pos.y,p.r*2.6,0,TAU); ctx.fillStyle=rgba(p.col,al*.12); ctx.fill();
    ctx.beginPath(); ctx.arc(pos.x,pos.y,p.r,0,TAU); ctx.fillStyle=rgba(p.col,al*.95); ctx.fill();
    keep.push(p);
  }
  particles=keep;
  ctx.globalCompositeOperation='source-over';

  /* nodes */
  for(const col of L.cols){
    const isDec=col.id==='decision';
    for(let i=0;i<col.nodes.length;i++){
      const n=col.nodes[i];
      const ph=(i*0.7+col.cx*0.01);
      const pv=clamp(n.v*(.82+.18*Math.sin(T*2.1+ph))+flash*(isDec?.08:.22),0,1.3);
      const r=R*(.78+pv*.52)*(isDec?.92:1);
      const c=n.color||col.color;
      if(!isDec){
        ctx.globalCompositeOperation='lighter';
        ctx.beginPath(); ctx.arc(n.x,n.y,r*3.4,0,TAU); ctx.fillStyle=rgba(c,.035+pv*.085); ctx.fill();
        ctx.beginPath(); ctx.arc(n.x,n.y,r*1.95,0,TAU); ctx.fillStyle=rgba(c,.055+pv*.16); ctx.fill();
        ctx.globalCompositeOperation='source-over';
      } else {
        ctx.beginPath(); ctx.arc(n.x,n.y,r*2.2,0,TAU); ctx.fillStyle=rgba(c,.05+pv*.08); ctx.fill();
      }
      ctx.beginPath(); ctx.arc(n.x,n.y,r,0,TAU);
      ctx.fillStyle=rgba(c,isDec?(.28+pv*.45):(.3+pv*.66)); ctx.fill();
      ctx.strokeStyle=rgba(CC.white,isDec?(.1+pv*.22):(.12+pv*.42)); ctx.lineWidth=1; ctx.stroke();
      if(!isDec && pv>.42){ ctx.beginPath(); ctx.arc(n.x,n.y,r*.38,0,TAU); ctx.fillStyle=rgba(CC.white,(pv-.42)*.8); ctx.fill(); }
    }
  }

  /* token labels */
  ctx.textBaseline='middle';
  for(const col of L.cols){
    if(!col.labels) continue;
    ctx.font='9px IBM Plex Mono,monospace'; ctx.textAlign='right';
    for(let i=0;i<col.nodes.length&&i<col.labels.length;i++){
      const n=col.nodes[i], lb=col.labels[i]; if(!lb) continue;
      const on=n.v>.3;
      const tx=n.x-R*1.6-6;
      const wpx=ctx.measureText(lb).width;
      ctx.fillStyle='rgba(3,8,11,.72)';
      roundRectFill(ctx,tx-wpx-5,n.y-7,wpx+8,14,3);
      ctx.fillStyle=on?rgba(col.color,.95):'rgba(99,122,131,.8)';
      ctx.fillText(lb,tx-1,n.y+.5);
    }
  }

  /* column headers + decision labels */
  ctx.textAlign='center';
  for(const col of L.cols){
    const act=clamp(+(a[col.act]||.12),0,1);
    ctx.font='600 10.5px IBM Plex Mono,monospace';
    ctx.fillStyle=rgba(col.color,.62+act*.38);
    ctx.fillText(col.title,col.cx,L.padT-32);
    ctx.font='8.5px IBM Plex Mono,monospace';
    ctx.fillStyle='rgba(88,111,120,.95)';
    ctx.fillText(col.sub,col.cx,L.padT-21);
    if(col.full>col.n&&col.id!=='decision'){
      ctx.font='8px IBM Plex Mono,monospace'; ctx.fillStyle='rgba(60,79,88,.95)';
      ctx.fillText(col.n+' of '+col.full,col.cx,L.H-L.padB+14);
    }
  }
  const Dn=L.by.decision;
  if(Dn){
    const ex=F.decision!=='refuse'&&F.decision!=='idle';
    ctx.font='600 9px IBM Plex Mono,monospace'; ctx.textAlign='center';
    ctx.fillStyle=rgba(CC.extract,ex?.62:.28);
    ctx.fillText('EXTRACT · span',Dn.cx,L.padT+L.innerH*.58);
    ctx.fillStyle=rgba(CC.refuse,ex?.22:.62);
    ctx.fillText('REFUSE · abstain',Dn.cx,L.padT+L.innerH*.93);
    ctx.font='8px IBM Plex Mono,monospace'; ctx.fillStyle='rgba(100,120,128,.9)';
    ctx.fillText('conf '+fmt(F.confidence,3),Dn.cx,L.padT+L.innerH*.68);
  }

  ctx.restore();

  /* soft decision wash — no additive bloom */
  if(hot&&F.decision&&F.decision!=='idle'){
    const c=F.decision==='refuse'?CC.refuse:CC.extract;
    const g2=ctx.createLinearGradient(W,0,W*.78,0);
    g2.addColorStop(0,rgba(c,.045+flash*.04)); g2.addColorStop(1,rgba(c,0));
    ctx.fillStyle=g2; ctx.fillRect(W*.78,0,W*.22,H);
  }
  /* scanline */
  ctx.fillStyle='rgba(63,224,212,.028)';
  const sy=((T*36)%(H+120))-60;
  ctx.fillRect(0,sy,W,2);
}

function roundRect(ctx,x,y,w,h,r){
  const rr=Math.min(r,w/2,h/2);
  ctx.moveTo(x+rr,y); ctx.arcTo(x+w,y,x+w,y+h,rr); ctx.arcTo(x+w,y+h,x,y+h,rr);
  ctx.arcTo(x,y+h,x,y,rr); ctx.arcTo(x,y,x+w,y,rr); ctx.closePath();
}
function roundRectFill(ctx,x,y,w,h,r){ ctx.beginPath(); roundRect(ctx,x,y,w,h,r); ctx.fill(); }

/* ═══════════════ interaction ═══════════════ */
function bindChrome(){
  const cv=$('cv');
  $('zIn').onclick=()=>{ view.s=Math.min(2.6,view.s*1.14); syncZ(); };
  $('zOut').onclick=()=>{ view.s=Math.max(.6,view.s/1.14); syncZ(); };
  const fit=()=>{ view={s:1,x:0,y:0}; syncZ(); layout=null; };
  $('zFit').onclick=fit; $('fitBtn').onclick=fit;
  cv.addEventListener('wheel',e=>{
    e.preventDefault();
    const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
    const bx=(mx-view.x)/view.s, by=(my-view.y)/view.s;
    view.s=clamp(view.s*(e.deltaY<0?1.09:1/1.09),.6,2.6);
    view.x=mx-bx*view.s; view.y=my-by*view.s; syncZ();
  },{passive:false});
  cv.addEventListener('mousedown',e=>{ drag=true; lx=e.clientX; ly=e.clientY; });
  window.addEventListener('mouseup',()=>drag=false);
  window.addEventListener('mousemove',e=>{ if(!drag) return; view.x+=e.clientX-lx; view.y+=e.clientY-ly; lx=e.clientX; ly=e.clientY; });
  window.addEventListener('resize',()=>{ layout=null; });
  $('presentBtn').onclick=togglePresent;
  window.addEventListener('keydown',e=>{
    if(e.target&&/INPUT|TEXTAREA/.test(e.target.tagName)) return;
    const k=e.key.toLowerCase();
    if(k==='p') togglePresent();
    else if(k==='l') cycleLog(1);
    else if(k==='f') fit();
  });
  syncZ();
}
function togglePresent(){
  document.body.classList.toggle('present');
  $('presentBtn').classList.toggle('active');
  layout=null;
  const box=$('log'); if(stick) requestAnimationFrame(()=>{ box.scrollTop=box.scrollHeight; });
}
function syncZ(){ $('zPct').textContent=Math.round(view.s*100)+'%'; }

boot().catch(e=>{ $('sq').textContent='API error: '+e; $('logmeta').textContent='API error: '+e; });
</script>
</body>
</html>
"""


class _ArchHandler(BaseHTTPRequestHandler):
    root: Path = Path(".")

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        # HTML first — never block the shell behind a torch.load
        if path in {"/", "/index.html"}:
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        # /api/live is hot path — never torch.load here
        if path == "/api/live":
            # Prefer on-disk config/ckpt dims; never let a finished micro-train
            # snapshot override product n_params in the live payload.
            cfg = _load_config(self.root, probe_ckpt=False)
            bundle = read_live_bundle(self.root)
            train = bundle.get("train") or {}
            if not (bundle.get("train_live") and train.get("status") == "running"):
                # Strip stale finished-train dims/losses so the UI shows product ckpt
                train = {
                    k: v
                    for k, v in train.items()
                    if k
                    not in {
                        "loss",
                        "gate_loss",
                        "span_loss",
                        "grad_norm",
                        "lr",
                        "step",
                        "epoch",
                        "epochs",
                        "sample_question",
                    }
                }
                train["status"] = "idle"
                train["dim"] = cfg.get("dim")
                train["n_params"] = cfg.get("n_params")
                train["arch"] = cfg.get("arch")
                bundle["train"] = train
                bundle["history"] = []
            if not bundle.get("forward"):
                bundle["forward"] = _idle_forward(cfg)
            if "events" not in bundle:
                bundle["events"] = _events(self.root)
            bundle["config"] = {
                "dim": cfg.get("dim"),
                "n_params": cfg.get("n_params"),
                "ckpt_n_params": cfg.get("ckpt_n_params"),
                "live_n_params": cfg.get("live_n_params"),
                "arch": cfg.get("arch"),
            }
            self._json(bundle)
            return
        if path == "/api/events":
            self._json({"events": _events(self.root), "ts": time.time()})
            return
        if path == "/api/health":
            cfg = _load_config(self.root, probe_ckpt=False)
            self._json(
                {
                    "ok": True,
                    "ts": time.time(),
                    "checkpoint": (self.root / "artifacts" / "cec.pt").exists(),
                    "dim": cfg.get("dim"),
                    "n_params": cfg.get("n_params"),
                }
            )
            return
        cfg = _load_config(self.root, probe_ckpt=True)
        if path == "/api/config":
            self._json(cfg)
            return
        if path == "/api/topology":
            self._json(_topology(cfg))
            return
        self._json({"error": "not_found"}, 404)


def run_architecture_dash(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7861,
    open_browser: bool = False,
) -> None:
    handler = type("ArchHandlerBound", (_ArchHandler,), {"root": Path(root)})
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"Live train monitor → {url}", flush=True)
    # Kept for CLI/API compatibility only. The monitor must never steal focus
    # or launch a browser from a training/watch process.
    _ = open_browser
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        server.server_close()
