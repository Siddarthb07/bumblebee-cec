#!/usr/bin/env python3
"""Watch dense train; on finish (or if already finished) run full benches.

Also restarts the architecture dash if it dies. Logs → artifacts/watchdog.log
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

LOG = ROOT / "artifacts" / "watchdog.log"
TRAIN_LOG = ROOT / "artifacts" / "train_dense.log"
LIVE = ROOT / "artifacts" / "live" / "train.json"
BENCH_LOG = ROOT / "artifacts" / "bench_after_dense.log"
DONE = ROOT / "artifacts" / "watchdog_done.json"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_train() -> dict:
    if not LIVE.exists():
        return {}
    try:
        return json.loads(LIVE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def train_proc_alive() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-f", "dim=640"], text=True)
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False


def dash_ok() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:7861/api/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_dash() -> None:
    if dash_ok():
        return
    log("dash down — restarting")
    subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "from pathlib import Path; from bumblebee.architecture_dash import run_architecture_dash; "
            "run_architecture_dash(Path('.'), host='127.0.0.1', port=7861, open_browser=False)",
        ],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        stdout=open(ROOT / "artifacts" / "arch_dash.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(2)


def restart_train() -> None:
    log("train process missing — restarting dim=640 epochs=4")
    cmd = (
        "import os; os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; "
        "os.environ['OPENBLAS_NUM_THREADS']='1'; "
        "from pathlib import Path; from bumblebee.train import train_model, clear_runtime_cache; "
        "root=Path('.'); clear_runtime_cache(); "
        "print('=== WATCHDOG RESTART DENSE dim=640 ===', flush=True); "
        "path=train_model(root, epochs=4, dim=640, max_p=128, max_q=48, lr=8e-4); "
        "print('=== SAVED', path, '===', flush=True); "
        "import json; cfg_path=root/'artifacts'/'config.json'; "
        "cfg=json.loads(cfg_path.read_text()) if cfg_path.exists() else {}; "
        "cfg['arch']='dense_cec'; cfg_path.write_text(json.dumps(cfg, indent=2))"
    )
    subprocess.Popen(
        [sys.executable, "-u", "-c", cmd],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        stdout=open(TRAIN_LOG, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def run_benches() -> dict:
    log("=== STARTING BENCHES ===")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from bumblebee.train import clear_runtime_cache, evaluate, ask_question_fast
    from bumblebee.calibration import fit_gate_temperature
    from bumblebee.results import bench_all
    from bumblebee.skills_bench import run_skills_bench
    from bumblebee.memory_bench import run_memory_bench
    from bumblebee.adversarial import run_adversarial_bench, write_adversarial_file
    from bumblebee.gate import AskResult

    root = ROOT
    clear_runtime_cache()
    cal = fit_gate_temperature(root)
    log(f"calibrate {cal}")
    clear_runtime_cache()
    rep = evaluate(root)
    log(f"eval in_domain_f1={rep.get('in_domain_f1')} n={rep.get('n')}")
    bench_all(root)
    log("bench_all done")
    sk = run_skills_bench(root, mode="auto")
    log(f"skills {sk.get('passed')}/{sk.get('n')}")
    mem = run_memory_bench(root, n_asks=8)
    log(
        f"memory params_m={mem['cec']['n_params_m']} peak_rss={mem['cec']['peak_rss_gb']}"
    )
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

    adv = run_adversarial_bench(root, ask_fn)
    log(f"adversarial keys={list(adv.keys())[:6]}")
    summary = {
        "eval_f1": rep.get("in_domain_f1"),
        "skills": f"{sk.get('passed')}/{sk.get('n')}",
        "params_m": mem["cec"]["n_params_m"],
        "ts": time.time(),
    }
    DONE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"=== ALL DONE {summary} ===")
    return summary


def main() -> None:
    log("watchdog start")
    ensure_dash()
    last_step = -1
    stalled = 0
    while True:
        ensure_dash()
        t = read_train()
        status = t.get("status")
        step = t.get("step")
        alive = train_proc_alive()
        log(
            f"status={status} step={step} ep={t.get('epoch')}/{t.get('epochs')} "
            f"D={t.get('dim')} alive={alive} loss={t.get('loss')}"
        )

        if status == "finished":
            if DONE.exists():
                log("already benched — idle")
                time.sleep(120)
                continue
            try:
                with open(BENCH_LOG, "a", encoding="utf-8") as bf:
                    # tee benches into log via print already
                    pass
                run_benches()
            except Exception as e:
                log(f"BENCH ERROR: {e!r}")
                time.sleep(60)
            continue

        if not alive and status == "running":
            # Process died mid-train — restart full train (no mid-ckpt)
            age = time.time() - float(t.get("ts") or 0)
            if age > 90:
                log(f"train dead (age={age:.0f}s) — restart")
                restart_train()
                time.sleep(30)
                continue

        if not alive and status not in {"running", "finished"}:
            # Never started / idle — start train
            log("no train — starting dim=640")
            restart_train()
            time.sleep(30)
            continue

        if step == last_step and status == "running":
            stalled += 1
        else:
            stalled = 0
            last_step = step
        if stalled >= 40 and alive:
            # ~40 min no step progress with process alive — likely hung
            log("train stalled 40 polls — killing and restarting")
            subprocess.call(["pkill", "-f", "dim=640"])
            time.sleep(5)
            restart_train()
            stalled = 0
            time.sleep(30)
            continue

        time.sleep(60)


if __name__ == "__main__":
    main()
