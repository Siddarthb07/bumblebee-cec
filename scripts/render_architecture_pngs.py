"""Render architecture mermaid diagrams to PNG via mermaid.ink."""

from __future__ import annotations

import base64
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "ARCHITECTURE_FLOWCHART.md"
OUT = ROOT / "docs" / "architecture_pngs"
MMD = OUT / "mmd"

TITLES = {
    1: "01_system_context",
    2: "02_inference_end_to_end",
    3: "03_cec_neural_stack",
    4: "04_softcorrelator_math",
    5: "05_coverage_gate_decide",
    6: "06_training_pipeline",
    7: "07_skill_pack",
    8: "08_module_map",
    9: "09_benchmark_surface",
}


def extract_diagrams(md: str) -> list[str]:
    return re.findall(r"```mermaid\n(.*?)```", md, flags=re.S)


def to_png(mermaid: str, dest: Path, retries: int = 4) -> None:
    # Force light theme + white background for readable exports
    if "%%{init:" not in mermaid:
        mermaid = (
            "%%{init: {'theme':'base', 'themeVariables': {"
            "'primaryColor':'#ebe4f5','primaryTextColor':'#1a1a1a',"
            "'lineColor':'#333333','background':'#ffffff'}}}%%\n"
            + mermaid
        )
    payload = base64.urlsafe_b64encode(mermaid.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{payload}?type=png&bgColor=!white"
    last_err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bumblebee-arch-render/1.0"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
            if len(data) < 100 or not data.startswith(b"\x89PNG"):
                raise RuntimeError(f"bad image payload ({len(data)} bytes)")
            dest.write_bytes(data)
            return
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"failed {dest.name}: {last_err}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MMD.mkdir(parents=True, exist_ok=True)
    diagrams = extract_diagrams(SRC.read_text(encoding="utf-8"))
    if len(diagrams) != 9:
        raise SystemExit(f"expected 9 mermaid blocks, found {len(diagrams)}")
    for i, diagram in enumerate(diagrams, start=1):
        name = TITLES[i]
        mmd_path = MMD / f"{name}.mmd"
        png_path = OUT / f"{name}.png"
        mmd_path.write_text(diagram.strip() + "\n", encoding="utf-8")
        print(f"rendering {name} ...")
        to_png(diagram.strip(), png_path)
        print(f"  wrote {png_path.relative_to(ROOT)} ({png_path.stat().st_size} bytes)")
    print("done")


if __name__ == "__main__":
    main()
