"""Generate the presentation-ready Bumblebee deep-dive PDF.

Includes architecture diagrams, element-by-element explanation, skills/live tools,
and an honest ~500M-parameter comparison vs dense transformer LLMs.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String, Polygon
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "bumblebee_deep_dive.pdf"
# Keep legacy path updated too
OUT_LEGACY = ROOT / "output" / "pdf" / "bumblebee_depth.pdf"

INK = HexColor("#1a1a1a")
MUTED = HexColor("#444444")
ACCENT = HexColor("#0b3d2e")
RULE = HexColor("#c8d5cf")
BAND = HexColor("#eef4f1")
BOX = HexColor("#1f4d3a")
BOX2 = HexColor("#2a6b4f")
WARN = HexColor("#8a5a00")
LLM = HexColor("#1a3a5c")


def styles():
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Title"], fontName="Times-Bold",
            fontSize=30, leading=36, textColor=ACCENT, alignment=TA_CENTER, spaceAfter=10,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base["Normal"], fontName="Helvetica",
            fontSize=11.5, leading=15, textColor=MUTED, alignment=TA_CENTER, spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Times-Bold",
            fontSize=15, leading=19, textColor=ACCENT, spaceBefore=14, spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Times-Bold",
            fontSize=12, leading=15, textColor=INK, spaceBefore=10, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Times-Roman",
            fontSize=10, leading=14, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontName="Times-Roman",
            fontSize=9.5, leading=13, textColor=INK, leftIndent=4,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=8.5, leading=11, textColor=MUTED, alignment=TA_CENTER,
            spaceBefore=3, spaceAfter=8,
        ),
        "tiny": ParagraphStyle(
            "tiny", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=10, textColor=MUTED,
        ),
    }


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(0.75 * inch, 0.6 * inch, LETTER[0] - 0.75 * inch, 0.6 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.75 * inch, 0.4 * inch, "Bumblebee CEC — deep dive (presentation)")
    canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.4 * inch, f"{doc.page}")
    canvas.restoreState()


def p(style, text):
    return Paragraph(text, style)


def bullets(style, items):
    return ListFlowable(
        [ListItem(Paragraph(i, style), leftIndent=10, bulletColor=ACCENT) for i in items],
        bulletType="bullet",
        start="•",
    )


def _box(d: Drawing, x, y, w, h, label, fill=BOX, fs=8):
    d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=white, strokeWidth=0.5, rx=4, ry=4))
    # wrap-ish single line centered
    d.add(String(x + w / 2, y + h / 2 - 3, label, fontSize=fs, fillColor=white, textAnchor="middle"))


def _arrow(d: Drawing, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=INK, strokeWidth=1.2))
    # simple arrow head
    if abs(x2 - x1) > abs(y2 - y1):
        # horizontal
        direction = 1 if x2 > x1 else -1
        d.add(Polygon(
            [x2, y2, x2 - 6 * direction, y2 + 3.5, x2 - 6 * direction, y2 - 3.5],
            fillColor=INK, strokeColor=INK, strokeWidth=0.5,
        ))
    else:
        direction = 1 if y2 > y1 else -1
        d.add(Polygon(
            [x2, y2, x2 - 3.5, y2 - 6 * direction, x2 + 3.5, y2 - 6 * direction],
            fillColor=INK, strokeColor=INK, strokeWidth=0.5,
        ))


def diagram_product_router() -> Drawing:
    d = Drawing(480, 160)
    d.add(String(240, 148, "Product inference path (no open-LM sidecar)", fontSize=10,
                 fillColor=ACCENT, textAnchor="middle", fontName="Helvetica-Bold"))
    _box(d, 10, 90, 70, 36, "User Q", BOX)
    _box(d, 100, 90, 70, 36, "Router", BOX2)
    _box(d, 190, 115, 85, 30, "Unsafe→refuse", WARN)
    _box(d, 190, 70, 85, 30, "Live tools", HexColor("#0d4f6c"))
    _box(d, 295, 115, 85, 30, "Skill/Know pack", BOX2)
    _box(d, 295, 70, 85, 30, "CEC core", BOX)
    _box(d, 400, 90, 70, 36, "Answer/Refuse", ACCENT)
    _arrow(d, 80, 108, 100, 108)
    _arrow(d, 170, 108, 190, 130)
    _arrow(d, 170, 108, 190, 85)
    _arrow(d, 275, 130, 295, 130)
    _arrow(d, 275, 85, 295, 85)
    _arrow(d, 380, 130, 400, 108)
    _arrow(d, 380, 85, 400, 108)
    d.add(String(240, 20,
                 "Live = CoinGecko / wttr.in / clock / calc  ·  Pack = curated Q→A  ·  CEC = GRU+correlator+gate",
                 fontSize=7.5, fillColor=MUTED, textAnchor="middle"))
    return d


def diagram_transformer() -> Drawing:
    d = Drawing(480, 175)
    d.add(String(240, 162, "Generic transformer LLM block (simplified)", fontSize=10,
                 fillColor=LLM, textAnchor="middle", fontName="Helvetica-Bold"))
    _box(d, 20, 100, 80, 40, "Tokens → Emb", LLM)
    _box(d, 130, 100, 100, 40, "Q, K, V proj", LLM)
    _box(d, 260, 100, 110, 40, "softmax(QKᵀ/√d)V", LLM)
    _box(d, 400, 100, 65, 40, "FFN", LLM)
    _arrow(d, 100, 120, 130, 120)
    _arrow(d, 230, 120, 260, 120)
    _arrow(d, 370, 120, 400, 120)
    _box(d, 130, 35, 220, 40, "Stack N layers → next-token logits → sample", HexColor("#2a5080"), fs=8)
    _arrow(d, 240, 100, 240, 75)
    d.add(String(240, 12, "Cost: O(n²) attention per layer · KV cache grows with decode · default = keep generating",
                 fontSize=7.5, fillColor=MUTED, textAnchor="middle"))
    return d


def diagram_cec_stack() -> Drawing:
    d = Drawing(480, 200)
    d.add(String(240, 188, "CEC neural stack (element-by-element)", fontSize=10,
                 fillColor=ACCENT, textAnchor="middle", fontName="Helvetica-Bold"))
    _box(d, 8, 130, 88, 36, "Q / P tokens", BOX)
    _box(d, 110, 130, 100, 36, "Hash Emb+GRU", BOX2)
    _box(d, 230, 145, 110, 32, "SoftCorrelator", BOX)
    _box(d, 230, 105, 110, 32, "CoverageGate", WARN)
    _box(d, 360, 145, 105, 32, "Extractive span", BOX2)
    _box(d, 360, 105, 105, 32, "Small answer GRU", BOX2)
    _box(d, 200, 40, 160, 36, "decide: answer + cite  |  refuse", ACCENT, fs=8)
    _arrow(d, 96, 148, 110, 148)
    _arrow(d, 210, 148, 230, 160)
    _arrow(d, 210, 148, 230, 120)
    _arrow(d, 340, 161, 360, 161)
    _arrow(d, 340, 121, 360, 121)
    _arrow(d, 280, 105, 280, 76)
    d.add(String(240, 12,
                 "No full-sequence self-attention · sparsemax mass over passage cells · abstain is first-class",
                 fontSize=7.5, fillColor=MUTED, textAnchor="middle"))
    return d


def diagram_compare() -> Drawing:
    d = Drawing(480, 150)
    d.add(String(120, 135, "~350–500M dense LLM", fontSize=9, fillColor=LLM, textAnchor="middle",
                 fontName="Helvetica-Bold"))
    d.add(String(360, 135, "Bumblebee LargeCEC ~472M", fontSize=9, fillColor=ACCENT, textAnchor="middle",
                 fontName="Helvetica-Bold"))
    _box(d, 30, 55, 180, 70, "Dense emb + N× Attn/FFN\nnext-token everywhere", LLM, fs=8)
    _box(d, 270, 55, 180, 70, "Sparse hash emb + narrow GRU\ncorrelator + gate + tools", BOX, fs=8)
    d.add(String(240, 20, "Similar headline params ≠ same compute graph or behavior",
                 fontSize=8, fillColor=MUTED, textAnchor="middle", fontName="Helvetica-Oblique"))
    return d


def styled_table(rows, col_widths):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BAND),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s = styles()
    cfg = load_json(ROOT / "artifacts" / "config.json") or {}
    bench = load_json(ROOT / "artifacts" / "bench_all.json") or {}
    eval_r = load_json(ROOT / "artifacts" / "eval_report.json") or {}
    skills = load_json(ROOT / "artifacts" / "bench_skills.json") or {}

    n_params = cfg.get("n_params") or (cfg.get("scale") or {}).get("estimated_params") or 472_000_000
    dim = cfg.get("dim", 192)
    emb_dim = cfg.get("emb_dim", 896)
    buckets = cfg.get("n_buckets", 524288)

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.85 * inch,
        title="Bumblebee CEC — Architecture Deep Dive",
        author="Bumblebee project",
    )
    story = []

    # ——— Cover ———
    story.append(Spacer(1, 1.4 * inch))
    story.append(p(s["cover_title"], "Bumblebee CEC"))
    story.append(p(s["cover_sub"],
                   "Transformer-free correlator QA with coverage-gated refusal,<br/>"
                   "skill/knowledge packs, and non-LLM live tools"))
    story.append(Spacer(1, 0.25 * inch))
    story.append(p(s["cover_sub"],
                   f"Deep-dive architecture note · ~{n_params/1e6:.0f}M Mac-fit LargeCEC<br/>"
                   "Presentation companion — honesty first, measured claims only"))
    story.append(Spacer(1, 0.9 * inch))
    story.append(p(s["cover_sub"],
                   "Contents: product path · transformer anatomy · CEC elements · "
                   "math · benchmarks · ~500M LLM comparison · limits · demo"))
    story.append(PageBreak())

    # ——— 1 Exec ———
    story.append(p(s["h1"], "1. Executive summary"))
    story.append(p(s["body"],
                   "Bumblebee is <b>not</b> a drop-in ChatGPT replacement. It is a "
                   "<b>Correlated Evidence Cells (CEC)</b> system: encode question and passage with "
                   "GRUs (no self-attention), bind evidence with a SoftCorrelator, estimate coverage, "
                   "then either answer with a citation or refuse. Product routing adds curated skill/"
                   "knowledge packs and deterministic live tools (prices, weather, time, arithmetic) — "
                   "still <b>no open transformer LLM sidecar</b>."))
    story.append(bullets(s["bullet"], [
        f"<b>Checkpoint:</b> LargeCEC arch={cfg.get('arch','large_hash_cec')}, "
        f"params≈{n_params/1e6:.1f}M (hash emb {buckets}×{emb_dim}, core dim={dim}).",
        "<b>Win condition:</b> coverage-gated refusal + faithful answers on a frozen pack, "
        "not open-world MMLU supremacy.",
        "<b>Honesty:</b> On the sealed pack BM25 leads in-domain F1 (0.136 vs 0.122). "
        "Answered-subset ECE is about 0.915, so refusal is not a calibrated probability.",
    ]))
    story.append(diagram_product_router())
    story.append(p(s["caption"], "Figure 1. Product inference path."))

    # ——— 2 Transformer ———
    story.append(p(s["h1"], "2. What a generic LLM transformer does"))
    story.append(p(s["h2"], "2.1 Next-token prediction"))
    story.append(p(s["body"],
                   "A typical LLM is trained to maximize likelihood of the next token given prior tokens. "
                   "After pretraining (and often instruction tuning / RLHF), the same mechanism is used "
                   "for chat. The model is rewarded for fluent continuation — <b>silence is not the default</b>."))
    story.append(diagram_transformer())
    story.append(p(s["caption"], "Figure 2. Transformer block sketch."))
    story.append(p(s["h2"], "2.2 Why this creates forced answering"))
    story.append(p(s["body"],
                   "Unless specially trained or constrained, transformers keep emitting tokens even when "
                   "evidence is missing. That yields hallucinations: high fluency, low grounding. RAG helps "
                   "retrieve context but often still answers on weak retrieval. Bumblebee’s thesis is that "
                   "<b>abstention must be architectural</b>, not only a prompt instruction."))

    # ——— 3 CEC elements ———
    story.append(PageBreak())
    story.append(p(s["h1"], "3. CEC architecture — every element"))
    story.append(diagram_cec_stack())
    story.append(p(s["caption"], "Figure 3. CEC forward path."))

    elements = [
        ("Hash / token embedding",
         "Maps token IDs to vectors. LargeCEC uses a large <b>sparse hash embedding</b> "
         f"({buckets} buckets × {emb_dim} dim) so most of the ~{n_params/1e6:.0f}M parameters "
         "sit in a table SparseAdam can update sparsely on Mac."),
        ("LocalStateEncoder (GRUCell)",
         "Reads tokens left-to-right with a GRU. Cost O(T·d²) locally — <b>no QKᵀ over the full "
         "sequence</b>. Produces passage cell states P and query final state q."),
        ("SoftCorrelator",
         "Learned affinity between q and each passage cell, then <b>sparsemax</b> mass m. "
         "Evidence e = Σ mₜ Pₜ. Sparse mass concentrates on supporting cells."),
        ("CoverageGate",
         "conf = σ(MLP([q;e])) × (1 − H(m)/log T_valid). Low concentration or weak bind → "
         "confidence below threshold → <b>refuse low_coverage</b>."),
        ("ExtractiveDecoder",
         "Predicts a start index / span for citation from passage tokens (faithfulness anchor)."),
        ("SmallAnswerLM (GRU)",
         "Closed-vocab next-token head conditioned on [q;e]. This is the only “next word” piece — "
         "tiny vs a full LLM, and gated by coverage + faithfulness."),
        ("gate.decide + garbage filters",
         "Refuse on empty/unfaithful/garbage decode (question echo, “known answer” collapse). "
         "Never substitutes gold labels at inference."),
        ("Skill / knowledge packs",
         "Deterministic Q→A for chat/math/code and educational explain topics (transformers, LLMs, "
         "CEC, RAG, …) so demos do not depend on undertrained freeform decode."),
        ("Live tools (non-LLM)",
         "CoinGecko BTC/ETH, wttr.in weather, local clock, safe arithmetic. Unsafe malware prompts hard-refuse."),
    ]
    for title, body in elements:
        story.append(p(s["h2"], title))
        story.append(p(s["body"], body))

    story.append(p(s["h2"], "3.1 SoftCorrelator equations"))
    story.append(p(s["body"],
                   "logitsₜ = (W_q q)ᵀ (W_k Pₜ) / τ &nbsp;&nbsp;→&nbsp;&nbsp; "
                   "m = sparsemax(logits) &nbsp;&nbsp;→&nbsp;&nbsp; e = Σₜ mₜ Pₜ. "
                   "Unlike transformer self-attention, mass is over <b>passage cells</b> for a fixed query, "
                   "not all-pairs token attention inside a generation window."))

    # ——— 4 Contrast table ———
    story.append(PageBreak())
    story.append(p(s["h1"], "4. Side-by-side: transformer LLM vs Bumblebee CEC"))
    story.append(diagram_compare())
    story.append(p(s["caption"], "Figure 4. Similar parameter headlines, different machines."))

    diff = [
        [p(s["bullet"], "<b>Axis</b>"), p(s["bullet"], "<b>~350–500M dense LLM</b>"),
         p(s["bullet"], "<b>Bumblebee LargeCEC</b>")],
        [p(s["bullet"], "Objective"), p(s["bullet"], "Next-token likelihood"),
         p(s["bullet"], "Evidence mass + answer / abstain")],
        [p(s["bullet"], "Core binder"), p(s["bullet"], "Softmax self-attention O(n²)"),
         p(s["bullet"], "GRU local state + sparsemax correlator")],
        [p(s["bullet"], "Params shape"), p(s["bullet"], "Dense emb + stacked Attn/FFN"),
         p(s["bullet"], "Sparse hash emb + narrow GRU core")],
        [p(s["bullet"], "Under ignorance"), p(s["bullet"], "Usually still generates"),
         p(s["bullet"], "Structured refuse")],
        [p(s["bullet"], "Grounding"), p(s["bullet"], "Optional (RAG/prompt)"),
         p(s["bullet"], "Citation + faithfulness required")],
        [p(s["bullet"], "Live facts"), p(s["bullet"], "Tools or parametric guess"),
         p(s["bullet"], "Explicit non-LLM tools")],
        [p(s["bullet"], "Open chat"), p(s["bullet"], "Strong if well trained"),
         p(s["bullet"], "Pack/knowledge + limited CEC decode")],
        [p(s["bullet"], "Serving memory"), p(s["bullet"], "KV cache grows with tokens"),
         p(s["bullet"], "No transformer KV cache")],
    ]
    story.append(styled_table(diff, [1.1 * inch, 2.6 * inch, 2.6 * inch]))
    story.append(p(s["caption"], "Table 1. Architectural and behavioral contrast."))

    # ——— 5 Benchmarks ———
    story.append(p(s["h1"], "5. Measured benchmarks (this repo)"))
    story.append(p(s["body"],
                   "Protocol: held-out in-domain + all OOD; predictions never use gold answers. "
                   "Lower AURC is better. Skills suite scores product routing (packs + tools + CEC)."))

    # Skills
    if skills:
        story.append(p(s["h2"], "5.1 Skills bench"))
        story.append(p(s["body"],
                       f"Overall <b>{skills.get('passed')}/{skills.get('n')}</b> "
                       f"({100*float(skills.get('accuracy') or 0):.0f}%). "
                       f"Model under test: {skills.get('model_under_test', 'cec')}."))
        sk_rows = [[p(s["bullet"], "<b>Category</b>"), p(s["bullet"], "<b>Passed</b>"),
                    p(s["bullet"], "<b>Acc</b>")]]
        for cat, sc in sorted((skills.get("by_category") or {}).items()):
            sk_rows.append([
                p(s["bullet"], cat),
                p(s["bullet"], f"{sc.get('passed')}/{sc.get('n')}"),
                p(s["bullet"], f"{sc.get('accuracy', 0):.2f}"),
            ])
        story.append(styled_table(sk_rows, [2.2 * inch, 1.2 * inch, 1.2 * inch]))
        story.append(p(s["caption"], "Table 2. Skills / live / refuse suite."))

    # Held-out CEC
    story.append(p(s["h2"], "5.2 Held-out corpus eval (CEC)"))
    if eval_r:
        erows = [[
            p(s["bullet"], "<b>In-domain F1</b>"),
            p(s["bullet"], "<b>OOD refuse P</b>"),
            p(s["bullet"], "<b>OOD refuse R</b>"),
            p(s["bullet"], "<b>ECE</b>"),
            p(s["bullet"], "<b>AURC</b>"),
            p(s["bullet"], "<b>Faithful</b>"),
        ], [
            p(s["bullet"], f"{eval_r.get('in_domain_f1_mean', float('nan')):.3f}"),
            p(s["bullet"], f"{eval_r.get('ood_refuse_precision', float('nan')):.3f}"),
            p(s["bullet"], f"{eval_r.get('ood_refuse_recall', float('nan')):.3f}"),
            p(s["bullet"], f"{eval_r.get('ece', float('nan')):.3f}"),
            p(s["bullet"], f"{eval_r.get('aurc', float('nan')):.3f}"),
            p(s["bullet"], f"{eval_r.get('faithfulness_rate', float('nan')):.3f}"),
        ]]
        story.append(styled_table(erows, [1.05 * inch] * 6))
        story.append(p(s["caption"], "Table 3. Latest artifacts/eval_report.json (LargeCEC)."))

    # vs baselines
    story.append(p(s["h2"], "5.3 Same-split baselines (bench_all)"))
    if bench:
        brows = [[
            p(s["bullet"], "<b>System</b>"),
            p(s["bullet"], "<b>F1</b>"),
            p(s["bullet"], "<b>Refuse P</b>"),
            p(s["bullet"], "<b>Refuse R</b>"),
            p(s["bullet"], "<b>ECE</b>"),
            p(s["bullet"], "<b>AURC</b>"),
        ]]
        for key in ("cec", "bm25", "embed"):
            if key not in bench:
                continue
            r = bench[key]
            brows.append([
                p(s["bullet"], key),
                p(s["bullet"], f"{r.get('in_domain_f1_mean', float('nan')):.3f}"),
                p(s["bullet"], f"{r.get('ood_refuse_precision', float('nan')):.3f}"),
                p(s["bullet"], f"{r.get('ood_refuse_recall', float('nan')):.3f}"),
                p(s["bullet"], f"{r.get('ece', float('nan')):.3f}"),
                p(s["bullet"], f"{r.get('aurc', float('nan')):.3f}"),
            ])
        story.append(styled_table(brows, [1.35 * inch, 0.75 * inch, 0.9 * inch, 0.9 * inch, 0.75 * inch, 0.75 * inch]))
        story.append(p(s["caption"],
                       "Table 4. Same retrieved protocol. BM25 leads in-domain F1 on the sealed pack."))
    story.append(p(s["body"],
                   "<b>How to say this in a pitch:</b> “On this frozen pack, retrieval can win extractive F1; "
                   "CEC’s job is coverage-gated refusal and faithful answering, and the gate is not yet calibrated. We publish both.”"))

    # ——— 6 Param-matched LLM ———
    story.append(PageBreak())
    story.append(p(s["h1"], "6. Comparison to ~350–500M parameter LLMs"))
    story.append(p(s["body"],
                   "Fairness note: headline parameter count is a weak equalizer. A dense "
                   "OPT-350M / GPT-2 Medium (355M) / Pythia-410M spends almost all parameters on "
                   "transformer blocks used every token. Bumblebee’s ~472M is dominated by a "
                   "<b>sparse hash embedding</b>; the recurrent core (dim=192) is far smaller. "
                   "Treat this as a <b>parameter-budget peer class</b>, not an apples-to-apples FLOPs match."))

    peer = [
        [p(s["bullet"], "<b>Model</b>"), p(s["bullet"], "<b>Params</b>"),
         p(s["bullet"], "<b>Arch</b>"), p(s["bullet"], "<b>Default behavior</b>")],
        [p(s["bullet"], "GPT-2 Medium"), p(s["bullet"], "355M"),
         p(s["bullet"], "Dense transformer"), p(s["bullet"], "Next-token; no native abstain")],
        [p(s["bullet"], "OPT-350M"), p(s["bullet"], "350M"),
         p(s["bullet"], "Dense transformer"), p(s["bullet"], "Next-token; open generation")],
        [p(s["bullet"], "Pythia-410M"), p(s["bullet"], "410M"),
         p(s["bullet"], "Dense transformer"), p(s["bullet"], "Next-token research LM")],
        [p(s["bullet"], "Bumblebee LargeCEC"), p(s["bullet"], f"{n_params/1e6:.0f}M"),
         p(s["bullet"], "Hash+GRU+correlator"), p(s["bullet"], "Answer if covered else refuse")],
    ]
    story.append(styled_table(peer, [1.6 * inch, 0.85 * inch, 1.5 * inch, 2.4 * inch]))
    story.append(p(s["caption"], "Table 5. Parameter-class peers (public sizes)."))

    story.append(p(s["h2"], "6.1 What we compare (and what we do not)"))
    story.append(bullets(s["bullet"], [
        "<b>Do compare:</b> abstention on OOD, citation faithfulness, Mac-fit trainability, "
        "linear-ish binding cost vs attention windows, tool-backed live facts without an LLM.",
        "<b>Do not claim:</b> beating OPT-350M on open-ended MMLU/HumanEval/chat Elo. "
        "Those models are trained on far more tokens for continuation; different objective.",
        "<b>In-repo proxy opponent:</b> BM25+abstain, hashing embed+abstain, and the tiny "
        "TransformerEncoder baseline on the <b>same</b> frozen eval JSONL (Table 4).",
        "<b>Behavioral expectation for a ~350M LLM on our OOD asks:</b> without a refuse "
        "policy it will usually invent; CEC’s measured OOD refuse recall is ≈ "
        f"{(eval_r or bench.get('cec') or {}).get('ood_refuse_recall', float('nan')):.2f}.",
    ]))

    story.append(p(s["h2"], "6.2 Qualitative head-to-head on product behaviors"))
    qh = [
        [p(s["bullet"], "<b>Prompt</b>"), p(s["bullet"], "<b>~350M LLM (typical)</b>"),
         p(s["bullet"], "<b>Bumblebee</b>")],
        [p(s["bullet"], "Explain transformers"),
         p(s["bullet"], "Fluent essay from weights"),
         p(s["bullet"], "Knowledge-pack grounded explanation")],
        [p(s["bullet"], "Live BTC price"),
         p(s["bullet"], "Stale/guess unless tooled"),
         p(s["bullet"], "CoinGecko tool quote")],
        [p(s["bullet"], "Far OOD trivia"),
         p(s["bullet"], "Often confident guess"),
         p(s["bullet"], "Refuse outside pack")],
        [p(s["bullet"], "Write malware"),
         p(s["bullet"], "Policy-dependent"),
         p(s["bullet"], "Hard refuse unsafe")],
        [p(s["bullet"], "Corpus QA"),
         p(s["bullet"], "May ignore notes"),
         p(s["bullet"], "Correlator + citation")],
    ]
    story.append(styled_table(qh, [1.5 * inch, 2.2 * inch, 2.2 * inch]))
    story.append(p(s["caption"], "Table 6. Product-behavior contrast (qualitative + measured refuse)."))

    # ——— 7 Limits ———
    story.append(p(s["h1"], "7. Limits (say these out loud)"))
    story.append(bullets(s["bullet"], [
        "Open-world chat quality is pack/tool limited; undertrained CEC decode is intentionally gated.",
        "In-domain F1 can lag BM25 on small factual packs — publish it.",
        "ECE can still be high; confidence calibration is an active weakness.",
        "Sparse ~500M ≠ dense ~500M compute; do not overclaim FLOPs parity.",
        "Live tools need network; they are not parametric world knowledge.",
        "Nothing here is a granted patent; prior art (RAG, selective prediction, sparsemax, GRUs) is acknowledged.",
    ]))

    # ——— 8 Demo ———
    story.append(p(s["h1"], "8. Demo & reproduction"))
    story.append(p(s["body"],
                   "<font face='Courier' size='8.5'>"
                   "pip install -e \".[dev]\"<br/>"
                   "python -m bumblebee.cli dash          # http://127.0.0.1:7860/<br/>"
                   "python -m bumblebee.cli bench --mode skills<br/>"
                   "python -m bumblebee.cli eval<br/>"
                   "python -m bumblebee.cli bench --mode all<br/>"
                   "python scripts/build_depth_pdf.py     # this PDF"
                   "</font>"))
    story.append(p(s["body"],
                   "Suggested live demo order: (1) explain LLM transformers (2) hi whats your name "
                   "(3) What is 17*19? (4) live Bitcoin price (5) far-OOD refuse (6) malware refuse."))

    story.append(p(s["h1"], "9. Bottom line"))
    story.append(p(s["body"],
                   "Transformers made continuation extraordinarily powerful and made fluent guessing "
                   "the default. Bumblebee is a measured counter-design: bind evidence with a correlator, "
                   "answer when covered, refuse when not, add packs/tools without smuggling in an open LLM. "
                   "At ~500M headline parameters it is a Mac-fit sparse CEC — compare it to dense 350–410M "
                   "LLMs on <b>abstention and grounding</b>, not on open-ended Elo."))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    OUT_LEGACY.write_bytes(OUT.read_bytes())
    print(f"Wrote {OUT}")
    print(f"Also updated {OUT_LEGACY}")
    return OUT


if __name__ == "__main__":
    build()
