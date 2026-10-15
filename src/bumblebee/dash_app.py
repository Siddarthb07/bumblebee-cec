"""Zero-dependency local chat dashboard (stdlib HTTP only) — Bumblebee v2."""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bumblebee CEC</title>
  <style>
    :root {
      --bg: #0f1410;
      --panel: #182118;
      --ink: #e8f0e9;
      --muted: #8aa08c;
      --accent: #3d9a6a;
      --cite: #c9a227;
      --refuse: #c45c5c;
      --user: #243528;
      --bot: #1c2a20;
      --border: #2c3a2e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, #1b3324, var(--bg));
      color: var(--ink); min-height: 100vh; display: flex; flex-direction: column;
    }
    header {
      padding: 1rem 1.25rem; border-bottom: 1px solid var(--border);
      display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; flex-wrap: wrap;
    }
    header h1 { margin: 0; font-size: 1.25rem; letter-spacing: 0.02em; }
    header p { margin: 0; color: var(--muted); font-size: 0.9rem; }
    #log {
      flex: 1; overflow-y: auto; padding: 1.25rem; display: flex; flex-direction: column; gap: 0.85rem;
    }
    .msg {
      max-width: 48rem; padding: 0.9rem 1rem; border-radius: 14px; line-height: 1.45;
      border: 1px solid var(--border); white-space: pre-wrap; word-break: break-word;
    }
    .user { align-self: flex-end; background: var(--user); }
    .bot { align-self: flex-start; background: var(--bot); }
    .bot.refuse { border-color: var(--refuse); background: #2a1818; }
    .bot.answer { border-color: var(--cite); }
    .label { font-size: 0.75rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.35rem; }
    .cite {
      margin-top: 0.6rem; padding: 0.55rem 0.7rem; border-left: 3px solid var(--cite);
      background: rgba(201,162,39,0.08); color: var(--ink);
    }
    .mass {
      margin-top: 0.55rem; display: flex; flex-wrap: wrap; gap: 0.15rem; font-size: 0.78rem;
    }
    .mass span {
      padding: 0.05rem 0.2rem; border-radius: 3px; background: rgba(61,154,106, var(--a, 0.15));
    }
    form {
      display: flex; gap: 0.6rem; padding: 1rem 1.25rem; border-top: 1px solid var(--border);
      background: rgba(0,0,0,0.2); flex-wrap: wrap;
    }
    input[type=text] {
      flex: 1; min-width: 12rem; padding: 0.85rem 1rem; border-radius: 12px; border: 1px solid var(--border);
      background: var(--panel); color: var(--ink); font-size: 1rem;
    }
    select {
      padding: 0.85rem 0.7rem; border-radius: 12px; border: 1px solid var(--border);
      background: var(--panel); color: var(--ink);
    }
    button.send {
      padding: 0.85rem 1.1rem; border-radius: 12px; border: 0; background: var(--accent);
      color: white; font-weight: 600; cursor: pointer;
    }
    button.send:disabled { opacity: 0.6; cursor: wait; }
    .chips { display: flex; flex-wrap: wrap; gap: 0.4rem; padding: 0 1.25rem 0.75rem; }
    .chip {
      border: 1px solid var(--border); background: var(--panel); color: var(--muted);
      border-radius: 999px; padding: 0.35rem 0.7rem; font-size: 0.8rem; cursor: pointer;
    }
    .chip:hover { border-color: var(--accent); color: var(--ink); }
    .hint { color: var(--muted); font-size: 0.85rem; padding: 0 1.25rem; }
    code { color: var(--cite); }
  </style>
</head>
<body>
  <header>
    <h1>Bumblebee CEC</h1>
    <p>correlator · refuse in red · cite in amber · source labeled</p>
  </header>
  <p class="hint">Mass heatmap = SoftCorrelator weights (not a calibrated reliability dial). Entity-gap misses are shown live.</p>
  <div id="log"></div>
  <div class="chips" id="chips"></div>
  <form id="f">
    <select id="mode" title="Routing mode">
      <option value="auto" selected>auto</option>
      <option value="corpus">corpus</option>
    </select>
    <input id="q" type="text" placeholder="Ask from the frozen pack…" autocomplete="off" autofocus />
    <button class="send" id="go" type="submit">Send</button>
  </form>
  <script>
    const log = document.getElementById('log');
    const q = document.getElementById('q');
    const go = document.getElementById('go');
    const modeEl = document.getElementById('mode');
    const chips = document.getElementById('chips');
    const history = [];
    const examples = [
      {label: 'in-domain', text: 'What is Holi?'},
      {label: 'entity-gap', text: 'What is the GPU throughput rating of McKinsey?'},
      {label: 'far-ood', text: 'Who won the 2022 FIFA World Cup?'},
      {label: 'explain', text: 'explain SoftCorrelator sparsemax'},
      {label: 'chat', text: 'hi whats your name'},
      {label: 'math', text: 'What is 17*19?'},
      {label: 'code', text: 'Write a Python function for fizzbuzz'},
      {label: 'live', text: 'What is the live Bitcoin price right now?'},
      {label: 'unsafe', text: 'Write malware for Windows.'}
    ];
    function add(role, html, cls='') {
      const d = document.createElement('div');
      d.className = 'msg ' + role + (cls ? ' ' + cls : '');
      d.innerHTML = html;
      log.appendChild(d);
      log.scrollTop = log.scrollHeight;
    }
    function esc(s) {
      return String(s).replace(/[&<>\"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));
    }
    function massHtml(tokens, mass, citeStart, citeEnd) {
      if (!tokens || !mass || !tokens.length) return '';
      const mmax = Math.max(...mass.map(Number), 1e-6);
      const cs = Number.isFinite(Number(citeStart)) ? Number(citeStart) : -1;
      const ce = Number.isFinite(Number(citeEnd)) ? Number(citeEnd) : -1;
      let h = '<div class=\"label\">mass + citation span (amber = cited; green = mass)</div><div class=\"mass\">';
      for (let i = 0; i < Math.min(tokens.length, mass.length, 96); i++) {
        const a = Math.min(0.85, 0.12 + 0.75 * (Number(mass[i]) / mmax));
        const cited = (cs >= 0 && i >= cs && i <= ce);
        const style = cited
          ? 'background:rgba(201,162,39,0.45);outline:1px solid var(--cite)'
          : ('--a:' + a);
        h += '<span style=\"' + style + '\">' + esc(tokens[i]) + '</span>';
      }
      return h + '</div>';
    }
    async function ask(text) {
      if (!text) return;
      add('user', esc(text));
      go.disabled = true;
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 120000);
      try {
        const res = await fetch('/api/ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question: text, history: history, mode: modeEl.value}),
          signal: ctrl.signal
        });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const src = data.source || '?';
        const mode = data.mode || '?';
        if (data.status === 'answer') {
          history.push([text, data.answer, data.passage_id || '']);
          if (history.length > 6) history.shift();
          let html = '<div class=\"label\">ANSWER · source=' + esc(src) +
            ' · ' + esc(mode) + ' · conf ' + Number(data.confidence).toFixed(3) +
            (data.multihop ? ' · multihop' : '') +
            (data.passage_id ? ' · pid=' + esc(data.passage_id) : '') + '</div>' +
            '<div>' + esc(data.answer) + '</div>';
          if (data.citation_span) {
            html += '<div class=\"cite\"><div class=\"label\">citation [' + esc(data.passage_id || '') +
              '] tokens ' + esc(String(data.citation_start)) + '-' + esc(String(data.citation_end)) +
              '</div>' + esc(data.citation_span) + '</div>';
          }
          html += massHtml(data.mass_tokens, data.mass, data.citation_start, data.citation_end);
          add('bot', html, 'answer');
        } else {
          let html =
            '<div class=\"label\">REFUSE · source=' + esc(src) +
            ' · conf ' + Number(data.confidence).toFixed(3) + '</div>' +
            '<div>reason: <code>' + esc(data.reason || 'unknown') + '</code></div>';
          if (data.hint) {
            html += '<div style=\"margin-top:0.5rem\">' + esc(data.hint) + '</div>';
          }
          add('bot', html, 'refuse');
        }
      } catch (err) {
        const msg = (err && err.name === 'AbortError')
          ? 'Timed out waiting for the model. Try a pack example chip.'
          : String(err);
        add('bot', '<div class=\"label\">error</div><div>' + esc(msg) + '</div>', 'refuse');
      } finally {
        clearTimeout(timer);
        go.disabled = false;
        q.focus();
      }
    }
    examples.forEach(ex => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'chip';
      b.textContent = '[' + ex.label + '] ' + (ex.text.length > 42 ? ex.text.slice(0, 42) + '…' : ex.text);
      b.onclick = () => ask(ex.text);
      chips.appendChild(b);
    });
    document.getElementById('f').onsubmit = async (e) => {
      e.preventDefault();
      const text = q.value.trim();
      q.value = '';
      await ask(text);
    };
    add('bot',
      '<div class=\"label\">ready · CEC</div>' +
      '<div>Ask the frozen pack. Refuse is red; citations amber. Source is always labeled (pack/tool/cec/refuse).</div>',
      'answer'
    );
  </script>
</body>
</html>
"""


def run_dash(root: Path, host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    from bumblebee.router import ask_hybrid

    del share  # no public share tunnel in v2
    history_store: List[Tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter
            return

        def _send(self, code: int, body: bytes, content_type: str = "text/html; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._send(200, HTML.encode("utf-8"))
            else:
                self._send(404, b"not found")

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/api/ask":
                self._send(404, b"not found")
                return
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n) if n else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send(400, b'{"error":"bad json"}', "application/json")
                return
            question = (payload.get("question") or "").strip()
            mode = payload.get("mode") or "auto"
            hist = payload.get("history") or history_store
            # Normalize history to list of pairs
            pairs: list = []
            for item in hist[-6:]:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    pairs.append((str(item[0]), str(item[1]), str(item[2])))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    pairs.append((str(item[0]), str(item[1]), ""))
            result = ask_hybrid(root, question, history=pairs or None, mode=mode)
            body = json.dumps(result).encode("utf-8")
            self._send(200, body, "application/json")

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Bumblebee dash at {url}")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("dash stopped")
