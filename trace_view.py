"""Offline HTML viewer for captured traces (traces/<id>_<arm>.json, v2).

Usage: python trace_view.py exam1 folded   ->  traces/exam1_folded.html
Self-contained output (inline CSS/JS, no CDN, no server). Open in a browser:
step buttons / arrow keys scrub the run; the workspace panel shows the block
cards the model saw, fold deletions struck through, new summary cards green.
"""

import json
import sys
from pathlib import Path

TRACE_DIR = Path(__file__).parent / "traces"

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{{TITLE}}</title>
<style>
:root{--blue:#4477AA;--orange:#EE7733;--green:#2E8B57;--red:#C44E52;--bg:#fafafa;--ink:#222}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:var(--bg);color:var(--ink)}
header{padding:14px 20px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:0;z-index:2}
h1{font-size:16px;margin:0 0 4px}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button.stepbtn{padding:6px 12px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
button.stepbtn.active{background:var(--orange);color:#fff;border-color:var(--orange)}
button.arrow{font-size:16px;padding:6px 12px}
main{display:grid;grid-template-columns:1fr 340px;gap:16px;padding:16px 20px;max-width:1200px}
.panel{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px}
.panel h2{font-size:13px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.05em;color:#777}
.card{border:1px solid #ccc;border-left:4px solid var(--blue);border-radius:6px;padding:8px 10px;margin:6px 0;white-space:pre-wrap;font-size:12.5px;line-height:1.45}
.card .tag{font-size:10.5px;color:#999;font-family:ui-monospace,monospace}
.card.retracted{border-left-color:var(--red);background:#fdf0f0;opacity:.62;text-decoration:line-through}
.card.new{border-left-color:var(--green);background:#eef7f0;text-decoration:none}
.card.prompt{border-left-color:#999}
.meta{font-size:12px;color:#555;line-height:1.6}
.meta b{color:var(--ink)}
.report{font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:pre-wrap;background:#f4f4f8;border-radius:6px;padding:8px}
.foldbox{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;background:#fff7e8;border:1px solid #e8d5a8;border-radius:6px;padding:8px;white-space:pre-wrap;margin-top:8px}
.tokbar{height:120px;display:flex;align-items:flex-end;gap:3px;margin-top:6px}
.tokbar .b{flex:1;background:var(--blue);border-radius:2px 2px 0 0;min-height:2px;position:relative}
.tokbar .b.active{background:var(--orange)}
.tokbar .b span{position:absolute;top:-14px;left:50%;transform:translateX(-50%);font-size:9px;color:#777}
.kpi{display:flex;gap:16px;font-size:12px;color:#555;margin-top:6px}
.kpi b{color:var(--ink)}
.answer{font-size:13px}
.gold{color:#777;font-size:12px}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head>
<body>
<header>
<h1>{{TITLE}}</h1>
<div class="controls">
<button class="stepbtn arrow" onclick="scrub(-1)">&#9664;</button>
<span id="steplabel" style="font-size:14px;font-weight:600;min-width:90px;text-align:center"></span>
<button class="stepbtn arrow" onclick="scrub(1)">&#9654;</button>
<span id="stepbtns"></span>
</div>
</header>
<main>
<section class="panel"><h2>Workspace — what the model sees at this step</h2>
<div id="workspace"></div>
<div id="foldinfo"></div>
</section>
<aside>
<div class="panel"><h2>Step</h2><div class="meta" id="stepmeta"></div></div>
<div class="panel" style="margin-top:12px"><h2>Report</h2><div class="report" id="report"></div></div>
<div class="panel" style="margin-top:12px"><h2>Prompt tokens by step</h2>
<div class="tokbar" id="tokbar"></div>
<div class="kpi" id="kpi"></div></div>
<div class="panel" style="margin-top:12px"><h2>Result</h2><div class="answer meta" id="answer"></div></div>
</aside>
</main>
<script>
const T = {{TRACE}};
let cur = 1;
const N = T.steps.length;
const btns = document.getElementById("stepbtns");
for (let i = 1; i <= N; i++) {
  const b = document.createElement("button");
  b.className = "stepbtn"; b.textContent = i; b.onclick = () => { cur = i; render(); };
  btns.appendChild(b);
}
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") scrub(1);
  if (e.key === "ArrowLeft") scrub(-1);
});
function scrub(d) { cur = Math.min(N, Math.max(1, cur + d)); render(); }
function key(b) { return b.start + "-" + b.end; }
function render() {
  document.getElementById("steplabel").textContent = "step " + cur + " / " + N;
  [...btns.children].forEach((b, i) => b.classList.toggle("active", i + 1 === cur));

  const s = T.steps[cur - 1];
  const before = cur > 1 ? T.steps[cur - 2].workspace_after : [];
  const after = s.workspace_after;
  const newKeys = new Set(after.map(key));
  const beforeKeys = new Set(before.map(key));
  const k = s.fold_applied ? s.fold.range[0] : null, last = s.fold_applied ? s.fold.range[1] : null;

  // workspace: the desk the model SAW this step (= before), plus fold outcome
  const ws = document.getElementById("workspace");
  if (T.arm === "baseline" || before.length === 0 && !s.fold_applied) {
    if (T.arm === "baseline") {
      ws.innerHTML = "<div class='meta'>Baseline arm: no workspace blocks — history is one ever-growing message list ("
        + s.prompt_tokens.toLocaleString() + " tokens this step).</div>";
    } else {
      ws.innerHTML = "<div class='meta'>Empty desk — first turn, nothing has happened yet.</div>";
    }
  } else {
    let html = "";
    for (const b of before) {
      const retracted = k !== null && b.start <= last && b.end >= k;
      const replaced = newKeys.has(key(b));
      html += card(b, retracted ? "retracted" : (replaced ? "" : "prompt"));
    }
    if (k !== null) {
      for (const b of after) if (!beforeKeys.has(key(b))) html += card(b, "new", "new summary block");
    }
    ws.innerHTML = html;
  }
  const fi = document.getElementById("foldinfo");
  if (s.fold_applied) {
    fi.innerHTML = "<div class='foldbox'>FOLD APPLIED  range [" + k + ", " + last + "]\\n"
      + s.fold.summary + "</div>";
  } else if (s.fold) {
    fi.innerHTML = "<div class='foldbox'>FOLD REJECTED (no-op) — range [" + s.fold.range[0] + ", " + s.fold.range[1]
      + "] invalid at this step</div>";
  } else if (s.fold_malformed) {
    fi.innerHTML = "<div class='foldbox'>FOLD MALFORMED — &lt;fold&gt; tag present but unparseable; ignored</div>";
  } else fi.innerHTML = "";

  document.getElementById("stepmeta").innerHTML =
    "<b>Action:</b> <span style='font-family:ui-monospace,monospace;font-size:12px'>" + esc(s.action) + "</span><br>" +
    "<b>Observation:</b> " + s.observation_chars.toLocaleString() + " chars<br>" +
    "<b>Prompt:</b> " + s.prompt_tokens.toLocaleString() + " tokens (approx)";
  document.getElementById("report").textContent = s.report || "(no report yet)";
  document.getElementById("answer").innerHTML =
    "<b>Predicted:</b> " + esc(T.answer || "(none)") + "<br><span class='gold'>Gold: " +
    T.golds.map(esc).join(" ; ") + "</span>";

  const tb = document.getElementById("tokbar");
  const maxT = Math.max(...T.steps.map(x => x.prompt_tokens));
  tb.innerHTML = T.steps.map(x =>
    "<div class='b" + (x.n === cur ? " active" : "") + "' style='height:" +
    Math.round(95 * x.prompt_tokens / maxT) + "%'><span>" + x.n + "</span></div>").join("");
  document.getElementById("kpi").innerHTML =
    "<div>peak <b>" + Math.max(...T.steps.map(x => x.prompt_tokens)).toLocaleString() + "</b></div>" +
    "<div>folds applied <b>" + T.steps.filter(x => x.fold_applied).length + "</b></div>" +
    "<div>total steps <b>" + N + "</b></div>";
}
function card(b, cls, note) {
  return "<div class='card " + cls + "'><span class='tag'>[steps " + b.start + "\\u2013" + b.end + "]"
    + (note ? " — " + note : "") + "</span>\\n" + esc(b.text.slice(0, 600))
    + (b.text.length > 600 ? "\\n…" : "") + "</div>";
}
function esc(s) { return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
render();
</script></body></html>"""


def main(exam_id: str, arm: str) -> None:
    trace = json.loads((TRACE_DIR / f"{exam_id}_{arm}.json").read_text())
    title = (f"{exam_id} — {arm} arm — "
             f"{len(trace['steps'])} steps, "
             f"peak {max(s['prompt_tokens'] for s in trace['steps']):,} tokens")
    html = (TEMPLATE
            .replace("{{TITLE}}", title)
            .replace("{{TRACE}}", json.dumps(trace)))
    out = TRACE_DIR / f"{exam_id}_{arm}.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB) — open it in a browser")


if __name__ == "__main__":
    exam_id = sys.argv[1] if len(sys.argv) > 1 else "exam1"
    arm = sys.argv[2] if len(sys.argv) > 2 else "folded"
    main(exam_id, arm)
