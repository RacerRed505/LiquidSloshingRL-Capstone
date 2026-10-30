"""Insert (or refresh) two ADDITIVE sections in every copy of the report.

1. "At a glance": a plain-language summary of the key results, placed right
   after the page header. It adds a summary; nothing already in the report
   is removed or replaced.
2. "05 Answers to the second review": documentation/review2_responses.md,
   converted to HTML, placed after section 04.

Each block sits between HTML comment markers, so re-running the script
replaces only its own earlier output. Run from the repo root:

    python -m scripts.add_review2_sections
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
REPORTS = [
    ROOT / "sloshing_report.html",
    ROOT / "sloshing_report_dx005.html",
    ROOT / "sloshing_report_dx003.html",
    ROOT / "for_advisor" / "2_interactive_report.html",
]
RESPONSES_MD = ROOT / "documentation" / "review2_responses.md"

GLANCE_START, GLANCE_END = "<!-- glance:start -->", "<!-- glance:end -->"
REVIEW_START, REVIEW_END = "<!-- review2:start -->", "<!-- review2:end -->"

GLANCE_HTML = """<!-- glance:start -->
  <style>
    #sec-glance .glance-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-bottom: 8px; }
    #sec-glance .glance-card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; box-shadow: var(--shadow); }
    #sec-glance .glance-card .big { font-family: "Fraunces", Georgia, serif; font-size: 26px; color: var(--ink); line-height: 1.1; }
    #sec-glance .glance-card .what { font-size: 13.5px; color: var(--ink-secondary); margin-top: 6px; }
  </style>
  <section id="sec-glance">
    <div class="section-head">
      <span class="num">&#9670;</span>
      <h2>At a glance: the results in plain words</h2>
    </div>
    <p class="dek">
      A water tank sits on a single pivot and a motor tilts it. A learned controller (reinforcement
      learning) decides, 50 times a second, how hard to twist the tank so the water stops sloshing.
      It was trained on a fast simplified water model, then tested on realistic particle-simulated
      water. This is a summary only: every detailed figure and number below is unchanged.
    </p>
    <div class="glance-grid">
      <div class="glance-card"><div class="big">1.2&nbsp;s</div><div class="what">Time for the learned controller to calm the sloshing on the simplified model. The formula-based LQR controller takes 3.4&ndash;3.7&nbsp;s; a standard PID controller does not react at all, because it cannot see the water.</div></div>
      <div class="glance-card"><div class="big">15&times; less</div><div class="what">Sloshing energy left by the learned controller on realistic water, compared with PID (0.0010&nbsp;J vs 0.0157&nbsp;J).</div></div>
      <div class="glance-card"><div class="big">0.7&nbsp;s</div><div class="what">How long LQR lasted on realistic water before losing control. A controller built on a simplified model can fail on the real physics.</div></div>
      <div class="glance-card"><div class="big">~5&nbsp;s</div><div class="what">How long the learned controller stays in control on realistic water. After that it loses control too; this is reported as a known limit, not hidden.</div></div>
      <div class="glance-card"><div class="big">3.8&times;</div><div class="what">Improvement from a short fine-tune on realistic tank physics. Training longer made it worse, so stopping early matters.</div></div>
      <div class="glance-card"><div class="big">within 6%</div><div class="what">How closely the simple pendulum model predicts the realistic water&rsquo;s sloshing frequency (1.34 vs 1.42&nbsp;Hz). The realistic simulation itself converges: the error halves with each finer grid.</div></div>
    </div>
  </section>
  <!-- glance:end -->"""


def review_section_html() -> str:
    md_text = RESPONSES_MD.read_text(encoding="utf-8")
    # Drop the document's own H1 and intro; the section header replaces them.
    body_md = md_text.split("\n---\n", 1)[1]
    html = markdown.markdown(body_md, extensions=["tables"])
    html = html.replace("<table>", '<table class="results-table">')
    html = re.sub(r"<code>(.*?)</code>", r'<span class="mono">\1</span>', html)
    # Demote headings one level so they sit under the section's own <h2>.
    html = html.replace("<h3>", '<h4 class="r2-h">').replace("</h3>", "</h4>")
    html = html.replace("<h2>", '<h3 class="r2-h">').replace("</h2>", "</h3>")
    return f"""<!-- review2:start -->
  <style>
    #sec-review2 .r2-body {{ font-size: 15px; color: var(--ink-secondary); overflow-wrap: anywhere; }}
    #sec-review2 .r2-body td, #sec-review2 .r2-body th {{ overflow-wrap: normal; }}
    #sec-review2 .r2-body h3.r2-h {{ font-size: 22px; color: var(--ink); margin: 34px 0 10px; }}
    #sec-review2 .r2-body h4.r2-h {{ font-size: 17px; color: var(--ink); margin: 22px 0 8px; font-family: "Fraunces", Georgia, serif; }}
    #sec-review2 .r2-body strong {{ color: var(--ink); }}
    #sec-review2 .r2-body table {{ width: 100%; border-collapse: collapse; margin: 12px 0 18px; font-size: 13.5px; }}
    #sec-review2 .r2-body hr {{ border: none; border-top: 1px solid var(--line); margin: 28px 0; }}
    #sec-review2 .r2-body .table-wrap {{ overflow-x: auto; }}
  </style>
  <section id="sec-review2">
    <div class="section-head">
      <span class="num">05</span>
      <h2>Answers to the second review</h2>
    </div>
    <p class="dek">
      Written answers to the instructor&rsquo;s second-review points that are not covered by a figure:
      why the torque is bounded, the problem statement, applications, significance, how the tank was
      sized, why water, and the three controllers explained for a non-specialist. The free-body diagram,
      block diagram and PPO flowchart are Figs.&nbsp;0b&ndash;0d in section 00; the hand calculation vs
      simulation comparison is in the Excel workbook.
    </p>
    <div class="r2-body">
{html.replace('<table class="results-table">', '<div class="table-wrap"><table class="results-table">').replace('</table>', '</table></div>')}
    </div>
  </section>
  <!-- review2:end -->"""


def _replace_or_insert(page: str, start: str, end: str, block: str, anchor: str, before: bool) -> str:
    if start in page:
        pattern = re.escape(start) + r".*?" + re.escape(end)
        return re.sub(pattern, lambda _: block, page, count=1, flags=re.S)
    idx = page.index(anchor)
    if not before:
        idx += len(anchor)
    return page[:idx] + "\n\n  " + block + "\n" + page[idx:]


def main() -> None:
    review_html = review_section_html()
    for path in REPORTS:
        page = path.read_text(encoding="utf-8")
        page = _replace_or_insert(page, GLANCE_START, GLANCE_END, GLANCE_HTML,
                                  '<section id="sec-overview">', before=True)
        # Section 04 is the last section before the footer.
        page = _replace_or_insert(page, REVIEW_START, REVIEW_END, review_html,
                                  "  <footer>", before=True)
        path.write_text(page, encoding="utf-8")
        print(f"updated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
