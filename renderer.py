"""
FirmBench — artifact renderers.

Renders agent-produced ad copy and feature specs into visual HTML files.
Uses Python string.Template (stdlib) — no Jinja2 dependency.
All output is self-contained HTML with inline CSS.
"""

import os
from pathlib import Path
from string import Template


# ----------------------------- ad card template ---------------------

_AD_CARD_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ad: $headline</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f1117;display:flex;justify-content:center;align-items:center;min-height:100vh;padding:20px}
.card{width:420px;border-radius:16px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.4);
  background:#1a1d27;color:#e0e0e0}
.card-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:32px 28px 24px;
  position:relative}
.card-header::after{content:'AD';position:absolute;top:12px;right:16px;font-size:11px;
  font-weight:700;letter-spacing:1px;color:rgba(255,255,255,0.5);
  border:1px solid rgba(255,255,255,0.3);padding:2px 8px;border-radius:4px}
h1{font-size:22px;font-weight:700;color:#fff;line-height:1.3;margin-bottom:8px}
.subtitle{font-size:13px;color:rgba(255,255,255,0.7)}
.card-body{padding:24px 28px}
.body-text{font-size:15px;line-height:1.6;color:#b0b0b0;margin-bottom:20px}
.cta{display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;
  font-size:14px;font-weight:600;padding:10px 24px;border-radius:8px;text-decoration:none;
  letter-spacing:0.5px}
.tags{padding:16px 28px;border-top:1px solid #2a2d3a;display:flex;gap:8px;flex-wrap:wrap}
.tag{font-size:11px;color:#667eea;background:rgba(102,126,234,0.12);padding:4px 10px;
  border-radius:12px;font-weight:500}
.meta{padding:12px 28px;border-top:1px solid #2a2d3a;font-size:11px;color:#555;
  display:flex;justify-content:space-between}
</style></head>
<body>
<div class="card">
  <div class="card-header">
    <h1>$headline</h1>
    <div class="subtitle">Sponsored · FirmBench</div>
  </div>
  <div class="card-body">
    <p class="body-text">$body</p>
    <a class="cta" href="#">$cta</a>
  </div>
  <div class="tags">$pain_tags</div>
  <div class="meta">
    <span>Craft: $craft</span>
    <span>Spend: $spend</span>
  </div>
</div>
</body></html>""")


# ----------------------------- feature page template ----------------

_FEATURE_PAGE_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>$feature_name</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f1117;color:#e0e0e0;min-height:100vh}
.hero{background:linear-gradient(135deg,#11998e 0%,#38ef7d 100%);padding:60px 40px;
  text-align:center}
.hero h1{font-size:32px;font-weight:800;color:#fff;margin-bottom:12px}
.hero p{font-size:16px;color:rgba(255,255,255,0.85);max-width:500px;margin:0 auto}
.content{max-width:600px;margin:0 auto;padding:40px 24px}
.description{font-size:16px;line-height:1.7;color:#b0b0b0;margin-bottom:32px}
h2{font-size:18px;font-weight:600;color:#fff;margin-bottom:16px}
.benefits{list-style:none;margin-bottom:32px}
.benefits li{padding:10px 0;border-bottom:1px solid #1a1d27;font-size:15px;color:#b0b0b0;
  padding-left:24px;position:relative}
.benefits li::before{content:'✓';position:absolute;left:0;color:#38ef7d;font-weight:700}
.cta-section{text-align:center;padding:32px 0}
.cta{display:inline-block;background:linear-gradient(135deg,#11998e,#38ef7d);color:#fff;
  font-size:16px;font-weight:600;padding:14px 32px;border-radius:8px;text-decoration:none}
.meta{text-align:center;padding:16px;font-size:12px;color:#444;border-top:1px solid #1a1d27}
</style></head>
<body>
<div class="hero">
  <h1>$feature_name</h1>
  <p>$tagline</p>
</div>
<div class="content">
  <p class="description">$description</p>
  <h2>Key Benefits</h2>
  <ul class="benefits">$benefits_html</ul>
  <div class="cta-section">
    <a class="cta" href="#">Get Started</a>
  </div>
</div>
<div class="meta">
  <span>Quality: $quality</span> · <span>Feature #$feature_id</span>
</div>
</body></html>""")


# ----------------------------- render functions ---------------------

def render_ad_card(headline: str, body: str, cta: str,
                   target_pains: set, pain_names: list,
                   craft: float = 1.0, spend: float = 0.0,
                   output_path: str = None) -> str:
    """Render an ad card as self-contained HTML. Returns the HTML string.
    If output_path is given, also writes to that file."""
    pain_tags = "".join(
        f'<span class="tag">{pain_names[p] if p < len(pain_names) else f"pain-{p}"}</span>'
        for p in sorted(target_pains)
    )
    html = _AD_CARD_TEMPLATE.substitute(
        headline=_esc(headline or "Untitled Ad"),
        body=_esc(body or "No description provided."),
        cta=_esc(cta or "Learn More"),
        pain_tags=pain_tags,
        craft=f"{craft:.2f}",
        spend=f"${spend:.0f}",
    )
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
    return html


def render_feature_page(feature_name: str, tagline: str, description: str,
                        benefits: list, quality: float = 1.0,
                        feature_id: int = 0,
                        output_path: str = None) -> str:
    """Render a feature landing page as self-contained HTML. Returns the HTML string.
    If output_path is given, also writes to that file."""
    benefits = benefits or ["Improved productivity"]
    benefits_html = "".join(f"<li>{_esc(b)}</li>" for b in benefits)
    html = _FEATURE_PAGE_TEMPLATE.substitute(
        feature_name=_esc(feature_name or "New Feature"),
        tagline=_esc(tagline or "Built for you."),
        description=_esc(description or "This feature helps you get more done."),
        benefits_html=benefits_html,
        quality=f"{quality:.2f}",
        feature_id=feature_id,
    )
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
    return html


def _esc(s: str) -> str:
    """Minimal HTML escaping for template substitution."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
