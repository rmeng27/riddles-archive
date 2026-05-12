"""CSS and visual helpers for the Streamlit app."""

from __future__ import annotations

from .characters import CHARACTERS


def global_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;700&family=Libre+Baskerville:wght@400;700&family=Luckiest+Guy&family=Quicksand:wght@400;600&family=Special+Elite&display=swap');

:root {
  --parchment: #f4ead2;
  --ink: #1f1712;
  --gold: #d7b377;
  --deep-green: #0d1f1a;
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(215,179,119,0.20), transparent 30%),
    radial-gradient(circle at bottom right, rgba(75,12,20,0.30), transparent 35%),
    linear-gradient(135deg, #050505 0%, #0d1f1a 45%, #1a090c 100%);
  color: var(--parchment);
}

.block-container { padding-top: 2rem; max-width: 1180px; }

h1, h2, h3 { font-family: 'Cormorant Garamond', Georgia, serif; letter-spacing: .03em; }

.archive-title {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: 3.1rem;
  line-height: 1;
  color: #f3dfb1;
  text-shadow: 0 0 18px rgba(215,179,119,.35);
  margin-bottom: .1rem;
}

.archive-subtitle {
  color: #cfbf9c;
  font-size: 1.02rem;
  margin-bottom: 1.4rem;
}

.diary-card {
  background: rgba(244,234,210,0.92);
  color: #22170f;
  border: 1px solid rgba(215,179,119,.65);
  box-shadow: 0 12px 40px rgba(0,0,0,.38);
  border-radius: 18px;
  padding: 1.15rem 1.25rem;
  margin: .85rem 0;
}

.user-card {
  background: rgba(28, 31, 30, 0.86);
  color: #f5ead3;
  border: 1px solid rgba(215,179,119,.35);
  border-radius: 18px;
  padding: 1rem 1.15rem;
  margin: .85rem 0;
}

.metric-card {
  background: rgba(0,0,0,.28);
  border: 1px solid rgba(215,179,119,.25);
  border-radius: 16px;
  padding: 1rem;
}

.small-muted { color: #d0c2a5; font-size: .88rem; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(13,31,26,.97), rgba(28,9,12,.97));
  border-right: 1px solid rgba(215,179,119,.25);
}

.stButton>button {
  border-radius: 999px;
  border: 1px solid rgba(215,179,119,.6);
  background: rgba(215,179,119,.12);
  color: #f3dfb1;
}

.stTextInput>div>div>input, .stTextArea textarea {
  background: rgba(244,234,210,.95);
  color: #1f1712;
  border-radius: 14px;
}

hr { border-color: rgba(215,179,119,.25); }

/* harmless wand-ish cursor fallback; browsers may ignore SVG cursor in some deployments */
* { cursor: default; }
</style>
"""


def character_response_card(character: str, content: str) -> str:
    cfg = CHARACTERS[character]
    safe_content = content.replace("\n", "<br>")
    return f"""
<div class="diary-card" style="border-left: 6px solid {cfg['accent']}; font-family: {cfg['font']};">
  <div style="font-size: .95rem; letter-spacing:.04em; color:{cfg['accent']}; font-weight:700; margin-bottom:.35rem;">
    {cfg['emoji']} {character}
  </div>
  <div style="font-size:1.08rem; line-height:1.62;">{safe_content}</div>
</div>
"""


def user_card(content: str) -> str:
    safe_content = content.replace("\n", "<br>")
    return f"""
<div class="user-card">
  <div style="font-size:.82rem; color:#d7b377; font-weight:700; margin-bottom:.3rem;">You</div>
  <div style="line-height:1.55;">{safe_content}</div>
</div>
"""
