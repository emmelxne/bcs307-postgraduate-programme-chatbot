"""Presentation helpers; programme and chat state remain in app.py."""
import base64
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"


def apply_style():
    st.html(f"<style>{(ASSETS / 'ui.css').read_text(encoding='utf-8')}</style>")


def render_header():
    logos = []
    # Each SVG is a viewport onto the unchanged, local combined screenshot.
    for filename, alt, css_class in [
        ("HelpUniversity.svg", "HELP University — university of achievers", "help-logo"),
        ("FCDT.svg", "Faculty of Computing and Digital Technology", "faculty-logo"),
    ]:
        path = ASSETS / filename
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            logos.append(f'<img class="{css_class}" src="data:image/svg+xml;base64,{encoded}" alt="{alt}">')
    st.html(
        '<header class="brand-header">'
        '<div class="brand-logos" aria-label="HELP University and FCDT">' + "".join(logos) + '</div>'
        '<h1>Postgraduate Programme Assistant</h1>'
        '<p>BCS307 student project · Based on the FCDT postgraduate handbook</p>'
        '</header>'
    )
