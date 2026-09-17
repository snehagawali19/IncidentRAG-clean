"""Embed the canvas node-network engine inside Streamlit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_ENGINE = Path(__file__).with_name("network_engine.html").read_text(encoding="utf-8")


def render_network(payload: dict[str, Any], *, height: int = 720) -> None:
    encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    html = _ENGINE.replace("__PAYLOAD__", encoded)
    components.html(html, height=height, scrolling=False)
