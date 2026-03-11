"""
Utility functions for the ChatGPT Codex Connector.
"""

import logging
import hashlib
import json
import re
from typing import Any, Dict, Optional


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger for the codex_connector package."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def make_cache_key(prompt: str, params: Dict[str, Any]) -> str:
    """Return a stable hash string suitable for use as a cache key."""
    payload = json.dumps({"prompt": prompt, **params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_code_blocks(text: str) -> list:
    """
    Extract fenced code blocks (``` … ```) from *text*.

    Returns a list of (language, code) tuples.  The language string may be
    empty when no language tag is present.
    """
    pattern = r"```(\w*)\n?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang, code.strip()) for lang, code in matches]


def strip_code_fences(text: str) -> str:
    """
    Return *text* with any surrounding code fences removed.

    Useful when the model wraps its output in a single fenced block and you
    only need the raw code.
    """
    blocks = extract_code_blocks(text)
    if blocks:
        return blocks[0][1]
    return text.strip()


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate *text* for display, adding an ellipsis when needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def format_code_response(response_text: str, language: str = "") -> str:
    """
    Wrap *response_text* in a Markdown fenced code block.

    If the text already contains fences, it is returned unchanged.
    """
    if "```" in response_text:
        return response_text
    lang_tag = language or ""
    return f"```{lang_tag}\n{response_text.strip()}\n```"


def estimate_token_count(text: str) -> int:
    """
    Rough token-count estimate (≈ 4 characters per token).

    This is not exact – use only for quick sanity checks.
    """
    return max(1, len(text) // 4)
