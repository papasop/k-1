"""
codex_connector – ChatGPT Codex Connector for papasop/k-1
==========================================================

Quick start::

    from codex_connector import CodexConnector

    connector = CodexConnector(api_key="sk-...")
    code = connector.generate("a Python function that sorts a list of dicts by key")
    print(code)
"""

from .config import Config
from .core import CodexConnector
from .utils import setup_logging

__version__ = "0.1.0"
__author__ = "papasop"
__all__ = [
    "CodexConnector",
    "Config",
    "setup_logging",
]
