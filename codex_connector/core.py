"""
Core Codex connector class.

Provides high-level methods for the most common code-related tasks:

- generate  – produce code from a natural-language description
- complete  – fill in / extend an incomplete code snippet
- explain   – describe what a piece of code does
- fix_bugs  – identify and repair defects
- optimize  – improve performance or readability
"""

import logging
from typing import Any, Dict, Optional

from .api_client import APIClient
from .config import Config
from .utils import strip_code_fences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_BASE = (
    "You are an expert software engineer and coding assistant. "
    "Respond with clean, well-commented code unless asked otherwise."
)

_PROMPTS: Dict[str, str] = {
    "generate": (
        "Generate complete, working {language} code for the following task.\n"
        "Return only the code – no surrounding explanations unless comments "
        "are needed for clarity.\n\n"
        "Task description:\n{description}"
    ),
    "complete": (
        "Complete the following {language} code snippet. "
        "Return only the completed code, preserving existing style.\n\n"
        "Code to complete:\n{code}"
    ),
    "explain": (
        "Explain what the following {language} code does. "
        "Be concise but thorough. Use plain language a junior developer "
        "would understand.\n\n"
        "Code:\n{code}"
    ),
    "fix_bugs": (
        "Identify and fix all bugs in the following {language} code. "
        "Return the corrected code followed by a brief explanation of "
        "each fix.\n\n"
        "Buggy code:\n{code}"
    ),
    "optimize": (
        "Optimize the following {language} code for {goal}. "
        "Return the improved code followed by a brief explanation of "
        "the changes.\n\n"
        "Original code:\n{code}"
    ),
}


class CodexConnector:
    """
    High-level interface to OpenAI's chat-completions API for code tasks.

    Parameters
    ----------
    config : Config, optional
        Supply a pre-built :class:`~codex_connector.config.Config` object.
        If omitted, one is constructed from environment variables.
    api_key : str, optional
        Convenience shortcut – overrides the ``OPENAI_API_KEY`` env var.
    **config_kwargs
        Any keyword argument accepted by :class:`~codex_connector.config.Config`.

    Examples
    --------
    >>> connector = CodexConnector(api_key="sk-...")
    >>> code = connector.generate("a Python function that reverses a string")
    >>> print(code)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        api_key: Optional[str] = None,
        **config_kwargs: Any,
    ) -> None:
        if config is None:
            if api_key:
                config_kwargs["api_key"] = api_key
            config = Config(**config_kwargs)

        config.validate()
        self.config = config
        self._client = APIClient(config)
        logger.info(
            "CodexConnector ready (model=%s, cache=%s)",
            config.model,
            config.cache_enabled,
        )

    # ------------------------------------------------------------------
    # Public synchronous methods
    # ------------------------------------------------------------------

    def generate(
        self,
        description: str,
        language: str = "Python",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate code from a natural-language *description*.

        Parameters
        ----------
        description : str
            What the code should do.
        language : str
            Target programming language (default: ``"Python"``).
        extra_params : dict, optional
            Additional parameters forwarded to the OpenAI API call.

        Returns
        -------
        str
            Raw code string (code fences stripped).
        """
        prompt = _PROMPTS["generate"].format(language=language, description=description)
        logger.debug("generate() prompt length=%d chars", len(prompt))
        raw = self._client.complete(_SYSTEM_BASE, prompt, extra_params)
        return strip_code_fences(raw)

    def complete(
        self,
        code: str,
        language: str = "Python",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Complete / extend an incomplete *code* snippet.

        Parameters
        ----------
        code : str
            Partial code to be completed.
        language : str
            Programming language of the snippet (default: ``"Python"``).
        extra_params : dict, optional
            Additional parameters forwarded to the OpenAI API call.

        Returns
        -------
        str
            Completed code string (code fences stripped).
        """
        prompt = _PROMPTS["complete"].format(language=language, code=code)
        logger.debug("complete() prompt length=%d chars", len(prompt))
        raw = self._client.complete(_SYSTEM_BASE, prompt, extra_params)
        return strip_code_fences(raw)

    def explain(
        self,
        code: str,
        language: str = "Python",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Explain what *code* does in plain English.

        Parameters
        ----------
        code : str
            Source code to explain.
        language : str
            Programming language of the code (default: ``"Python"``).
        extra_params : dict, optional
            Additional parameters forwarded to the OpenAI API call.

        Returns
        -------
        str
            Plain-text explanation.
        """
        prompt = _PROMPTS["explain"].format(language=language, code=code)
        logger.debug("explain() prompt length=%d chars", len(prompt))
        return self._client.complete(_SYSTEM_BASE, prompt, extra_params)

    def fix_bugs(
        self,
        code: str,
        language: str = "Python",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Identify and fix bugs in *code*.

        Parameters
        ----------
        code : str
            Potentially buggy source code.
        language : str
            Programming language (default: ``"Python"``).
        extra_params : dict, optional
            Additional parameters forwarded to the OpenAI API call.

        Returns
        -------
        str
            Fixed code followed by a summary of changes.
        """
        prompt = _PROMPTS["fix_bugs"].format(language=language, code=code)
        logger.debug("fix_bugs() prompt length=%d chars", len(prompt))
        return self._client.complete(_SYSTEM_BASE, prompt, extra_params)

    def optimize(
        self,
        code: str,
        language: str = "Python",
        goal: str = "performance and readability",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Optimize *code* for a given *goal*.

        Parameters
        ----------
        code : str
            Source code to optimize.
        language : str
            Programming language (default: ``"Python"``).
        goal : str
            Optimization objective, e.g. ``"performance"``, ``"readability"``,
            or ``"memory usage"`` (default: ``"performance and readability"``).
        extra_params : dict, optional
            Additional parameters forwarded to the OpenAI API call.

        Returns
        -------
        str
            Optimized code followed by a summary of improvements.
        """
        prompt = _PROMPTS["optimize"].format(
            language=language, code=code, goal=goal
        )
        logger.debug("optimize() prompt length=%d chars", len(prompt))
        return self._client.complete(_SYSTEM_BASE, prompt, extra_params)

    # ------------------------------------------------------------------
    # Public asynchronous methods
    # ------------------------------------------------------------------

    async def generate_async(
        self,
        description: str,
        language: str = "Python",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Async version of :meth:`generate`."""
        prompt = _PROMPTS["generate"].format(language=language, description=description)
        raw = await self._client.complete_async(_SYSTEM_BASE, prompt, extra_params)
        return strip_code_fences(raw)

    async def complete_async(
        self,
        code: str,
        language: str = "Python",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Async version of :meth:`complete`."""
        prompt = _PROMPTS["complete"].format(language=language, code=code)
        raw = await self._client.complete_async(_SYSTEM_BASE, prompt, extra_params)
        return strip_code_fences(raw)

    async def explain_async(
        self,
        code: str,
        language: str = "Python",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Async version of :meth:`explain`."""
        prompt = _PROMPTS["explain"].format(language=language, code=code)
        return await self._client.complete_async(_SYSTEM_BASE, prompt, extra_params)

    async def fix_bugs_async(
        self,
        code: str,
        language: str = "Python",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Async version of :meth:`fix_bugs`."""
        prompt = _PROMPTS["fix_bugs"].format(language=language, code=code)
        return await self._client.complete_async(_SYSTEM_BASE, prompt, extra_params)

    async def optimize_async(
        self,
        code: str,
        language: str = "Python",
        goal: str = "performance and readability",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Async version of :meth:`optimize`."""
        prompt = _PROMPTS["optimize"].format(
            language=language, code=code, goal=goal
        )
        return await self._client.complete_async(_SYSTEM_BASE, prompt, extra_params)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the in-memory response cache."""
        self._client.clear_cache()
