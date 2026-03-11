"""
OpenAI API client for the ChatGPT Codex Connector.

Handles authentication, retries, optional response caching,
and both synchronous and async invocation.
"""

import logging
import time
from typing import Any, Dict, Optional

from .config import Config
from .utils import make_cache_key

logger = logging.getLogger(__name__)


class APIClient:
    """
    Thin wrapper around the OpenAI chat-completions endpoint.

    Parameters
    ----------
    config : Config
        Connector configuration (API key, model, timeouts, …).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._cache: Dict[str, str] = {}
        self._client: Optional[Any] = None
        self._async_client: Optional[Any] = None
        self._init_client()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Lazily initialise the openai SDK client."""
        try:
            import openai  # noqa: PLC0415  (local import intentional)
            self._client = openai.OpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
            self._async_client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
            logger.debug("OpenAI client initialised (model=%s)", self.config.model)
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required. Install it with: pip install openai"
            ) from exc

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _get_params(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if extra:
            params.update(extra)
        return params

    # ------------------------------------------------------------------
    # Public synchronous interface
    # ------------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a chat-completion request and return the response text.

        Retries up to ``config.max_retries`` times on transient errors.
        Results are optionally cached when ``config.cache_enabled`` is True.
        """
        params = self._get_params(extra_params)
        cache_key = make_cache_key(system_prompt + user_prompt, params)

        if self.config.cache_enabled and cache_key in self._cache:
            logger.debug("Cache hit for key %s", cache_key[:12])
            return self._cache[cache_key]

        messages = self._build_messages(system_prompt, user_prompt)
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    messages=messages, **params
                )
                result = response.choices[0].message.content or ""
                if self.config.cache_enabled:
                    self._cache[cache_key] = result
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "API request failed (attempt %d/%d): %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * attempt)

        raise RuntimeError(
            f"API request failed after {self.config.max_retries} attempts"
        ) from last_error

    # ------------------------------------------------------------------
    # Public asynchronous interface
    # ------------------------------------------------------------------

    async def complete_async(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Async variant of :meth:`complete`.

        Useful when integrating the connector into async frameworks
        (e.g. FastAPI, asyncio-based pipelines).
        """
        import asyncio  # noqa: PLC0415

        params = self._get_params(extra_params)
        cache_key = make_cache_key(system_prompt + user_prompt, params)

        if self.config.cache_enabled and cache_key in self._cache:
            logger.debug("Cache hit (async) for key %s", cache_key[:12])
            return self._cache[cache_key]

        messages = self._build_messages(system_prompt, user_prompt)
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self._async_client.chat.completions.create(
                    messages=messages, **params
                )
                result = response.choices[0].message.content or ""
                if self.config.cache_enabled:
                    self._cache[cache_key] = result
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Async API request failed (attempt %d/%d): %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * attempt)

        raise RuntimeError(
            f"Async API request failed after {self.config.max_retries} attempts"
        ) from last_error

    def clear_cache(self) -> None:
        """Evict all cached responses."""
        self._cache.clear()
        logger.debug("Response cache cleared")
