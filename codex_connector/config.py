"""
Configuration management for the ChatGPT Codex Connector.

Loads settings from environment variables and optional .env file.
"""

import os
import logging
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; rely on environment variables already set

logger = logging.getLogger(__name__)


class Config:
    """Central configuration object for the Codex connector."""

    # OpenAI settings
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    timeout: int

    # Retry / resilience
    max_retries: int
    retry_delay: float

    # Caching
    cache_enabled: bool
    cache_ttl: int

    # Logging
    log_level: str

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        cache_enabled: Optional[bool] = None,
        cache_ttl: Optional[int] = None,
        log_level: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("CODEX_MODEL", "gpt-4o")
        self.max_tokens = max_tokens or int(os.environ.get("CODEX_MAX_TOKENS", "2048"))
        self.temperature = temperature if temperature is not None else float(
            os.environ.get("CODEX_TEMPERATURE", "0.2")
        )
        self.timeout = timeout or int(os.environ.get("CODEX_TIMEOUT", "60"))
        self.max_retries = max_retries or int(os.environ.get("CODEX_MAX_RETRIES", "3"))
        self.retry_delay = retry_delay if retry_delay is not None else float(
            os.environ.get("CODEX_RETRY_DELAY", "1.0")
        )
        self.cache_enabled = cache_enabled if cache_enabled is not None else (
            os.environ.get("CODEX_CACHE_ENABLED", "false").lower() == "true"
        )
        self.cache_ttl = cache_ttl or int(os.environ.get("CODEX_CACHE_TTL", "3600"))
        self.log_level = log_level or os.environ.get("CODEX_LOG_LEVEL", "INFO")

        if not self.api_key:
            logger.warning(
                "OPENAI_API_KEY is not set. "
                "Set the environment variable or pass api_key= to Config()."
            )

    def validate(self) -> None:
        """Raise ValueError if required settings are missing."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key= to Config()."
            )
