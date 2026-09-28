"""LLM clients. The pipeline only sees the LLMClient protocol: complete(system, user) -> str."""

from __future__ import annotations

import threading
from typing import Any

from rag.core.config import LLMProvider, Settings
from rag.core.interfaces import LLMClient


class OpenAILLMClient:
    """OpenAI chat completions.

    - Retries on rate limits, timeouts and 5xx are handled by the OpenAI SDK (max_retries).
    - temperature=0 makes evaluation runs reproducible. Some models (reasoning models)
      reject a temperature; the first such refusal switches it off for later calls.
    - Thread safe: the SDK client is thread safe and the fallback flag is lock protected.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float | None = 0.0,
        max_retries: int = 3,
        timeout: float = 60.0,
        client: Any = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout = timeout
        self._client = client
        self._lock = threading.Lock()

    @property
    def client(self) -> Any:
        with self._lock:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key, max_retries=self._max_retries, timeout=self._timeout)
            return self._client

    def complete(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "temperature" not in kwargs or not _rejects_temperature(exc):
                raise
            with self._lock:
                self.temperature = None
            kwargs.pop("temperature")
            response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def _rejects_temperature(exc: Exception) -> bool:
    return type(exc).__name__ == "BadRequestError" and "temperature" in str(exc).lower()


def make_llm_client(settings: Settings) -> LLMClient:
    """The configured provider's client, using settings.llm_model."""
    if settings.llm_provider is LLMProvider.OPENAI:
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAILLMClient(settings.llm_model, api_key=api_key)
    raise NotImplementedError(
        f"llm_provider={settings.llm_provider.value!r} is not implemented yet; set LLM_PROVIDER=openai"
    )
