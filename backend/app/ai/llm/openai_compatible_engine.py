"""
OpenAI-Compatible LLM Engine (backend/app/ai/llm/openai_compatible_engine.py)
----------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Universal Inference Gateway: Connects to any OpenAI-API compliant server
  including local Ollama (`localhost:11434/v1`), vLLM, LMStudio, Groq, and DeepInfra.
- Full-Duplex SSE Streaming: Consumes server-sent completion events and yields token deltas
  in real time for seamless TTS ingestion.
- Connection Timeout & Fault Tolerance: Bounded HTTP client timeouts prevent deadlocks
  if the local daemon is unbooted.
"""

import time
from typing import AsyncGenerator, List, Optional, Any
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import logger, timed_step
from app.ai.llm.base import LLMEngine, LLMMessage, LLMResponse


class OpenAICompatibleLLMEngine(LLMEngine):
    """
    Inference backend targeting OpenAI-compatible endpoints (Ollama, vLLM, Groq).
    """

    engine_name: str = "openai_compatible"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_sec: float = 15.0
    ):
        self.base_url = base_url or settings.LLM_BASE_URL or settings.OLLAMA_BASE_URL
        self.api_key = api_key or settings.LLM_API_KEY or "ollama"
        self.model_name = model_name or settings.LLM_MODEL
        self.timeout_sec = timeout_sec

        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout_sec,
                max_retries=0
            )
        return self._client

    def is_available(self) -> bool:
        # Client initialized
        return True

    @timed_step("OpenAI-Compatible LLM Generation")
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any
    ) -> LLMResponse:
        client = self._get_client()
        start_t = time.perf_counter()

        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        resp = await client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )

        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        choice = resp.choices[0]
        usage = resp.usage

        return LLMResponse(
            content=choice.message.content or "",
            model_name=self.model_name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason or "stop"
        )

    async def stream_generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        stream = await client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
