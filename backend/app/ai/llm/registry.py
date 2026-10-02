"""
LLM Engine Registry & Resilient Fallback Router (backend/app/ai/llm/registry.py)
---------------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Strategy & Registry Pattern: Decouples conversational services from specific LLM providers.
- Resilient Fallback Proxy: Transparently intercepts network or connection errors
  from local/remote LLMs (e.g. unbooted Ollama server) and falls back to LocalFallbackLLMEngine
  without bubbling 500 errors to the client.
"""

from typing import AsyncGenerator, Dict, List, Optional, Type, Any
from app.core.config import settings
from app.core.logging import logger
from app.ai.llm.base import LLMEngine, LLMMessage, LLMResponse
from app.ai.llm.fallback_engine import LocalFallbackLLMEngine
from app.ai.llm.openai_compatible_engine import OpenAICompatibleLLMEngine


class ResilientLLMProxy(LLMEngine):
    """
    Wraps a primary LLMEngine with automatic fallback to LocalFallbackLLMEngine
    upon connection or inference failures.
    """

    def __init__(self, primary: LLMEngine, fallback: LLMEngine):
        self.primary = primary
        self.fallback = fallback
        self.engine_name = f"{primary.engine_name}_resilient"
        self.model_name = primary.model_name

    def is_available(self) -> bool:
        return True

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any
    ) -> LLMResponse:
        try:
            return await self.primary.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as exc:
            logger.warning(
                f"Primary LLM engine '{self.primary.engine_name}' error: {exc}. "
                "Failing over to LocalFallbackLLMEngine."
            )
            return await self.fallback.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

    async def stream_generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        try:
            # Attempt to stream from primary
            async for token in self.primary.stream_generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                yield token
        except Exception as exc:
            logger.warning(
                f"Primary LLM stream '{self.primary.engine_name}' error: {exc}. "
                "Failing over to LocalFallbackLLMEngine stream."
            )
            async for token in self.fallback.stream_generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                yield token


class LLMRegistry:
    """Registry maintaining active Large Language Model inference engines."""

    def __init__(self):
        self._engines: Dict[str, LLMEngine] = {}
        self._fallback = LocalFallbackLLMEngine()

        # Register standard engines
        self._engines["fallback"] = self._fallback
        self._engines["ollama"] = OpenAICompatibleLLMEngine(
            base_url=settings.OLLAMA_BASE_URL,
            model_name=settings.LLM_MODEL
        )
        self._engines["openai_compatible"] = OpenAICompatibleLLMEngine(
            base_url=settings.LLM_BASE_URL or settings.OLLAMA_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model_name=settings.LLM_MODEL
        )

    def register_engine(self, name: str, engine: LLMEngine) -> None:
        self._engines[name.lower()] = engine

    def get_engine(self, name: Optional[str] = None, enable_resilience: bool = True) -> LLMEngine:
        """
        Resolves the requested engine or defaults to configured engine.
        Wraps non-fallback engines in ResilientLLMProxy to guarantee 100% uptime.
        """
        engine_key = (name or settings.DEFAULT_LLM_ENGINE).lower()

        primary = self._engines.get(engine_key, self._fallback)

        if engine_key == "fallback" or not enable_resilience:
            return primary

        return ResilientLLMProxy(primary=primary, fallback=self._fallback)

    def available_engines(self) -> List[str]:
        return list(self._engines.keys())


# Global singleton registry
llm_registry = LLMRegistry()


def get_llm_engine(name: Optional[str] = None, enable_resilience: bool = True) -> LLMEngine:
    """Convenience getter for the active LLM engine."""
    return llm_registry.get_engine(name=name, enable_resilience=enable_resilience)
