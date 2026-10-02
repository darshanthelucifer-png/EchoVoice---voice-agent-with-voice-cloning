"""
LLM Abstract Base Class & Data Models (backend/app/ai/llm/base.py)
------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Abstract Base Class (`abc.ABC`): Formal contract for pluggable conversational LLM engines
  (Ollama, Groq, vLLM, Hugging Face, and local fallback).
- Asynchronous Generators (`AsyncGenerator`): Streaming token-by-token emission
  for low-latency voice-to-voice pipelining.
- Dataclasses: Strictly typed message turns and generation telemetry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Any


@dataclass(frozen=True)
class LLMMessage:
    """Represents a single conversational turn in the chat context."""
    role: str       # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Standardized response payload returned by all LLM engines."""
    content: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMEngine(ABC):
    """Abstract Base Class for Large Language Model generation backends."""

    engine_name: str = "base"
    model_name: str = "base-model"

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Executes non-streaming generation over conversational message turns.
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """
        Streams generated tokens asynchronously as they are produced.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the inference backend is initialized and reachable."""
        pass
