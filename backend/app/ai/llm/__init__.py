"""
EchoVoice Conversational LLM Core Package (backend/app/ai/llm/__init__.py)
------------------------------------------------------------------------
Exports base LLM interfaces, OpenAI/Ollama integrations, fallback engines,
conversational personas, and sliding window memory.
"""

from app.ai.llm.base import LLMEngine, LLMMessage, LLMResponse
from app.ai.llm.fallback_engine import LocalFallbackLLMEngine
from app.ai.llm.openai_compatible_engine import OpenAICompatibleLLMEngine
from app.ai.llm.prompts import build_system_prompt, PERSONA_PROMPTS
from app.ai.llm.memory import conversation_memory, ConversationMemory
from app.ai.llm.registry import llm_registry, LLMRegistry, get_llm_engine

__all__ = [
    "LLMEngine",
    "LLMMessage",
    "LLMResponse",
    "LocalFallbackLLMEngine",
    "OpenAICompatibleLLMEngine",
    "build_system_prompt",
    "PERSONA_PROMPTS",
    "conversation_memory",
    "ConversationMemory",
    "llm_registry",
    "LLMRegistry",
    "get_llm_engine",
]
