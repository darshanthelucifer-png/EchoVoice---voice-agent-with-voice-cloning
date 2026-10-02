"""
Local Fallback LLM Engine (backend/app/ai/llm/fallback_engine.py)
-----------------------------------------------------------------
PYTHON & CONVERSATIONAL AI CONCEPTS DEMONSTRATED:
- Graceful Degradation / Standalone Generator: Provides deterministic, context-aware responses
  when external local engines (Ollama) or remote cloud endpoints (Groq/vLLM) are unavailable.
- RAG Context Extraction: Intelligently inspects the system prompt or history for retrieved
  knowledge passages (`[Source X: ...]`) and synthesizes spoken-style factual answers.
- Async Token Streaming Simulation: Yields realistic word chunks over `AsyncGenerator`
  to test real-time audio playback pipelines without external model overhead.
"""

import asyncio
import re
import time
from typing import AsyncGenerator, List, Optional, Any

from app.core.logging import logger, timed_step
from app.ai.llm.base import LLMEngine, LLMMessage, LLMResponse


class LocalFallbackLLMEngine(LLMEngine):
    """
    Zero-dependency, offline conversational engine with RAG knowledge extraction
    and spoken-style response generation.
    """

    engine_name: str = "fallback"
    model_name: str = "echovoice-local-fallback"

    def is_available(self) -> bool:
        return True

    def _extract_rag_context(self, messages: List[LLMMessage]) -> Optional[str]:
        """Extracts text passages if RAG context was injected into the prompt."""
        for msg in messages:
            if "[Source " in msg.content or "Context:" in msg.content:
                return msg.content
        return None

    def _synthesize_response(self, messages: List[LLMMessage]) -> str:
        """Determines the most contextually relevant conversational reply."""
        user_query = ""
        for m in reversed(messages):
            if m.role == "user":
                user_query = m.content.strip().lower()
                break

        rag_context = self._extract_rag_context(messages)

        # 1. Answer using retrieved RAG Knowledge if present
        if rag_context:
            # Extract plain text from [Source ...] blocks
            cleaned_context = re.sub(r"\[Source \d+:[^\]]+\]", "", rag_context).strip()
            sentences = [s.strip() for s in re.split(r"[.!?।॥\n]", cleaned_context) if len(s.strip()) > 15]

            # Find sentences most relevant to user query words
            query_words = set(re.findall(r"\w+", user_query)) - {"what", "is", "the", "how", "why", "are", "tell", "me", "about"}
            scored_sentences = []
            for s in sentences:
                score = sum(1 for w in query_words if w in s.lower())
                scored_sentences.append((score, s))

            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_facts = [s for score, s in scored_sentences if score > 0][:2]

            if top_facts:
                return f"{top_facts[0]}. {top_facts[1] if len(top_facts) > 1 else ''}".strip()
            elif sentences:
                return f"Based on your documents, {sentences[0].lower()}."

        # 2. Pattern-based conversational replies
        if not user_query:
            return "Hello! I am EchoVoice, your real-time conversational AI voice agent. How can I help you today?"

        if any(w in user_query for w in ["hello", "hi", "hey", "greetings"]):
            return "Hello there! I'm EchoVoice. I can answer questions from your knowledge base or speak with a custom cloned voice. What would you like to explore?"

        if "who are you" in user_query or "what is your name" in user_query:
            return "I am EchoVoice, an open-source real-time conversational assistant featuring neural voice cloning and retrieval-augmented generation."

        if any(w in user_query for w in ["voice clone", "clone", "voice profile"]):
            return "With EchoVoice, you can upload a twenty-second audio sample, verify ethical consent, and synthesize speech in that custom voice."

        if any(w in user_query for w in ["rag", "knowledge", "document", "faiss"]):
            return "EchoVoice uses FAISS dense vector search and sentence-transformers to recall facts from your uploaded PDF and text documents in real time."

        if any(w in user_query for w in ["how are you", "how's it going"]):
            return "I'm running smoothly with ultra-low latency and ready to assist you. What can I do for you?"

        if any(w in user_query for w in ["thank", "thanks"]):
            return "You're very welcome! Let me know if you need anything else."

        # 3. Default fallback conversational answer
        return (
            f"I understood your request about '{user_query[:50]}'. "
            "I'm operating in conversational mode and ready to synthesize audio or answer questions."
        )

    @timed_step("Local Fallback LLM Generation")
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any
    ) -> LLMResponse:
        start_t = time.perf_counter()
        reply_text = self._synthesize_response(messages)
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

        prompt_len = sum(len(m.content.split()) for m in messages)
        comp_len = len(reply_text.split())

        return LLMResponse(
            content=reply_text,
            model_name=self.model_name,
            prompt_tokens=prompt_len,
            completion_tokens=comp_len,
            latency_ms=latency_ms,
            finish_reason="stop"
        )

    async def stream_generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        reply_text = self._synthesize_response(messages)
        words = reply_text.split()

        for i, word in enumerate(words):
            # Yield word with trailing space
            token = word + (" " if i < len(words) - 1 else "")
            yield token
            # Realistic token interval (15ms)
            await asyncio.sleep(0.015)
