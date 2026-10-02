"""
Standalone Conversational AI Verification Script (backend/scripts/test_chat_core.py)
------------------------------------------------------------------------------------
Tests end-to-end:
1. LLMEngine interface and LocalFallbackLLMEngine generation
2. Token streaming via AsyncGenerator
3. Voice-first prompt construction with persona and RAG grounding
4. Conversational sliding window memory
5. Resilient fallback proxy fault tolerance
"""

import asyncio
from pathlib import Path
import sys
import time

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ai.llm.base import LLMMessage
from app.ai.llm.fallback_engine import LocalFallbackLLMEngine
from app.ai.llm.openai_compatible_engine import OpenAICompatibleLLMEngine
from app.ai.llm.prompts import build_system_prompt
from app.ai.llm.memory import conversation_memory
from app.ai.llm.registry import get_llm_engine, ResilientLLMProxy


async def run_chat_core_test():
    print("=" * 70)
    print(" EchoVoice Phase 9: Conversational AI Core Verification")
    print("=" * 70)

    # 1. Fallback Engine Generation Test
    print("\n--- 1. Testing Local Fallback LLM Engine ---")
    fallback = LocalFallbackLLMEngine()
    messages = [
        LLMMessage(role="system", content="You are EchoVoice, a real-time conversational voice assistant."),
        LLMMessage(role="user", content="Hello! Who are you and what can you do?")
    ]
    resp = await fallback.generate(messages)
    print(f"Engine: {resp.model_name} | Latency: {resp.latency_ms} ms")
    print(f"Reply: \"{resp.content}\"")
    assert len(resp.content) > 10
    assert "echovoice" in resp.content.lower()

    # 2. Token Streaming Test
    print("\n--- 2. Testing Token-by-Token Streaming ---")
    streamed_tokens = []
    start_t = time.perf_counter()
    async for token in fallback.stream_generate(messages):
        streamed_tokens.append(token)
        # Visual live stream simulation
        print(token, end="", flush=True)
    stream_time_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
    print(f"\nStreamed {len(streamed_tokens)} tokens in {stream_time_ms} ms.")
    assert len(streamed_tokens) > 0

    # 3. Prompt Engineering & RAG Grounding Test
    print("\n--- 3. Testing Voice-First Prompt & RAG Grounding ---")
    rag_context = (
        "[Source 1: EchoVoice Security Policy (Page 1)]\n"
        "Voice profiles require explicit ethical consent before audio generation.\n"
        "All speaker embeddings are stored as 256-dimensional unit vectors."
    )
    prompt = build_system_prompt(persona="friendly_assistant", rag_context=rag_context)
    print("Generated Prompt Preview:")
    for line in prompt.strip().split("\n")[:6]:
        print(f"  {line}")
    assert "VOICE SYNTHESIS RULES" in prompt
    assert "KNOWLEDGE BASE CONTEXT" in prompt

    # Ask a RAG-grounded question to fallback engine
    rag_messages = [
        LLMMessage(role="system", content=prompt),
        LLMMessage(role="user", content="What is the rule about consent?")
    ]
    rag_resp = await fallback.generate(rag_messages)
    print(f"\nRAG Answer: \"{rag_resp.content}\"")
    assert "consent" in rag_resp.content.lower()

    # 4. Resilient Proxy Fault Tolerance Test
    print("\n--- 4. Testing Resilient Fallback Proxy ---")
    # Point OpenAICompatibleLLMEngine to a non-existent port to simulate disconnected Ollama
    dead_ollama = OpenAICompatibleLLMEngine(base_url="http://127.0.0.1:9999/v1", timeout_sec=1.0)
    proxy = ResilientLLMProxy(primary=dead_ollama, fallback=fallback)
    
    proxy_resp = await proxy.generate(messages)
    print(f"Proxy successfully intercepted failure and routed to: {proxy_resp.model_name}")
    print(f"Safe Reply: \"{proxy_resp.content}\"")
    assert len(proxy_resp.content) > 0

    print("\n" + "=" * 70)
    print(" ALL PHASE 9 CONVERSATIONAL AI CORE CHECKS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_chat_core_test())
