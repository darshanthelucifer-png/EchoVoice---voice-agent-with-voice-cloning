"""
Conversational AI Core Unit & Integration Tests (backend/tests/test_chat.py)
----------------------------------------------------------------------------
PYTHON & CONVERSATIONAL AI CONCEPTS TESTED:
1. LLM Engine Interfaces: Asserts ABC compliance for fallback and OpenAI-compatible engines.
2. Voice Prompt Engineering: Verifies oral rhythm rules and dynamic RAG grounding.
3. Sliding Window Memory: Tests context window bounds and message history reconstruction.
4. Fault-Tolerant Failover: Verifies ResilientLLMProxy seamless failover on network timeouts.
5. REST & SSE Streaming API: Tests synchronous turns, Server-Sent Events, and conversation CRUD.
"""

import json
import pytest
from httpx import AsyncClient

from app.ai.llm.base import LLMEngine, LLMMessage, LLMResponse
from app.ai.llm.fallback_engine import LocalFallbackLLMEngine
from app.ai.llm.openai_compatible_engine import OpenAICompatibleLLMEngine
from app.ai.llm.prompts import build_system_prompt
from app.ai.llm.memory import conversation_memory
from app.ai.llm.registry import llm_registry, get_llm_engine, ResilientLLMProxy
from app.models.conversation import Message as DBMessage


# ==============================================================================
# 1. Interface & Engine Tests
# ==============================================================================

def test_llm_engine_interfaces():
    """Verify all LLM engines adhere strictly to the LLMEngine ABC contract."""
    fallback = LocalFallbackLLMEngine()
    openai_compat = OpenAICompatibleLLMEngine()

    for engine in [fallback, openai_compat]:
        assert isinstance(engine, LLMEngine)
        assert engine.is_available() is True
        assert isinstance(engine.engine_name, str)
        assert isinstance(engine.model_name, str)


@pytest.mark.asyncio
async def test_local_fallback_generation():
    """Verify fallback engine generates spoken conversational replies and measures latency."""
    engine = LocalFallbackLLMEngine()
    messages = [
        LLMMessage(role="system", content="You are EchoVoice."),
        LLMMessage(role="user", content="Hello, how are you?")
    ]

    response = await engine.generate(messages, temperature=0.7)
    assert isinstance(response, LLMResponse)
    assert len(response.content) > 0
    assert response.latency_ms >= 0.0
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_local_fallback_streaming():
    """Verify fallback engine streams token chunks over AsyncGenerator."""
    engine = LocalFallbackLLMEngine()
    messages = [
        LLMMessage(role="user", content="Who are you?")
    ]

    tokens = []
    async for token in engine.stream_generate(messages):
        assert isinstance(token, str)
        tokens.append(token)

    assert len(tokens) > 0
    full_text = "".join(tokens)
    assert "EchoVoice" in full_text


# ==============================================================================
# 2. Prompt Engineering & Memory Tests
# ==============================================================================

def test_voice_prompt_builder():
    """Verify voice-first prompt constraints and RAG knowledge block injection."""
    rag_context = "[Source 1: Policy]\nVoice consent is strictly required."
    prompt = build_system_prompt(persona="friendly_assistant", rag_context=rag_context)

    assert "VOICE SYNTHESIS RULES" in prompt
    assert "Never use markdown syntax" in prompt
    assert "KNOWLEDGE BASE CONTEXT" in prompt
    assert "Voice consent is strictly required." in prompt


def test_conversation_memory_sliding_window():
    """Verify memory manager enforces maximum sliding window turns."""
    history = [
        DBMessage(role="user", content=f"User query {i}") if i % 2 == 0
        else DBMessage(role="assistant", content=f"Assistant reply {i}")
        for i in range(20)
    ]

    messages = conversation_memory.build_messages(
        system_prompt="System Directive",
        history=history,
        current_user_message="Current turn"
    )

    # 1 system + 10 history + 1 current = 12 total messages
    assert len(messages) == 12
    assert messages[0].role == "system"
    assert messages[-1].role == "user"
    assert messages[-1].content == "Current turn"


# ==============================================================================
# 3. Fault-Tolerant Proxy Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_resilient_proxy_failover():
    """Verify proxy transparently routes to local fallback when primary endpoint is unreachable."""
    fallback = LocalFallbackLLMEngine()
    dead_client = OpenAICompatibleLLMEngine(base_url="http://127.0.0.1:9999/v1", timeout_sec=0.1)
    proxy = ResilientLLMProxy(primary=dead_client, fallback=fallback)

    messages = [LLMMessage(role="user", content="What is voice cloning?")]
    resp = await proxy.generate(messages)

    assert isinstance(resp, LLMResponse)
    assert len(resp.content) > 0
    assert resp.model_name == fallback.model_name


# ==============================================================================
# 4. REST API Endpoint Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_chat_api_lifecycle(client: AsyncClient):
    """
    Test full conversational API lifecycle:
    1. Register user and authenticate
    2. Create conversation via POST /api/v1/chat/conversations
    3. Execute synchronous turn via POST /api/v1/chat/completions
    4. Execute SSE streaming turn via POST /api/v1/chat/completions (stream=True)
    5. List conversations via GET /api/v1/chat/conversations
    6. Retrieve conversation history via GET /api/v1/chat/conversations/{id}
    7. Delete conversation via DELETE /api/v1/chat/conversations/{id}
    """
    # 1. Register test user
    user_payload = {
        "email": "chat_tester@echovoice.ai",
        "password": "Password123!",
        "full_name": "Chat Tester"
    }
    reg_res = await client.post("/api/v1/auth/register", json=user_payload)
    assert reg_res.status_code in (200, 201)
    token = reg_res.json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Create conversation
    conv_res = await client.post(
        "/api/v1/chat/conversations",
        headers=headers,
        json={"title": "Test Voice Session", "persona": "friendly_assistant"}
    )
    assert conv_res.status_code == 200
    conv_data = conv_res.json()["data"]
    conv_id = conv_data["id"]
    assert conv_data["title"] == "Test Voice Session"

    # 3. Synchronous chat completion
    chat_payload = {
        "message": "Hello EchoVoice, what are you?",
        "conversation_id": conv_id,
        "engine": "fallback",
        "stream": False
    }
    comp_res = await client.post("/api/v1/chat/completions", headers=headers, json=chat_payload)
    assert comp_res.status_code == 200
    comp_data = comp_res.json()["data"]
    assert len(comp_data["message"]) > 0
    assert comp_data["conversation_id"] == conv_id

    # 4. Streaming SSE chat completion
    stream_payload = {
        "message": "How does voice cloning work?",
        "conversation_id": conv_id,
        "engine": "fallback",
        "stream": True
    }
    stream_res = await client.post("/api/v1/chat/completions", headers=headers, json=stream_payload)
    assert stream_res.status_code == 200
    assert "text/event-stream" in stream_res.headers.get("content-type", "")
    assert len(stream_res.text) > 0
    assert "data: " in stream_res.text

    # 5. List conversations
    list_res = await client.get("/api/v1/chat/conversations", headers=headers)
    assert list_res.status_code == 200
    convs = list_res.json()["data"]
    assert len(convs) >= 1
    assert any(c["id"] == conv_id for c in convs)

    # 6. Retrieve conversation history (must contain recorded messages)
    get_res = await client.get(f"/api/v1/chat/conversations/{conv_id}", headers=headers)
    assert get_res.status_code == 200
    history = get_res.json()["data"]
    assert len(history["messages"]) >= 2  # At least 1 user turn + 1 assistant turn

    # 7. Delete conversation
    del_res = await client.delete(f"/api/v1/chat/conversations/{conv_id}", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["data"] is True
