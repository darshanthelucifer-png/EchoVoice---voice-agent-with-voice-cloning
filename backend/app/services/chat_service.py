"""
Conversational Chat Service (backend/app/services/chat_service.py)
-----------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Service Layer Orchestration: Coordinates relational session history, semantic RAG retrieval,
  conversational memory windowing, and LLM text generation.
- Full-Duplex SSE Stream Accumulation: Yields token deltas immediately to the client
  while buffering the full response for atomic database turn persistence.
"""

import json
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any
import uuid
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger, timed_step
from app.models.conversation import Conversation, Message
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from app.ai.llm.base import LLMMessage
from app.ai.llm.prompts import build_system_prompt
from app.ai.llm.memory import conversation_memory
from app.ai.llm.registry import get_llm_engine
from app.services.rag_service import rag_service


class ChatService:
    """
    Coordinates multi-turn dialogue management, knowledge grounding,
    and LLM inference.
    """

    async def create_conversation(
        self,
        db: AsyncSession,
        user_id: str,
        title: Optional[str] = None,
        persona: str = "friendly_assistant"
    ) -> Conversation:
        """Initializes a new conversational session."""
        conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title or "New Voice Conversation",
            persona=persona
        )
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
        return conv

    async def get_conversation(
        self,
        db: AsyncSession,
        conversation_id: str,
        user_id: str
    ) -> Optional[Conversation]:
        """Retrieves a conversation by ID with loaded message history."""
        query = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def list_conversations(
        self,
        db: AsyncSession,
        user_id: str
    ) -> List[Conversation]:
        """Lists all conversations for a user ordered by recency."""
        query = select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.created_at.desc())
        result = await db.execute(query)
        return list(result.scalars().all())

    async def delete_conversation(
        self,
        db: AsyncSession,
        conversation_id: str,
        user_id: str
    ) -> bool:
        """Deletes a conversation and its messages."""
        conv = await self.get_conversation(db, conversation_id, user_id)
        if not conv:
            return False
        await db.delete(conv)
        await db.commit()
        return True

    @timed_step("Conversational Chat Turn")
    async def process_chat_turn(
        self,
        db: AsyncSession,
        user_id: str,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Executes a complete synchronous conversation turn:
        1. Resolves/creates conversation session
        2. Retrieves RAG knowledge context if enabled
        3. Formulates voice-first prompt with sliding window memory
        4. Queries LLM engine
        5. Persists turn to database
        """
        # 1. Resolve or create conversation
        if request.conversation_id:
            conv = await self.get_conversation(db, request.conversation_id, user_id)
            if not conv:
                conv = await self.create_conversation(db, user_id, persona=request.persona)
        else:
            conv = await self.create_conversation(db, user_id, persona=request.persona)

        # 2. RAG Retrieval
        rag_context: Optional[str] = None
        rag_sources: List[str] = []
        if request.use_rag:
            try:
                retrieval = await rag_service.query_knowledge_base(
                    user_id=user_id,
                    query=request.message,
                    collection_name=request.collection_name
                )
                if retrieval.has_context:
                    rag_context = retrieval.context_text
                    rag_sources = [c.title for c in retrieval.chunks]
            except Exception as exc:
                logger.debug(f"RAG retrieval skipped or failed: {exc}")

        # 3. Build system prompt & memory
        system_prompt = build_system_prompt(persona=request.persona or conv.persona, rag_context=rag_context)
        llm_messages = conversation_memory.build_messages(
            system_prompt=system_prompt,
            history=conv.messages,
            current_user_message=request.message
        )

        # 4. LLM Generation
        engine = get_llm_engine(request.engine)
        llm_response = await engine.generate(
            messages=llm_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # 5. Persist turns in DB
        user_msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv.id,
            role="user",
            content=request.message
        )
        assistant_msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv.id,
            role="assistant",
            content=llm_response.content,
            latency_ms=llm_response.latency_ms
        )
        db.add(user_msg)
        db.add(assistant_msg)
        await db.commit()

        return ChatCompletionResponse(
            message=llm_response.content,
            conversation_id=conv.id,
            model_name=llm_response.model_name,
            latency_ms=llm_response.latency_ms,
            rag_sources=rag_sources
        )

    async def stream_chat_turn(
        self,
        db: AsyncSession,
        user_id: str,
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Server-Sent Events (SSE) generator streaming token deltas.
        Persists the completed turn to the database at stream conclusion.
        """
        # Resolve conversation
        if request.conversation_id:
            conv = await self.get_conversation(db, request.conversation_id, user_id)
            if not conv:
                conv = await self.create_conversation(db, user_id, persona=request.persona)
        else:
            conv = await self.create_conversation(db, user_id, persona=request.persona)

        # RAG Retrieval
        rag_context = None
        if request.use_rag:
            try:
                retrieval = await rag_service.query_knowledge_base(
                    user_id=user_id,
                    query=request.message,
                    collection_name=request.collection_name
                )
                if retrieval.has_context:
                    rag_context = retrieval.context_text
            except Exception:
                pass

        system_prompt = build_system_prompt(persona=request.persona or conv.persona, rag_context=rag_context)
        llm_messages = conversation_memory.build_messages(
            system_prompt=system_prompt,
            history=conv.messages,
            current_user_message=request.message
        )

        engine = get_llm_engine(request.engine)
        accumulated_text: List[str] = []

        try:
            async for token in engine.stream_generate(
                messages=llm_messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                accumulated_text.append(token)
                event_data = {
                    "token": token,
                    "conversation_id": conv.id,
                    "done": False
                }
                yield f"data: {json.dumps(event_data)}\n\n"

            # Final SSE completion packet
            full_reply = "".join(accumulated_text).strip()
            final_data = {
                "token": "",
                "conversation_id": conv.id,
                "done": True,
                "full_message": full_reply
            }
            yield f"data: {json.dumps(final_data)}\n\n"

            # Persist turns
            user_msg = Message(
                id=str(uuid.uuid4()),
                conversation_id=conv.id,
                role="user",
                content=request.message
            )
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                conversation_id=conv.id,
                role="assistant",
                content=full_reply
            )
            db.add(user_msg)
            db.add(assistant_msg)
            await db.commit()

        except Exception as exc:
            logger.error(f"Chat streaming error: {exc}", exc_info=True)
            err_data = {"error": str(exc), "done": True}
            yield f"data: {json.dumps(err_data)}\n\n"


# Global singleton service
chat_service = ChatService()
