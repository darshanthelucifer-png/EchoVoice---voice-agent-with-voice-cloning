"""
Conversational Agent & Chat Router (backend/app/api/routes/chat.py)
--------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Server-Sent Events (SSE) Streaming (`StreamingResponse`): Streams real-time token events
  to the frontend with `text/event-stream` media type.
- RESTful Conversation Management: Full CRUD for multi-turn conversational histories.
- Dependency Injection: Secures chat sessions through `get_current_user` and database session `get_db`.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ConversationCreate,
    ConversationRead
)
from app.services.chat_service import chat_service
from app.ai.llm.registry import llm_registry

router = APIRouter(prefix="/chat", tags=["Conversational Agent"])


@router.get("/engines", summary="List registered LLM engines")
async def list_llm_engines():
    """Returns available LLM inference backends."""
    return APIResponse(
        success=True,
        data={
            "engines": llm_registry.available_engines(),
            "default_engine": "fallback"
        }
    )


@router.post(
    "/completions",
    response_model=APIResponse[ChatCompletionResponse],
    summary="Generate conversational assistant response"
)
async def chat_completions(
    request: ChatCompletionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Processes a conversational turn with RAG factual grounding and persona conditioning.
    If stream=True, returns a Server-Sent Events (SSE) stream of token deltas.
    """
    if request.stream:
        return StreamingResponse(
            chat_service.stream_chat_turn(
                db=db,
                user_id=current_user.id,
                request=request
            ),
            media_type="text/event-stream"
        )

    response = await chat_service.process_chat_turn(
        db=db,
        user_id=current_user.id,
        request=request
    )

    return APIResponse(
        success=True,
        message="Turn completed successfully.",
        data=response
    )


@router.post(
    "/conversations",
    response_model=APIResponse[ConversationRead],
    summary="Create a new conversation session"
)
async def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Initializes a new conversational session for tracking multi-turn dialogue."""
    conv = await chat_service.create_conversation(
        db=db,
        user_id=current_user.id,
        title=payload.title,
        persona=payload.persona
    )
    return APIResponse(
        success=True,
        message="Conversation initialized.",
        data=ConversationRead.model_validate(conv)
    )


@router.get(
    "/conversations",
    response_model=APIResponse[List[ConversationRead]],
    summary="List all user conversations"
)
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Lists all conversation sessions belonging to the authenticated user."""
    convs = await chat_service.list_conversations(db=db, user_id=current_user.id)
    return APIResponse(
        success=True,
        data=[ConversationRead.model_validate(c) for c in convs]
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=APIResponse[ConversationRead],
    summary="Get conversation with message history"
)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves conversation details and full turn history."""
    conv = await chat_service.get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found."
        )
    return APIResponse(
        success=True,
        data=ConversationRead.model_validate(conv)
    )


@router.delete(
    "/conversations/{conversation_id}",
    response_model=APIResponse[bool],
    summary="Delete conversation session"
)
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Deletes a conversation and its messages."""
    deleted = await chat_service.delete_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found."
        )
    return APIResponse(
        success=True,
        message="Conversation deleted.",
        data=True
    )
