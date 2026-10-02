"""
Conversational Memory & Sliding Window Buffer (backend/app/ai/llm/memory.py)
-----------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Sliding Window Context Management: Retains recent conversational turns to preserve
  multi-turn dialogue coherence while preventing context window exhaustion.
- Message Pruning & System Role Preservation: Ensures that system instructions and RAG
  knowledge blocks are always retained at index 0.
"""

from typing import List, Optional
from app.ai.llm.base import LLMMessage
from app.models.conversation import Message as DBMessage


class ConversationMemory:
    """
    Manages conversational turns with a sliding window buffer.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns

    def build_messages(
        self,
        system_prompt: str,
        history: List[DBMessage],
        current_user_message: str
    ) -> List[LLMMessage]:
        """
        Assembles a bounded list of LLMMessage turns:
        [System Message] + [Sliding Window History] + [Current User Message]
        """
        messages: List[LLMMessage] = [
            LLMMessage(role="system", content=system_prompt)
        ]

        # Retain last max_turns from history
        recent_history = history[-self.max_turns:] if len(history) > self.max_turns else history

        for m in recent_history:
            if m.role in ("user", "assistant"):
                messages.append(LLMMessage(role=m.role, content=m.content))

        # Add current user message
        messages.append(LLMMessage(role="user", content=current_user_message))

        return messages


# Global singleton memory manager
conversation_memory = ConversationMemory()
