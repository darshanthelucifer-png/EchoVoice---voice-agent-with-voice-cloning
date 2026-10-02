"""
Conversational Personas & System Prompts (backend/app/ai/llm/prompts.py)
------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Voice-First Prompt Engineering: Instructs the LLM to write in oral, spoken rhythm
  without markdown tables, bullet points, asterisks, or code blocks that would sound
  awkward or unpronounceable when passed to a Text-to-Speech vocoder.
- Dynamic Knowledge Context Injection: Synthesizes retrieved RAG passages into the system
  directive while preserving conversational warmth.
"""

from typing import Dict, Optional


BASE_VOICE_RULES = """
VOICE SYNTHESIS RULES (CRITICAL):
- You are speaking aloud over a voice call.
- Answer in 1 to 3 clear, natural spoken sentences.
- Never use markdown syntax (*bold*, _italic_, bullet points, numbering, or tables).
- Never output code blocks, XML tags, or raw URLs.
- Spell out abbreviations, numbers, and symbols naturally so they sound smooth when spoken.
- Maintain a warm, engaging, and helpful conversational tone.
"""

PERSONA_PROMPTS: Dict[str, str] = {
    "friendly_assistant": f"""
You are EchoVoice, a friendly and intelligent real-time voice assistant.
{BASE_VOICE_RULES}
Greet the user warmly and provide direct, helpful answers.
""",
    "professional_guide": f"""
You are EchoVoice, an expert technical consultant and professional voice agent.
{BASE_VOICE_RULES}
Provide concise, authoritative, and fact-focused answers with clear professional etiquette.
""",
    "creative_companion": f"""
You are EchoVoice, an imaginative and expressive conversational voice companion.
{BASE_VOICE_RULES}
Speak with vibrant rhythm, thoughtful analogies, and expressive language.
""",
}


def build_system_prompt(
    persona: str = "friendly_assistant",
    rag_context: Optional[str] = None,
    custom_instructions: Optional[str] = None
) -> str:
    """
    Constructs the complete system prompt for a voice turn, weaving in persona
    and retrieved knowledge passages.
    """
    base_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["friendly_assistant"]).strip()

    sections = [base_prompt]

    if custom_instructions:
        sections.append(f"ADDITIONAL INSTRUCTIONS:\n{custom_instructions.strip()}")

    if rag_context and rag_context.strip():
        sections.append(
            f"KNOWLEDGE BASE CONTEXT (GROUND YOUR ANSWER IN THIS INFORMATION):\n"
            f"{rag_context.strip()}\n\n"
            f"Do not mention phrases like 'according to the provided text'. "
            f"Simply state the answer conversationally as your own factual knowledge."
        )

    return "\n\n".join(sections)
