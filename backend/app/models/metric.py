"""
Metrics Model (backend/app/models/metric.py)
--------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Detailed latency telemetry schema: Captures granular millisecond metrics
  across the entire real-time conversational voice pipeline
  (VAD -> ASR -> RAG Retrieval -> LLM TTFT -> TTS First Audio -> Total).
- Quality scores: Tracks Word Error Rate (WER) and speaker cosine similarity.
"""

from sqlalchemy import Column, String, Float, ForeignKey
from app.core.database import Base
from app.models.base import BaseModelMixin


class LatencyMetric(Base, BaseModelMixin):
    """
    Stores per-stage profiling metrics for live analytics dashboard and latency reports.
    """
    __tablename__ = "latency_metrics"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    turn_id = Column(String(36), index=True, nullable=False)
    session_type = Column(String(50), default="realtime_voice", nullable=False)

    # Granular timing in milliseconds
    vad_ms = Column(Float, default=0.0, nullable=False)
    asr_ms = Column(Float, default=0.0, nullable=False)
    retrieval_ms = Column(Float, default=0.0, nullable=False)
    llm_first_token_ms = Column(Float, default=0.0, nullable=False)
    llm_total_ms = Column(Float, default=0.0, nullable=False)
    tts_first_chunk_ms = Column(Float, default=0.0, nullable=False)
    tts_total_ms = Column(Float, default=0.0, nullable=False)
    total_latency_ms = Column(Float, default=0.0, nullable=False)

    # Quality scores
    wer_score = Column(Float, nullable=True)
    similarity_score = Column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<LatencyMetric turn_id={self.turn_id} total_ms={self.total_latency_ms:.2f}>"
