"""
EchoVoice Audio AI Subsystem (backend/app/ai/audio/__init__.py)
--------------------------------------------------------------
Exports audio processing, VAD, quality analysis, and cleanup modules.
"""

from app.ai.audio.vad import SileroVADProcessor, vad_processor
from app.ai.audio.quality import (
    AudioQualityAnalyzer,
    AudioQualityReport,
    quality_analyzer,
)
from app.ai.audio.cleanup import (
    AudioCleanupPipeline,
    CleanupConfig,
    CleanupResult,
    cleanup_pipeline,
)
from app.ai.audio.similarity import (
    SpeakerSimilarityEvaluator,
    SpeakerSimilarityResult,
    speaker_similarity_evaluator,
)

__all__ = [
    "SileroVADProcessor",
    "vad_processor",
    "AudioQualityAnalyzer",
    "AudioQualityReport",
    "quality_analyzer",
    "AudioCleanupPipeline",
    "CleanupConfig",
    "CleanupResult",
    "cleanup_pipeline",
    "SpeakerSimilarityEvaluator",
    "SpeakerSimilarityResult",
    "speaker_similarity_evaluator",
]
