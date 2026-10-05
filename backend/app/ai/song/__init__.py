"""
EchoVoice Song Studio AI Package (backend/app/ai/song/__init__.py)
------------------------------------------------------------------
Provides neural stem separation, vocal dereverberation/de-bleeding,
musical key/BPM/F0 pitch trajectory analysis, singing voice conversion (SVC),
vocal post-processing, multitrack mixdown, broadcast remastering,
and acoustic/biometric quality gate certification.
"""

from app.ai.song.separator import StemSeparationEngine, SeparationResult, stem_separation_engine
from app.ai.song.dereverb import VocalDereverberator, DereverbResult, vocal_dereverberator
from app.ai.song.analyzer import SongMusicalAnalyzer, SongAnalysisResult, song_musical_analyzer
from app.ai.song.svc_engine import (
    SingingVoiceConversionEngine,
    SVCResult,
    SVCSettings,
    singing_voice_engine
)
from app.ai.song.vocal_postprocess import (
    VocalPostProcessor,
    VocalPostProcessResult,
    VocalPostProcessConfig,
    vocal_post_processor
)
from app.ai.song.remaster import (
    SongRemasterEngine,
    SongRemasterResult,
    MixdownSettings,
    song_remaster_engine
)
from app.ai.song.quality_gate import (
    SongQualityGate,
    QualityGateReport,
    OctaveErrorSection,
    song_quality_gate
)
from app.ai.song.legal_gate import (
    SongLegalGate,
    LegalVerificationResult,
    song_legal_gate
)
from app.ai.song.experimental import (
    ExperimentalSongFeatures,
    LyricTranslationResult,
    ACEStepGenerationResult,
    experimental_song_features
)

__all__ = [
    "StemSeparationEngine",
    "SeparationResult",
    "stem_separation_engine",
    "VocalDereverberator",
    "DereverbResult",
    "vocal_dereverberator",
    "SongMusicalAnalyzer",
    "SongAnalysisResult",
    "song_musical_analyzer",
    "SingingVoiceConversionEngine",
    "SVCResult",
    "SVCSettings",
    "singing_voice_engine",
    "VocalPostProcessor",
    "VocalPostProcessResult",
    "VocalPostProcessConfig",
    "vocal_post_processor",
    "SongRemasterEngine",
    "SongRemasterResult",
    "MixdownSettings",
    "song_remaster_engine",
    "SongQualityGate",
    "QualityGateReport",
    "OctaveErrorSection",
    "song_quality_gate",
    "SongLegalGate",
    "LegalVerificationResult",
    "song_legal_gate",
    "ExperimentalSongFeatures",
    "LyricTranslationResult",
    "ACEStepGenerationResult",
    "experimental_song_features",
]
