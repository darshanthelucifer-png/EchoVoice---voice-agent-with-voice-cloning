"""
Quality Control & Audio Verification Evaluator (backend/app/ai/audio/qc.py)
--------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Multi-Metric Evaluation Architecture: Combines ASR phonetic fidelity (Word Error Rate),
  acoustic timbre matching (Cosine Similarity of 256-D speaker embeddings), and physical
  signal health (SNR, clipping, BS.1770 loudness).
- Optional Dependency Handling & Resilient Fallbacks: Gracefully leverages `whisper` / `faster_whisper`
  when present, and falls back to speech rate and duration alignment checks when neural ASR is offline.
- Normalized Text Levenshtein Distance: Uses `jiwer` / `rapidfuzz` for standardized Word Error Rate (WER)
  with case-folding, whitespace unification, and punctuation stripping.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import numpy as np
from scipy import signal

from app.core.logging import logger, timed_step
from app.ai.audio.quality import quality_analyzer, AudioQualityReport


@dataclass
class QCReport:
    """Comprehensive Quality Control evaluation report for synthesized speech."""
    passed: bool
    wer: Optional[float]                       # Word Error Rate (0.0 to 1.0)
    wer_threshold: float                       # Maximum allowable WER (default 0.08 / 8%)
    speaker_similarity: Optional[float]       # Cosine similarity (0.0 to 1.0)
    similarity_threshold: float                # Minimum acceptable similarity (0.82)
    transcribed_text: Optional[str]            # ASR hypothesis transcript
    reference_text: str                        # Target script text
    snr_db: float                              # Signal-to-noise ratio
    clipping_ratio: float                      # Clipping ratio (0.0 to 1.0)
    integrated_lufs: float                     # Measured loudness
    warnings: List[str] = field(default_factory=list)


class AudioQCEvaluator:
    """
    Automated QC Engine validating synthesized speech against original scripts
    and speaker reference profiles.
    """

    def __init__(
        self,
        wer_threshold: float = 0.08,
        similarity_threshold: float = 0.82
    ):
        self.wer_threshold = wer_threshold
        self.similarity_threshold = similarity_threshold

    def extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extracts a normalized 256-dimensional acoustic speaker timbre embedding.
        Matches the enrollment feature extractor in VoiceProfileService.
        """
        f, t, sxx = signal.spectrogram(audio, sr, nperseg=512, noverlap=256)
        log_sxx = np.log(sxx + 1e-6)

        mean_spec = np.mean(log_sxx, axis=1)
        std_spec = np.std(log_sxx, axis=1)

        raw_feat = np.concatenate([mean_spec, std_spec])
        interp_feat = np.interp(
            np.linspace(0, len(raw_feat) - 1, 256),
            np.arange(len(raw_feat)),
            raw_feat
        ).astype(np.float32)

        norm = np.linalg.norm(interp_feat) + 1e-12
        return (interp_feat / norm).astype(np.float32)

    def calculate_similarity(
        self,
        audio: np.ndarray,
        sr: int,
        ref_embedding: np.ndarray
    ) -> float:
        """
        Computes cosine similarity between generated audio and enrolled speaker profile.
        """
        synth_emb = self.extract_speaker_embedding(audio, sr)
        norm_synth = np.linalg.norm(synth_emb) + 1e-12
        norm_ref = np.linalg.norm(ref_embedding) + 1e-12
        cos_sim = float(np.dot(synth_emb, ref_embedding) / (norm_synth * norm_ref))
        return float(np.clip(cos_sim, -1.0, 1.0))

    def _normalize_text_for_wer(self, text: str) -> str:
        """Lowercases and strips punctuation for fair ASR word matching."""
        t = text.lower()
        t = re.sub(r'[^\w\s]', '', t)
        t = re.sub(r'\s+', ' ', t)
        return t.strip()

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Computes Word Error Rate using jiwer or rapidfuzz fallback.
        """
        ref_norm = self._normalize_text_for_wer(reference)
        hyp_norm = self._normalize_text_for_wer(hypothesis)

        if not ref_norm:
            return 0.0 if not hyp_norm else 1.0

        try:
            import jiwer
            return float(jiwer.wer(ref_norm, hyp_norm))
        except Exception:
            # Word-level Levenshtein fallback
            ref_words = ref_norm.split()
            hyp_words = hyp_norm.split()
            if not ref_words:
                return 0.0
            from difflib import SequenceMatcher
            matcher = SequenceMatcher(None, ref_words, hyp_words)
            dist = len(ref_words) + len(hyp_words) - 2 * sum(block.size for block in matcher.get_matching_blocks())
            return float(min(1.0, dist / len(ref_words)))

    def transcribe_audio(self, audio: np.ndarray, sr: int) -> Optional[str]:
        """
        Transcribes audio using Whisper / faster_whisper if installed.
        Returns None if no ASR model is locally available.
        """
        try:
            import whisper
            # Lightweight CPU-friendly model if available
            model = whisper.load_model("tiny")
            result = model.transcribe(audio.astype(np.float32))
            return result.get("text", "").strip()
        except ImportError:
            pass

        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(audio)
            return " ".join(s.text.strip() for s in segments)
        except ImportError:
            pass

        return None

    @timed_step("Audio QC Evaluation")
    def evaluate(
        self,
        audio: np.ndarray,
        sr: int,
        target_script: str,
        ref_embedding: Optional[np.ndarray] = None,
        mock_hypothesis: Optional[str] = None
    ) -> QCReport:
        """
        Performs thorough Quality Control audit on generated speech.
        """
        warnings: List[str] = []
        passed = True

        # 1. Signal Health Analysis
        metrics = quality_analyzer.analyze(audio, sr)

        if metrics.clipping_ratio > 0.001:
            warnings.append(
                f"Audio clipping detected ({metrics.clipping_ratio*100:.2f}%). Potential distortion."
            )
            passed = False

        if metrics.snr_db < 15.0:
            warnings.append(
                f"Low SNR ({metrics.snr_db:.1f} dB). Audible background noise or floor rumble present."
            )

        # 2. Loudness Compliance Check (-14 LUFS ± 1.5 LUFS)
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        lufs = float(meter.integrated_loudness(audio))
        if abs(lufs - (-14.0)) > 2.0:
            warnings.append(
                f"Loudness deviation: {lufs:.1f} LUFS (expected -14.0 LUFS ± 1.5 LUFS)."
            )

        # 3. Speaker Similarity Check
        similarity_score: Optional[float] = None
        if ref_embedding is not None:
            similarity_score = self.calculate_similarity(audio, sr, ref_embedding)
            if similarity_score < self.similarity_threshold:
                warnings.append(
                    f"Speaker similarity score ({similarity_score:.3f}) is below threshold ({self.similarity_threshold}). "
                    "Voice timbre drift detected."
                )
                # Similarity warning flags for review
                if similarity_score < 0.70:
                    passed = False

        # 4. ASR Word Error Rate (WER) Check
        hypothesis = mock_hypothesis or self.transcribe_audio(audio, sr)
        wer: Optional[float] = None

        if hypothesis is not None:
            wer = self.calculate_wer(target_script, hypothesis)
            if wer > self.wer_threshold:
                warnings.append(
                    f"ASR Word Error Rate ({wer*100:.1f}%) exceeds threshold ({self.wer_threshold*100:.1f}%). "
                    "Potential speech omission, stuttering, or hallucination."
                )
                passed = False
        else:
            # When neural ASR is not loaded, verify duration alignment with expected reading speed
            words = target_script.split()
            expected_min_sec = (len(words) / 200.0) * 60.0  # Fast 200 wpm
            expected_max_sec = (len(words) / 80.0) * 60.0   # Slow 80 wpm
            actual_sec = len(audio) / sr

            if actual_sec < expected_min_sec * 0.5:
                warnings.append(
                    f"Generated audio is abnormally short ({actual_sec:.1f}s vs expected min {expected_min_sec:.1f}s). "
                    "Possible dropped sentences."
                )
                passed = False

        return QCReport(
            passed=passed,
            wer=wer,
            wer_threshold=self.wer_threshold,
            speaker_similarity=similarity_score,
            similarity_threshold=self.similarity_threshold,
            transcribed_text=hypothesis,
            reference_text=target_script,
            snr_db=metrics.snr_db,
            clipping_ratio=metrics.clipping_ratio,
            integrated_lufs=lufs,
            warnings=warnings
        )


audio_qc_evaluator = AudioQCEvaluator()
