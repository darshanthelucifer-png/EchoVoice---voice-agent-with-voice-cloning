"""
Speaker Similarity & Biometric Likeness Evaluation (backend/app/ai/audio/similarity.py)
--------------------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Singleton & Lazy Loading Pattern: Neural embedding models (ECAPA-TDNN & WavLM-SV)
  are loaded into memory on first request, avoiding multi-second boot overhead.
- Dual-Embedding Acoustic Ensembling: Fuses SpeechBrain ECAPA-TDNN (x-vector channel
  attention) and Microsoft WavLM-SV (self-supervised transformer representations)
  to measure fine-grained vocal timbre identity with high discriminatory precision.
- Vectorized Cosine Similarity & L2 Normalization: Computes inner product on unit
  hyperspheres with PyTorch/NumPy tensors for mathematically rigorous distance metrics.
- Pydantic/Dataclass Structured Results: Fully type-hinted outputs for API and CLI consumers.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

from app.core.config import settings
from app.core.logging import logger, timed_step
from app.ai.audio.cleanup import AudioCleanupPipeline


@dataclass
class SpeakerSimilarityResult:
    """
    Structured outcome of biometric speaker verification comparison.
    Contains both raw model similarities, an ensembled composite score,
    and a definitive tier threshold evaluation.
    """
    ecapa_similarity: float
    wavlm_similarity: float
    composite_score: float
    target_met: bool
    target_threshold: float
    likeness_tier: str
    verdict: str
    reference_duration_sec: float
    candidate_duration_sec: float
    device_used: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Type alias for downstream compatibility
SimilarityResult = SpeakerSimilarityResult


class SpeakerSimilarityEvaluator:
    """
    Evaluation judge assessing acoustic likeness between a speaker's reference recording
    and synthesized or converted speech.
    Uses two state-of-the-art open-source speaker recognition models:
    1. ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) - 192-dim x-vector with channel attention
    2. WavLM-SV (microsoft/wavlm-base-plus-sv) - 512-dim transformer-based speaker representation
    """

    def __init__(
        self,
        ecapa_model_id: Optional[str] = None,
        wavlm_model_id: Optional[str] = None,
        target_score: Optional[float] = None
    ):
        self.ecapa_model_id = ecapa_model_id or settings.SPEAKER_SIMILARITY_ECAPA_MODEL
        self.wavlm_model_id = wavlm_model_id or settings.SPEAKER_SIMILARITY_WAVLM_MODEL
        self.target_score = target_score or settings.SIMILARITY_TARGET_SCORE

        self._ecapa_model = None
        self._wavlm_model = None
        self._wavlm_extractor = None

    def _get_device(self) -> torch.device:
        pref = getattr(settings, "EVAL_DEVICE", "auto").lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif pref == "auto" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_ecapa(self) -> Any:
        """Lazy loader for SpeechBrain ECAPA-TDNN classifier."""
        if self._ecapa_model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            device = self._get_device()
            logger.info(f"Loading ECAPA-TDNN model '{self.ecapa_model_id}' on {device}...")
            cache_dir = settings.DATA_DIR / "models" / "ecapa_voxceleb"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._ecapa_model = EncoderClassifier.from_hparams(
                source=self.ecapa_model_id,
                savedir=str(cache_dir),
                run_opts={"device": str(device)}
            )
        return self._ecapa_model

    def _load_wavlm(self) -> Tuple[Any, Any]:
        """Lazy loader for Microsoft WavLM-SV model and feature extractor."""
        if self._wavlm_model is None or self._wavlm_extractor is None:
            from transformers import AutoFeatureExtractor, WavLMForXVector
            device = self._get_device()
            logger.info(f"Loading WavLM-SV model '{self.wavlm_model_id}' on {device}...")
            self._wavlm_extractor = AutoFeatureExtractor.from_pretrained(self.wavlm_model_id)
            self._wavlm_model = WavLMForXVector.from_pretrained(self.wavlm_model_id).to(device)
            self._wavlm_model.eval()
        return self._wavlm_model, self._wavlm_extractor

    def extract_ecapa_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extracts a 192-dimensional speaker embedding using ECAPA-TDNN.
        Audio is resampled to 16 kHz mono and converted to tensor.
        """
        target_sr = 16000
        if sr != target_sr:
            audio, sr = AudioCleanupPipeline.load_audio(audio, target_sr=target_sr)

        device = self._get_device()
        ecapa = self._load_ecapa()

        tensor_audio = torch.from_numpy(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = ecapa.encode_batch(tensor_audio)
            # Squeeze to 1D: shape [192]
            emb_vec = emb.squeeze().cpu().numpy().astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(emb_vec) + 1e-12
        return emb_vec / norm

    def extract_wavlm_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extracts a 512-dimensional speaker embedding using WavLM-SV.
        Audio is resampled to 16 kHz mono.
        """
        target_sr = 16000
        if sr != target_sr:
            audio, sr = AudioCleanupPipeline.load_audio(audio, target_sr=target_sr)

        device = self._get_device()
        wavlm, extractor = self._load_wavlm()

        inputs = extractor(audio, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = wavlm(**inputs)
            emb = outputs.embeddings.squeeze().cpu().numpy().astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(emb) + 1e-12
        return emb / norm

    @timed_step("Compute Speaker Similarity")
    def compute_similarity(
        self,
        ref_audio: np.ndarray,
        cand_audio: np.ndarray,
        sr_ref: int = 16000,
        sr_cand: int = 16000
    ) -> SpeakerSimilarityResult:
        """
        Computes cosine similarity between reference recording and candidate speech audio
        using both ECAPA-TDNN and WavLM-SV neural speaker verification models.
        """
        ref_dur = len(ref_audio) / sr_ref
        cand_dur = len(cand_audio) / sr_cand

        device_name = str(self._get_device())

        # 1. ECAPA-TDNN Similarity
        ref_ecapa = self.extract_ecapa_embedding(ref_audio, sr_ref)
        cand_ecapa = self.extract_ecapa_embedding(cand_audio, sr_cand)
        ecapa_cos = float(np.dot(ref_ecapa, cand_ecapa))
        # Clamp to [0.0, 1.0] for standard percentage interpretation
        ecapa_score = max(0.0, min(1.0, ecapa_cos))

        # 2. WavLM-SV Similarity
        ref_wavlm = self.extract_wavlm_embedding(ref_audio, sr_ref)
        cand_wavlm = self.extract_wavlm_embedding(cand_audio, sr_cand)
        wavlm_cos = float(np.dot(ref_wavlm, cand_wavlm))
        wavlm_score = max(0.0, min(1.0, wavlm_cos))

        # 3. Composite Ensemble (0.60 ECAPA + 0.40 WavLM)
        composite = round(0.60 * ecapa_score + 0.40 * wavlm_score, 4)

        # 4. Target Threshold & Likeness Tier Determination
        target_met = ecapa_score >= self.target_score

        if ecapa_score >= self.target_score:
            tier = "Tier 3: Near-Identical Likeness"
            verdict = (
                f"PASSED target likeness threshold (ECAPA: {ecapa_score:.3f} >= {self.target_score}). "
                "Synthesized voice matches the speaker's biometric vocal timbre with high fidelity."
            )
        elif ecapa_score >= 0.75:
            tier = "Tier 2: Close Likeness"
            verdict = (
                f"Close likeness achieved (ECAPA: {ecapa_score:.3f}). Timbre strongly resembles the speaker, "
                f"but falls just short of the Tier 3 target ({self.target_score})."
            )
        elif ecapa_score >= 0.60:
            tier = "Tier 1: Approximate Timbre"
            verdict = (
                f"Approximate timbre match (ECAPA: {ecapa_score:.3f}). Vocal register and coarse formants are present, "
                "but noticeable acoustic drift is detectable."
            )
        else:
            tier = "Divergent: Low Likeness"
            verdict = (
                f"FAILED likeness match (ECAPA: {ecapa_score:.3f} < 0.60). The generated voice sounds like a different speaker. "
                "Check reference audio quality, background noise, or model tier."
            )

        return SpeakerSimilarityResult(
            ecapa_similarity=round(ecapa_score, 4),
            wavlm_similarity=round(wavlm_score, 4),
            composite_score=composite,
            target_met=target_met,
            target_threshold=self.target_score,
            likeness_tier=tier,
            verdict=verdict,
            reference_duration_sec=round(ref_dur, 2),
            candidate_duration_sec=round(cand_dur, 2),
            device_used=device_name
        )

    def compare_files(
        self,
        ref_path: Union[str, Path],
        cand_path: Union[str, Path]
    ) -> SpeakerSimilarityResult:
        """Loads two audio files and computes speaker likeness metrics."""
        ref_audio, sr_ref = AudioCleanupPipeline.load_audio(ref_path)
        cand_audio, sr_cand = AudioCleanupPipeline.load_audio(cand_path)
        return self.compute_similarity(ref_audio, cand_audio, sr_ref, sr_cand)


# Global singleton instance
speaker_similarity_evaluator = SpeakerSimilarityEvaluator()
