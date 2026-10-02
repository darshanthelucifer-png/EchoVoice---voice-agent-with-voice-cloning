"""
Faster-Whisper CTranslate2 ASR Engine (backend/app/ai/asr/faster_whisper_engine.py)
-----------------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- CTranslate2 INT8 Quantization: Runs Whisper up to 4x faster with 70% less memory usage
  compared to vanilla PyTorch Whisper.
- Thread Delegation (`asyncio.to_thread`): Offloads CPU-intensive neural decoding to a worker
  thread, keeping the ASGI event loop responsive for WebSocket streaming.
- Lazy Singleton Instantiation: Models are loaded into memory on first inference request.
- Polyphase Resampling: Resamples arbitrary input audio to 16 kHz required by Whisper.
"""

import asyncio
import math
from pathlib import Path
import time
from typing import List, Optional, Union
import numpy as np
from scipy import signal
import soundfile as sf
import torch

from app.core.config import settings
from app.core.logging import logger, timed_step
from app.ai.asr.base import ASREngine, ASRResult, ASRToken


class FasterWhisperEngine(ASREngine):
    """
    High-performance Whisper engine powered by CTranslate2.
    Supports INT8 CPU inference and FP16 CUDA acceleration.
    """

    engine_name: str = "faster_whisper"

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ):
        self.model_size = model_size or "tiny"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self._model = None
        self._lock = asyncio.Lock()

    def _ensure_model_loaded(self):
        """Loads CTranslate2 Whisper model into memory if not already initialized."""
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(
                f"Loading faster-whisper model '{self.model_size}' "
                f"on {self.device} (compute_type={self.compute_type})..."
            )
            start_t = time.perf_counter()
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4
            )
            load_ms = (time.perf_counter() - start_t) * 1000.0
            logger.info(f"faster-whisper '{self.model_size}' loaded in {load_ms:.1f} ms.")

    def _resample_to_16k(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Resamples input audio array to 16,000 Hz if needed."""
        if sr == 16_000:
            return audio

        gcd = math.gcd(sr, 16_000)
        up = 16_000 // gcd
        down = sr // gcd
        resampled = signal.resample_poly(audio, up, down).astype(np.float32)
        return resampled

    def _sync_transcribe(
        self,
        audio_16k: np.ndarray,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        word_timestamps: bool = True
    ) -> ASRResult:
        """Synchronous decoding called within thread pool."""
        self._ensure_model_loaded()

        start_time = time.perf_counter()
        duration_sec = len(audio_16k) / 16_000.0

        if duration_sec < 0.1:
            return ASRResult(
                text="",
                language=language or "en",
                duration_seconds=duration_sec,
                latency_ms=0.0,
                rtf=0.0,
                words=[],
                confidence=1.0,
                engine_name=self.engine_name
            )

        # Ensure float32 normalized [-1.0, 1.0]
        if audio_16k.dtype != np.float32:
            audio_16k = audio_16k.astype(np.float32)

        peak = np.max(np.abs(audio_16k))
        if peak > 1.0:
            audio_16k = audio_16k / peak

        # Decode segments using CTranslate2
        segments, info = self._model.transcribe(
            audio_16k,
            language=language,
            initial_prompt=prompt,
            word_timestamps=word_timestamps,
            vad_filter=False,  # Upstream Silero VAD handles speech gating
            beam_size=1,       # Greedy decoding for lowest conversational latency
            temperature=0.0    # Strict deterministic decoding to eliminate fallback temperature loops
        )

        full_text_parts: List[str] = []
        token_list: List[ASRToken] = []
        avg_probs: List[float] = []

        for segment in segments:
            full_text_parts.append(segment.text.strip())
            avg_probs.append(segment.avg_logprob)

            if word_timestamps and segment.words:
                for w in segment.words:
                    token_list.append(
                        ASRToken(
                            word=w.word.strip(),
                            start_sec=round(w.start, 3),
                            end_sec=round(w.end, 3),
                            probability=round(w.probability, 3)
                        )
                    )

        full_text = " ".join(full_text_parts).strip()
        elapsed_sec = time.perf_counter() - start_time
        latency_ms = elapsed_sec * 1000.0
        rtf = elapsed_sec / max(duration_sec, 1e-4)

        # Confidence heuristic from logprob
        mean_logprob = float(np.mean(avg_probs)) if avg_probs else 0.0
        confidence = float(np.clip(math.exp(mean_logprob), 0.0, 1.0))

        return ASRResult(
            text=full_text,
            language=info.language if hasattr(info, "language") else (language or "en"),
            duration_seconds=round(duration_sec, 3),
            latency_ms=round(latency_ms, 2),
            rtf=round(rtf, 3),
            words=token_list,
            confidence=round(confidence, 3),
            engine_name=f"{self.engine_name}-{self.model_size}"
        )

    @timed_step("Faster-Whisper Transcription")
    async def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16_000,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        word_timestamps: bool = True
    ) -> ASRResult:
        """Asynchronously transcribes audio off the event loop."""
        audio_16k = self._resample_to_16k(audio, sr)
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_transcribe,
                audio_16k=audio_16k,
                language=language,
                prompt=prompt,
                word_timestamps=word_timestamps
            )

    async def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        word_timestamps: bool = True
    ) -> ASRResult:
        """Loads and transcribes an audio file on disk."""
        data, sr = sf.read(str(file_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return await self.transcribe(
            data, sr=sr, language=language, word_timestamps=word_timestamps
        )
