"""
VAD-Gated Real-Time Audio Streaming Pipeline (backend/app/ai/asr/stream.py)
--------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- State Machine Architecture: Transitions between IDLE, SPEAKING, and PAUSE states
  based on neural Silero VAD probability scores.
- Ring Buffer Pre-Roll: Keeps a 200ms circular pre-roll history to prevent clipping
  the initial plosive consonants (e.g. 'P', 'T', 'K') of spoken phrases.
- Low-Latency Interim & Final Transcript Dispatching: Yields async event dicts
  suitable for WebSocket streaming directly to the client.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import AsyncGenerator, Dict, List, Optional, Any
import numpy as np

from app.core.logging import logger
from app.ai.audio.vad import vad_processor
from app.ai.asr.base import ASREngine, ASRResult
from app.ai.asr.registry import get_asr_engine


class StreamEventType(str, Enum):
    SPEECH_STARTED = "speech_started"
    INTERIM = "interim"
    FINAL = "final"
    SILENCE = "silence"


@dataclass
class StreamConfig:
    """Config for streaming VAD and transcription behavior."""
    sample_rate: int = 16_000
    vad_threshold: float = 0.50
    silence_timeout_sec: float = 0.45       # 450ms pause triggers end-of-utterance finalization
    min_speech_duration_sec: float = 0.30   # Minimum speech length to transcribe
    max_speech_duration_sec: float = 25.0   # Maximum utterance buffer before forced finalization
    interim_interval_sec: float = 0.60      # How often to emit partial hypotheses while speaking
    pre_roll_sec: float = 0.20              # 200ms pre-roll audio buffer


class VADAudioStreamProcessor:
    """
    Consumes streaming audio chunks, evaluates voice activity via Silero VAD,
    and coordinates low-latency interim and final speech recognition events.
    """

    def __init__(
        self,
        engine: Optional[ASREngine] = None,
        config: Optional[StreamConfig] = None
    ):
        self.engine = engine or get_asr_engine()
        self.config = config or StreamConfig()

        self.sr = self.config.sample_rate
        self.is_speaking = False
        self.silence_start_time: Optional[float] = None
        self.silence_samples: int = 0
        self.last_interim_time: float = 0.0

        # Audio Buffers
        self.speech_frames: List[np.ndarray] = []
        self.pre_roll_buffer: List[np.ndarray] = []
        self.pre_roll_max_samples = int(self.config.pre_roll_sec * self.sr)
        self.pre_roll_current_samples = 0

    def _push_pre_roll(self, chunk: np.ndarray) -> None:
        """Maintains fixed-size pre-roll history before speech trigger."""
        self.pre_roll_buffer.append(chunk)
        self.pre_roll_current_samples += len(chunk)

        while self.pre_roll_current_samples > self.pre_roll_max_samples and len(self.pre_roll_buffer) > 1:
            popped = self.pre_roll_buffer.pop(0)
            self.pre_roll_current_samples -= len(popped)

    async def process_chunk(
        self,
        chunk: np.ndarray,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes an incoming 1D float32 audio chunk (e.g. 50ms - 100ms).
        Returns a list of generated streaming event dictionaries.
        """
        events: List[Dict[str, Any]] = []

        if len(chunk) == 0:
            return events

        # 1. Evaluate Voice Activity
        chunk_is_speech = vad_processor.is_speech(chunk, self.sr)

        # 2. State Machine: Not Speaking (IDLE / SILENCE)
        if not self.is_speaking:
            if chunk_is_speech:
                # Transition to SPEAKING
                self.is_speaking = True
                self.silence_start_time = None
                self.last_interim_time = time.perf_counter()

                # Prepend pre-roll buffer so initial syllable is preserved
                if self.pre_roll_buffer:
                    self.speech_frames = list(self.pre_roll_buffer)
                    self.pre_roll_buffer.clear()
                    self.pre_roll_current_samples = 0
                else:
                    self.speech_frames = []

                self.speech_frames.append(chunk)
                events.append({
                    "type": StreamEventType.SPEECH_STARTED.value,
                    "timestamp": time.time()
                })
            else:
                # Still silent: update pre-roll buffer
                self._push_pre_roll(chunk)

        # 3. State Machine: Currently Speaking
        else:
            self.speech_frames.append(chunk)
            now = time.perf_counter()
            current_speech_samples = sum(len(f) for f in self.speech_frames)
            current_duration_sec = current_speech_samples / float(self.sr)

            if chunk_is_speech:
                self.silence_start_time = None
                self.silence_samples = 0

                # Check if it is time to emit an INTERIM hypothesis
                if (now - self.last_interim_time) >= self.config.interim_interval_sec and current_duration_sec >= 0.5:
                    self.last_interim_time = now
                    combined = np.concatenate(self.speech_frames)
                    # Interim decode without word timestamps for maximum speed
                    interim_result = await self.engine.transcribe(
                        combined, sr=self.sr, language=language, word_timestamps=False
                    )
                    if not interim_result.is_empty:
                        events.append({
                            "type": StreamEventType.INTERIM.value,
                            "text": interim_result.text,
                            "duration_seconds": round(current_duration_sec, 2),
                            "timestamp": time.time()
                        })
            else:
                # User paused speaking: track silence duration (audio samples + wall clock)
                if self.silence_start_time is None:
                    self.silence_start_time = now
                self.silence_samples += len(chunk)

                wall_elapsed = now - self.silence_start_time
                audio_elapsed = self.silence_samples / float(self.sr)
                silence_elapsed = max(wall_elapsed, audio_elapsed)

                # Check if pause exceeded threshold OR maximum buffer length reached
                if (
                    silence_elapsed >= self.config.silence_timeout_sec
                    or current_duration_sec >= self.config.max_speech_duration_sec
                ):
                    # Finalize utterance
                    final_events = await self._finalize_utterance(language=language)
                    events.extend(final_events)

        return events

    async def _finalize_utterance(
        self,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Transcribes the accumulated speech buffer and emits the final transcription."""
        events: List[Dict[str, Any]] = []

        if not self.speech_frames:
            self.is_speaking = False
            self.silence_start_time = None
            self.silence_samples = 0
            return events

        combined = np.concatenate(self.speech_frames)
        total_duration = len(combined) / float(self.sr)

        # Reset states
        self.speech_frames.clear()
        self.is_speaking = False
        self.silence_start_time = None
        self.silence_samples = 0

        if total_duration >= self.config.min_speech_duration_sec:
            # Full transcription with word-level timestamps
            result: ASRResult = await self.engine.transcribe(
                combined,
                sr=self.sr,
                language=language,
                word_timestamps=True
            )

            if not result.is_empty:
                events.append({
                    "type": StreamEventType.FINAL.value,
                    "text": result.text,
                    "language": result.language,
                    "duration_seconds": result.duration_seconds,
                    "latency_ms": result.latency_ms,
                    "rtf": result.rtf,
                    "confidence": result.confidence,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start_sec,
                            "end": w.end_sec,
                            "probability": w.probability
                        }
                        for w in result.words
                    ],
                    "timestamp": time.time()
                })

        return events

    async def flush(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Forces immediate finalization of any pending speech."""
        return await self._finalize_utterance(language=language)

    def reset(self) -> None:
        """Resets all streaming buffers and VAD state flags."""
        self.speech_frames.clear()
        self.pre_roll_buffer.clear()
        self.pre_roll_current_samples = 0
        self.is_speaking = False
        self.silence_start_time = None
        self.silence_samples = 0
        self.last_interim_time = 0.0
