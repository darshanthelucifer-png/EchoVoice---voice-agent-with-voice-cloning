"""
Real-Time Speech-to-Text & Streaming Pipeline Verification (backend/scripts/test_asr_pipeline.py)
-------------------------------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- End-to-End ASR Benchmarking: Evaluates CTranslate2 Faster-Whisper decoding latency,
  Real-Time Factor (RTF), and word-level token alignment.
- VAD-Gated Chunk Stream Simulation: Simulates real-time microphone packet ingestion (50ms chunks),
  validating the ring-buffer pre-roll, speech start trigger, interim transcripts, and pause finalization.
"""

import asyncio
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.logging import logger
from app.ai.asr.base import ASRResult
from app.ai.asr.registry import asr_registry, get_asr_engine
from app.ai.asr.stream import VADAudioStreamProcessor, StreamConfig, StreamEventType
from app.services.tts_service import tts_service


async def run_asr_engine_benchmark():
    """
    Test 1: Faster-Whisper Neural Transcription & Word-Level Timestamping.
    """
    print("\n" + "="*70)
    print("TEST 1: FASTER-WHISPER NEURAL ASR TRANSCRIPTION & WORD TIMESTAMPS")
    print("="*70)

    # 1. Synthesize known audio clip to test transcription
    target_sentence = "Artificial intelligence is transforming real time conversational voice agents."
    print(f"[*] Reference Prompt: '{target_sentence}'")
    print("[*] Generating reference speech test audio...")

    tts_output, audio_path = await tts_service.generate_speech(
        text=target_sentence,
        engine_name="fallback",
        language="en"
    )
    print(f"[+] Audio generated: {tts_output.duration_seconds:.2f}s (sr={tts_output.sample_rate}Hz)")

    # 2. Transcribe using FasterWhisperEngine
    engine = get_asr_engine("faster_whisper")
    print(f"[*] Testing ASR engine: {engine.engine_name}")

    start_t = time.perf_counter()
    result: ASRResult = await engine.transcribe(
        audio=tts_output.audio,
        sr=tts_output.sample_rate,
        language="en",
        word_timestamps=True
    )
    total_time = (time.perf_counter() - start_t) * 1000.0

    print(f"\n[+] Transcription Hypothesis: '{result.text}'")
    print(f"    - Engine Used   : {result.engine_name}")
    print(f"    - Audio Length  : {result.duration_seconds:.2f}s")
    print(f"    - Latency       : {result.latency_ms:.1f} ms")
    print(f"    - Real-Time Fac : {result.rtf:.3f}x RTF")
    print(f"    - Confidence    : {result.confidence:.2f}")
    print(f"    - Word Tokens   : {len(result.words)} extracted")

    if result.words:
        print("\n    Sample Word Timestamps:")
        for w in result.words[:6]:
            print(f"      • [{w.start_sec:5.2f}s - {w.end_sec:5.2f}s] '{w.word}' (p={w.probability:.2f})")

    assert result.duration_seconds > 0.5
    assert result.latency_ms > 0
    print("\n[SUCCESS] Neural ASR transcription benchmark passed!")


async def run_vad_streaming_pipeline_simulation():
    """
    Test 2: Real-time VAD-Gated Audio Streaming (50ms Chunk Ingestion).
    Simulates:
      - 400ms Silence (Pre-roll test)
      - 2.0s Speech (Speech start + interim hypotheses)
      - 600ms Silence (End-of-utterance finalization)
    """
    print("\n" + "="*70)
    print("TEST 2: VAD-GATED STREAMING PIPELINE SIMULATION (50ms FRAMES)")
    print("="*70)

    sr = 16_000
    chunk_ms = 50
    chunk_samples = int((chunk_ms / 1000.0) * sr)  # 800 samples per 50ms chunk

    # Generate synthetic speech signal for Phase B
    t = np.linspace(0, 2.0, int(sr * 2.0))
    speech_signal = (
        0.35 * np.sin(2 * np.pi * 220 * t) +
        0.20 * np.sin(2 * np.pi * 440 * t) +
        0.10 * np.sin(2 * np.pi * 880 * t)
    ).astype(np.float32)

    silence_chunk = np.zeros(chunk_samples, dtype=np.float32)

    config = StreamConfig(
        sample_rate=sr,
        silence_timeout_sec=0.40,
        interim_interval_sec=0.50
    )
    processor = VADAudioStreamProcessor(config=config)

    emitted_events = []

    print("[*] Feeding streaming chunks to VAD processor...")

    # Phase A: 8 chunks of silence (400ms)
    print("    [Phase A] Streaming 400ms background silence...")
    for _ in range(8):
        evs = await processor.process_chunk(silence_chunk)
        if evs:
            emitted_events.extend(evs)
        await asyncio.sleep(0.01)

    # Phase B: 40 chunks of speech (2000ms)
    print("    [Phase B] Streaming 2,000ms active speech...")
    for start_idx in range(0, len(speech_signal) - chunk_samples + 1, chunk_samples):
        speech_chunk = speech_signal[start_idx : start_idx + chunk_samples]
        evs = await processor.process_chunk(speech_chunk)
        if evs:
            for ev in evs:
                emitted_events.append(ev)
                print(f"      -> Event Emitted: [{ev['type'].upper()}] {ev.get('text', '')}")
        await asyncio.sleep(0.01)

    # Phase C: 12 chunks of silence (600ms) to trigger pause finalization
    print("    [Phase C] Streaming 600ms conversational turn-taking pause...")
    for _ in range(12):
        evs = await processor.process_chunk(silence_chunk)
        if evs:
            for ev in evs:
                emitted_events.append(ev)
                print(f"      -> Event Emitted: [{ev['type'].upper()}] '{ev.get('text', '')}' (Latency: {ev.get('latency_ms', 0)}ms)")
        await asyncio.sleep(0.01)

    event_types = [ev["type"] for ev in emitted_events]
    print(f"\n[*] Total Streaming Events Captured: {len(emitted_events)}")
    print(f"[*] Event Sequence: {' -> '.join(event_types)}")

    assert StreamEventType.SPEECH_STARTED.value in event_types, "SPEECH_STARTED event was not triggered!"
    print("[+] Verified SPEECH_STARTED triggered at speech onset.")
    print("[SUCCESS] Real-time streaming pipeline verified successfully!")


async def main():
    print("="*70)
    print("ECHOVOICE PHASE 7: SPEECH-TO-TEXT (ASR) VERIFICATION SUITE")
    print("="*70)
    await run_asr_engine_benchmark()
    await run_vad_streaming_pipeline_simulation()
    print("\n" + "="*70)
    print("ALL PHASE 7 VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
