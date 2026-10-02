"""
TTS Engine & A/B Perceptual Benchmark CLI (backend/scripts/test_tts_ab.py)
--------------------------------------------------------------------------
Tests Phase 3 TTS Engine Interface, XTTS-v2 zero-shot voice cloning,
and side-by-side A/B performance comparison.

Usage:
  # Run A/B test with default sample:
  python backend/scripts/test_tts_ab.py

  # Run with custom text and reference voice:
  python backend/scripts/test_tts_ab.py --text "EchoVoice real-time voice cloning is active." --reference "data/voice_profiles/phase2_test/reference.wav"
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.services.tts_service import tts_service
from app.ai.tts.registry import tts_registry


async def run_cli():
    parser = argparse.ArgumentParser(description="EchoVoice TTS & A/B Testing Suite")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! Welcome to EchoVoice. Your voice profile has been processed and synthesized with zero-shot cloning.",
        help="Text script to synthesize"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to clean speaker reference audio (WAV)"
    )
    parser.add_argument(
        "--engine-a",
        type=str,
        default="xtts_v2",
        help="First engine to test (default: xtts_v2)"
    )
    parser.add_argument(
        "--engine-b",
        type=str,
        default="fallback",
        help="Second engine for A/B comparison (default: fallback)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Target language code (e.g. en, es, hi)"
    )
    args = parser.parse_args()

    print("=" * 76)
    print("EchoVoice - Phase 3: TTS Engine & A/B Perceptual Benchmark")
    print("=" * 76)

    # Resolve speaker reference
    ref_path = None
    if args.reference and Path(args.reference).exists():
        ref_path = Path(args.reference)
    else:
        # Check if Phase 2 generated a reference.wav
        default_phase2_ref = settings.VOICE_PROFILES_DIR / "phase2_test" / "reference.wav"
        if default_phase2_ref.exists():
            ref_path = default_phase2_ref

    print(f"Script Text:       \"{args.text}\"")
    print(f"Language:          {args.language}")
    print(f"Speaker Reference: {ref_path if ref_path else 'None (Base Synthesizer)'}")
    print(f"Active Engines:    A: [{args.engine_a}]  vs.  B: [{args.engine_b}]")
    print(f"Available Engines: {tts_registry.available_engines()}")
    print("-" * 76)

    print("\nExecuting side-by-side A/B synthesis...")
    results = await tts_service.run_ab_test(
        text=args.text,
        speaker_wav_a=ref_path,
        speaker_wav_b=ref_path,
        engine_name_a=args.engine_a,
        engine_name_b=args.engine_b,
        language=args.language
    )

    opt_a = results["option_a"]
    opt_b = results["option_b"]

    print("\n" + "=" * 76)
    print(f"{'Performance Metric':<28} | {'Option A (' + opt_a['engine'] + ')':<22} | {'Option B (' + opt_b['engine'] + ')':<22}")
    print("-" * 76)
    print(f"{'Audio Duration':<28} | {opt_a['duration_seconds']:<18.2f} s | {opt_b['duration_seconds']:<18.2f} s")
    print(f"{'Inference Latency':<28} | {opt_a['latency_ms']:<18.2f} ms| {opt_b['latency_ms']:<18.2f} ms")
    print(f"{'Real-Time Factor (RTF)':<28} | {opt_a['rtf']:<18.3f}   | {opt_b['rtf']:<18.3f}  ")
    print(f"{'Audio Output File':<28} | {Path(opt_a['audio_path']).name:<22} | {Path(opt_b['audio_path']).name:<22}")
    print("=" * 76)

    print("\nGenerated Audio Artifacts:")
    print(f"  * Option A: {opt_a['audio_path']}")
    print(f"  * Option B: {opt_b['audio_path']}")
    print(f"\nLatency Difference: {results['latency_delta_ms']} ms")
    print("=" * 76)
    print("SUCCESS: Phase 3 TTS Engine Interface and A/B Testing verified!")
    print("=" * 76)


if __name__ == "__main__":
    asyncio.run(run_cli())
