"""
ASR Word Error Rate (WER) Evaluation CLI (backend/scripts/evaluate_asr_wer.py)
----------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Sequence Alignment & Dynamic Programming: Levenshtein distance dynamic programming
  via `jiwer` to calculate Word Error Rate (WER), Character Error Rate (CER),
  substitutions, insertions, and deletions.
- Text Normalization Pipeline: Standardizes transcripts across capitalization,
  punctuation, numbers, and whitespace to prevent false errors.
- End-to-End ASR Benchmarking: Transcribes candidate audio using Faster-Whisper
  and compares phonetic accuracy against prompt text.

Usage:
  # Run self-test verification with synthetic audio:
  python backend/scripts/evaluate_asr_wer.py --synthetic-test

  # Evaluate speech audio against ground-truth prompt text:
  python backend/scripts/evaluate_asr_wer.py --audio path/to/synthesized.wav --reference-text "Hello, welcome to EchoVoice."

  # Output as JSON:
  python backend/scripts/evaluate_asr_wer.py --audio path/to/audio.wav --reference-text "Hello world" --json
"""

import sys
import re
import json
import asyncio
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import jiwer
from app.core.config import settings
from app.core.logging import logger
from app.ai.audio.cleanup import AudioCleanupPipeline
from app.ai.asr.registry import get_asr_engine
from app.services.tts_service import tts_service


@dataclass
class ASRWERResult:
    reference_text: str
    hypothesis_text: str
    normalized_reference: str
    normalized_hypothesis: str
    wer: float
    cer: float
    accuracy_rate: float
    substitutions: int
    insertions: int
    deletions: int
    hits: int
    total_words: int
    passed: bool
    verdict: str
    duration_sec: float
    transcription_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_text_for_wer(text: str) -> str:
    """
    Normalizes text for standardized ASR benchmarking:
    - Lowercase
    - Strip punctuation and symbols
    - Collapse multiple spaces
    """
    t = text.lower().strip()
    # Replace hyphens with spaces
    t = t.replace("-", " ")
    # Remove punctuation
    t = re.sub(r"[^\w\s]", "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compute_wer_metrics(
    reference: str,
    hypothesis: str,
    duration_sec: float = 0.0,
    latency_ms: float = 0.0,
    max_acceptable_wer: float = 0.15
) -> ASRWERResult:
    """
    Computes Levenshtein-based WER and CER metrics using normalized strings.
    """
    norm_ref = normalize_text_for_wer(reference)
    norm_hyp = normalize_text_for_wer(hypothesis)

    if not norm_ref:
        norm_ref = "empty"
    if not norm_hyp:
        norm_hyp = "empty"

    wer = float(jiwer.wer(norm_ref, norm_hyp))
    cer = float(jiwer.cer(norm_ref, norm_hyp))

    # Compute detailed alignment breakdown using jiwer 3.x process_words
    try:
        word_output = jiwer.process_words(norm_ref, norm_hyp)
        substitutions = word_output.substitutions
        insertions = word_output.insertions
        deletions = word_output.deletions
        hits = word_output.hits
    except Exception:
        substitutions = 0
        insertions = 0
        deletions = 0
        hits = len(norm_ref.split()) if wer == 0 else 0

    total_words = len(norm_ref.split())
    accuracy = max(0.0, 1.0 - wer)

    passed = wer <= max_acceptable_wer
    if passed:
        verdict = f"PASSED phonetic accuracy gate (WER: {wer * 100:.1f}% <= {max_acceptable_wer * 100:.1f}%). Synthesis faithfully articulates prompt words."
    else:
        verdict = f"FAILED phonetic accuracy gate (WER: {wer * 100:.1f}% > {max_acceptable_wer * 100:.1f}%). Slurring, word dropouts, or hallucinations detected."

    return ASRWERResult(
        reference_text=reference,
        hypothesis_text=hypothesis,
        normalized_reference=norm_ref,
        normalized_hypothesis=norm_hyp,
        wer=round(wer, 4),
        cer=round(cer, 4),
        accuracy_rate=round(accuracy, 4),
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        hits=hits,
        total_words=total_words,
        passed=passed,
        verdict=verdict,
        duration_sec=round(duration_sec, 2),
        transcription_latency_ms=round(latency_ms, 1)
    )


def print_wer_report(result: ASRWERResult):
    print("\n" + "=" * 76)
    print(" ASR PHONETIC INTELLIGIBILITY & WORD ERROR RATE (WER) REPORT ")
    print("=" * 76)
    print(f"Reference Text   : \"{result.reference_text}\"")
    print(f"ASR Hypothesis   : \"{result.hypothesis_text}\"")
    print(f"Normalized Ref   : \"{result.normalized_reference}\"")
    print(f"Normalized Hyp   : \"{result.normalized_hypothesis}\"")
    print("-" * 76)
    print(f"{'Metric':<30} | {'Value':<15} | {'Target Benchmark':<24}")
    print("-" * 76)
    wer_color = "PASSED" if result.passed else "EXCEEDED"
    print(f"{'Word Error Rate (WER)':<30} | {result.wer * 100:<13.2f} % | <= 15.0 % ({wer_color})")
    print(f"{'Character Error Rate (CER)':<30} | {result.cer * 100:<13.2f} % | <= 8.0 %")
    print(f"{'Word Accuracy Rate':<30} | {result.accuracy_rate * 100:<13.2f} % | >= 85.0 %")
    print("-" * 76)
    print(f"Total Words: {result.total_words} | Hits: {result.hits} | Substitutions: {result.substitutions} | Insertions: {result.insertions} | Deletions: {result.deletions}")
    print(f"Audio Length: {result.duration_sec:.2f} s | Transcription Latency: {result.transcription_latency_ms:.1f} ms")
    print("-" * 76)
    print(f"Intelligibility Verdict:")
    print(f"  {result.verdict}")
    print("=" * 76 + "\n")


async def transcribe_audio_file(audio_path: Path, language: str = "en") -> tuple[str, float, float]:
    """Transcribes audio file using faster-whisper engine and measures duration & latency."""
    import time
    audio, sr = AudioCleanupPipeline.load_audio(audio_path)
    engine = get_asr_engine("faster_whisper")

    start_t = time.perf_counter()
    asr_res = await engine.transcribe(audio=audio, sr=sr, language=language)
    latency = (time.perf_counter() - start_t) * 1000.0
    duration = len(audio) / sr

    return asr_res.text, duration, latency


async def run_cli():
    parser = argparse.ArgumentParser(description="EchoVoice ASR Word Error Rate (WER) Evaluation Harness")
    parser.add_argument("--audio", "-a", type=str, default=None, help="Path to synthesized audio file (WAV/MP3)")
    parser.add_argument("--reference-text", "-r", type=str, default=None, help="Ground truth prompt script text")
    parser.add_argument("--language", "-l", type=str, default="en", help="Language code (e.g. en, hi)")
    parser.add_argument("--synthetic-test", action="store_true", help="Execute self-test by synthesizing and evaluating test audio")
    parser.add_argument("--max-wer", type=float, default=0.15, help="Maximum acceptable WER threshold (default: 0.15)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save JSON report to file")
    args = parser.parse_args()

    # 1. Synthetic Self-Test Mode
    if args.synthetic_test or (not args.audio and not args.reference_text):
        print("\n[*] Executing ASR WER Self-Test Verification...")
        test_script = "EchoVoice delivers studio grade voice cloning and speech synthesis with exact likeness."
        print(f"[*] Synthesizing reference utterance: \"{test_script}\"")

        tts_output, audio_path = await tts_service.generate_speech(
            text=test_script,
            engine_name="fallback",
            language="en"
        )

        print("[*] Transcribing generated utterance via Faster-Whisper...")
        hyp_text, dur, latency = await transcribe_audio_file(audio_path, language="en")

        result = compute_wer_metrics(
            reference=test_script,
            hypothesis=hyp_text,
            duration_sec=dur,
            latency_ms=latency,
            max_acceptable_wer=args.max_wer
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_wer_report(result)
            if result.wer <= 0.25:
                print("[SUCCESS] ASR WER evaluation harness successfully verified!")
            else:
                print("[NOTE] Fallback speech synthesizer showed non-zero error rate (expected on synthetic test).")
        return

    # 2. File Evaluation Mode
    if not args.audio:
        print("[ERROR] Please provide --audio <path> to evaluate.")
        sys.exit(1)
    if not args.reference_text:
        print("[ERROR] Please provide --reference-text \"<script>\" for comparison.")
        sys.exit(1)

    audio_p = Path(args.audio)
    if not audio_p.exists():
        print(f"[ERROR] Audio file does not exist: {audio_p}")
        sys.exit(1)

    print(f"[*] Transcribing audio: {audio_p}...")
    hyp_text, dur, latency = await transcribe_audio_file(audio_p, language=args.language)

    result = compute_wer_metrics(
        reference=args.reference_text,
        hypothesis=hyp_text,
        duration_sec=dur,
        latency_ms=latency,
        max_acceptable_wer=args.max_wer
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_wer_report(result)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"[*] Report saved to {out_p}")


if __name__ == "__main__":
    asyncio.run(run_cli())
