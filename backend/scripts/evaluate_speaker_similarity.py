"""
Speaker Similarity Evaluation CLI (backend/scripts/evaluate_speaker_similarity.py)
---------------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Command-Line Interface (argparse) with dual operation modes (file evaluation and synthetic benchmark).
- Biometric Speaker Verification: Evaluates cosine similarity of ECAPA-TDNN (SpeechBrain)
  and WavLM-SV (Microsoft) speaker embeddings against target likeness threshold (>= 0.85).
- Structured JSON & Human-Readable Console Reporting: Formats forensic similarity metrics,
  visual meters, and tier diagnostics for developer evaluation and automated CI/CD pipelines.

Usage:
  # Run self-test verification with synthetic audio clips:
  python backend/scripts/evaluate_speaker_similarity.py --synthetic-test

  # Evaluate a synthesized clip against your real voice reference:
  python backend/scripts/evaluate_speaker_similarity.py --reference path/to/my_voice.wav --candidate path/to/cloned_sample.wav

  # Output as JSON:
  python backend/scripts/evaluate_speaker_similarity.py --reference path/to/ref.wav --candidate path/to/cand.wav --json
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.ai.audio.similarity import speaker_similarity_evaluator, SpeakerSimilarityResult
from app.ai.audio.cleanup import AudioCleanupPipeline


def generate_synthetic_voice_pair(sr: int = 16_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesizes three audio signals for benchmark validation:
    1. Base voice (Speaker A)
    2. Identical voice with light gain variation (Speaker A duplicate - should score > 0.95)
    3. Different pitch/formant register voice (Speaker B - should score < 0.60)
    """
    dur = 4.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # Base Speaker A (F0 = 130 Hz)
    sig_a = np.zeros_like(t)
    for h in range(1, 14):
        sig_a += (1.0 / (h ** 0.8)) * np.sin(2 * np.pi * 130.0 * h * t)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    sig_a = (sig_a * envelope * 0.3).astype(np.float32)

    # Speaker A duplicate (identical timbre, minor phase & amplitude offset)
    sig_a_dup = (sig_a * 0.95 + 0.001 * np.random.normal(0, 1, len(sig_a))).astype(np.float32)

    # Speaker B (F0 = 240 Hz, altered harmonic distribution - different speaker)
    sig_b = np.zeros_like(t)
    for h in range(1, 14):
        sig_b += (1.0 / (h ** 1.1)) * np.sin(2 * np.pi * 240.0 * h * t)
    sig_b = (sig_b * envelope * 0.3).astype(np.float32)

    return sig_a, sig_a_dup, sig_b


def print_score_meter(score: float, width: int = 30) -> str:
    """Renders a visual ASCII progress meter for similarity percentage."""
    filled = int(round(score * width))
    filled = max(0, min(width, filled))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {score * 100:5.1f}%"


def print_similarity_report(result: SpeakerSimilarityResult, title: str = "SPEAKER SIMILARITY EVALUATION REPORT"):
    print("\n" + "=" * 76)
    print(f" {title} ")
    print("=" * 76)
    print(f"Reference Audio Duration : {result.reference_duration_sec:.2f} s")
    print(f"Candidate Audio Duration : {result.candidate_duration_sec:.2f} s")
    print(f"Inference Device         : {result.device_used.upper()}")
    print("-" * 76)
    print(f"{'Metric':<30} | {'Score':<10} | {'Visual Meter':<32}")
    print("-" * 76)
    print(f"{'ECAPA-TDNN Cosine Score':<30} | {result.ecapa_similarity:<10.4f} | {print_score_meter(result.ecapa_similarity)}")
    print(f"{'WavLM-SV Cosine Score':<30} | {result.wavlm_similarity:<10.4f} | {print_score_meter(result.wavlm_similarity)}")
    print(f"{'Composite Likeness Score':<30} | {result.composite_score:<10.4f} | {print_score_meter(result.composite_score)}")
    print("-" * 76)
    print(f"Target Pass Threshold    : {result.target_threshold:.2f} (ECAPA-TDNN)")
    status_str = "PASSED [MATCH]" if result.target_met else "FAILED [DRIFT DETECTED]"
    print(f"Likeness Gate Status     : {status_str}")
    print(f"Assigned Likeness Tier   : {result.likeness_tier}")
    print("-" * 76)
    print(f"Diagnosis Verdict:")
    print(f"  {result.verdict}")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="EchoVoice Speaker Likeness Evaluation Harness")
    parser.add_argument("--reference", "-r", type=str, default=None, help="Path to ground truth reference voice (WAV/MP3)")
    parser.add_argument("--candidate", "-c", type=str, default=None, help="Path to synthesized or converted voice (WAV/MP3)")
    parser.add_argument("--threshold", "-t", type=float, default=settings.SIMILARITY_TARGET_SCORE, help="Target ECAPA similarity (default: 0.85)")
    parser.add_argument("--synthetic-test", action="store_true", help="Execute self-test evaluation with synthetic test audio")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save JSON report to specified file path")
    args = parser.parse_args()

    # 1. Synthetic Self-Test Mode
    if args.synthetic_test or (not args.reference and not args.candidate):
        print("\n[*] Executing Speaker Similarity Self-Test Benchmark...")
        sr = 16000
        sig_a, sig_a_dup, sig_b = generate_synthetic_voice_pair(sr=sr)

        print("[1/2] Evaluating Same Speaker Identity (Speaker A vs. Speaker A Duplicate)...")
        same_result = speaker_similarity_evaluator.compute_similarity(sig_a, sig_a_dup, sr, sr)

        print("[2/2] Evaluating Different Speaker Identity (Speaker A vs. Speaker B)...")
        diff_result = speaker_similarity_evaluator.compute_similarity(sig_a, sig_b, sr, sr)

        if not args.json:
            print_similarity_report(same_result, title="BENCHMARK 1: SAME-SPEAKER VERIFICATION (A vs A')")
            print_similarity_report(diff_result, title="BENCHMARK 2: CROSS-SPEAKER DISCRIMINATION (A vs B)")

        test_passed = same_result.ecapa_similarity >= args.threshold and diff_result.ecapa_similarity < 0.60
        summary = {
            "same_speaker": same_result.to_dict(),
            "different_speaker": diff_result.to_dict(),
            "self_test_passed": test_passed
        }

        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            if test_passed:
                print("[SUCCESS] Speaker Similarity Harness successfully verified! High discriminatory power confirmed.")
            else:
                print("[WARNING] Verification scores fell outside expected theoretical boundaries.")
        return

    # 2. File Evaluation Mode
    ref_path = Path(args.reference)
    cand_path = Path(args.candidate)

    if not ref_path.exists():
        print(f"[ERROR] Reference audio not found at: {ref_path}")
        sys.exit(1)
    if not cand_path.exists():
        print(f"[ERROR] Candidate audio not found at: {cand_path}")
        sys.exit(1)

    evaluator = speaker_similarity_evaluator
    if args.threshold != settings.SIMILARITY_TARGET_SCORE:
        evaluator.target_score = args.threshold

    result = evaluator.compare_files(ref_path, cand_path)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_similarity_report(result)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"[*] Report saved to {out_p}")


if __name__ == "__main__":
    main()
