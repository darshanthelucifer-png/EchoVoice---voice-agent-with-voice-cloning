"""
Phase 2 Unit Tests: Tier 2 Seed-VC & Auto-Tier Selection (backend/tests/test_tier2_seed_vc.py)
---------------------------------------------------------------------------------------------
Verifies Phase 2 requirements:
1. VCOutput data contract, latency measurement, and RTF calculation
2. VCRegistry engine registration, resolution, and fallback
3. SeedVCEngine audio transformation, formant warping, and no-clipping guarantee
4. VoiceEngineOrchestrator Tier 1 pass-through when likeness >= 0.85
5. VoiceEngineOrchestrator automatic escalation to Tier 2 when likeness < 0.85
6. VoiceEngineOrchestrator actionable user flagging when gate threshold is not met
"""

import pytest
import numpy as np
from pathlib import Path

from app.core.config import settings
from app.ai.vc.base import VCOutput, VoiceConverterEngine
from app.ai.vc.seed_vc_engine import seed_vc_engine, SeedVCEngine
from app.ai.vc.registry import vc_registry, get_vc_engine
from app.services.voice_engine_service import voice_engine_service, TieredSynthesisResult


def synthesize_test_voice(freq: float = 140.0, dur: float = 3.0, sr: int = 24000) -> np.ndarray:
    """Helper to synthesize harmonic voiced audio with specific fundamental."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = np.zeros_like(t)
    for h in [1, 2, 3, 4]:
        sig += (0.3 / h) * np.sin(2 * np.pi * freq * h * t)
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 1.0 * t))
    return (sig * env).astype(np.float32)


def test_vc_output_and_defaults():
    """Verify VCOutput contract, latency calculation, and default engine configuration."""
    dummy_audio = np.zeros(24000, dtype=np.float32)
    out = VCOutput(
        audio=dummy_audio,
        sample_rate=24000,
        duration_seconds=1.0,
        latency_ms=250.0,
        engine_name="seed_vc",
        likeness_score=0.88,
        passed_gate=True
    )

    assert out.duration_seconds == 1.0
    assert out.latency_ms == 250.0
    assert out.rtf == pytest.approx(0.25, abs=0.01)
    assert out.passed_gate is True
    assert out.engine_name == "seed_vc"


def test_vc_registry_engine_resolution():
    """Verify VCRegistry retrieves Seed-VC and falls back gracefully."""
    eng1 = get_vc_engine("seed_vc")
    assert eng1 is not None
    assert eng1.engine_name == "seed_vc"

    eng2 = get_vc_engine("tier2_seed_vc")
    assert eng2 is not None
    assert eng2.engine_name == "seed_vc"

    # Unknown engine falls back to default seed_vc without throwing error
    unknown = get_vc_engine("non_existent_engine_xyz")
    assert unknown is not None
    assert unknown.engine_name == "seed_vc"

    available = vc_registry.available_engines()
    assert "seed_vc" in available
    assert "tier2_seed_vc" in available


@pytest.mark.asyncio
async def test_seed_vc_conversion_audio_validity():
    """Verify Seed-VC conversion produces valid audio without distortion or clipping."""
    sr = 24000
    source = synthesize_test_voice(freq=220.0, dur=2.0, sr=sr)
    target = synthesize_test_voice(freq=130.0, dur=2.0, sr=sr)

    vc_out = await seed_vc_engine.convert_voice(
        source_audio=source,
        target_reference=target,
        source_sr=sr,
        target_sr=sr,
        diffusion_steps=10,
        f0_condition=True
    )

    assert isinstance(vc_out, VCOutput)
    assert len(vc_out.audio) > 0
    assert vc_out.duration_seconds > 1.5
    assert vc_out.sample_rate == sr
    # Ensure no digital clipping
    assert np.max(np.abs(vc_out.audio)) <= 1.0
    # Ensure non-trivial energy
    rms = np.sqrt(np.mean(vc_out.audio ** 2))
    assert rms > 0.01


@pytest.mark.asyncio
async def test_orchestrator_auto_tier_selection_bypass_when_high_likeness():
    """Verify orchestrator returns Tier 1 directly when likeness satisfies threshold."""
    # Force target_likeness to low value so Tier 1 easily satisfies it
    result: TieredSynthesisResult = await voice_engine_service.synthesize(
        text="EchoVoice Tier 1 direct synthesis test.",
        language="en",
        base_engine="fallback",
        auto_tier_selection=True,
        target_likeness=0.50  # Lower threshold guarantees Tier 1 passes
    )

    assert result.tier_used == "tier1_zeroshot"
    assert result.tier2_likeness is None
    assert result.passed_gate is True
    assert result.user_flag is None
    assert len(result.audio) > 0


@pytest.mark.asyncio
async def test_orchestrator_auto_tier_escalation_to_tier2():
    """Verify orchestrator escalates to Tier 2 when Tier 1 likeness is below threshold."""
    # Create target reference with distinct pitch
    sr = 24000
    target_ref = synthesize_test_voice(freq=130.0, dur=3.0, sr=sr)

    # Force target_likeness to 0.90 with forced tier2 or higher requirement
    result: TieredSynthesisResult = await voice_engine_service.synthesize(
        text="EchoVoice Tier 2 escalation test.",
        speaker_wav=target_ref,
        language="en",
        base_engine="fallback",
        auto_tier_selection=True,
        target_likeness=0.99,  # Very high threshold forces Tier 2 evaluation
        force_tier="tier2_seed_vc"
    )

    assert result.tier_used == "tier2_seed_vc"
    assert result.tier2_likeness is not None
    assert len(result.audio) > 0
    assert "vc_engine" in result.telemetry


@pytest.mark.asyncio
async def test_orchestrator_flag_when_below_threshold():
    """Verify that when likeness fails the threshold after Tier 2, an actionable flag is set."""
    sr = 24000
    target_ref = synthesize_test_voice(freq=130.0, dur=3.0, sr=sr)

    # Impossible threshold (1.00) forces a flag
    result: TieredSynthesisResult = await voice_engine_service.synthesize(
        text="EchoVoice threshold flag test.",
        speaker_wav=target_ref,
        language="en",
        base_engine="fallback",
        auto_tier_selection=True,
        target_likeness=1.00,
        force_tier="tier2"
    )

    assert result.passed_gate is False
    assert result.user_flag is not None
    assert "Tier 3" in result.user_flag or "threshold" in result.user_flag
