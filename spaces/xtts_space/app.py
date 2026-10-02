"""
Hugging Face Space Gradio Wrapper for XTTS-v2 (spaces/xtts_space/app.py)
------------------------------------------------------------------------
Deploy this script on a free Hugging Face Space (with zero-gpu or T4 GPU).
Exposes a clean Gradio API interface that EchoVoice backend calls via `gradio_client`.
"""

import os
import tempfile
from pathlib import Path
import gradio as gr
import torch
import soundfile as sf
import numpy as np

# Load XTTS-v2 model on Space startup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Coqui XTTS-v2 on device: {device}...")

try:
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("XTTS-v2 model loaded successfully!")
except Exception as e:
    tts = None
    print(f"Error loading TTS: {e}")


def predict(prompt: str, language: str, audio_file_pth: str, speed: float = 1.0):
    """
    Generates speech conditioned on audio_file_pth speaker timbre.
    """
    if tts is None:
        raise gr.Error("TTS model failed to load on this space.")

    if not audio_file_pth or not Path(audio_file_pth).exists():
        raise gr.Error("Valid reference audio file is required.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_wav = tmp.name

    tts.tts_to_file(
        text=prompt,
        speaker_wav=audio_file_pth,
        language=language.lower(),
        speed=float(speed),
        file_path=output_wav
    )

    return output_wav


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Text Prompt", value="Hello, this is EchoVoice speaking in your voice."),
        gr.Dropdown(
            label="Language",
            choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"],
            value="en"
        ),
        gr.Audio(label="Reference Voice Audio", type="filepath"),
        gr.Slider(label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1),
    ],
    outputs=gr.Audio(label="Synthesized Cloned Speech", type="filepath"),
    title="EchoVoice XTTS-v2 Zero-Shot Voice Cloning API",
    description="Free serverless voice cloning worker for the EchoVoice real-time conversational agent."
)

if __name__ == "__main__":
    demo.queue().launch()
