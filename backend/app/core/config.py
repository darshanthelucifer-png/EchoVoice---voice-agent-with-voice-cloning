"""
Core Configuration Module (backend/app/core/config.py)
------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Pydantic v2 `BaseSettings` & Data Validation: Strictly validates configuration types
  at runtime, parsing environment variables automatically.
- Dataclasses (`@dataclass(frozen=True)`): Provides lightweight, immutable value objects
  for mastering presets with zero boilerplate.
- Pathlib Object-Oriented Paths: Cross-platform directory path management (Windows & POSIX).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class MasterPreset:
    """
    Immutable preset configuration for YouTube-standard audio mastering.
    Python concept: dataclasses with frozen=True enforce read-only value semantics.
    """
    sample_rate: int = 48_000
    target_lufs: float = -14.0
    true_peak_db: float = -1.5
    low_shelf_db: float = 2.0
    presence_db: float = 2.5
    high_shelf_db: float = 1.5
    saturation_drive_db: float = 1.0
    saturation_mix: float = 0.15
    reverb_mix: float = 0.0
    pitch_shift_semitones: float = 0.0


# Registry of mastering presets
MASTERING_PRESETS: Dict[str, MasterPreset] = {
    "youtube_voiceover": MasterPreset(
        sample_rate=48_000,
        target_lufs=-14.0,
        true_peak_db=-1.5,
        low_shelf_db=2.0,
        presence_db=2.5,
        high_shelf_db=1.5,
        saturation_drive_db=1.0,
        saturation_mix=0.15,
        reverb_mix=0.0,
    ),
    "podcast_warm": MasterPreset(
        sample_rate=48_000,
        target_lufs=-16.0,
        true_peak_db=-1.5,
        low_shelf_db=3.0,
        presence_db=1.8,
        high_shelf_db=1.0,
        saturation_drive_db=1.5,
        saturation_mix=0.20,
        reverb_mix=0.04,
    ),
    "deep_narrator": MasterPreset(
        sample_rate=48_000,
        target_lufs=-14.0,
        true_peak_db=-1.5,
        low_shelf_db=4.0,
        presence_db=2.0,
        high_shelf_db=1.2,
        saturation_drive_db=2.0,
        saturation_mix=0.25,
        reverb_mix=0.06,
        pitch_shift_semitones=-1.5,
    ),
    "clean_neutral": MasterPreset(
        sample_rate=48_000,
        target_lufs=-14.0,
        true_peak_db=-1.5,
        low_shelf_db=0.0,
        presence_db=0.0,
        high_shelf_db=0.0,
        saturation_drive_db=0.0,
        saturation_mix=0.0,
        reverb_mix=0.0,
    ),
}


class Settings(BaseSettings):
    """
    Central application settings loaded from environment variables and `.env`.
    All model IDs, endpoints, and credentials live here so the entire architecture
    can be reconfigured without code changes.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # --- Hugging Face Credentials ---
    HF_TOKEN: str = Field(default="", description="Read-only Hugging Face access token")

    # --- Application Identity & Environment ---
    APP_NAME: str = "EchoVoice"
    APP_VERSION: str = "1.0.0"
    APP_ENV: Literal["development", "testing", "production"] = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # --- Security & JWT Authentication ---
    JWT_SECRET: str = Field(
        default="echovoice_super_secret_dev_key_change_in_production_32bytes_min",
        description="Secret key for signing JWT tokens"
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # --- Database Configuration ---
    # Default: local SQLite with async driver (aiosqlite)
    # Swappable via .env to: postgresql+asyncpg://user:pass@host:5432/dbname
    DATABASE_URL: str = "sqlite+aiosqlite:///./echovoice.db"

    # --- Storage Directories ---
    DATA_DIR: Path = Path("./data")
    UPLOADS_DIR: Path = Path("./data/uploads")
    VOICE_PROFILES_DIR: Path = Path("./data/voice_profiles")
    EXPORTS_DIR: Path = Path("./data/exports")
    CHECKPOINTS_DIR: Path = Path("./data/checkpoints")
    VECTOR_STORE_DIR: Path = Path("./data/vector_store")
    DOCUMENTS_DIR: Path = Path("./data/documents")

    # --- Open-Source AI Models (Strategy / Registry pattern) ---
    ASR_MODEL: str = "openai/whisper-large-v3-turbo"
    ASR_FALLBACK_MODEL: str = "distil-whisper/distil-large-v3"
    LLM_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
    LLM_FALLBACK_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    LLM_BASE_URL: Optional[str] = None
    LLM_API_KEY: str = Field(default="", description="Optional API key for OpenAI-compatible LLM endpoint")
    DEFAULT_LLM_ENGINE: str = "fallback"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 256
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    TTS_ENGINE: str = "xtts_v2"
    TRANSLATE_MODEL: str = "facebook/nllb-200-distilled-600M"
    ENHANCE_MODEL: str = "ResembleAI/resemble-enhance"

    # --- RAG & Vector Store Defaults ---
    RAG_CHUNK_SIZE: int = 600
    RAG_CHUNK_OVERLAP: int = 100
    RAG_TOP_K: int = 4
    RAG_SCORE_THRESHOLD: float = 0.30

    # --- Audio Mastering Defaults ---
    DEFAULT_MASTERING_PRESET: str = "youtube_voiceover"
    TARGET_LUFS: float = -14.0
    TRUE_PEAK_DB: float = -1.5

    # --- Rate Limiting ---
    RATE_LIMIT_DEFAULT: str = "100/minute"
    RATE_LIMIT_AUTH: str = "20/minute"

    def ensure_directories(self) -> None:
        """
        Creates all required data, upload, profile, export, checkpoint, and vector store directories
        if they do not already exist.
        """
        for directory in [
            self.DATA_DIR,
            self.UPLOADS_DIR,
            self.VOICE_PROFILES_DIR,
            self.EXPORTS_DIR,
            self.CHECKPOINTS_DIR,
            self.VECTOR_STORE_DIR,
            self.DOCUMENTS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def active_mastering_preset(self) -> MasterPreset:
        return MASTERING_PRESETS.get(
            self.DEFAULT_MASTERING_PRESET,
            MASTERING_PRESETS["youtube_voiceover"]
        )


# Global settings singleton
settings = Settings()
