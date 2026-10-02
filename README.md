# EchoVoice: Real-Time Conversational Voice Agent & Studio Voice Cloning

EchoVoice is a production-grade, real-time conversational voice agent with Retrieval-Augmented Generation (RAG) and studio-quality voice cloning. Built entirely on free, open-source Hugging Face models with no paid APIs or credit cards required.

---

## Architecture Overview

```
echovoice/
├── backend/
│   ├── app/
│   │   ├── main.py                  # ASGI app, CORS, routers, rate limiter, lifespan
│   │   ├── api/routes/ (auth, health, voice_profiles, tts_jobs, chat, knowledge, metrics, ws_realtime)
│   │   ├── core/ (config.py, security.py, database.py, rate_limit.py, logging.py)
│   │   ├── models/ (user, voice_profile, tts_job, conversation, document, metric)
│   │   ├── schemas/ (auth, user, common, voice_profile, tts_job)
│   │   ├── services/ (base.py, auth_service.py)
│   │   ├── ai/ (audio, asr, llm, rag, tts, translate)
│   │   ├── workers/ (job_runner.py)
│   │   └── utils/
│   ├── scripts/ (verify_phase1.py)
│   ├── tests/ (conftest.py, test_auth.py, test_health.py)
│   ├── requirements.txt · .env.example
├── pytest.ini
└── README.md
```

---

## Phase 1 Completed: FastAPI Skeleton, Configuration, DB, Auth & Logging

- **Modern Asynchronous Stack:** FastAPI with ASGI lifespan context managers.
- **Config & Model Swapping:** Pydantic v2 `BaseSettings` + Dataclass presets (`youtube_voiceover`, `podcast_warm`, `deep_narrator`, `clean_neutral`).
- **Async Database & ORM:** SQLAlchemy 2.0 with async engine and `aiosqlite` (swappable to PostgreSQL via `DATABASE_URL`).
- **Authentication & Security:** Pure `bcrypt` password hashing (clean and warning-free on Python 3.11+), signed JWT bearer tokens (`PyJWT`), and protected endpoints with OpenAPI Swagger integration.
- **Structured Logging & Telemetry:** Custom logger with `@timed_step` decorator and `timed_block` context manager for profiling latency across the voice pipeline.
- **Rate Limiting:** `slowapi` limiter attached to API routes.

---

## Quickstart

### 1. Local Setup
```bash
# Clone the repository and navigate into backend
cd backend

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 3. Run Automated Tests
```bash
pytest backend/tests/ -v
```

### 4. Run Phase 1 Verification Script
```bash
python backend/scripts/verify_phase1.py
```

### 5. Launch the Development Server
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive API documentation will be available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **Health Check:** `http://localhost:8000/health`
