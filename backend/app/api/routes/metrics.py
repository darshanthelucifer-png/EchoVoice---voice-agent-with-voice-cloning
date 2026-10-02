"""
Latency & Metrics Router (backend/app/api/routes/metrics.py)
------------------------------------------------------------
Phase 10 Implementation Target: Latency waterfall chart telemetry,
WER/similarity metrics, and cache hit rate analytics.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/metrics", tags=["Latency & Telemetry"])


@router.get("/summary", summary="Retrieve latency telemetry summary (Phase 10)")
async def get_metrics_summary():
    return {"message": "Metrics profiler initialized. Live dashboard in Phase 10."}
