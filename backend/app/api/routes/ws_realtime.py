"""
WebSocket Realtime Router (backend/app/api/routes/ws_realtime.py)
-----------------------------------------------------------------
Phase 7 & 9 Implementation Target: Streaming audio chunks via WebSocket,
VAD speech boundary events, and real-time barge-in interruption.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["Realtime WebSocket"])


@router.websocket("/ws/stream")
async def websocket_realtime_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_json({
            "type": "connection_established",
            "message": "EchoVoice Realtime WebSocket ready. Full streaming in Phase 7 & 9."
        })
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "echo", "received": data})
    except WebSocketDisconnect:
        pass
