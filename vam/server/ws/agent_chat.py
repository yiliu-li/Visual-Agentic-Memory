from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Optional

from fastapi import APIRouter, WebSocket

from vam.protocol import RetrieveToolArgs, SummarizeToolArgs
from vam.agent import (
    stream_agent_response,
    stream_retrieve_tool_request,
    stream_summarize_tool_request,
)


router = APIRouter()


async def _safe_send_text(websocket: WebSocket, payload: dict) -> bool:
    try:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except Exception:
        return False


@router.websocket("/ws/agent")
async def agent_chat(websocket: WebSocket):
    await websocket.accept()
    history = []
    ping_task: Optional[asyncio.Task[None]] = None

    async def _keep_alive() -> None:
        try:
            while True:
                await asyncio.sleep(20)
                if not await _safe_send_text(websocket, {"type": "ping"}):
                    break
        except asyncio.CancelledError:
            return

    ping_task = asyncio.create_task(_keep_alive())

    try:
        await _safe_send_text(
            websocket,
            {
                "type": "info",
                "message": "retrieval-vqa agent ready",
            },
        )

        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                continue

            text = msg.get("text")
            if text is None:
                continue

            payload = None
            with contextlib.suppress(Exception):
                payload = json.loads(text)
            if not isinstance(payload, dict):
                continue

            msg_type = str(payload.get("type") or "")
            if msg_type == "image":
                await _safe_send_text(websocket, {"type": "info", "message": "image ignored in retrieval-only mode"})
                continue

            if msg_type == "config":
                await _safe_send_text(websocket, {"type": "info", "message": "config ignored in retrieval-only mode"})
                continue

            if msg_type == "reset":
                history.clear()
                await _safe_send_text(websocket, {"type": "info", "message": "chat reset"})
                continue

            if msg_type == "tool":
                tool_name = str(payload.get("name") or "").strip()
                tool_args = payload.get("args") or {}
                if tool_name == "retrieve":
                    try:
                        transcript = RetrieveToolArgs.model_validate(tool_args).question
                        await _safe_send_text(websocket, {"type": "user_text", "text": transcript})
                        async for event in stream_retrieve_tool_request(transcript=transcript, history=history):
                            await _safe_send_text(websocket, event)
                    except Exception as e:
                        await _safe_send_text(websocket, {"type": "error", "message": f"tool failed: {type(e).__name__}: {e}"})
                    continue

                if tool_name == "summarize":
                    try:
                        args = SummarizeToolArgs.model_validate(tool_args)
                        async for event in stream_summarize_tool_request(args=args, history=history):
                            await _safe_send_text(websocket, event)
                    except Exception as e:
                        await _safe_send_text(websocket, {"type": "error", "message": f"tool failed: {type(e).__name__}: {e}"})
                    continue

                await _safe_send_text(websocket, {"type": "error", "message": f"unknown tool: {tool_name}"})
                continue

            if msg_type != "text":
                continue

            transcript = str(payload.get("text") or "").strip()
            if not transcript:
                continue

            try:
                async for event in stream_agent_response(transcript=transcript, history=history):
                    await _safe_send_text(websocket, event)
            except Exception as e:
                await _safe_send_text(websocket, {"type": "error", "message": f"text handling failed: {type(e).__name__}: {e}"})

    finally:
        if ping_task is not None:
            ping_task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()
