import asyncio
from typing import Dict, Optional

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: Dict[str, WebSocket] = {}

    async def register(self, session_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._sessions[session_id] = websocket

    async def unregister(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def get(self, session_id: str) -> Optional[WebSocket]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def send_text(self, session_id: str, text: str) -> bool:
        ws = await self.get(session_id)
        if ws is None:
            return False
        await ws.send_text(text)
        return True

    async def send_bytes(self, session_id: str, data: bytes) -> bool:
        ws = await self.get(session_id)
        if ws is None:
            return False
        await ws.send_bytes(data)
        return True


manager = ConnectionManager()