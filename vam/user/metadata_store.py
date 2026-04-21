import asyncio
from typing import Dict, Optional, List
from datetime import datetime

from vam.models import UserMetadata, HabitRecord, MemoryItem, ChatMessage


class MetadataStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._store: Dict[str, UserMetadata] = {}

    async def get_metadata(self, user_id: str) -> UserMetadata:
        async with self._lock:
            meta = self._store.get(user_id)
            if meta is None:
                meta = UserMetadata(user_id=user_id)
                self._store[user_id] = meta
            return meta

    async def add_chat_message(self, user_id: str, role: str, content: str, *, metadata: Optional[Dict[str, str]] = None) -> ChatMessage:
        msg = ChatMessage(role=role, content=content, timestamp=datetime.utcnow(), metadata=metadata)
        async with self._lock:
            meta = self._store.get(user_id)
            if meta is None:
                meta = UserMetadata(user_id=user_id)
                self._store[user_id] = meta
            meta.chat_history.append(msg)
            return msg

    async def get_chat_history(self, user_id: str) -> List[ChatMessage]:
        async with self._lock:
            meta = self._store.get(user_id)
            return list(meta.chat_history) if meta else []

    async def record_habit(self, record: HabitRecord) -> HabitRecord:
        async with self._lock:
            meta = self._store.get(record.user_id)
            if meta is None:
                meta = UserMetadata(user_id=record.user_id)
                self._store[record.user_id] = meta
            meta.habits.append(record)
            return record

    async def add_memory(self, item: MemoryItem) -> MemoryItem:
        async with self._lock:
            meta = self._store.get(item.user_id)
            if meta is None:
                meta = UserMetadata(user_id=item.user_id)
                self._store[item.user_id] = meta
            meta.memories.append(item)
            return item


store = MetadataStore()
