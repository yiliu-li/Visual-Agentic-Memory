from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from vam.models import HabitRecord, MemoryItem
from vam.llm.router import LLMRouter
from vam import prompts
from vam.user.metadata_store import store as metadata_store


@dataclass
class SummarizeResult:
    event_document: str
    habits: List[HabitRecord]


class SessionMemoryManager:
    def __init__(self) -> None:
        self._router = LLMRouter()
        self._last_summarized_index: Dict[str, int] = {}

    async def maybe_summarize_on_gap(self, *, user_id: str, now: Optional[datetime] = None) -> Optional[SummarizeResult]:
        now = now or datetime.utcnow()
        history = await metadata_store.get_chat_history(user_id)
        if not history:
            self._last_summarized_index.setdefault(user_id, 0)
            return None

        last_msg_ts = history[-1].timestamp
        if (now - last_msg_ts) <= timedelta(minutes=5):
            return None

        start_idx = self._last_summarized_index.get(user_id, 0)
        segment = history[start_idx:]
        if not segment:
            self._last_summarized_index[user_id] = len(history)
            return None

        event_document, habits = await self._summarize_segment(user_id=user_id, segment=segment)
        await metadata_store.add_memory(
            MemoryItem(
                user_id=user_id,
                key=f"event_document:{now.isoformat()}",
                value=event_document,
                created_at=now,
            )
        )
        for h in habits:
            await metadata_store.record_habit(h)

        self._last_summarized_index[user_id] = len(history)
        return SummarizeResult(event_document=event_document, habits=habits)

    async def summarize_now(self, *, user_id: str, now: Optional[datetime] = None) -> Optional[SummarizeResult]:
        now = now or datetime.utcnow()
        history = await metadata_store.get_chat_history(user_id)
        if not history:
            self._last_summarized_index.setdefault(user_id, 0)
            return None

        start_idx = self._last_summarized_index.get(user_id, 0)
        segment = history[start_idx:]
        if not segment:
            self._last_summarized_index[user_id] = len(history)
            return None

        event_document, habits = await self._summarize_segment(user_id=user_id, segment=segment)
        await metadata_store.add_memory(
            MemoryItem(
                user_id=user_id,
                key=f"event_document:{now.isoformat()}",
                value=event_document,
                created_at=now,
            )
        )
        for h in habits:
            await metadata_store.record_habit(h)
        self._last_summarized_index[user_id] = len(history)
        return SummarizeResult(event_document=event_document, habits=habits)

    async def _summarize_segment(self, *, user_id: str, segment: List[Any]) -> Tuple[str, List[HabitRecord]]:
        lines: List[str] = []
        for m in segment:
            role = getattr(m, "role", "unknown")
            content = getattr(m, "content", "")
            lines.append(f"{role}: {content}")

        prompt = {
            "role": "user",
            "content": prompts.session_summarize_payload(chat="\n".join(lines)),
        }

        text, _ = await self._router.chat([prompt], route="light", temperature=0.2, max_tokens=500)
        raw = (text or "").strip()
        event_document = raw
        habits: List[HabitRecord] = []
        try:
            obj = json.loads(raw)
            event_document = str(obj.get("event_document") or "").strip() or raw
            for item in obj.get("habits") or []:
                if not isinstance(item, str):
                    continue
                text = item.strip()
                if not text:
                    continue
                habits.append(
                    HabitRecord(
                        user_id=user_id,
                        text=text,
                        timestamp=datetime.utcnow(),
                    )
                )
        except Exception:
            pass
        return event_document, habits


manager = SessionMemoryManager()
