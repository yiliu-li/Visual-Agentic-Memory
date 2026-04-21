from typing import Optional, List, Literal, Dict
from datetime import datetime
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, str]] = None

class HabitRecord(BaseModel):
    user_id: str
    text: str
    timestamp: datetime

class MemoryItem(BaseModel):
    user_id: str
    key: str
    value: str
    created_at: datetime
    expires_at: Optional[datetime] = None

class UserMetadata(BaseModel):
    user_id: str
    chat_history: List[ChatMessage] = []
    habits: List[HabitRecord] = []
    memories: List[MemoryItem] = []
