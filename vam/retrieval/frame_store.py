from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import math
import sqlite3
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

try:
    import fcntl
except Exception:
    fcntl = None

try:
    import msvcrt
except Exception:
    msvcrt = None

try:
    import torch
except Exception:
    torch = None

import anyio
import numpy as np

from vam.llm.openrouter import OpenRouterClient, build_mm_user_content
from vam import prompts
from vam.vision.backends import VisionBackend, get_backend
from vam.config import get_settings


logger = logging.getLogger(__name__)
_DB_SCHEMA_VERSION = 1


@contextlib.contextmanager
def _exclusive_file_lock(target_path: str):
    lock_path = f"{target_path}.lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    with os.fdopen(fd, "r+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            if msvcrt is None:
                raise RuntimeError("msvcrt is required for file locking on Windows")
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            if fcntl is None:
                raise RuntimeError("fcntl is required for file locking on POSIX")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _normalize_data_uri(data_uri_or_b64: str) -> str:
    if data_uri_or_b64.startswith("data:"):
        return data_uri_or_b64
    compact = "".join((data_uri_or_b64 or "").split())
    raw = _b64decode(compact)
    mime = _sniff_image_mime(raw) or "image/jpeg"
    return f"data:{mime};base64,{compact}"


def _b64decode(value: str) -> bytes:
    v = "".join((value or "").split())
    try:
        return base64.b64decode(v, validate=True)
    except Exception:
        pad = (-len(v)) % 4
        v2 = v + ("=" * pad)
        return base64.urlsafe_b64decode(v2)


def _sniff_image_mime(raw: bytes) -> str:
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "image/gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return ""


def _video_file_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def _tensor_to_blob(t: Optional[torch.Tensor]) -> Optional[bytes]:
    if t is None:
        return None
    a = t.detach()
    if a.dtype != torch.float16:
        a = a.to(dtype=torch.float16)
    return a.contiguous().cpu().numpy().tobytes()


def _blob_to_tensor(raw: Optional[bytes]) -> Optional[torch.Tensor]:
    if raw is None or torch is None:
        return None
    arr = np.frombuffer(raw, dtype=np.float16).copy()
    return torch.from_numpy(arr)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    if torch is None:
        raise RuntimeError("torch is required for embeddings")
    a = a.float()
    b = b.float()
    an = torch.linalg.vector_norm(a)
    bn = torch.linalg.vector_norm(b)
    if an.item() == 0.0 or bn.item() == 0.0:
        return 0.0
    return (torch.dot(a, b) / (an * bn)).item()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return max(0.0, min(2.0, 1.0 - cos_sim(a, b)))


def otsu_threshold(values: List[float], *, bins: int = 64) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return None
    vmin = min(vals)
    vmax = max(vals)
    if vmax <= vmin:
        return float(vmin)
    hist, bin_edges = np.histogram(np.asarray(vals, dtype=np.float32), bins=max(8, int(bins)), range=(vmin, vmax))
    total = float(hist.sum())
    if total <= 0.0:
        return None
    prob = hist.astype(np.float64) / total
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_sq = np.where(denom > 0.0, ((mu_t * omega - mu) ** 2) / denom, 0.0)
    idx = int(np.argmax(sigma_b_sq))
    return float(centers[idx])


def _normalize_text_payload(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _normalize_multiline_payload(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines: List[str] = []
    for raw_line in raw.split("\n"):
        line = re.sub(r"^\s*(?:[-*]+|\d+[.)])\s*", "", raw_line)
        line = _normalize_text_payload(line)
        if line:
            lines.append(line)
    if lines:
        return "\n".join(lines)
    return _normalize_text_payload(text)


def _strip_event_prefix(text: str) -> str:
    normalized = _normalize_multiline_payload(text)
    if not normalized:
        return ""
    lines = normalized.split("\n")
    if lines and lines[0].lower().startswith("[event]") and "|" in lines[0]:
        lines[0] = _normalize_text_payload(lines[0].split("|", 1)[1])
    return "\n".join(line for line in lines if line)


def _split_retrieval_lines(text: str) -> List[str]:
    compact = _strip_event_prefix(text)
    if not compact:
        return []
    lines: List[str] = []
    seen: set[str] = set()
    for raw_line in compact.split("\n"):
        line = _normalize_text_payload(raw_line)
        if not line or line in seen:
            continue
        lines.append(line)
        seen.add(line)
    return lines


def _split_retrieval_sentences(text: str) -> List[str]:
    compact = _normalize_text_payload(_strip_event_prefix(text))
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?])\s+", compact)
    sentences = [_normalize_text_payload(part) for part in parts if _normalize_text_payload(part)]
    return sentences or [compact]


def _chunk_text_for_retrieval(
    text: str,
    *,
    max_chars: int = 240,
    max_sentences: int = 2,
    max_chunks: int = 6,
) -> List[str]:
    natural_lines = _split_retrieval_lines(text)
    chunks: List[str] = []
    if len(natural_lines) >= 2:
        for line in natural_lines:
            if len(line) <= int(max_chars):
                chunks.append(line)
                continue
            sentences = _split_retrieval_sentences(line)
            current: List[str] = []
            current_len = 0
            for sentence in sentences:
                added_len = len(sentence) if not current else (1 + len(sentence))
                if current and (current_len + added_len > int(max_chars) or len(current) >= int(max_sentences)):
                    chunks.append(" ".join(current))
                    current = [sentence]
                    current_len = len(sentence)
                else:
                    current.append(sentence)
                    current_len += added_len
            if current:
                chunks.append(" ".join(current))
    else:
        sentences = _split_retrieval_sentences(text)
        if not sentences:
            return []
        current = []
        current_len = 0
        for sentence in sentences:
            added_len = len(sentence) if not current else (1 + len(sentence))
            if current and (current_len + added_len > int(max_chars) or len(current) >= int(max_sentences)):
                chunks.append(" ".join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += added_len
        if current:
            chunks.append(" ".join(current))
    deduped: List[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        compact = _normalize_text_payload(chunk)
        if not compact or compact in seen:
            continue
        deduped.append(compact)
        seen.add(compact)
        if len(deduped) >= int(max_chunks):
            break
    return deduped


def _tensor_to_base64(t: Optional[torch.Tensor]) -> Optional[str]:
    raw = _tensor_to_blob(t)
    if raw is None:
        return None
    return base64.b64encode(raw).decode("ascii")


def _base64_to_tensor(value: Any) -> Optional[torch.Tensor]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return _blob_to_tensor(base64.b64decode(value.encode("ascii")))
    except Exception:
        return None


@dataclass
class FrameRecord:
    frame_id: str
    t: float
    image_data_uri: str
    img_emb: torch.Tensor
    absolute_t: Optional[float] = None
    memory_tier: Optional[str] = None
    event_id: Optional[str] = None


@dataclass
class MemoryDocument:
    doc_id: str
    layer: str
    kind: str
    start_t: float
    end_t: float
    text: str
    representative_frame_id: Optional[str] = None
    frame_ids: Optional[List[str]] = None
    absolute_start_t: Optional[float] = None
    absolute_end_t: Optional[float] = None
    emb: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class FrameStore:
    def __init__(self, *, backend: VisionBackend, persist_path: str) -> None:
        self._lock = asyncio.Lock()
        self._persist_lock = asyncio.Lock()
        self._frames: List[FrameRecord] = []
        self._memory_documents: List[MemoryDocument] = []
        self._backend = backend
        self._dim: Optional[int] = None
        cfg = get_settings()
        self._vlm_client = OpenRouterClient(model_id=cfg.openrouter_model_id_light or cfg.openrouter_model_id)
        self._persist_path = (persist_path or "").strip()
        self._legacy_json_path = self._persist_path[:-8] + ".json" if self._persist_path.lower().endswith(".sqlite3") else ""
        self._dedup_distance_history: List[float] = []
        self._event_distance_history: List[float] = []
        self._recent_event_doc_id: Optional[str] = None
        self._latest_dedup_filter: Optional[Dict[str, Any]] = None
        self._latest_event_filter: Optional[Dict[str, Any]] = None
        self._persist_revision: int = 0
        if self._persist_path:
            self._init_db()
            self._maybe_migrate_legacy_json()
            self._load_from_db()

    def embedding_name(self) -> str:
        return getattr(self._backend, "name", "unknown")

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._persist_path, timeout=30.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        path = self._persist_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._connect_db() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if user_version > _DB_SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported memory store schema version {user_version}; this build supports up to {_DB_SCHEMA_VERSION}"
                )
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_meta (
                    store_key INTEGER PRIMARY KEY CHECK (store_key = 1),
                    backend TEXT,
                    dim INTEGER,
                    revision INTEGER NOT NULL DEFAULT 0,
                    recent_event_doc_id TEXT,
                    latest_dedup_filter TEXT,
                    latest_event_filter TEXT
                );
                INSERT OR IGNORE INTO store_meta (
                    store_key, backend, dim, revision, recent_event_doc_id, latest_dedup_filter, latest_event_filter
                ) VALUES (1, NULL, NULL, 0, NULL, NULL, NULL);

                CREATE TABLE IF NOT EXISTS frames (
                    frame_id TEXT PRIMARY KEY,
                    t REAL NOT NULL,
                    absolute_t REAL,
                    image_data_uri TEXT NOT NULL,
                    img_emb BLOB NOT NULL,
                    memory_tier TEXT,
                    event_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_frames_t ON frames(t);
                CREATE INDEX IF NOT EXISTS idx_frames_absolute_t ON frames(absolute_t);
                CREATE INDEX IF NOT EXISTS idx_frames_event_id ON frames(event_id);

                CREATE TABLE IF NOT EXISTS memory_documents (
                    doc_id TEXT PRIMARY KEY,
                    layer TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    start_t REAL NOT NULL,
                    end_t REAL NOT NULL,
                    absolute_start_t REAL,
                    absolute_end_t REAL,
                    text TEXT NOT NULL,
                    representative_frame_id TEXT,
                    frame_ids_json TEXT,
                    emb BLOB,
                    metadata_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memory_documents_layer_kind ON memory_documents(layer, kind);
                CREATE INDEX IF NOT EXISTS idx_memory_documents_start_end ON memory_documents(start_t, end_t);
                CREATE INDEX IF NOT EXISTS idx_memory_documents_absolute_start_end ON memory_documents(absolute_start_t, absolute_end_t);
                CREATE INDEX IF NOT EXISTS idx_memory_documents_rep_frame ON memory_documents(representative_frame_id);
                CREATE INDEX IF NOT EXISTS idx_memory_documents_layer_kind_start_end ON memory_documents(layer, kind, start_t, end_t);
                CREATE INDEX IF NOT EXISTS idx_memory_documents_layer_kind_abs_start_end ON memory_documents(layer, kind, absolute_start_t, absolute_end_t);
                """
            )
            if user_version < _DB_SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version = {_DB_SCHEMA_VERSION}")

    def _load_legacy_json_state(
        self,
        path: str,
    ) -> Tuple[List[FrameRecord], List[MemoryDocument], Optional[int], Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]], int]:
        if torch is None:
            return [], [], None, None, None, None, 0
        with _exclusive_file_lock(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        frames_in = data.get("frames") if isinstance(data, dict) else None
        if not isinstance(frames_in, list):
            return [], [], None, None, None, None, 0

        meta = data.get("meta", {})
        revision = int(meta.get("revision") or 0) if isinstance(meta, dict) else 0
        recent_event_doc_id = meta.get("recent_event_doc_id") if isinstance(meta, dict) else None
        latest_dedup_filter = (meta.get("latest_dedup_filter") or meta.get("latest_moment_filter")) if isinstance(meta, dict) else None
        latest_event_filter = meta.get("latest_event_filter") if isinstance(meta, dict) else None

        frames: List[FrameRecord] = []
        dim: Optional[int] = None
        for item in frames_in:
            if not isinstance(item, dict):
                continue
            frame_id = str(item.get("frame_id") or "")
            if not frame_id:
                continue
            absolute_t = item.get("absolute_t")
            img_emb = None
            img_b16 = item.get("img_emb_b16")
            if isinstance(img_b16, str) and img_b16:
                img_emb = _blob_to_tensor(base64.b64decode(img_b16.encode("ascii")))
            else:
                img_emb_list = item.get("img_emb")
                if isinstance(img_emb_list, list) and img_emb_list:
                    img_emb = torch.tensor([float(x) for x in img_emb_list], dtype=torch.float16)
            if img_emb is None:
                continue
            frames.append(
                FrameRecord(
                    frame_id=frame_id,
                    t=float(item.get("t") or 0.0),
                    image_data_uri=str(item.get("image_data_uri") or ""),
                    img_emb=img_emb,
                    absolute_t=float(absolute_t) if absolute_t is not None else None,
                    memory_tier=item.get("memory_tier"),
                    event_id=item.get("event_id"),
                )
            )
        if frames:
            counts: Dict[int, int] = {}
            for frame in frames:
                d = int(frame.img_emb.numel())
                counts[d] = counts.get(d, 0) + 1
            dim = max(counts.items(), key=lambda x: x[1])[0]
            frames = [frame for frame in frames if int(frame.img_emb.numel()) == dim]

        documents: List[MemoryDocument] = []
        for item in (data.get("memory_documents") or []):
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get("text") or "").split()).strip()
            if not text:
                continue
            emb = None
            emb_b16 = item.get("emb_b16")
            if isinstance(emb_b16, str) and emb_b16:
                emb = _blob_to_tensor(base64.b64decode(emb_b16.encode("ascii")))
            documents.append(
                MemoryDocument(
                    doc_id=str(item.get("doc_id") or uuid.uuid4().hex),
                    layer=str(item.get("layer") or "long"),
                    kind=str(item.get("kind") or "event"),
                    start_t=float(item.get("start_t") or 0.0),
                    end_t=float(item.get("end_t") or 0.0),
                    text=text,
                    representative_frame_id=item.get("representative_frame_id"),
                    frame_ids=item.get("frame_ids") or [],
                    absolute_start_t=float(item["absolute_start_t"]) if item.get("absolute_start_t") is not None else None,
                    absolute_end_t=float(item["absolute_end_t"]) if item.get("absolute_end_t") is not None else None,
                    emb=emb,
                    metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else None,
                )
            )
        return frames, documents, dim, recent_event_doc_id, latest_dedup_filter, latest_event_filter, revision

    def _load_from_db(self) -> None:
        if torch is None:
            return
        try:
            with self._connect_db() as conn:
                meta = conn.execute(
                    """
                    SELECT dim, revision, recent_event_doc_id, latest_dedup_filter, latest_event_filter
                    FROM store_meta WHERE store_key = 1
                    """
                ).fetchone()
                frames_rows = conn.execute(
                    """
                    SELECT frame_id, t, absolute_t, image_data_uri, img_emb, memory_tier, event_id
                    FROM frames
                    ORDER BY t ASC, frame_id ASC
                    """
                ).fetchall()
                doc_rows = conn.execute(
                    """
                    SELECT doc_id, layer, kind, start_t, end_t, absolute_start_t, absolute_end_t,
                           text, representative_frame_id, frame_ids_json, emb, metadata_json
                    FROM memory_documents
                    ORDER BY start_t ASC, doc_id ASC
                    """
                ).fetchall()

            self._dim = int(meta["dim"]) if meta and meta["dim"] is not None else None
            self._persist_revision = int(meta["revision"]) if meta else 0
            self._recent_event_doc_id = str(meta["recent_event_doc_id"]) if meta and meta["recent_event_doc_id"] is not None else None
            self._latest_dedup_filter = json.loads(meta["latest_dedup_filter"]) if meta and meta["latest_dedup_filter"] else None
            self._latest_event_filter = json.loads(meta["latest_event_filter"]) if meta and meta["latest_event_filter"] else None

            self._frames = [
                FrameRecord(
                    frame_id=str(row["frame_id"]),
                    t=float(row["t"]),
                    image_data_uri=str(row["image_data_uri"]),
                    img_emb=_blob_to_tensor(row["img_emb"]),
                    absolute_t=float(row["absolute_t"]) if row["absolute_t"] is not None else None,
                    memory_tier=row["memory_tier"],
                    event_id=row["event_id"],
                )
                for row in frames_rows
                if _blob_to_tensor(row["img_emb"]) is not None
            ]
            self._memory_documents = [
                MemoryDocument(
                    doc_id=str(row["doc_id"]),
                    layer=str(row["layer"]),
                    kind=str(row["kind"]),
                    start_t=float(row["start_t"]),
                    end_t=float(row["end_t"]),
                    text=str(row["text"]),
                    representative_frame_id=row["representative_frame_id"],
                    frame_ids=json.loads(row["frame_ids_json"]) if row["frame_ids_json"] else [],
                    absolute_start_t=float(row["absolute_start_t"]) if row["absolute_start_t"] is not None else None,
                    absolute_end_t=float(row["absolute_end_t"]) if row["absolute_end_t"] is not None else None,
                    emb=_blob_to_tensor(row["emb"]),
                    metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else None,
                )
                for row in doc_rows
            ]
        except Exception as exc:
            raise RuntimeError(f"failed to load memory store database '{self._persist_path}': {type(exc).__name__}: {exc}") from exc

    def _db_is_empty(self) -> bool:
        with self._connect_db() as conn:
            frames_count = int(conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
            docs_count = int(conn.execute("SELECT COUNT(*) FROM memory_documents").fetchone()[0])
            meta = conn.execute(
                "SELECT revision, recent_event_doc_id, latest_dedup_filter, latest_event_filter FROM store_meta WHERE store_key = 1"
            ).fetchone()
        if meta is None:
            return frames_count == 0 and docs_count == 0
        return (
            frames_count == 0
            and docs_count == 0
            and int(meta["revision"] or 0) == 0
            and meta["recent_event_doc_id"] is None
            and meta["latest_dedup_filter"] is None
            and meta["latest_event_filter"] is None
        )

    def _maybe_migrate_legacy_json(self) -> None:
        legacy_path = (self._legacy_json_path or "").strip()
        if not legacy_path or not os.path.exists(legacy_path):
            return
        if not self._db_is_empty():
            return
        frames, documents, dim, recent_event_doc_id, latest_dedup_filter, latest_event_filter, revision = self._load_legacy_json_state(legacy_path)
        self._dim = dim
        self._recent_event_doc_id = recent_event_doc_id
        self._latest_dedup_filter = dict(latest_dedup_filter or {}) or None
        self._latest_event_filter = dict(latest_event_filter or {}) or None
        self._persist_revision = 0
        written_revision = self._persist_to_db_sync(frames, documents, recent_event_doc_id, 0)
        self._persist_revision = int(written_revision)

    def _persist_to_db_sync(
        self,
        frames: List[FrameRecord],
        documents: List[MemoryDocument],
        recent_event_doc_id: Optional[str],
        expected_revision: int,
    ) -> int:
        path = self._persist_path
        if not path:
            return int(expected_revision)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._connect_db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current_revision = int(
                conn.execute("SELECT revision FROM store_meta WHERE store_key = 1").fetchone()[0]
            )
            if int(current_revision) != int(expected_revision):
                conn.execute("ROLLBACK")
                raise RuntimeError(
                    f"memory store changed in database (expected revision {expected_revision}, found {current_revision})"
                )
            next_revision = int(current_revision) + 1
            conn.execute("DELETE FROM frames")
            conn.executemany(
                """
                INSERT INTO frames (frame_id, t, absolute_t, image_data_uri, img_emb, memory_tier, event_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f.frame_id,
                        float(f.t),
                        float(f.absolute_t) if f.absolute_t is not None else None,
                        f.image_data_uri,
                        _tensor_to_blob(f.img_emb),
                        f.memory_tier,
                        f.event_id,
                    )
                    for f in frames
                ],
            )
            conn.execute("DELETE FROM memory_documents")
            conn.executemany(
                """
                INSERT INTO memory_documents (
                    doc_id, layer, kind, start_t, end_t, absolute_start_t, absolute_end_t,
                    text, representative_frame_id, frame_ids_json, emb, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        d.doc_id,
                        d.layer,
                        d.kind,
                        float(d.start_t),
                        float(d.end_t),
                        float(d.absolute_start_t) if d.absolute_start_t is not None else None,
                        float(d.absolute_end_t) if d.absolute_end_t is not None else None,
                        d.text,
                        d.representative_frame_id,
                        json.dumps(list(d.frame_ids or []), ensure_ascii=False),
                        _tensor_to_blob(d.emb),
                        json.dumps(d.metadata or {}, ensure_ascii=False) if d.metadata else None,
                    )
                    for d in documents
                ],
            )
            conn.execute(
                """
                UPDATE store_meta
                SET backend = ?, dim = ?, revision = ?, recent_event_doc_id = ?, latest_dedup_filter = ?, latest_event_filter = ?
                WHERE store_key = 1
                """,
                (
                    self.embedding_name(),
                    int(self._dim or 0) if self._dim is not None else None,
                    next_revision,
                    recent_event_doc_id,
                    json.dumps(self._latest_dedup_filter or {}, ensure_ascii=False) if self._latest_dedup_filter else None,
                    json.dumps(self._latest_event_filter or {}, ensure_ascii=False) if self._latest_event_filter else None,
                ),
            )
            conn.commit()
            return next_revision

    async def set_latest_filters(
        self,
        *,
        dedup_filter: Optional[Dict[str, Any]] = None,
        event_filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._reload_from_db()
        async with self._lock:
            if dedup_filter is not None:
                self._latest_dedup_filter = dict(dedup_filter)
            if event_filter is not None:
                self._latest_event_filter = dict(event_filter)
        await self._persist()

    async def get_latest_filters(self) -> Dict[str, Any]:
        await self._reload_from_db()
        async with self._lock:
            return {
                "dedup_filter": dict(self._latest_dedup_filter or {}),
                "event_filter": dict(self._latest_event_filter or {}),
            }

    async def _persist(self) -> None:
        if not self._persist_path:
            return
        if torch is None:
            return
        async with self._persist_lock:
            async with self._lock:
                snap = list(self._frames)
                documents = list(self._memory_documents)
                recent_event_doc_id = self._recent_event_doc_id
                expected_revision = int(self._persist_revision)
            written_revision = await anyio.to_thread.run_sync(
                self._persist_to_db_sync,
                snap,
                documents,
                recent_event_doc_id,
                expected_revision,
            )
            async with self._lock:
                self._persist_revision = int(written_revision)

    async def _reload_from_db(self) -> None:
        if not self._persist_path or torch is None:
            return
        await anyio.to_thread.run_sync(self._load_from_db)

    def _frame_from_row(self, row: sqlite3.Row) -> Optional[FrameRecord]:
        emb = _blob_to_tensor(row["img_emb"])
        if emb is None:
            return None
        return FrameRecord(
            frame_id=str(row["frame_id"]),
            t=float(row["t"]),
            image_data_uri=str(row["image_data_uri"]),
            img_emb=emb,
            absolute_t=float(row["absolute_t"]) if row["absolute_t"] is not None else None,
            memory_tier=row["memory_tier"],
            event_id=row["event_id"],
        )

    def _document_from_row(self, row: sqlite3.Row) -> MemoryDocument:
        return MemoryDocument(
            doc_id=str(row["doc_id"]),
            layer=str(row["layer"]),
            kind=str(row["kind"]),
            start_t=float(row["start_t"]),
            end_t=float(row["end_t"]),
            text=str(row["text"]),
            representative_frame_id=row["representative_frame_id"],
            frame_ids=json.loads(row["frame_ids_json"]) if row["frame_ids_json"] else [],
            absolute_start_t=float(row["absolute_start_t"]) if row["absolute_start_t"] is not None else None,
            absolute_end_t=float(row["absolute_end_t"]) if row["absolute_end_t"] is not None else None,
            emb=_blob_to_tensor(row["emb"]),
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else None,
        )

    def _fetch_frames_sync(
        self,
        *,
        frame_ids: Optional[List[str]] = None,
        min_time: float = 0.0,
        max_time: float = float("inf"),
        time_mode: str = "auto",
    ) -> List[FrameRecord]:
        where: List[str] = []
        params: List[Any] = []
        if frame_ids:
            placeholders = ",".join("?" for _ in frame_ids)
            where.append(f"frame_id IN ({placeholders})")
            params.extend(frame_ids)
        if min_time > 0.0 or max_time < float("inf"):
            use_absolute = self._should_use_absolute_time_range(
                min_time=min_time,
                max_time=max_time,
                time_mode=time_mode,
            )
            if use_absolute:
                where.append("absolute_t IS NOT NULL AND absolute_t >= ? AND absolute_t <= ?")
            else:
                where.append("t >= ? AND t <= ?")
            params.extend([float(min_time), float(max_time)])
        sql = """
            SELECT frame_id, t, absolute_t, image_data_uri, img_emb, memory_tier, event_id
            FROM frames
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY t ASC, frame_id ASC"
        with self._connect_db() as conn:
            rows = conn.execute(sql, params).fetchall()
        frames: List[FrameRecord] = []
        for row in rows:
            frame = self._frame_from_row(row)
            if frame is not None:
                frames.append(frame)
        return frames

    def _fetch_documents_sync(
        self,
        *,
        doc_id: Optional[str] = None,
        layer: Optional[str] = None,
        kind: Optional[str] = None,
        min_time: float = 0.0,
        max_time: float = float("inf"),
        time_mode: str = "auto",
    ) -> List[MemoryDocument]:
        where: List[str] = []
        params: List[Any] = []
        if doc_id:
            where.append("doc_id = ?")
            params.append(doc_id)
        if layer:
            where.append("layer = ?")
            params.append(str(layer))
        if kind:
            where.append("kind = ?")
            params.append(str(kind))
        if min_time > 0.0 or max_time < float("inf"):
            use_absolute = self._should_use_absolute_time_range(
                min_time=min_time,
                max_time=max_time,
                time_mode=time_mode,
            )
            if use_absolute:
                where.append("COALESCE(absolute_end_t, absolute_start_t) >= ? AND COALESCE(absolute_start_t, absolute_end_t) <= ?")
            else:
                where.append("end_t >= ? AND start_t <= ?")
            params.extend([float(min_time), float(max_time)])
        sql = """
            SELECT doc_id, layer, kind, start_t, end_t, absolute_start_t, absolute_end_t,
                   text, representative_frame_id, frame_ids_json, emb, metadata_json
            FROM memory_documents
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY start_t ASC, doc_id ASC"
        with self._connect_db() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._document_from_row(row) for row in rows]

    def _fetch_recent_event_doc_id_sync(self) -> Optional[str]:
        with self._connect_db() as conn:
            row = conn.execute(
                "SELECT recent_event_doc_id FROM store_meta WHERE store_key = 1"
            ).fetchone()
        if row is None or row["recent_event_doc_id"] is None:
            return None
        return str(row["recent_event_doc_id"])

    def _record_distance(self, history: List[float], distance: float, *, limit: int = 512) -> None:
        if not math.isfinite(float(distance)):
            return
        history.append(float(distance))
        if len(history) > limit:
            del history[: len(history) - limit]

    def _adaptive_distance_threshold(
        self,
        *,
        history: List[float],
        fallback_similarity_threshold: float,
        candidate_distances: Optional[List[float]] = None,
        min_samples: int = 8,
    ) -> Dict[str, float | str]:
        values = list(history[-256:])
        if candidate_distances:
            values.extend(float(v) for v in candidate_distances if math.isfinite(float(v)))
        fallback_distance = max(0.0, min(2.0, 1.0 - float(fallback_similarity_threshold)))
        if len(values) < int(min_samples):
            return {
                "distance_threshold": float(fallback_distance),
                "similarity_threshold": float(1.0 - fallback_distance),
                "method": "fallback",
            }
        threshold = otsu_threshold(values)
        if threshold is None:
            threshold = fallback_distance
            method = "fallback"
        else:
            method = "otsu"
        threshold = max(0.0, min(2.0, float(threshold)))
        return {
            "distance_threshold": float(threshold),
            "similarity_threshold": float(1.0 - threshold),
            "method": method,
        }

    def adaptive_dedup_threshold(self, *, candidate_distances: Optional[List[float]] = None) -> Dict[str, float | str]:
        settings = get_settings()
        return self._adaptive_distance_threshold(
            history=self._dedup_distance_history,
            fallback_similarity_threshold=float(settings.video_similarity_threshold),
            candidate_distances=candidate_distances,
        )

    def adaptive_event_threshold(self, *, candidate_distances: Optional[List[float]] = None) -> Dict[str, float | str]:
        settings = get_settings()
        return self._adaptive_distance_threshold(
            history=self._event_distance_history,
            fallback_similarity_threshold=float(settings.video_event_threshold),
            candidate_distances=candidate_distances,
        )

    def event_boundary_threshold(self, *, candidate_distances: Optional[List[float]] = None) -> Dict[str, float | str]:
        settings = get_settings()
        values = list(self._event_distance_history[-256:])
        if candidate_distances:
            values.extend(float(v) for v in candidate_distances if math.isfinite(float(v)))
        fallback_distance = max(0.0, min(2.0, 1.0 - float(settings.video_event_threshold)))
        if len(values) < 5:
            return {
                "distance_threshold": float(fallback_distance),
                "method": "fallback",
                "otsu_threshold": float(fallback_distance),
                "median": float(fallback_distance),
            }
        arr = np.asarray(values, dtype=np.float32)
        median = float(np.median(arr))
        otsu = otsu_threshold(values)
        if otsu is None:
            otsu = fallback_distance
            method = "fallback"
        else:
            method = "relaxed_otsu"
        threshold = float(median + 0.6 * (float(otsu) - median))
        if method == "relaxed_otsu":
            threshold = max(0.0, min(float(otsu), float(threshold)))
        else:
            threshold = max(fallback_distance * 0.75, min(float(otsu), float(threshold)))
        return {
            "distance_threshold": float(threshold),
            "method": method,
            "median": float(median),
            "otsu_threshold": float(otsu),
        }

    def detect_event_boundaries(self, distances: List[float]) -> Tuple[List[int], Dict[str, float | str]]:
        info = self.event_boundary_threshold(candidate_distances=distances)
        threshold = float(info["distance_threshold"])
        if not distances:
            return [], info
        boundaries: List[int] = []
        for i, distance in enumerate(distances):
            prev_d = distances[i - 1] if i > 0 else float("-inf")
            next_d = distances[i + 1] if i + 1 < len(distances) else float("-inf")
            is_peak = distance >= prev_d and distance >= next_d
            if distance >= threshold and is_peak:
                boundaries.append(i + 1)
        if not boundaries and distances:
            strongest_idx = int(np.argmax(np.asarray(distances, dtype=np.float32)))
            if float(distances[strongest_idx]) >= threshold:
                boundaries.append(strongest_idx + 1)
        return boundaries, info

    def _duration_capped_event_boundaries(
        self,
        recs: List[FrameRecord],
        distances: List[float],
        natural_boundaries: List[int],
        *,
        threshold: float,
        max_duration_s: float,
    ) -> Tuple[List[int], int]:
        if not recs:
            return [], 0
        if max_duration_s <= 0.0:
            return sorted(set(int(b) for b in natural_boundaries if 0 < int(b) < len(recs))), 0

        boundary_set = {int(b) for b in natural_boundaries if 0 < int(b) < len(recs)}
        extra_splits = 0
        start_idx = 0

        while start_idx < len(recs) - 1:
            next_natural = min((b for b in boundary_set if b > start_idx), default=len(recs))
            segment_end = next_natural
            segment_duration = float(recs[segment_end - 1].t) - float(recs[start_idx].t)
            if segment_duration <= max_duration_s:
                start_idx = segment_end
                continue

            max_end_idx = start_idx
            while max_end_idx + 1 < segment_end and (float(recs[max_end_idx + 1].t) - float(recs[start_idx].t)) <= max_duration_s:
                max_end_idx += 1
            if max_end_idx <= start_idx:
                max_end_idx = min(start_idx + 1, segment_end - 1)

            candidate_peaks: List[int] = []
            best_idx = start_idx
            best_distance = float("-inf")
            for dist_idx in range(start_idx, max_end_idx):
                distance = float(distances[dist_idx])
                prev_d = float(distances[dist_idx - 1]) if dist_idx > 0 else float("-inf")
                next_d = float(distances[dist_idx + 1]) if dist_idx + 1 < len(distances) else float("-inf")
                is_peak = distance >= prev_d and distance >= next_d
                if is_peak and distance >= threshold:
                    candidate_peaks.append(dist_idx)
                if distance >= best_distance:
                    best_distance = distance
                    best_idx = dist_idx

            split_idx = (candidate_peaks[-1] + 1) if candidate_peaks else (best_idx + 1)
            split_idx = max(start_idx + 1, min(split_idx, segment_end - 1))
            if split_idx in boundary_set:
                start_idx = split_idx
                continue

            boundary_set.add(split_idx)
            extra_splits += 1
            start_idx = split_idx

        return sorted(boundary_set), extra_splits

    def segment_event_frames(
        self,
        recs: List[FrameRecord],
    ) -> Tuple[List[List[FrameRecord]], Dict[str, float | str], List[float]]:
        if not recs:
            return [], self.event_boundary_threshold(candidate_distances=[]), []
        distances = [float(cosine_distance(recs[i - 1].img_emb, recs[i].img_emb)) for i in range(1, len(recs))]
        boundaries, info = self.detect_event_boundaries(distances)
        settings = get_settings()
        max_duration_s = float(getattr(settings, "video_event_max_duration_s", 300.0) or 0.0)
        boundaries, extra_splits = self._duration_capped_event_boundaries(
            recs,
            distances,
            boundaries,
            threshold=float(info["distance_threshold"]),
            max_duration_s=max_duration_s,
        )
        info = dict(info)
        info["max_duration_s"] = float(max_duration_s)
        info["duration_cap_splits"] = int(extra_splits)
        segments: List[List[FrameRecord]] = []
        start_idx = 0
        for boundary_idx in boundaries:
            segment = recs[start_idx:boundary_idx]
            if segment:
                segments.append(segment)
            start_idx = boundary_idx
        tail = recs[start_idx:]
        if tail:
            segments.append(tail)
        return segments, info, distances

    def record_dedup_distance(self, distance: float) -> None:
        self._record_distance(self._dedup_distance_history, distance)

    def record_event_distance(self, distance: float) -> None:
        self._record_distance(self._event_distance_history, distance)

    async def add_frame(self, *, t: float, image_base64: str, frame_id: Optional[str] = None, absolute_t: Optional[float] = None) -> FrameRecord | None:
        if torch is None:
            raise RuntimeError("torch is required for embeddings")
        await self._reload_from_db()
        image_data_uri = _normalize_data_uri(image_base64)
        def _embed_sync(img: str) -> Tuple[List[float], int]:
            return self._backend.embed_image_base64(img, instruction=prompts.EMBED_INSTRUCTION_REPRESENT_USER_INPUT)

        emb_list, _ = await anyio.to_thread.run_sync(_embed_sync, image_data_uri)
        d = int(len(emb_list))
        if self._dim is None:
            self._dim = d
        if int(self._dim) != d:
            raise RuntimeError(f"embedding dim mismatch: store_dim={self._dim} new_dim={d}")
        emb = torch.tensor(emb_list, dtype=torch.float16)

        distance: Optional[float] = None
        threshold_info = self.adaptive_dedup_threshold()
        async with self._lock:
            if self._frames:
                last_emb = self._frames[-1].img_emb
                distance = cosine_distance(last_emb, emb)
                if distance <= float(threshold_info["distance_threshold"]):
                    # Current frame (D) is similar to last stored frame (A/B/C)
                    # We want to keep "both sides" of a similar chunk.
                    # Logic:
                    # If D is similar to LastSaved, we temporarily hold D.
                    # If Next (E) is ALSO similar to LastSaved, we can drop D (the middle one) and hold E.
                    # If Next (E) is DIFFERENT, we must save D (the end of the similar chunk) and then save E.

                    # Implementation complexity: add_frame is single-item. We can't know "Next" yet.
                    # To support "keep start and end of similar sequence", we need a buffer or state.
                    # But here we are in a simple append-only store.
                    # Simplified approximation for single-frame add:
                    # Always replace the last frame if it was also "similar to the one before it"?
                    # No, that's too complex for this stateless method.
                    # For add_frame (single), we will stick to simple deduplication (drop if similar).
                    # The complex "keep ends" logic is better handled in add_frames (batch).
                    self.record_dedup_distance(distance)
                    return None

            rec = FrameRecord(
                frame_id=frame_id or uuid.uuid4().hex,
                t=t,
                image_data_uri=image_data_uri,
                img_emb=emb,
                absolute_t=absolute_t,
                memory_tier="recent",
            )
            self._frames.append(rec)
        if distance is not None:
            self.record_dedup_distance(distance)
        await self._persist()
        return rec

    async def add_frames(
        self,
        *,
        ts: List[float],
        images_base64: List[str],
        frame_ids: Optional[List[Optional[str]]] = None,
        absolute_ts: Optional[List[Optional[float]]] = None,
        batch_size: int = 4,
        similarity_threshold: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        embed_error_callback: Optional[Callable[[int, int, Exception], None]] = None,
    ) -> List[FrameRecord]:
        if torch is None:
            raise RuntimeError("torch is required for embeddings")
        await self._reload_from_db()
        if not ts or not images_base64:
            return []
        if len(ts) != len(images_base64):
            raise ValueError("ts and images_base64 must have the same length")
        if frame_ids is not None and len(frame_ids) != len(images_base64):
            raise ValueError("frame_ids must match images_base64 length")
        if absolute_ts is not None and len(absolute_ts) != len(images_base64):
            raise ValueError("absolute_ts must match images_base64 length")

        image_data_uris = [_normalize_data_uri(v) for v in images_base64]
        total_images = len(image_data_uris)
        processed_images = 0
        async with self._lock:
            existing_frame_ids = {str(frame.frame_id) for frame in self._frames}
            current_ref_emb: Optional[torch.Tensor] = self._frames[-1].img_emb if self._frames else None

        def _embed_batch_sync(chunk: List[str]) -> List[List[float]]:
            res, _ = self._backend.embed_images_base64_batch(
                chunk,
                instruction=prompts.EMBED_INSTRUCTION_REPRESENT_USER_INPUT,
                batch_size=max(1, int(batch_size)),
            )
            return res

        recs: List[FrameRecord] = []
        settings = get_settings()
        fallback_similarity_threshold = float(similarity_threshold) if similarity_threshold is not None else float(settings.video_similarity_threshold)

        for i in range(0, len(image_data_uris), batch_size):
            chunk_items: List[Tuple[int, float, str, str, Optional[float]]] = []
            batch_end = min(i + batch_size, len(image_data_uris))
            for global_idx in range(i, batch_end):
                fid = str((frame_ids[global_idx] if frame_ids is not None else None) or uuid.uuid4().hex)
                if fid in existing_frame_ids:
                    processed_images += 1
                    continue
                abs_t = absolute_ts[global_idx] if absolute_ts is not None else None
                chunk_items.append(
                    (
                        global_idx,
                        float(ts[global_idx]),
                        image_data_uris[global_idx],
                        fid,
                        abs_t,
                    )
                )

            if not chunk_items:
                if progress_callback is not None:
                    try:
                        progress_callback(int(processed_images), int(total_images))
                    except Exception:
                        pass
                continue

            chunk_imgs = [item[2] for item in chunk_items]
            try:
                emb_list = await anyio.to_thread.run_sync(_embed_batch_sync, chunk_imgs)
            except Exception as exc:
                processed_images += len(chunk_items)
                if progress_callback is not None:
                    try:
                        progress_callback(int(processed_images), int(total_images))
                    except Exception:
                        pass
                if embed_error_callback is not None:
                    try:
                        embed_error_callback(int(i), int(len(chunk_items)), exc)
                    except Exception:
                        pass
                continue
            processed_images += len(chunk_items)
            if progress_callback is not None:
                try:
                    progress_callback(int(processed_images), int(total_images))
                except Exception:
                    pass

            dims = {len(v) for v in emb_list}
            if len(dims) != 1:
                raise RuntimeError(f"embedding dim mismatch in batch: dims={sorted(dims)}")
            d = int(next(iter(dims)))
            if self._dim is None and d > 0:
                self._dim = d
            if self._dim is not None and d > 0 and int(self._dim) != d:
                raise RuntimeError(f"embedding dim mismatch: store_dim={self._dim} new_dim={d}")

            chunk_embs = torch.tensor(emb_list, dtype=torch.float16)
            batch_recs: List[FrameRecord] = []
            for j, emb in enumerate(chunk_embs):
                _, t_value, image_data_uri, fid, abs_t = chunk_items[j]
                batch_recs.append(
                    FrameRecord(
                        frame_id=fid,
                        t=t_value,
                        absolute_t=abs_t,
                        image_data_uri=image_data_uri,
                        img_emb=emb,
                        memory_tier="recent",
                    )
                )

            candidate_distances: List[float] = []
            prev_emb = current_ref_emb
            for rec in batch_recs:
                if prev_emb is not None:
                    candidate_distances.append(cosine_distance(prev_emb, rec.img_emb))
                prev_emb = rec.img_emb
            threshold_info = self._adaptive_distance_threshold(
                history=self._dedup_distance_history,
                fallback_similarity_threshold=fallback_similarity_threshold,
                candidate_distances=candidate_distances,
            )
            distance_threshold = float(threshold_info["distance_threshold"])
            filtered_batch_recs: List[FrameRecord] = []
            pending_similar_rec: Optional[FrameRecord] = None

            for rec in batch_recs:
                if current_ref_emb is None:
                    filtered_batch_recs.append(rec)
                    current_ref_emb = rec.img_emb
                    existing_frame_ids.add(str(rec.frame_id))
                    pending_similar_rec = None
                    continue

                distance = cosine_distance(current_ref_emb, rec.img_emb)
                self.record_dedup_distance(distance)

                if distance <= distance_threshold:
                    pending_similar_rec = rec
                else:
                    if pending_similar_rec is not None:
                        filtered_batch_recs.append(pending_similar_rec)
                        current_ref_emb = pending_similar_rec.img_emb
                        existing_frame_ids.add(str(pending_similar_rec.frame_id))
                        pending_similar_rec = None
                    filtered_batch_recs.append(rec)
                    current_ref_emb = rec.img_emb
                    existing_frame_ids.add(str(rec.frame_id))

            if pending_similar_rec is not None:
                filtered_batch_recs.append(pending_similar_rec)
                current_ref_emb = pending_similar_rec.img_emb
                existing_frame_ids.add(str(pending_similar_rec.frame_id))

            if filtered_batch_recs:
                async with self._lock:
                    self._frames.extend(filtered_batch_recs)
                await self._persist()
                recs.extend(filtered_batch_recs)

        return recs

    async def list_frames(self) -> List[FrameRecord]:
        return await anyio.to_thread.run_sync(partial(self._fetch_frames_sync))

    def _should_use_absolute_time_range(
        self,
        *,
        min_time: float,
        max_time: float,
        time_mode: str,
    ) -> bool:
        if time_mode == "absolute":
            return True
        if time_mode == "relative":
            return False
        return bool(min_time > 1e8 or (max_time > 1e8 and max_time != float("inf")))

    def _frame_in_time_range(
        self,
        frame: FrameRecord,
        *,
        min_time: float,
        max_time: float,
        time_mode: str,
    ) -> bool:
        if min_time <= 0.0 and max_time == float("inf"):
            return True
        use_absolute = self._should_use_absolute_time_range(
            min_time=min_time,
            max_time=max_time,
            time_mode=time_mode,
        )
        if use_absolute:
            return frame.absolute_t is not None and float(min_time) <= float(frame.absolute_t) <= float(max_time)
        return float(min_time) <= float(frame.t) <= float(max_time)

    def _document_in_time_range(
        self,
        doc: MemoryDocument,
        *,
        min_time: float,
        max_time: float,
        time_mode: str,
    ) -> bool:
        if min_time <= 0.0 and max_time == float("inf"):
            return True
        use_absolute = self._should_use_absolute_time_range(
            min_time=min_time,
            max_time=max_time,
            time_mode=time_mode,
        )
        if use_absolute:
            if doc.absolute_start_t is None and doc.absolute_end_t is None:
                return False
            start_v = float(doc.absolute_start_t if doc.absolute_start_t is not None else doc.absolute_end_t)
            end_v = float(doc.absolute_end_t if doc.absolute_end_t is not None else doc.absolute_start_t)
        else:
            start_v = float(doc.start_t)
            end_v = float(doc.end_t)
        return end_v >= float(min_time) and start_v <= float(max_time)

    def _matches_summary_filter(
        self,
        doc: MemoryDocument,
        summary_filter: Optional[Dict[str, Any]],
    ) -> bool:
        if not summary_filter:
            return True
        if str(doc.kind) != "summary":
            return False
        metadata = dict(doc.metadata or {})
        summary_structure = summary_filter.get("summary_structure")
        if summary_structure is not None:
            stored_structure = str(metadata.get("summary_structure") or metadata.get("summary_type") or "").strip().lower()
            if stored_structure != str(summary_structure).strip().lower():
                return False
        granularity = summary_filter.get("granularity_seconds")
        if granularity is not None:
            try:
                if float(metadata.get("granularity_seconds")) != float(granularity):
                    return False
            except Exception:
                return False
        return True

    async def list_summary_structures(self) -> List[Dict[str, Any]]:
        documents = await self.list_memory_documents_in_time_range(kind="summary")
        grouped: Dict[Tuple[str, float, str], Dict[str, Any]] = {}
        for doc in documents:
            metadata = dict(doc.metadata or {})
            summary_structure = str(metadata.get("summary_structure") or metadata.get("summary_type") or "").strip().lower()
            granularity = float(metadata.get("granularity_seconds") or 0.0)
            focus = " ".join(str(metadata.get("focus") or "").split()).strip()
            key = (summary_structure, granularity, focus)
            if key not in grouped:
                grouped[key] = {
                    "summary_structure": summary_structure,
                    "granularity_seconds": granularity,
                    "focus": focus,
                    "count": 0,
                    "min_start_t": float(doc.start_t),
                    "max_end_t": float(doc.end_t),
                }
            grouped[key]["count"] = int(grouped[key]["count"]) + 1
            grouped[key]["min_start_t"] = min(float(grouped[key]["min_start_t"]), float(doc.start_t))
            grouped[key]["max_end_t"] = max(float(grouped[key]["max_end_t"]), float(doc.end_t))
        return sorted(
            grouped.values(),
            key=lambda item: (
                str(item.get("summary_structure") or ""),
                float(item.get("granularity_seconds") or 0.0),
                str(item.get("focus") or ""),
            ),
        )

    async def list_frames_in_time_range(
        self,
        *,
        min_time: float = 0.0,
        max_time: float = float("inf"),
        time_mode: str = "auto",
    ) -> List[FrameRecord]:
        return await anyio.to_thread.run_sync(
            partial(
                self._fetch_frames_sync,
                min_time=float(min_time),
                max_time=float(max_time),
                time_mode=str(time_mode),
            )
        )

    async def list_frames_by_ids(self, frame_ids: List[str]) -> List[FrameRecord]:
        normalized = [str(frame_id).strip() for frame_id in frame_ids if str(frame_id).strip()]
        if not normalized:
            return []
        return await anyio.to_thread.run_sync(partial(self._fetch_frames_sync, frame_ids=normalized))

    def _frame_time_value(self, frame: FrameRecord) -> Optional[float]:
        if frame.absolute_t is not None:
            return float(frame.absolute_t)
        return float(frame.t)

    def _use_absolute_time_axis(self, frames: List[FrameRecord]) -> bool:
        return sum(1 for f in frames if f.absolute_t is not None) >= 2

    def _is_event_frame(self, frame: FrameRecord) -> bool:
        return bool(frame.event_id) and str(frame.memory_tier or "") == "long"

    def _tier_for_age(self, age_s: float) -> str:
        cfg = get_settings()
        if age_s <= float(cfg.memory_recent_window_s):
            return "recent"
        if age_s <= float(cfg.memory_mid_window_s):
            return "mid"
        return "long"

    def _min_gap_for_tier(self, tier: str) -> float:
        cfg = get_settings()
        if tier == "recent":
            return max(0.001, float(cfg.memory_recent_min_gap_s))
        if tier == "mid":
            return max(0.001, float(cfg.memory_mid_min_gap_s))
        return max(0.001, float(cfg.memory_long_min_gap_s))

    def _compression_for_tier(self, tier: str) -> Tuple[Optional[int], Optional[int]]:
        cfg = get_settings()
        if tier == "mid":
            return int(cfg.memory_mid_max_side), int(cfg.memory_mid_jpeg_quality)
        if tier == "long":
            return int(cfg.memory_long_max_side), int(cfg.memory_long_jpeg_quality)
        return None, None

    def _compress_data_uri(self, image_data_uri: str, *, max_side: int, jpeg_quality: int) -> str:
        if not image_data_uri.startswith("data:image"):
            return image_data_uri
        try:
            import cv2
        except Exception:
            return image_data_uri
        try:
            header, b64 = image_data_uri.split(",", 1)
            raw = _b64decode(b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return image_data_uri
            h, w = img.shape[:2]
            largest = max(h, w)
            if largest > max_side > 0:
                scale = float(max_side) / float(largest)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not ok:
                return image_data_uri
            payload = base64.b64encode(buf.tobytes()).decode("ascii")
            return f"data:image/jpeg;base64,{payload}"
        except Exception:
            return image_data_uri

    async def apply_retention_policy(self) -> Dict[str, Any]:
        await self._reload_from_db()
        async with self._lock:
            frames = list(self._frames)
        if len(frames) <= 2:
            return {"removed": 0, "kept": len(frames), "compressed": 0}

        use_absolute = self._use_absolute_time_axis(frames)
        keyed: List[Tuple[float, FrameRecord]] = []
        for frame in frames:
            tv = float(frame.absolute_t) if use_absolute and frame.absolute_t is not None else float(frame.t)
            keyed.append((tv, frame))
        keyed.sort(key=lambda item: item[0])
        latest_t = keyed[-1][0]

        keep_ids: set[str] = set()
        tier_by_id: Dict[str, str] = {}
        bucket_last: Dict[Tuple[str, int], str] = {}
        compressed_targets: List[Tuple[str, str, int, int]] = []

        if keyed:
            keep_ids.add(keyed[0][1].frame_id)
            keep_ids.add(keyed[-1][1].frame_id)

        for ts, frame in keyed:
            age_s = max(0.0, float(latest_t - ts))
            tier = self._tier_for_age(age_s)
            tier_by_id[frame.frame_id] = tier
            if self._is_event_frame(frame):
                keep_ids.add(frame.frame_id)
            else:
                gap = self._min_gap_for_tier(tier)
                bucket = int(math.floor(ts / gap))
                bucket_last[(tier, bucket)] = frame.frame_id

        keep_ids.update(bucket_last.values())

        for _, frame in keyed:
            tier = tier_by_id.get(frame.frame_id, "recent")
            max_side, jpeg_quality = self._compression_for_tier(tier)
            if frame.frame_id in keep_ids and max_side and jpeg_quality:
                compressed_targets.append((frame.frame_id, frame.image_data_uri, max_side, jpeg_quality))

        compressed_map: Dict[str, str] = {}
        for frame_id, image_data_uri, max_side, jpeg_quality in compressed_targets:
            new_uri = await anyio.to_thread.run_sync(
                partial(
                    self._compress_data_uri,
                    image_data_uri,
                    max_side=max_side,
                    jpeg_quality=jpeg_quality,
                )
            )
            if new_uri != image_data_uri:
                compressed_map[frame_id] = new_uri

        async with self._lock:
            before = len(self._frames)
            new_frames: List[FrameRecord] = []
            compressed = 0
            for frame in self._frames:
                if frame.frame_id not in keep_ids:
                    continue
                frame.memory_tier = tier_by_id.get(frame.frame_id, "recent")
                if frame.frame_id in compressed_map:
                    frame.image_data_uri = compressed_map[frame.frame_id]
                    compressed += 1
                new_frames.append(frame)
            self._frames = new_frames
            after = len(self._frames)

        if before != after or compressed:
            await self._persist()
        return {
            "removed": int(before - after),
            "kept": int(after),
            "compressed": int(compressed),
            "time_axis": "absolute" if use_absolute else "relative",
        }

    async def get_frame_by_id(self, frame_id: str) -> Optional[FrameRecord]:
        if not frame_id:
            return None
        frames = await anyio.to_thread.run_sync(partial(self._fetch_frames_sync, frame_ids=[str(frame_id)]))
        return frames[0] if frames else None

    def _format_time_range_text(
        self,
        *,
        start_t: float,
        end_t: float,
        absolute_start_t: Optional[float],
        absolute_end_t: Optional[float],
    ) -> str:
        parts = [f"relative_time={start_t:.1f}s->{end_t:.1f}s"]
        if absolute_start_t is not None and absolute_end_t is not None:
            parts.append(f"absolute_time={absolute_start_t:.3f}->{absolute_end_t:.3f}")
        return " | ".join(parts)

    async def _embed_text_for_memory(self, text: str) -> Optional[torch.Tensor]:
        if not text:
            return None
        emb = await self.embed_texts(texts=[text])
        return emb[0]

    def _build_retrieval_fields(
        self,
        *,
        layer: str,
        kind: str,
        start_t: float,
        end_t: float,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        meta = dict(metadata or {})
        lines = _split_retrieval_lines(text)
        sentences = _split_retrieval_sentences(text)
        focus = _normalize_text_payload(str(meta.get("focus") or ""))
        summary_structure = _normalize_text_payload(str(meta.get("summary_structure") or meta.get("summary_type") or ""))
        fields: List[Dict[str, str]] = []
        if focus:
            fields.append({"name": "focus", "text": focus})
        if summary_structure:
            fields.append({"name": "summary_structure", "text": summary_structure})
        if lines:
            fields.append({"name": "lead", "text": lines[0]})
            if len(lines) > 1 and lines[-1] != lines[0]:
                fields.append({"name": "tail", "text": lines[-1]})
        elif sentences:
            fields.append({"name": "lead", "text": sentences[0]})
            if len(sentences) > 1 and sentences[-1] != sentences[0]:
                fields.append({"name": "tail", "text": sentences[-1]})
        fields.append(
            {
                "name": "time_range",
                "text": f"{kind} memory from {float(start_t):.1f}s to {float(end_t):.1f}s in {layer} layer",
            }
        )
        deduped: List[Dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for field in fields:
            name = _normalize_text_payload(field.get("name") or "")
            value = _normalize_text_payload(field.get("text") or "")
            key = (name, value)
            if not name or not value or key in seen:
                continue
            deduped.append({"name": name, "text": value})
            seen.add(key)
        return deduped[:6]

    async def _build_retrieval_payload(
        self,
        *,
        layer: str,
        kind: str,
        start_t: float,
        end_t: float,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        chunks = _chunk_text_for_retrieval(text)
        fields = self._build_retrieval_fields(
            layer=layer,
            kind=kind,
            start_t=start_t,
            end_t=end_t,
            text=text,
            metadata=metadata,
        )
        unique_texts: List[str] = []
        for chunk in chunks:
            if chunk not in unique_texts:
                unique_texts.append(chunk)
        for field in fields:
            field_text = field["text"]
            if field_text not in unique_texts:
                unique_texts.append(field_text)
        if not unique_texts:
            return None
        emb_tensor = await self.embed_texts(texts=unique_texts)
        serialized_embeddings = {
            text_value: _tensor_to_base64(emb_tensor[idx])
            for idx, text_value in enumerate(unique_texts)
        }
        return {
            "version": 1,
            "chunks": [
                {
                    "text": chunk,
                    "emb_b16": serialized_embeddings.get(chunk),
                }
                for chunk in chunks
            ],
            "fields": [
                {
                    "name": field["name"],
                    "text": field["text"],
                    "emb_b16": serialized_embeddings.get(field["text"]),
                }
                for field in fields
            ],
        }

    def _retrieval_entries_from_doc(self, doc: MemoryDocument, entry_kind: str) -> List[Dict[str, Any]]:
        metadata = dict(doc.metadata or {})
        retrieval = metadata.get("_retrieval")
        parsed: List[Dict[str, Any]] = []
        if isinstance(retrieval, dict):
            raw_entries = retrieval.get(entry_kind)
            if isinstance(raw_entries, list):
                for raw_entry in raw_entries:
                    if not isinstance(raw_entry, dict):
                        continue
                    text = _normalize_text_payload(str(raw_entry.get("text") or ""))
                    if not text:
                        continue
                    entry: Dict[str, Any] = {"text": text}
                    if entry_kind == "fields":
                        entry["name"] = _normalize_text_payload(str(raw_entry.get("name") or ""))
                    emb = _base64_to_tensor(raw_entry.get("emb_b16"))
                    if emb is not None:
                        entry["emb"] = emb
                    parsed.append(entry)
        if parsed:
            return parsed
        if entry_kind == "chunks":
            return [{"text": chunk} for chunk in _chunk_text_for_retrieval(doc.text)]
        return [
            {"name": field["name"], "text": field["text"]}
            for field in self._build_retrieval_fields(
                layer=doc.layer,
                kind=doc.kind,
                start_t=doc.start_t,
                end_t=doc.end_t,
                text=doc.text,
                metadata=doc.metadata,
            )
        ]

    def _best_retrieval_entry_match(
        self,
        *,
        doc: MemoryDocument,
        entry_kind: str,
        query_vec: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        entries = self._retrieval_entries_from_doc(doc, entry_kind)
        if not entries:
            return None
        qd = int(query_vec.numel())
        best: Optional[Dict[str, Any]] = None
        for entry in entries:
            text = _normalize_text_payload(str(entry.get("text") or ""))
            if not text:
                continue
            emb = entry.get("emb")
            emb_score: Optional[float] = None
            if isinstance(emb, torch.Tensor) and int(emb.numel()) == qd:
                emb_score = float(cos_sim(query_vec, emb))
            if emb_score is None:
                continue
            score = float(emb_score)
            candidate = {
                "text": text,
                "name": _normalize_text_payload(str(entry.get("name") or "")),
                "score": float(score),
                "embedding_score": float(emb_score) if emb_score is not None else None,
                "source": entry_kind[:-1] if entry_kind.endswith("s") else entry_kind,
            }
            if best is None or float(candidate["score"]) > float(best["score"]):
                best = candidate
        return best

    async def add_memory_document(
        self,
        *,
        layer: str,
        kind: str,
        start_t: float,
        end_t: float,
        text: str,
        representative_frame_id: Optional[str] = None,
        frame_ids: Optional[List[str]] = None,
        absolute_start_t: Optional[float] = None,
        absolute_end_t: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryDocument:
        content = _normalize_multiline_payload(text)
        dense_content = _normalize_text_payload(content)
        doc_metadata = dict(metadata or {})
        retrieval_payload = await self._build_retrieval_payload(
            layer=layer,
            kind=kind,
            start_t=float(start_t),
            end_t=float(end_t),
            text=content,
            metadata=doc_metadata,
        )
        if retrieval_payload:
            doc_metadata["_retrieval"] = retrieval_payload
        doc = MemoryDocument(
            doc_id=uuid.uuid4().hex,
            layer=layer,
            kind=kind,
            start_t=float(start_t),
            end_t=float(end_t),
            text=content,
            representative_frame_id=representative_frame_id,
            frame_ids=list(frame_ids or []),
            absolute_start_t=absolute_start_t,
            absolute_end_t=absolute_end_t,
            emb=await self._embed_text_for_memory(dense_content),
            metadata=doc_metadata or None,
        )
        async with self._lock:
            self._memory_documents.append(doc)
            if layer == "long" and kind == "event":
                self._recent_event_doc_id = doc.doc_id
        await self._persist()
        return doc

    def _matches_exact_summary_window(
        self,
        doc: MemoryDocument,
        *,
        start_t: float,
        end_t: float,
        time_mode: str,
        granularity_s: float,
        summary_structure: Optional[str],
        focus: str,
    ) -> bool:
        if str(doc.kind) != "summary":
            return False

        metadata = dict(doc.metadata or {})
        stored_focus = " ".join(str(metadata.get("focus") or "").split()).strip()
        stored_structure = str(metadata.get("summary_structure") or metadata.get("summary_type") or "").strip().lower()
        target_structure = str(summary_structure or "").strip().lower()

        try:
            stored_granularity = float(metadata.get("granularity_seconds") or 0.0)
        except Exception:
            stored_granularity = 0.0

        if stored_focus != focus:
            return False
        if stored_structure != target_structure:
            return False
        if abs(stored_granularity - float(granularity_s)) > 1e-6:
            return False

        use_absolute = self._should_use_absolute_time_range(
            min_time=start_t,
            max_time=end_t,
            time_mode=time_mode,
        )
        if use_absolute:
            if doc.absolute_start_t is None or doc.absolute_end_t is None:
                return False
            return abs(float(doc.absolute_start_t) - float(start_t)) <= 1e-6 and abs(float(doc.absolute_end_t) - float(end_t)) <= 1e-6
        return abs(float(doc.start_t) - float(start_t)) <= 1e-6 and abs(float(doc.end_t) - float(end_t)) <= 1e-6

    async def upsert_summary_document(
        self,
        *,
        start_t: float,
        end_t: float,
        time_mode: str,
        granularity_s: float,
        summary_structure: Optional[str],
        prompt: str,
        text: str,
        representative_frame_id: Optional[str],
        frame_ids: List[str],
        absolute_start_t: Optional[float],
        absolute_end_t: Optional[float],
    ) -> MemoryDocument:
        content = _normalize_multiline_payload(text)
        dense_content = _normalize_text_payload(content)
        normalized_focus = " ".join((prompt or "").split()).strip()
        normalized_structure = " ".join((summary_structure or "").split()).strip().lower()
        match_start_t = float(absolute_start_t if absolute_start_t is not None and self._should_use_absolute_time_range(min_time=start_t, max_time=end_t, time_mode=time_mode) else start_t)
        match_end_t = float(absolute_end_t if absolute_end_t is not None and self._should_use_absolute_time_range(min_time=start_t, max_time=end_t, time_mode=time_mode) else end_t)
        metadata: Dict[str, Any] = {
            "granularity_seconds": float(granularity_s),
            "focus": normalized_focus,
            "source_kinds": ["event", "frame"],
        }
        if normalized_structure:
            metadata["summary_structure"] = normalized_structure

        retrieval_payload = await self._build_retrieval_payload(
            layer="summary",
            kind="summary",
            start_t=float(start_t),
            end_t=float(end_t),
            text=content,
            metadata=metadata,
        )
        if retrieval_payload:
            metadata["_retrieval"] = retrieval_payload
        summary_emb = await self._embed_text_for_memory(dense_content)

        documents = await self.list_memory_documents_in_time_range(
            min_time=match_start_t,
            max_time=match_end_t,
            time_mode=time_mode,
            kind="summary",
            summary_filter={
                "summary_structure": normalized_structure or None,
                "granularity_seconds": float(granularity_s),
            },
        )
        matches = [
            doc
            for doc in documents
            if self._matches_exact_summary_window(
                doc,
                start_t=match_start_t,
                end_t=match_end_t,
                time_mode=time_mode,
                granularity_s=float(granularity_s),
                summary_structure=normalized_structure or None,
                focus=normalized_focus,
            )
        ]

        if matches:
            target = max(matches, key=lambda item: (float(item.end_t), float(item.start_t), str(item.doc_id)))
            async with self._lock:
                if len(matches) > 1:
                    duplicate_ids = {doc.doc_id for doc in matches if doc.doc_id != target.doc_id}
                    self._memory_documents = [doc for doc in self._memory_documents if doc.doc_id not in duplicate_ids]
                target.layer = "summary"
                target.kind = "summary"
                target.start_t = float(start_t)
                target.end_t = float(end_t)
                target.text = content
                target.representative_frame_id = representative_frame_id
                target.frame_ids = list(frame_ids)
                target.absolute_start_t = absolute_start_t
                target.absolute_end_t = absolute_end_t
                target.emb = summary_emb
                target.metadata = metadata or None
            await self._persist()
            return target

        return await self.add_memory_document(
            layer="summary",
            kind="summary",
            start_t=float(start_t),
            end_t=float(end_t),
            text=content,
            representative_frame_id=representative_frame_id,
            frame_ids=list(frame_ids),
            absolute_start_t=absolute_start_t,
            absolute_end_t=absolute_end_t,
            metadata=metadata,
        )

    async def list_memory_documents(self) -> List[MemoryDocument]:
        return await anyio.to_thread.run_sync(partial(self._fetch_documents_sync))

    async def list_memory_documents_in_time_range(
        self,
        *,
        min_time: float = 0.0,
        max_time: float = float("inf"),
        time_mode: str = "auto",
        layer: Optional[str] = None,
        kind: Optional[str] = None,
        summary_filter: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryDocument]:
        documents = await anyio.to_thread.run_sync(
            partial(
                self._fetch_documents_sync,
                layer=layer,
                kind=kind,
                min_time=float(min_time),
                max_time=float(max_time),
                time_mode=str(time_mode),
            )
        )
        out: List[MemoryDocument] = []
        for doc in documents:
            if not self._matches_summary_filter(doc, summary_filter):
                continue
            out.append(doc)
        return out

    async def get_recent_event_frames(self) -> List[FrameRecord]:
        recent_doc = await self.get_recent_event_document()
        if recent_doc is None:
            return []
        frame_ids = [str(frame_id) for frame_id in (recent_doc.frame_ids or []) if str(frame_id).strip()]
        if not frame_ids:
            return []
        return await anyio.to_thread.run_sync(partial(self._fetch_frames_sync, frame_ids=frame_ids))

    async def get_recent_event_document(self) -> Optional[MemoryDocument]:
        recent_doc_id = await anyio.to_thread.run_sync(partial(self._fetch_recent_event_doc_id_sync))
        if not recent_doc_id:
            return None
        documents = await anyio.to_thread.run_sync(partial(self._fetch_documents_sync, doc_id=recent_doc_id))
        return documents[0] if documents else None

    async def search_memory_documents(
        self,
        *,
        query: str,
        top_k: int = 5,
        layer: Optional[str] = None,
        kind: Optional[str] = None,
        min_time: float = 0.0,
        max_time: float = float("inf"),
        time_mode: str = "auto",
        summary_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if torch is None:
            return []
        q_emb = await self.embed_texts(texts=[query])
        q_vec = q_emb[0]
        documents = await anyio.to_thread.run_sync(
            partial(
                self._fetch_documents_sync,
                layer=layer,
                kind=kind,
                min_time=float(min_time),
                max_time=float(max_time),
                time_mode=str(time_mode),
            )
        )
        if not documents:
            return []
        representative_ids = [
            str(doc.representative_frame_id)
            for doc in documents
            if doc.representative_frame_id
        ]
        frames = await anyio.to_thread.run_sync(partial(self._fetch_frames_sync, frame_ids=representative_ids))
        frame_lookup = {f.frame_id: f for f in frames}
        qd = int(q_vec.numel())
        results: List[Dict[str, Any]] = []

        for doc in documents:
            if not self._matches_summary_filter(doc, summary_filter):
                continue
            full_score: Optional[float] = None
            if doc.emb is not None and int(doc.emb.numel()) == qd:
                full_score = float(cos_sim(q_vec, doc.emb))
            best_chunk = self._best_retrieval_entry_match(
                doc=doc,
                entry_kind="chunks",
                query_vec=q_vec,
            )
            best_field = self._best_retrieval_entry_match(
                doc=doc,
                entry_kind="fields",
                query_vec=q_vec,
            )
            score_candidates = [
                float(full_score) if full_score is not None else float("-inf"),
                float(best_chunk["score"]) if best_chunk is not None else float("-inf"),
                float(best_field["score"]) if best_field is not None else float("-inf"),
            ]
            score = max(score_candidates)
            if not math.isfinite(score):
                continue
            match_text = doc.text
            match_source = "document"
            match_name = ""
            if best_chunk is not None and float(best_chunk["score"]) >= score and (
                full_score is None or float(best_chunk["score"]) > float(full_score) + 0.01
            ):
                match_text = str(best_chunk["text"])
                match_source = str(best_chunk["source"])
                match_name = str(best_chunk.get("name") or "")
            elif best_field is not None and float(best_field["score"]) >= score and (
                full_score is None or float(best_field["score"]) > float(full_score) + 0.005
            ):
                match_text = str(best_field["text"])
                match_source = str(best_field["source"])
                match_name = str(best_field.get("name") or "")
            frame = frame_lookup.get(str(doc.representative_frame_id or ""))
            results.append(
                {
                    "frame_id": doc.representative_frame_id,
                    "t": float(doc.start_t),
                    "absolute_t": doc.absolute_start_t,
                    "image": frame.image_data_uri if frame is not None else "",
                    "caption": match_text,
                    "document_text": doc.text,
                    "score": score,
                    "source": "memory_document",
                    "doc_id": doc.doc_id,
                    "layer": doc.layer,
                    "kind": doc.kind,
                    "match_source": match_source,
                    "match_name": match_name,
                    "score_breakdown": {
                        "document": float(full_score) if full_score is not None else None,
                        "chunk": float(best_chunk["score"]) if best_chunk is not None else None,
                        "field": float(best_field["score"]) if best_field is not None else None,
                    },
                    "frame_ids": list(doc.frame_ids or []),
                    "metadata": dict(doc.metadata or {}),
                    "time_range": {
                        "start_t": float(doc.start_t),
                        "end_t": float(doc.end_t),
                        "absolute_start_t": doc.absolute_start_t,
                        "absolute_end_t": doc.absolute_end_t,
                    },
                }
            )
        results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return results[: max(1, int(top_k))]

    async def summarize_time_range(
        self,
        *,
        min_time: float,
        max_time: float,
        time_mode: str = "auto",
        granularity_s: float = 60.0,
        summary_structure: Optional[str] = None,
        prompt: str = "",
        include_event_docs: bool = True,
        max_frames_per_window: int = 6,
    ) -> List[MemoryDocument]:
        bucket_s = max(5.0, float(granularity_s))
        start_v = float(min_time)
        end_v = float(max_time)
        if not math.isfinite(start_v):
            start_v = 0.0
        if not math.isfinite(end_v):
            frames = await self.list_frames()
            documents = await self.list_memory_documents()
            if self._should_use_absolute_time_range(
                min_time=start_v,
                max_time=max_time,
                time_mode=time_mode,
            ):
                abs_values = [float(f.absolute_t) for f in frames if f.absolute_t is not None]
                abs_values.extend(
                    float(doc.absolute_end_t)
                    for doc in documents
                    if doc.absolute_end_t is not None
                )
                end_v = max(abs_values) if abs_values else start_v
            else:
                rel_values = [float(f.t) for f in frames]
                rel_values.extend(float(doc.end_t) for doc in documents)
                end_v = max(rel_values) if rel_values else start_v
        if end_v <= start_v:
            return []

        created: List[MemoryDocument] = []
        cursor = start_v
        while cursor < end_v:
            window_start = float(cursor)
            window_end = float(min(end_v, cursor + bucket_s))
            frames = await self.list_frames_in_time_range(
                min_time=window_start,
                max_time=window_end,
                time_mode=time_mode,
            )
            docs = await self.list_memory_documents_in_time_range(
                min_time=window_start,
                max_time=window_end,
                time_mode=time_mode,
                kind="event" if include_event_docs else None,
            )
            if frames or docs:
                summary_text = await self.describe_summary_window(
                    frames=frames[: max(1, int(max_frames_per_window))],
                    documents=docs,
                    start_t=window_start,
                    end_t=window_end,
                    time_mode=time_mode,
                    summary_structure=summary_structure,
                    prompt=prompt,
                    granularity_s=bucket_s,
                )
                if summary_text:
                    representative_frame_id = frames[-1].frame_id if frames else (docs[-1].representative_frame_id if docs else None)
                    frame_ids: List[str] = [f.frame_id for f in frames]
                    if not frame_ids:
                        for doc in docs:
                            for frame_id in (doc.frame_ids or []):
                                if frame_id not in frame_ids:
                                    frame_ids.append(frame_id)
                    absolute_start_t = None
                    absolute_end_t = None
                    if self._should_use_absolute_time_range(
                        min_time=window_start,
                        max_time=window_end,
                        time_mode=time_mode,
                    ):
                        absolute_start_t = window_start
                        absolute_end_t = window_end
                        rel_start_t = float(frames[0].t) if frames else float(docs[0].start_t) if docs else 0.0
                        rel_end_t = float(frames[-1].t) if frames else float(docs[-1].end_t) if docs else rel_start_t
                    else:
                        rel_start_t = window_start
                        rel_end_t = window_end
                    created.append(
                        await self.upsert_summary_document(
                            start_t=rel_start_t,
                            end_t=rel_end_t,
                            time_mode=time_mode,
                            granularity_s=bucket_s,
                            summary_structure=summary_structure,
                            prompt=prompt,
                            text=summary_text,
                            representative_frame_id=representative_frame_id,
                            frame_ids=list(frame_ids),
                            absolute_start_t=absolute_start_t,
                            absolute_end_t=absolute_end_t,
                        )
                    )
            cursor = window_end
        return created

    async def finalize_event_memory(
        self,
        *,
        frames: List[FrameRecord],
        source_video_path: Optional[str] = None,
    ) -> Optional[MemoryDocument]:
        if not frames:
            return None
        await self._reload_from_db()
        event_id = frames[0].event_id or f"event-{uuid.uuid4().hex[:12]}"
        for frame in frames:
            frame.event_id = event_id
            frame.memory_tier = "long" if frame is frames[-1] else "mid"
        frame_updates = {frame.frame_id: frame for frame in frames}
        async with self._lock:
            for cached_frame in self._frames:
                updated = frame_updates.get(cached_frame.frame_id)
                if updated is None:
                    continue
                cached_frame.event_id = updated.event_id
                cached_frame.memory_tier = updated.memory_tier
        start_t = float(frames[0].t)
        end_t = float(frames[-1].t)
        absolute_start_t = frames[0].absolute_t
        absolute_end_t = frames[-1].absolute_t
        caption = await self.describe_memory_segment(
            frames=frames,
            start_t=start_t,
            end_t=end_t,
            absolute_start_t=absolute_start_t,
            absolute_end_t=absolute_end_t,
            source_video_path=source_video_path,
        )
        if not caption:
            caption = (
                f"Event {event_id} at {start_t:.1f}s to {end_t:.1f}s. "
                "Visual event summary could not be generated."
            )
        representative = frames[-1]
        return await self.add_memory_document(
            layer="long",
            kind="event",
            start_t=start_t,
            end_t=end_t,
            text=f"[event] id={event_id} | {caption}",
            representative_frame_id=representative.frame_id,
            frame_ids=[f.frame_id for f in frames],
            absolute_start_t=absolute_start_t,
            absolute_end_t=absolute_end_t,
        )

    async def clear_with_stats(self) -> Dict[str, int]:
        await self._reload_from_db()
        async with self._lock:
            frame_count = len(self._frames)
            document_count = len(self._memory_documents)
            event_count = sum(1 for doc in self._memory_documents if str(doc.kind) == "event")
            summary_count = sum(1 for doc in self._memory_documents if str(doc.kind) == "summary")
            self._frames = []
            self._memory_documents = []
            self._recent_event_doc_id = None
        self._latest_dedup_filter = None
        self._latest_event_filter = None
        await self._persist()
        return {
            "frames": int(frame_count),
            "documents": int(document_count),
            "events": int(event_count),
            "summaries": int(summary_count),
        }

    async def clear(self) -> int:
        stats = await self.clear_with_stats()
        return int(stats["frames"])

    async def remove_frames(self, frame_ids: List[str]) -> int:
        if not frame_ids:
            return 0
        await self._reload_from_db()
        remove = set(frame_ids)
        async with self._lock:
            before = len(self._frames)
            self._frames = [f for f in self._frames if f.frame_id not in remove]
            after = len(self._frames)
        await self._persist()
        return int(before - after)

    async def describe_memory_segment(
        self,
        *,
        frames: List[FrameRecord],
        start_t: float,
        end_t: float,
        absolute_start_t: Optional[float] = None,
        absolute_end_t: Optional[float] = None,
        source_video_path: Optional[str] = None,
    ) -> str:
        if not frames:
            return ""
        system_prompt = prompts.memory_segment_caption_system()
        user_prompt = prompts.memory_segment_caption_user_payload(
            start_t=start_t,
            end_t=end_t,
            absolute_start_t=absolute_start_t,
            absolute_end_t=absolute_end_t,
        )
        image_uris = [f.image_data_uri for f in frames[: min(len(frames), 8)]]
        video_uris: List[str] = []
        if source_video_path and os.path.exists(source_video_path):
            video_uri = self._build_event_video_clip_data_uri(source_video_path, start_t, end_t)
            if video_uri:
                video_uris.append(video_uri)
                print(
                    "[EventVideo] attach clip "
                    f"range={start_t:.1f}s->{end_t:.1f}s "
                    f"frames={len(frames)} "
                    f"uri_len={len(video_uri)}"
                )
            else:
                print(
                    "[EventVideo] clip build failed "
                    f"range={start_t:.1f}s->{end_t:.1f}s "
                    f"source={source_video_path}"
                )
        else:
            print(
                "[EventVideo] source video missing "
                f"range={start_t:.1f}s->{end_t:.1f}s "
                f"source={source_video_path}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_mm_user_content(user_prompt, image_uris, video_uris),
            },
        ]
        try:
            text, _ = await self._vlm_client.chat(messages, temperature=0.1, max_tokens=700)
        except Exception as e:
            if video_uris and "support input video" in str(e):
                print("[EventVideo] provider rejected video input, retrying with frames only")
                retry_messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": build_mm_user_content(user_prompt, image_uris, []),
                    },
                ]
                text, _ = await self._vlm_client.chat(retry_messages, temperature=0.1, max_tokens=700)
            else:
                raise
        return _normalize_multiline_payload(text)

    async def describe_summary_window(
        self,
        *,
        frames: List[FrameRecord],
        documents: List[MemoryDocument],
        start_t: float,
        end_t: float,
        time_mode: str,
        summary_structure: Optional[str],
        prompt: str,
        granularity_s: float,
    ) -> str:
        system_prompt = prompts.memory_summary_system()
        user_prompt = prompts.memory_summary_user_payload(
            start_t=start_t,
            end_t=end_t,
            time_mode=time_mode,
            summary_structure=summary_structure,
            granularity_s=granularity_s,
            prompt=prompt,
            event_documents=[
                {
                    "doc_id": doc.doc_id,
                    "layer": doc.layer,
                    "kind": doc.kind,
                    "text": doc.text,
                    "start_t": doc.start_t,
                    "end_t": doc.end_t,
                    "absolute_start_t": doc.absolute_start_t,
                    "absolute_end_t": doc.absolute_end_t,
                }
                for doc in documents[:8]
            ],
        )
        image_uris = [frame.image_data_uri for frame in frames[:8] if frame.image_data_uri]
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_mm_user_content(user_prompt, image_uris),
            },
        ]
        try:
            text, _ = await self._vlm_client.chat(messages, temperature=0.1, max_tokens=700)
            summary = _normalize_multiline_payload(text)
        except Exception:
            summary = self._fallback_summary_window_text(
                documents=documents,
                frames=frames,
                start_t=start_t,
                end_t=end_t,
                time_mode=time_mode,
                prompt=prompt,
            )
        if not summary:
            return ""
        return _normalize_multiline_payload(summary)

    def _fallback_summary_window_text(
        self,
        *,
        documents: List[MemoryDocument],
        frames: List[FrameRecord],
        start_t: float,
        end_t: float,
        time_mode: str,
        prompt: str,
    ) -> str:
        time_label = self._format_time_range_text(
            start_t=start_t,
            end_t=end_t,
            absolute_start_t=start_t if time_mode == "absolute" else None,
            absolute_end_t=end_t if time_mode == "absolute" else None,
        )
        doc_snippets: List[str] = []
        for doc in documents[:3]:
            text = " ".join((doc.text or "").split()).strip()
            if text:
                doc_snippets.append(text)
        frame_bits: List[str] = []
        for frame in frames[:4]:
            frame_bits.append(f"frame@{float(frame.t):.1f}s")
        pieces = [f"Summary window {time_label}."]
        if prompt:
            pieces.append(f"Focus: {prompt}.")
        if doc_snippets:
            pieces.append("Event context: " + " ".join(doc_snippets))
        if frame_bits:
            pieces.append("Representative frames: " + ", ".join(frame_bits) + ".")
        if not doc_snippets and not frame_bits:
            pieces.append("No visual evidence was available in this window.")
        return " ".join(pieces)

    def _build_event_video_clip_data_uri(self, source_video_path: str, start_t: float, end_t: float) -> Optional[str]:
        path = (source_video_path or "").strip()
        if not path or not os.path.exists(path):
            print(f"[EventVideo] source path not found: {path}")
            return None
        clip_start = max(0.0, float(start_t))
        clip_end = max(clip_start + 0.1, float(end_t))
        if clip_end - clip_start < 0.5:
            clip_end = clip_start + 0.5
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            clip_path = tmp.name
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{clip_start:.3f}",
                "-to",
                f"{clip_end:.3f}",
                "-i",
                path,
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-movflags",
                "+faststart",
                clip_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not os.path.exists(clip_path) or os.path.getsize(clip_path) <= 0:
                err = (proc.stderr or "").strip()
                if len(err) > 400:
                    err = err[-400:]
                print(
                    "[EventVideo] ffmpeg clip build failed "
                    f"code={proc.returncode} "
                    f"range={clip_start:.1f}s->{clip_end:.1f}s "
                    f"stderr={err}"
                )
                return None
            clip_size = os.path.getsize(clip_path)
            print(
                "[EventVideo] built clip "
                f"range={clip_start:.1f}s->{clip_end:.1f}s "
                f"bytes={clip_size}"
            )
            return _video_file_to_data_uri(clip_path)
        finally:
            try:
                if os.path.exists(clip_path):
                    os.unlink(clip_path)
            except Exception:
                pass

    async def inspect_frames(self, frames: List[FrameRecord], query: str) -> str:
        """
        Perform Visual Inspection over multiple frames to answer a specific query.
        Uses the internal Multimodal Inspector client.
        """
        system_prompt = (
            "You are the Multimodal Inspector for a Visual Agentic Memory system. "
            "You are performing Visual Inspection over a sequence of video frames. "
            "Analyze the frames strictly with respect to the query. "
            "Describe what happens across the frames when it is relevant to the query. "
            "If the frames contain relevant visual evidence, describe it in detail. "
            "If the frames are irrelevant to the query, clearly say so."
        )

        # Build the Multimodal Inspector prompt over multiple frames.
        image_uris = [f.image_data_uri for f in frames]
        user_prompt = f"User Query: {query}\n\nPlease examine these {len(frames)} frames and answer the query."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_mm_user_content(user_prompt, image_uris),
            },
        ]

        # Multi-frame Visual Inspection typically needs a slightly larger response budget.
        text, _ = await self._vlm_client.chat(messages, temperature=0.1, max_tokens=500)
        return (text or "").strip()

    async def inspect_frame(self, frame: FrameRecord, query: str) -> str:
        """Compatibility wrapper for single-frame Visual Inspection."""
        return await self.inspect_frames([frame], query)

    async def judge_frames_for_anchor(
        self,
        *,
        frames: List[FrameRecord],
        target_event: str,
        candidate_hint: str = "",
        verification_prompt: Optional[str] = None,
        candidate_window: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not frames:
            return {
                "match": False,
                "confidence": 0.0,
                "observed_event": "",
                "reason": "No frames were available for anchor adjudication.",
                "raw": "",
            }

        messages = [
            {"role": "system", "content": prompts.anchor_event_judge_system()},
            {
                "role": "user",
                "content": build_mm_user_content(
                    prompts.anchor_event_judge_user_prompt(
                        target_event=target_event,
                        candidate_hint=candidate_hint,
                        verification_prompt=verification_prompt,
                        candidate_window=candidate_window,
                    ),
                    [frame.image_data_uri for frame in frames],
                ),
            },
        ]
        text, _ = await self._vlm_client.chat(messages, temperature=0.0, max_tokens=220)
        raw = (text or "").strip()
        payload: Dict[str, Any] = {}
        candidate_text = raw
        if candidate_text.startswith("```json"):
            candidate_text = candidate_text[7:]
        if candidate_text.startswith("```"):
            candidate_text = candidate_text[3:]
        if candidate_text.endswith("```"):
            candidate_text = candidate_text[:-3]
        candidate_text = candidate_text.strip()
        try:
            loaded = json.loads(candidate_text)
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            match = re.search(r"\{[\s\S]*\}", candidate_text)
            if match:
                try:
                    loaded = json.loads(match.group(0))
                    if isinstance(loaded, dict):
                        payload = loaded
                except Exception:
                    payload = {}

        observed_event = str(payload.get("observed_event") or "").strip()
        reason = str(payload.get("reason") or raw).strip()
        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        match_value = payload.get("match", False)
        if isinstance(match_value, str):
            is_match = match_value.strip().lower() in {"true", "yes", "1"}
        else:
            is_match = bool(match_value)
        return {
            "match": bool(is_match),
            "confidence": float(confidence),
            "observed_event": observed_event,
            "reason": reason,
            "raw": raw,
        }

    def topk_by_image_text(
        self,
        *,
        frames: List[FrameRecord],
        query_text_emb: torch.Tensor,
        top_k: int,
    ) -> List[Tuple[FrameRecord, float]]:
        scored: List[Tuple[FrameRecord, float]] = []
        qd = int(query_text_emb.numel())
        for f in frames:
            if int(f.img_emb.numel()) != qd:
                continue
            scored.append((f, cos_sim(query_text_emb, f.img_emb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, top_k)]

    def topk_by_image_text_batch(
        self,
        *,
        frames: List[FrameRecord],
        query_text_embs: torch.Tensor,
        top_k: int,
    ) -> List[List[Tuple[FrameRecord, float]]]:
        if torch is None:
            raise RuntimeError("torch is required for embeddings")
        if not frames:
            return [[] for _ in range(int(query_text_embs.shape[0]))]
        q = query_text_embs.float()
        if q.ndim == 1:
            q = q.unsqueeze(0)
        qd = int(q.shape[1])
        kept = [f for f in frames if int(f.img_emb.numel()) == qd]
        if not kept:
            return [[] for _ in range(int(q.shape[0]))]
        imgs = torch.stack([f.img_emb.float() for f in kept], dim=0)
        qn = torch.linalg.vector_norm(q, dim=1, keepdim=True)
        inorm = torch.linalg.vector_norm(imgs, dim=1, keepdim=True)
        q = torch.where(qn == 0.0, q, q / qn)
        imgs = torch.where(inorm == 0.0, imgs, imgs / inorm)
        scores = q @ imgs.T
        k = max(1, min(int(top_k), int(scores.shape[1])))
        top_scores, top_idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
        out: List[List[Tuple[FrameRecord, float]]] = []
        for i in range(int(top_scores.shape[0])):
            pairs: List[Tuple[FrameRecord, float]] = []
            for j in range(k):
                idx = int(top_idx[i, j].item())
                pairs.append((kept[idx], float(top_scores[i, j].item())))
            out.append(pairs)
        return out

    async def embed_texts(self, *, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        if torch is None:
            raise RuntimeError("torch is required for embeddings")
        def _embed_sync(ts: List[str]) -> Tuple[List[List[float]], int]:
            return self._backend.embed_texts_batch(
                ts,
                instruction=prompts.EMBED_INSTRUCTION_RETRIEVE_RELEVANT,
                batch_size=int(batch_size),
            )

        emb_list, _ = await anyio.to_thread.run_sync(_embed_sync, texts)
        return torch.tensor(emb_list, dtype=torch.float32)

def _resolve_persist_path(base_path: str) -> str:
    base = (base_path or "").strip()
    if not base:
        return ""
    lower = base.lower()
    if lower.endswith(".sqlite3"):
        return base
    if lower.endswith(".sqlite"):
        return base
    if lower.endswith(".db"):
        return base
    if lower.endswith(".json"):
        root = base[:-5]
        return f"{root}.sqlite3"
    return f"{base}.sqlite3"


_stores: dict[str, FrameStore] = {}


def get_store(*, persist_path: Optional[str] = None) -> FrameStore:
    cfg = get_settings()
    base_persist_path = str(persist_path or "").strip() or str(cfg.frame_store_path or "")
    resolved_persist_path = _resolve_persist_path(base_persist_path)
    key = hashlib.sha1(resolved_persist_path.encode("utf-8")).hexdigest()[:10]
    if key in _stores:
        return _stores[key]
    store = FrameStore(backend=get_backend(), persist_path=resolved_persist_path)
    _stores[key] = store
    return store


def resolve_memory_store(*, persist_path: Optional[str] = None) -> FrameStore:
    """Resolve the retrieval store used by the agent."""
    return get_store(persist_path=persist_path)


store = get_store()
