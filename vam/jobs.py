from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Job:
    job_id: str
    status: str
    phase: str
    current: int
    total: int
    logs: List[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    started_at: float
    updated_at: float


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._max_logs = 250

    def create(self, *, phase: str = "queued") -> str:
        jid = uuid.uuid4().hex
        now = time.time()
        initial_phase = (phase or "queued").strip() or "queued"
        initial_status = "queued" if initial_phase == "queued" else "running"
        with self._lock:
            self._jobs[jid] = Job(
                job_id=jid,
                status=initial_status,
                phase=initial_phase,
                current=0,
                total=0,
                logs=[],
                result=None,
                error=None,
                started_at=now,
                updated_at=now,
            )
        return jid

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            return Job(
                job_id=j.job_id,
                status=j.status,
                phase=j.phase,
                current=j.current,
                total=j.total,
                logs=list(j.logs),
                result=j.result,
                error=j.error,
                started_at=j.started_at,
                updated_at=j.updated_at,
            )

    def log(self, job_id: str, msg: str) -> None:
        msg = (msg or "").strip()
        if not msg:
            return
        now = time.time()
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return
            j.logs.append(msg)
            if len(j.logs) > self._max_logs:
                j.logs = j.logs[-self._max_logs :]
            j.updated_at = now

    def progress(self, job_id: str, *, phase: Optional[str] = None, current: Optional[int] = None, total: Optional[int] = None) -> None:
        now = time.time()
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return
            if phase is not None:
                j.phase = phase
            if current is not None:
                j.current = int(current)
            if total is not None:
                j.total = int(total)
            if j.status == "queued":
                if (phase is not None and str(phase) != "queued") or int(j.current) > 0 or int(j.total) > 0:
                    j.status = "running"
            j.updated_at = now

    def done(self, job_id: str, *, result: Dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return
            j.status = "done"
            j.phase = "done"
            if j.total > 0 and j.current < j.total:
                j.current = j.total
            j.result = result
            j.updated_at = now

    def fail(self, job_id: str, *, error: str) -> None:
        now = time.time()
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return
            j.status = "error"
            j.phase = "error"
            j.error = (error or "").strip() or "unknown error"
            j.updated_at = now


manager = JobManager()
