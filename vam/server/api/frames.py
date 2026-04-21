from __future__ import annotations

import errno
import os
import tempfile
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from vam.config import get_settings
from vam.retrieval.frame_store import get_store
from vam.video import (
    parse_absolute_time,
    format_absolute_time,
    index_video_async as core_index_video_async,
)

router = APIRouter(prefix="/api/frames", tags=["Frames"])

class FrameIn(BaseModel):
    t: float
    absolute_t: Optional[float] = None
    image_base64: str
    frame_id: Optional[str] = None

class AddFramesRequest(BaseModel):
    frames: List[FrameIn]

class IndexVideoPathRequest(BaseModel):
    video_path: str
    fps: Optional[float] = None
    max_frames: Optional[int] = None
    laplacian_min: Optional[float] = None
    diff_threshold: Optional[float] = None
    ssim_threshold: Optional[float] = None
    hist_threshold: Optional[float] = None
    similarity_threshold: Optional[float] = None
    frame_id_prefix: str = "v"
    backend: Optional[str] = None
    store_path: Optional[str] = None
    video_start_time: Optional[str] = None
    debug: bool = False

async def _save_upload_to_temp(video: UploadFile, *, suffix: str = "") -> str:
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await video.read(8 * 1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        return tmp_path
    except OSError as e:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if int(getattr(e, "errno", -1)) == int(errno.ENOSPC):
            raise HTTPException(status_code=507, detail="insufficient storage: temp disk is full")
        raise HTTPException(status_code=500, detail=f"failed to save upload: {type(e).__name__}: {e}")


def _query_string(request: Request, name: str) -> Optional[str]:
    raw = request.query_params.get(name)
    if raw is None:
        return None
    text = str(raw).strip()
    return text if text else None


def _query_float(request: Request, name: str) -> Optional[float]:
    raw = _query_string(request, name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid float for {name}: {raw}") from exc


def _query_int(request: Request, name: str) -> Optional[int]:
    raw = _query_string(request, name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid integer for {name}: {raw}") from exc


def _query_bool(request: Request, name: str) -> Optional[bool]:
    raw = _query_string(request, name)
    if raw is None:
        return None
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise HTTPException(status_code=422, detail=f"invalid boolean for {name}: {raw}")

@router.post("/index_video_async")
async def index_video_async(
    request: Request,
    video: UploadFile = File(...),
    fps: Optional[float] = Form(default=None),
    max_frames: Optional[int] = Form(default=None),
    laplacian_min: Optional[float] = Form(default=None),
    diff_threshold: Optional[float] = Form(default=None),
    ssim_threshold: Optional[float] = Form(default=None),
    hist_threshold: Optional[float] = Form(default=None),
    similarity_threshold: Optional[float] = Form(default=None),
    frame_id_prefix: Optional[str] = Form(default="v"),
    backend: Optional[str] = Form(default=None),
    store_path: Optional[str] = Form(default=None),
    video_start_time: Optional[str] = Form(default=None),
    debug: Optional[bool] = Form(default=None),
):
    fps = fps if fps is not None else _query_float(request, "fps")
    max_frames = max_frames if max_frames is not None else _query_int(request, "max_frames")
    laplacian_min = laplacian_min if laplacian_min is not None else _query_float(request, "laplacian_min")
    diff_threshold = diff_threshold if diff_threshold is not None else _query_float(request, "diff_threshold")
    ssim_threshold = ssim_threshold if ssim_threshold is not None else _query_float(request, "ssim_threshold")
    hist_threshold = hist_threshold if hist_threshold is not None else _query_float(request, "hist_threshold")
    similarity_threshold = similarity_threshold if similarity_threshold is not None else _query_float(request, "similarity_threshold")
    frame_id_prefix = (frame_id_prefix or _query_string(request, "frame_id_prefix") or "v").strip() or "v"
    backend = backend if backend is not None else _query_string(request, "backend")
    store_path = store_path if store_path is not None else _query_string(request, "store_path")
    video_start_time = video_start_time if video_start_time is not None else _query_string(request, "video_start_time")
    debug = debug if debug is not None else _query_bool(request, "debug")

    settings = get_settings()
    fps = fps if fps is not None else settings.video_fps
    max_frames = max_frames if max_frames is not None else settings.video_max_frames
    laplacian_min = laplacian_min if laplacian_min is not None else settings.video_laplacian_min
    diff_threshold = diff_threshold if diff_threshold is not None else settings.video_diff_threshold
    ssim_threshold = ssim_threshold if ssim_threshold is not None else settings.video_ssim_threshold
    hist_threshold = hist_threshold if hist_threshold is not None else settings.video_hist_threshold

    video_absolute_start_time = parse_absolute_time(video_start_time)
    
    suffix = ""
    if video.filename and "." in video.filename:
        suffix = "." + video.filename.rsplit(".", 1)[-1]

    tmp_path = await _save_upload_to_temp(video, suffix=suffix)
    
    job_id = await core_index_video_async(
        tmp_path=tmp_path,
        cleanup_temp=True,
        fps=fps,
        max_frames=max_frames,
        laplacian_min=laplacian_min,
        diff_threshold=diff_threshold,
        ssim_threshold=ssim_threshold,
        hist_threshold=hist_threshold,
        similarity_threshold=similarity_threshold,
        frame_id_prefix=frame_id_prefix,
        backend=backend,
        video_absolute_start_time=video_absolute_start_time,
        debug=bool(debug),
        store_path=store_path,
    )
    return {"job_id": job_id}

@router.post("/index_video_async_by_path")
async def index_video_async_by_path(req: IndexVideoPathRequest):
    settings = get_settings()
    fps = req.fps if req.fps is not None else settings.video_fps
    max_frames = req.max_frames if req.max_frames is not None else settings.video_max_frames
    laplacian_min = req.laplacian_min if req.laplacian_min is not None else settings.video_laplacian_min
    diff_threshold = req.diff_threshold if req.diff_threshold is not None else settings.video_diff_threshold
    ssim_threshold = req.ssim_threshold if req.ssim_threshold is not None else settings.video_ssim_threshold
    hist_threshold = req.hist_threshold if req.hist_threshold is not None else settings.video_hist_threshold
    
    video_path = str(req.video_path or "").strip()
    if not video_path:
        raise HTTPException(status_code=400, detail="video_path is required")
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"video not found: {video_path}")
    
    video_absolute_start_time = parse_absolute_time(req.video_start_time)
    
    job_id = await core_index_video_async(
        tmp_path=video_path,
        cleanup_temp=False,
        fps=float(fps),
        max_frames=max_frames,
        laplacian_min=float(laplacian_min),
        diff_threshold=float(diff_threshold),
        ssim_threshold=float(ssim_threshold),
        hist_threshold=float(hist_threshold),
        similarity_threshold=req.similarity_threshold,
        frame_id_prefix=req.frame_id_prefix or "v",
        backend=req.backend,
        video_absolute_start_time=video_absolute_start_time,
        debug=bool(req.debug),
        store_path=req.store_path,
    )
    return {"job_id": job_id}

@router.post("/add")
async def add_frames(req: AddFramesRequest, backend: Optional[str] = None, store_path: Optional[str] = Query(default=None)):
    if not req.frames:
        raise HTTPException(status_code=400, detail="frames is empty")
    store = get_store(backend, persist_path=store_path)
    ts = [float(f.t) for f in req.frames]
    absolute_ts = [float(f.absolute_t) if f.absolute_t is not None else None for f in req.frames]
    imgs = [f.image_base64 for f in req.frames]
    fids = [f.frame_id for f in req.frames]
    recs = await store.add_frames(ts=ts, images_base64=imgs, frame_ids=fids, absolute_ts=absolute_ts)
    await store.apply_retention_policy()
    kept_ids = {f.frame_id for f in await store.list_frames()}
    recs = [r for r in recs if r.frame_id in kept_ids]
    out = [{"frame_id": r.frame_id, "t": r.t, "absolute_t": r.absolute_t, "absolute_time_str": format_absolute_time(r.absolute_t)} for r in recs]
    return {"added": out, "count": len(out)}

@router.get("/count")
async def count_frames(backend: Optional[str] = None, store_path: Optional[str] = Query(default=None)):
    store = get_store(backend, persist_path=store_path)
    frames = await store.list_frames()
    return {"count": len(frames)}

async def _reset_store_payload(*, backend: Optional[str], store_path: Optional[str]) -> dict:
    store = get_store(backend, persist_path=store_path)
    stats = await store.clear_with_stats()
    return {
        "cleared": int(stats["frames"]),
        "cleared_frames": int(stats["frames"]),
        "cleared_documents": int(stats["documents"]),
        "cleared_events": int(stats["events"]),
        "cleared_summaries": int(stats["summaries"]),
        "scope": "store",
        "message": "Cleared the full store, including frames and memory documents.",
    }


@router.post("/clear")
async def clear_frames(backend: Optional[str] = None, store_path: Optional[str] = Query(default=None)):
    return await _reset_store_payload(backend=backend, store_path=store_path)


@router.post("/reset_store")
async def reset_store(backend: Optional[str] = None, store_path: Optional[str] = Query(default=None)):
    return await _reset_store_payload(backend=backend, store_path=store_path)
