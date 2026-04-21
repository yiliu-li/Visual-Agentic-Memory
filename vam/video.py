from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import datetime
from typing import Callable, List, Optional

import anyio

from vam.config import get_settings
from vam.retrieval.frame_store import get_store, cosine_distance
from vam.jobs import manager as job_manager

def parse_absolute_time(val: str | float | None) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val = val.strip()
    if not val:
        return None
    if len(val) == 14 and val.isdigit():
        try:
            dt = datetime.datetime.strptime(val, "%Y%m%d%H%M%S")
            return dt.timestamp()
        except ValueError:
            pass
    try:
        return float(val)
    except ValueError:
        return None

def format_absolute_time(val: float | None) -> str | None:
    if val is None:
        return None
    try:
        dt = datetime.datetime.fromtimestamp(val)
        return dt.strftime("%Y%m%d%H%M%S")
    except Exception:
        return None

def _parse_ffprobe_rate(raw: str | None) -> float:
    if not raw:
        return 0.0
    text = str(raw).strip()
    if "/" in text:
        num_s, den_s = text.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
            if den != 0.0:
                return num / den
        except Exception:
            return 0.0
        return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0

def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is required but was not found in PATH")
    return path

def _probe_video_metadata(video_path: str) -> dict:
    ffprobe = _require_binary("ffprobe")
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-print_format",
            "json",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "ffprobe failed").strip())
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams") or []
    format_info = payload.get("format") or {}
    video_stream = next((s for s in streams if str(s.get("codec_type")) == "video"), {})
    duration = 0.0
    for raw in (video_stream.get("duration"), format_info.get("duration")):
        try:
            duration = float(raw)
            if duration > 0.0:
                break
        except Exception:
            continue
    src_fps = max(
        _parse_ffprobe_rate(video_stream.get("avg_frame_rate")),
        _parse_ffprobe_rate(video_stream.get("r_frame_rate")),
    )
    if src_fps <= 0.0:
        src_fps = 30.0
    frame_count = 0
    for raw in (video_stream.get("nb_frames"),):
        try:
            frame_count = int(raw)
            if frame_count > 0:
                break
        except Exception:
            continue
    if frame_count <= 0 and duration > 0.0:
        frame_count = max(1, int(round(duration * src_fps)))
    return {
        "duration": float(duration),
        "src_fps": float(src_fps),
        "frame_count": int(frame_count),
        "streams": streams,
    }

def _calculate_histogram_similarity(cv2_module, np_module, img1, img2) -> float:
    hist1 = [cv2_module.calcHist([img1], [i], None, [256], [0, 256]) for i in range(3)]
    hist2 = [cv2_module.calcHist([img2], [i], None, [256], [0, 256]) for i in range(3)]
    hist1 = [cv2_module.normalize(h, h).flatten() for h in hist1]
    hist2 = [cv2_module.normalize(h, h).flatten() for h in hist2]
    correlations = [cv2_module.compareHist(h1, h2, cv2_module.HISTCMP_CORREL) for h1, h2 in zip(hist1, hist2)]
    return float(np_module.mean(correlations))

def extract_data_uris_ffmpeg(
    video_path: str,
    fps: float,
    max_frames: Optional[int],
    laplacian_min: float,
    diff_threshold: float,
    ssim_threshold: float,
    hist_threshold: float,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    debug: bool = False,
) -> tuple[List[tuple[float, str]], float, bool, dict]:
    try:
        import cv2
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:
        raise RuntimeError(f"Required libraries missing: {e}")

    ffmpeg = _require_binary("ffmpeg")
    meta = _probe_video_metadata(video_path)
    src_fps = float(meta["src_fps"])
    frame_count = int(meta["frame_count"])
    duration = float(meta["duration"])

    step = 1.0 / float(fps)
    dropped_blur: List[float] = []
    dropped_diff: List[float] = []
    dropped_hist: List[float] = []
    dropped_ssim: List[float] = []
    keep_list_max = 200
    last_kept_t: Optional[float] = None
    truncated = False
    out: List[tuple[float, str]] = []
    prev_gray = None
    prev_frame = None

    cache_root = tempfile.gettempdir()
    cache_key = hashlib.sha1(
        f"{os.path.abspath(video_path)}|{os.path.getmtime(video_path)}|{os.path.getsize(video_path)}|{fps}".encode("utf-8")
    ).hexdigest()[:16]
    frames_dir = os.path.join(cache_root, f"ffmpeg_frames_{cache_key}")
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = sorted(
        os.path.join(frames_dir, name)
        for name in os.listdir(frames_dir)
        if name.lower().endswith(".jpg")
    )
    if not frame_paths:
        pattern = os.path.join(frames_dir, "frame_%06d.jpg")
        proc = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-map",
                "0:v:0",
                "-vf",
                f"fps={fps}",
                "-vsync",
                "vfr",
                "-q:v",
                "2",
                pattern,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "ffmpeg frame extraction failed").strip())
        frame_paths = sorted(
            os.path.join(frames_dir, name)
            for name in os.listdir(frames_dir)
            if name.lower().endswith(".jpg")
        )

    if max_frames is not None and len(frame_paths) > int(max_frames):
        frame_paths = frame_paths[: int(max_frames)]
        truncated = True

    selection_key = hashlib.sha1(
        f"{fps}|{max_frames}|{laplacian_min}|{diff_threshold}|{ssim_threshold}|{hist_threshold}".encode("utf-8")
    ).hexdigest()[:16]
    selection_cache_path = os.path.join(frames_dir, f"selection_{selection_key}.json")

    if os.path.isfile(selection_cache_path):
        try:
            with open(selection_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_frames = cached.get("frames", [])
            cached_report = cached.get("report", {})
            cached_src_fps = cached.get("src_fps", src_fps)
            cached_truncated = cached.get("truncated", False)
            
            total_for_progress = max(1, len(cached_frames))
            for idx, item in enumerate(cached_frames):
                frame_name = item.get("name")
                frame_t = item.get("t")
                if not frame_name: continue
                
                frame_path = os.path.join(frames_dir, frame_name)
                with open(frame_path, "rb") as rf:
                    b64 = base64.b64encode(rf.read()).decode("ascii")
                out.append((frame_t, f"data:image/jpeg;base64,{b64}"))
                
                if progress_cb is not None:
                    progress_cb(idx + 1, total_for_progress)
            
            if progress_cb is not None:
                progress_cb(total_for_progress, total_for_progress)
                
            return out, float(cached_src_fps), bool(cached_truncated), cached_report
        except Exception:
            pass

    sampled = len(frame_paths)
    total_for_progress = max(1, sampled)
    kept_frame_items = []
    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        t = idx * step
        if duration > 0.0:
            t = min(t, duration)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap < float(laplacian_min):
            if len(dropped_blur) < keep_list_max:
                dropped_blur.append(float(t))
            continue

        is_redundant = False
        if prev_gray is not None and prev_frame is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(np.mean(diff))
            if mean_diff < float(diff_threshold):
                is_redundant = True
                if len(dropped_diff) < keep_list_max:
                    dropped_diff.append(float(t))
            if not is_redundant and hist_threshold > 0:
                if _calculate_histogram_similarity(cv2, np, frame, prev_frame) > float(hist_threshold):
                    is_redundant = True
                    if len(dropped_hist) < keep_list_max:
                        dropped_hist.append(float(t))
            if not is_redundant and ssim_threshold > 0:
                small_curr = cv2.resize(gray, (64, 64))
                small_prev = cv2.resize(prev_gray, (64, 64))
                if ssim(small_curr, small_prev, data_range=255) > float(ssim_threshold):
                    is_redundant = True
                    if len(dropped_ssim) < keep_list_max:
                        dropped_ssim.append(float(t))

        if is_redundant:
            continue

        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok2:
            raise RuntimeError("failed to encode frame")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        out.append((float(t), f"data:image/jpeg;base64,{b64}"))
        
        kept_frame_items.append({
            "name": os.path.basename(frame_path),
            "t": float(t)
        })
        
        prev_gray = gray
        prev_frame = frame
        last_kept_t = float(t)
        if progress_cb is not None:
            progress_cb(idx + 1, total_for_progress)

    if progress_cb is not None:
        progress_cb(total_for_progress, total_for_progress)

    report = {
        "extract_backend": "ffmpeg",
        "step_s": float(step),
        "scanned": int(frame_count),
        "sampled": int(sampled),
        "kept_pre_embed": int(len(out)),
        "dedup_filter": None,
        "event_filter": None,
        "video_meta": {
            "duration_s": float(duration),
            "source_fps": float(src_fps),
            "streams": int(len(meta["streams"] or [])),
        },
        "dropped": {
            "blur": {"count": int(len(dropped_blur)), "t": dropped_blur},
            "pixel_diff": {"count": int(len(dropped_diff)), "t": dropped_diff},
            "hist": {"count": int(len(dropped_hist)), "t": dropped_hist},
            "ssim": {"count": int(len(dropped_ssim)), "t": dropped_ssim},
        },
        "max_frames": {
            "enabled": max_frames is not None,
            "max": int(max_frames) if max_frames is not None else None,
            "last_kept_t": last_kept_t,
            "truncated": bool(truncated),
        },
    }
    
    try:
        cache_data = {
            "frames": kept_frame_items,
            "report": report,
            "src_fps": src_fps,
            "truncated": truncated
        }
        with open(selection_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
    except Exception:
        pass
        
    return out, float(src_fps), bool(truncated), report

async def index_video_async(
    *,
    tmp_path: str,
    cleanup_temp: bool,
    fps: float,
    max_frames: Optional[int],
    laplacian_min: float,
    diff_threshold: float,
    ssim_threshold: float,
    hist_threshold: float,
    similarity_threshold: Optional[float],
    frame_id_prefix: str,
    backend: Optional[str],
    video_absolute_start_time: Optional[float],
    debug: bool,
    store_path: Optional[str] = None,
) -> str:
    if not os.path.isfile(tmp_path):
        raise FileNotFoundError(f"video not found: {tmp_path}")

    job_id = job_manager.create(phase="queued")
    job_manager.log(job_id, "queued")

    async def _run() -> None:
        try:
            enable_event_finalize = str(os.getenv("VAM_ENABLE_EVENT_FINALIZE", "1")).strip() == "1"
            event_finalize_timeout_s = float(os.getenv("VAM_EVENT_FINALIZE_TIMEOUT_S", "120"))
            store = get_store(backend, persist_path=store_path)
            throttle = {"t": 0.0}

            def _prog(phase: str, current: int, total: int) -> None:
                now = time.time()
                if current < total and (now - throttle["t"]) < 0.25:
                    return
                throttle["t"] = now
                job_manager.progress(job_id, phase=phase, current=current, total=total)
            
            job_manager.progress(job_id, phase="extract", current=0, total=1)
            try:
                frames, src_fps, truncated, report = await anyio.to_thread.run_sync(
                    extract_data_uris_ffmpeg,
                    tmp_path,
                    float(fps),
                    max_frames,
                    float(laplacian_min),
                    float(diff_threshold),
                    float(ssim_threshold),
                    float(hist_threshold),
                    lambda current, total: _prog("extract", current, total),
                    debug,
                )
            except Exception as e:
                job_manager.log(job_id, f"error in extract: {str(e)}")
                job_manager.fail(job_id, error=str(e))
                return
            
            job_manager.log(job_id, f"extracted={len(frames)} truncated={bool(truncated)}")

            added: List[dict] = []
            job_manager.progress(job_id, phase="embed", current=0, total=max(1, len(frames)))
            if frames:
                ts = [float(t) for t, _ in frames]
                absolute_ts_all = None
                if video_absolute_start_time is not None:
                    absolute_ts_all = [float(video_absolute_start_time) + t for t in ts]
                imgs = [data_uri for _, data_uri in frames]
                fids = [f"{frame_id_prefix}-{i:06d}" for i in range(1, len(frames) + 1)]
                failed_frame_ids: set[str] = set()
                failed_batches: List[dict] = []

                def _embed_error_callback(start_idx: int, batch_len: int, exc: Exception) -> None:
                    failed = []
                    for offset in range(int(batch_len)):
                        global_idx = int(start_idx) + offset
                        if global_idx >= len(fids):
                            break
                        fid = str(fids[global_idx])
                        failed_frame_ids.add(fid)
                        if len(failed) < 50:
                            failed.append({"frame_id": fid, "t": float(ts[global_idx])})
                    failed_batches.append({
                        "start_idx": int(start_idx),
                        "batch_len": int(batch_len),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "failed_frames": failed,
                    })

                settings = get_settings()
                st = float(similarity_threshold) if similarity_threshold is not None else float(settings.video_similarity_threshold)
                report["embedding_similarity_threshold"] = float(st)
                
                recs = await store.add_frames(
                    ts=ts,
                    images_base64=imgs,
                    frame_ids=fids,
                    absolute_ts=absolute_ts_all,
                    similarity_threshold=float(st),
                    progress_callback=lambda current, total: _prog("embed", current, total),
                    embed_error_callback=_embed_error_callback,
                )
                
                report["embed_failures"] = {
                    "count": int(len(failed_frame_ids)),
                    "batches": int(len(failed_batches)),
                    "details": failed_batches[:200],
                }
                
                try:
                    dedup_distances = [float(cosine_distance(recs[k - 1].img_emb, recs[k].img_emb)) for k in range(1, len(recs))]
                    report["dedup_filter"] = store.adaptive_dedup_threshold(candidate_distances=dedup_distances)
                except Exception:
                    report["dedup_filter"] = store.adaptive_dedup_threshold()
                
                report["embedding_dedup_threshold"] = report["dedup_filter"]
                report["embed_dedup"] = {"count": 0, "dropped": []}
                chunk_keep_ids = {r.frame_id for r in recs}
                for fid, t in zip(fids, ts):
                    if fid in failed_frame_ids:
                        continue
                    if fid not in chunk_keep_ids:
                        report["embed_dedup"]["count"] = int(report["embed_dedup"]["count"]) + 1
                        if len(report["embed_dedup"]["dropped"]) < 200:
                            report["embed_dedup"]["dropped"].append({"frame_id": fid, "t": float(t)})

                added = [{"frame_id": r.frame_id, "t": r.t, "absolute_t": r.absolute_t, "absolute_time_str": format_absolute_time(r.absolute_t)} for r in recs]
                segments, boundary_info, _ = store.segment_event_frames(recs)
                
                if enable_event_finalize:
                    job_manager.progress(job_id, phase="event_finalize", current=0, total=max(1, len(segments)))
                    for idx, segment in enumerate(segments, start=1):
                        await asyncio.wait_for(
                            store.finalize_event_memory(frames=segment, source_video_path=tmp_path),
                            timeout=max(1.0, float(event_finalize_timeout_s)),
                        )
                        job_manager.progress(job_id, phase="event_finalize", current=idx, total=max(1, len(segments)))
                
                report["event_filter"] = {
                    "threshold": boundary_info,
                    "dropped": 0,
                    "removed": 0,
                    "dropped_detail": [],
                    "kept": int(len(recs)),
                    "segments": int(len(segments)),
                    "source_signal": "embedding_distance_between_frames",
                }
                await store.set_latest_filters(
                    dedup_filter=report.get("dedup_filter"),
                    event_filter=report.get("event_filter"),
                )
                report["kept_post_event"] = int(len(recs))
                report["retention"] = await store.apply_retention_policy()
                kept_ids = {f.frame_id for f in await store.list_frames()}
                added = [item for item in added if item["frame_id"] in kept_ids]

            result = {
                "count": len(added),
                "added": added,
                "fps": float(fps),
                "source_fps": float(src_fps),
                "truncated": bool(truncated),
                "filter_report": report,
            }
            job_manager.done(job_id, result=result)
        except Exception as e:
            job_manager.log(job_id, f"error in runtime: {str(e)}")
            job_manager.fail(job_id, error=str(e))
        finally:
            if cleanup_temp:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    task = asyncio.create_task(_run())
    job_manager.attach_task(job_id, task)
    return job_id
