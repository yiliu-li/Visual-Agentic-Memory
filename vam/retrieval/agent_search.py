from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

try:
    import torch
except Exception:
    torch = None

from vam.config import get_settings
from vam.protocol import PlannerActionAdapter, PlannerQuerySpec
from vam.llm.openrouter import OpenRouterClient
from vam import prompts
from vam.retrieval.frame_store import store as default_store, FrameStore

logger = logging.getLogger(__name__)


def _format_time_context(frames: List[Any]) -> str:
    if not frames:
        return "No indexed frames available"
    has_abs = any(f.absolute_t is not None for f in frames)
    if has_abs:
        abs_values = [float(f.absolute_t) for f in frames if f.absolute_t is not None]
        if abs_values:
            return f"Absolute Timestamps available (Range: {min(abs_values)} to {max(abs_values)})"
    rel_values = [float(f.t) for f in frames]
    if rel_values:
        return f"Relative Time (0s start) only (Range: 0.0 to {max(rel_values):.1f}s)"
    return "Relative Time (0s start) only"


def _normalize_time_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in {"relative", "absolute", "auto"}:
        return mode
    return "auto"


def _format_result_time_range(result: Dict[str, Any]) -> str:
    tr = result.get("time_range") if isinstance(result.get("time_range"), dict) else {}
    rel_start = tr.get("start_t")
    rel_end = tr.get("end_t")
    abs_start = tr.get("absolute_start_t")
    abs_end = tr.get("absolute_end_t")
    parts: List[str] = []
    if rel_start is not None or rel_end is not None:
        start_v = float(rel_start if rel_start is not None else rel_end if rel_end is not None else 0.0)
        end_v = float(rel_end if rel_end is not None else rel_start if rel_start is not None else start_v)
        parts.append(f"rel={start_v:.1f}s->{end_v:.1f}s")
    if abs_start is not None or abs_end is not None:
        start_v = float(abs_start if abs_start is not None else abs_end if abs_end is not None else 0.0)
        end_v = float(abs_end if abs_end is not None else abs_start if abs_start is not None else start_v)
        parts.append(f"abs={start_v:.3f}->{end_v:.3f}")
    if not parts:
        if result.get("absolute_t") is not None:
            parts.append(f"abs@{float(result['absolute_t']):.3f}")
        parts.append(f"t={float(result.get('t', 0.0) or 0.0):.1f}s")
    return " ".join(parts)


def _extract_result_time_window(
    result: Dict[str, Any],
    *,
    time_mode: str,
) -> Tuple[Optional[float], Optional[float], str]:
    requested_mode = _normalize_time_mode(time_mode)
    tr = result.get("time_range") if isinstance(result.get("time_range"), dict) else {}
    has_absolute = any(
        result.get(key) is not None
        for key in ("absolute_t",)
    ) or any(tr.get(key) is not None for key in ("absolute_start_t", "absolute_end_t"))
    use_absolute = requested_mode == "absolute" or (requested_mode == "auto" and has_absolute)
    resolved_mode = "absolute" if use_absolute else "relative"
    if use_absolute:
        start_v = tr.get("absolute_start_t")
        end_v = tr.get("absolute_end_t")
        point_v = result.get("absolute_t")
    else:
        start_v = tr.get("start_t")
        end_v = tr.get("end_t")
        point_v = result.get("t")
    if start_v is None and point_v is not None:
        start_v = point_v
    if end_v is None and point_v is not None:
        end_v = point_v
    if start_v is None and end_v is not None:
        start_v = end_v
    if end_v is None and start_v is not None:
        end_v = start_v
    if start_v is None or end_v is None:
        return None, None, resolved_mode
    start_f = float(start_v)
    end_f = float(end_v)
    if end_f < start_f:
        start_f, end_f = end_f, start_f
    return start_f, end_f, resolved_mode


def _resolve_context_result_ref(context: List[Dict[str, Any]], ref: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(ref, dict):
        return None
    try:
        turn_idx = int(ref.get("turn_idx", 1)) - 1
        result_idx = int(ref.get("result_idx", 0))
    except Exception:
        return None
    if not (0 <= turn_idx < len(context)):
        return None
    results = context[turn_idx].get("results") or []
    if not (0 <= result_idx < len(results)):
        return None
    candidate = results[result_idx]
    return candidate if isinstance(candidate, dict) else None


def _build_context_string(context: List[Dict[str, Any]], frames: List[Any]) -> str:
    lines = [f"Time Context: {_format_time_context(frames)}", ""]
    if not context:
        lines.append("(No findings yet)")
        return "\n".join(lines)

    for i, item in enumerate(context, start=1):
        lines.append(f"Turn {i} Query: '{item['query']}'")
        notes = item.get("notes") or []
        for note in notes:
            note_text = " ".join(str(note or "").split()).strip()
            if note_text:
                lines.append(f"  -> Note: {note_text}")
        results = item.get("results") or []
        if not results:
            lines.append("  -> No relevant matches found.")
            lines.append("")
            continue
        ordered_candidates = _format_ordered_candidate_refs(results, time_mode="auto", turn_idx=i)
        if ordered_candidates:
            lines.append(f"  -> Ordered candidate refs by time: {ordered_candidates}")
        for ridx, result in enumerate(results):
            desc = (result.get("inspection") or result.get("caption") or "No description").strip()
            source = str(result.get("source") or "unknown")
            layer = str(result.get("layer") or "-")
            kind = str(result.get("kind") or "-")
            time_bits = _format_result_time_range(result)
            ref_bits = f"ref=({i},{ridx})"
            identity_bits = []
            if result.get("doc_id"):
                identity_bits.append(f"doc_id={result.get('doc_id')}")
            if result.get("frame_id"):
                identity_bits.append(f"frame_id={result.get('frame_id')}")
            identity_text = " " + " ".join(identity_bits) if identity_bits else ""
            lines.append(
                f"  -> Result {ridx}: {ref_bits} {time_bits} "
                f"score={float(result.get('score', 0.0)):.2f} source={source} layer={layer} kind={kind}{identity_text} {desc}"
            )
        lines.append("")
    return "\n".join(lines)


def _build_chat_history_string(chat_history: Optional[List[Dict[str, Any]]]) -> str:
    if not chat_history:
        return "(No prior chat turns)"
    lines: List[str] = []
    for idx, item in enumerate(chat_history[-8:], start=1):
        role = str(item.get("role") or "unknown").strip() or "unknown"
        content = " ".join(str(item.get("content") or "").split()).strip()
        if not content:
            continue
        lines.append(f"{idx}. {role}: {content}")
    return "\n".join(lines) if lines else "(No prior chat turns)"


def _format_summary_structures(structures: List[Dict[str, Any]]) -> str:
    if not structures:
        return "(No summary structures currently available)"
    lines: List[str] = []
    for item in structures:
        summary_structure = str(item.get("summary_structure") or "").strip()
        granularity = float(item.get("granularity_seconds") or 0.0)
        focus = " ".join(str(item.get("focus") or "").split()).strip()
        count = int(item.get("count") or 0)
        min_start = float(item.get("min_start_t") or 0.0)
        max_end = float(item.get("max_end_t") or 0.0)
        focus_part = f" focus='{focus}'" if focus else ""
        structure_part = f" structure={summary_structure}" if summary_structure else ""
        lines.append(
            f"- granularity={granularity:.1f}s windows={count} range={min_start:.1f}s->{max_end:.1f}s{structure_part}{focus_part}"
        )
    return "\n".join(lines)


def _collect_hits(context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for turn_item in context:
        for result in (turn_item.get("results") or []):
            if isinstance(result, dict):
                hits.append(result)
    return hits


def _best_hit(context: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    hits = _collect_hits(context)
    if not hits:
        return None
    best: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    for hit in hits:
        try:
            score = float(hit.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        if best is None or score > best_score:
            best = hit
            best_score = score
    return best


def _has_evidence(context: List[Dict[str, Any]], *, min_score: float = 0.12) -> bool:
    best = _best_hit(context)
    if not best:
        return False
    try:
        score = float(best.get("score", 0.0) or 0.0)
    except Exception:
        score = 0.0
    return score >= float(min_score)


async def _resolve_visual_reference(
    *,
    store: FrameStore,
    context: List[Dict[str, Any]],
    visual_ref: Any,
) -> Tuple[Optional[List[float]], Optional[str]]:
    if not isinstance(visual_ref, dict):
        return None, None
    try:
        turn_idx = int(visual_ref.get("turn_idx", 1)) - 1
        result_idx = int(visual_ref.get("result_idx", 0))
    except Exception:
        return None, None
    if not (0 <= turn_idx < len(context)):
        return None, None
    turn_results = context[turn_idx].get("results") or []
    if not (0 <= result_idx < len(turn_results)):
        return None, None

    result = turn_results[result_idx]
    frame_id = str(result.get("frame_id") or "").strip()
    if not frame_id:
        return None, None
    frame = await store.get_frame_by_id(frame_id)
    if frame is None or frame.img_emb is None:
        return None, None
    return frame.img_emb.float().tolist(), f"Visual Match (Turn {turn_idx + 1} Result {result_idx})"


def _frame_from_result(result: Dict[str, Any]) -> "FrameRecord":
    from .frame_store import FrameRecord

    return FrameRecord(
        frame_id=str(result.get("frame_id") or "temp"),
        t=float(result.get("t", 0.0) or 0.0),
        image_data_uri=str(result.get("image") or ""),
        img_emb=None,
        absolute_t=result.get("absolute_t"),
    )


def _unique_texts(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        out.append(text)
        seen.add(key)
    return out


def _anchor_query_list(anchor: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    query = str(anchor.get("query") or "").strip()
    if query:
        values.append(query)
    for item in (anchor.get("query_variants") or []):
        text = str(item or "").strip()
        if text:
            values.append(text)
    return _unique_texts(values)


def _candidate_window_text(result: Dict[str, Any]) -> str:
    return _format_result_time_range(result)


def _candidate_hint_text(result: Dict[str, Any]) -> str:
    parts = [
        str(result.get("caption") or "").strip(),
        str(result.get("inspection") or "").strip(),
        str(result.get("document_text") or "").strip(),
    ]
    return " | ".join(part for part in parts if part)


def _anchor_source_groups(anchor: Dict[str, Any]) -> List[List[str]]:
    explicit_groups = anchor.get("candidate_source_groups")
    if isinstance(explicit_groups, list):
        groups: List[List[str]] = []
        seen: set[tuple[str, ...]] = set()
        for group in explicit_groups:
            if not isinstance(group, list):
                continue
            cleaned = [str(source).strip().lower() for source in group if str(source).strip()]
            if not cleaned:
                continue
            key = tuple(cleaned)
            if key in seen:
                continue
            groups.append(cleaned)
            seen.add(key)
        if groups:
            return groups
    requested_sources = [
        str(source).strip().lower()
        for source in (anchor.get("sources") or [])
        if str(source).strip()
    ]
    return [requested_sources] if requested_sources else [[]]


def _anchor_candidate_rank(item: Dict[str, Any]) -> Tuple[float, float]:
    score = float(item.get("score", 0.0) or 0.0)
    confidence = 0.0
    if isinstance(item.get("anchor_review"), dict):
        confidence = float(item.get("anchor_review", {}).get("confidence", 0.0) or 0.0)
    return confidence, score


def _result_time_sort_key(result: Dict[str, Any], *, time_mode: str) -> float:
    start_t, end_t, resolved_mode = _extract_result_time_window(result, time_mode=time_mode)
    if start_t is not None:
        return float(start_t)
    if resolved_mode == "absolute" and result.get("absolute_t") is not None:
        return float(result.get("absolute_t") or 0.0)
    return float(result.get("t", 0.0) or 0.0)


def _summarize_result_times(
    results: List[Dict[str, Any]],
    *,
    time_mode: str,
    limit: int = 6,
    turn_idx: Optional[int] = None,
) -> List[str]:
    if not results:
        return []
    ordered = sorted(results, key=lambda item: _result_time_sort_key(item, time_mode=time_mode))
    windows: List[str] = []
    seen: set[str] = set()
    for result in ordered:
        label = _format_result_time_range(result)
        if label in seen:
            continue
        windows.append(label)
        seen.add(label)
        if len(windows) >= max(1, int(limit)):
            break
    if not windows:
        return []
    notes = ["Returned time candidates: " + " | ".join(windows)]
    ordered_candidates = _format_ordered_candidate_refs(results, time_mode=time_mode, turn_idx=turn_idx, limit=limit)
    if ordered_candidates:
        notes.append("Candidate occurrences ordered by time: " + ordered_candidates)
    if len(windows) >= 2:
        notes.append(
            "If the question depends on first, second, last, before, after, or between occurrences, decide from these candidate refs and returned times first."
        )
        notes.append(
            "If visual confirmation is still needed, inspect only the ambiguous candidate ref or the narrow interval between the chosen refs, not a broad superset window."
        )
    return notes


def _sample_frames_evenly(frames: List[Any], max_frames: int) -> List[Any]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return list(frames)
    if max_frames == 1:
        return [frames[len(frames) // 2]]
    selected: List[Any] = []
    used: set[int] = set()
    last_index = len(frames) - 1
    for i in range(max_frames):
        idx = int(round(i * last_index / (max_frames - 1)))
        if idx in used:
            continue
        selected.append(frames[idx])
        used.add(idx)
    if len(selected) < max_frames:
        for idx, frame in enumerate(frames):
            if idx in used:
                continue
            selected.append(frame)
            used.add(idx)
            if len(selected) >= max_frames:
                break
    return selected[:max_frames]


def _format_ordered_candidate_refs(
    results: List[Dict[str, Any]],
    *,
    time_mode: str,
    turn_idx: Optional[int] = None,
    limit: int = 6,
) -> str:
    if len(results) < 2:
        return ""
    ordered_pairs = sorted(
        list(enumerate(results)),
        key=lambda pair: _result_time_sort_key(pair[1], time_mode=time_mode),
    )
    parts: List[str] = []
    seen: set[Tuple[str, str]] = set()
    for ridx, result in ordered_pairs:
        label = _format_result_time_range(result)
        identity = str(result.get("doc_id") or result.get("frame_id") or "")
        key = (label, identity)
        if key in seen:
            continue
        seen.add(key)
        occurrence_idx = len(parts) + 1
        ref_text = f"ref=({turn_idx},{ridx}) " if turn_idx is not None else ""
        parts.append(f"#{occurrence_idx} {ref_text}{label}")
        if len(parts) >= max(1, int(limit)):
            break
    return " | ".join(parts)


def _summarize_anchor_events(events: List[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    for event in events:
        event_type = str(event.get("type") or "")
        role = str(event.get("role") or "anchor")
        if event_type == "anchor_search_start":
            queries = event.get("queries") or ([event.get("query")] if event.get("query") else [])
            sources = event.get("sources")
            notes.append(
                f"{role} attempt started with queries={queries} sources={sources} top_k={event.get('top_k')} inspect_k={event.get('inspect_k')}."
            )
        elif event_type == "anchor_candidate_review_done":
            candidates = event.get("candidates") or []
            verdicts: List[str] = []
            for candidate in candidates[:4]:
                review = candidate.get("anchor_review") or {}
                verdicts.append(
                    f"{_candidate_window_text(candidate)} match={review.get('match')} confidence={review.get('confidence')} caption={candidate.get('caption')!r}"
                )
            if verdicts:
                notes.append(f"{role} review results: " + " | ".join(verdicts))
        elif event_type == "anchor_search_failed":
            notes.append(f"{role} attempt failed.")
        elif event_type == "time_range_resolved":
            resolved = event.get("resolved") or {}
            notes.append(
                f"{role} resolved to min={resolved.get('min')} max={resolved.get('max')} mode={resolved.get('mode')}."
            )
    return notes


async def _frames_for_anchor_result(
    *,
    store: FrameStore,
    result: Dict[str, Any],
    time_mode: str,
    max_frames: int = 6,
) -> List[Any]:
    frames: List[Any] = []
    frame_ids = [str(frame_id).strip() for frame_id in (result.get("frame_ids") or []) if str(frame_id).strip()]
    if frame_ids:
        frames = await store.list_frames_by_ids(frame_ids)
    if not frames:
        start_t, end_t, resolved_mode = _extract_result_time_window(result, time_mode=time_mode)
        if start_t is not None and end_t is not None:
            min_time = float(start_t)
            max_time = float(end_t)
            if max_time <= min_time:
                pad = 1.5
                if resolved_mode != "absolute":
                    min_time = max(0.0, min_time - pad)
                else:
                    min_time = min_time - pad
                max_time = float(end_t) + pad
            frames = await store.list_frames_in_time_range(
                min_time=min_time,
                max_time=max_time,
                time_mode=resolved_mode,
            )
    if not frames:
        frame_id = str(result.get("frame_id") or "").strip()
        if frame_id:
            frame = await store.get_frame_by_id(frame_id)
            if frame is not None:
                if _source_bucket(result) == "frame":
                    center = float(frame.absolute_t) if _normalize_time_mode(time_mode) == "absolute" and frame.absolute_t is not None else float(frame.t)
                    pad = 2.0
                    min_time = center - pad
                    max_time = center + pad
                    if _normalize_time_mode(time_mode) != "absolute":
                        min_time = max(0.0, min_time)
                    nearby_frames = await store.list_frames_in_time_range(
                        min_time=min_time,
                        max_time=max_time,
                        time_mode=time_mode,
                    )
                    frames = nearby_frames or [frame]
                else:
                    frames = [frame]
    if not frames:
        return []
    use_absolute = _normalize_time_mode(time_mode) == "absolute" and any(getattr(frame, "absolute_t", None) is not None for frame in frames)
    frames.sort(
        key=lambda frame: float(frame.absolute_t) if use_absolute and frame.absolute_t is not None else float(frame.t)
    )
    return _sample_frames_evenly(frames, max_frames)


async def _adjudicate_anchor_candidates(
    *,
    store: FrameStore,
    candidates: List[Dict[str, Any]],
    anchor: Dict[str, Any],
    role: str,
    time_mode: str,
    default_target_event: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    inspect_k = max(0, min(len(candidates), int(anchor.get("inspect_k", anchor.get("top_k", 3)) or 0)))
    if inspect_k <= 0:
        return list(candidates), []
    review_events: List[Dict[str, Any]] = [
        {
            "type": "anchor_candidate_review_start",
            "role": role,
            "inspect_k": inspect_k,
            "candidate_count": len(candidates),
        }
    ]
    reviewed = list(candidates)
    target_event = " ".join(default_target_event.split()).strip()
    verification_prompt = str(anchor.get("verification_prompt") or "").strip() or None
    review_candidates = list(reviewed[:inspect_k])
    inspect_k = len(review_candidates)

    async def _review_one(candidate: Dict[str, Any]) -> Dict[str, Any]:
        frames = await _frames_for_anchor_result(
            store=store,
            result=candidate,
            time_mode=time_mode,
            max_frames=6,
        )
        judgment = await store.judge_frames_for_anchor(
            frames=frames,
            target_event=target_event,
            candidate_hint=_candidate_hint_text(candidate),
            verification_prompt=verification_prompt,
            candidate_window=_candidate_window_text(candidate),
        )
        output = dict(candidate)
        output["anchor_review"] = {
            "match": bool(judgment.get("match")),
            "confidence": float(judgment.get("confidence", 0.0) or 0.0),
            "observed_event": str(judgment.get("observed_event") or "").strip(),
            "reason": str(judgment.get("reason") or "").strip(),
        }
        return output

    reviewed_subset = await asyncio.gather(*[_review_one(candidate) for candidate in review_candidates])
    reviewed_lookup = {
        str(candidate.get("doc_id") or candidate.get("frame_id") or id(candidate)): candidate
        for candidate in reviewed_subset
    }
    rewritten: List[Dict[str, Any]] = []
    for candidate in reviewed:
        key = str(candidate.get("doc_id") or candidate.get("frame_id") or id(candidate))
        rewritten.append(reviewed_lookup.get(key, candidate))
    reviewed = rewritten
    review_events.append(
        {
            "type": "anchor_candidate_review_done",
            "role": role,
            "candidates": [
                {
                    "frame_id": candidate.get("frame_id"),
                    "t": candidate.get("t"),
                    "absolute_t": candidate.get("absolute_t"),
                    "score": candidate.get("score"),
                    "caption": candidate.get("caption"),
                    "time_range": candidate.get("time_range"),
                    "anchor_review": candidate.get("anchor_review"),
                }
                for candidate in review_candidates
            ],
        }
    )
    return reviewed, review_events


async def _inspect_results(
    *,
    store: FrameStore,
    results: List[Dict[str, Any]],
    inspect_k: int,
    prompt: str,
    joint: bool,
) -> List[Dict[str, Any]]:
    inspect_n = max(0, min(len(results), int(inspect_k)))
    if inspect_n == 0:
        return list(results)

    inspected = list(results[:inspect_n])
    untouched = list(results[inspect_n:])

    if joint:
        frames = [_frame_from_result(r) for r in inspected]
        joint_result = await store.inspect_frames(frames, prompt)
        for r in inspected:
            r["inspection"] = f"[Joint Analysis]: {joint_result}"
        return inspected + untouched

    inspections = await asyncio.gather(
        *[store.inspect_frame(_frame_from_result(r), prompt) for r in inspected]
    )
    for r, inspection in zip(inspected, inspections):
        r["inspection"] = inspection
    return inspected + untouched


async def hybrid_event_frame_search(
    query: str,
    top_k: int = 5,
    threshold: float = 0.65,
    store_instance: Optional[FrameStore] = None,
    min_time: float = 0.0,
    max_time: float = float('inf'),
    time_mode: str = "auto",
    image_query_emb: Optional[List[float]] = None,
    sources: Optional[List[str]] = None,
    summary_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid Agentic Retrieval over the Parallel Representation.
    Searches Spatial Representation frame embeddings and Temporal Representation documents.
    If image_query_emb is provided, use that for visual similarity instead of a text query.
    time_mode: "auto", "relative", or "absolute".
    """
    if torch is None:
        return []

    store = store_instance or default_store
    all_frames = await store.list_frames()
    source_set = {str(source).strip().lower() for source in (sources or []) if str(source).strip()}
    use_frame = not source_set or "frame" in source_set or image_query_emb is not None
    use_event = not source_set or "event" in source_set
    use_summary = not source_set or "summary" in source_set

    if min_time > 0 or max_time < float('inf'):
        frames = await store.list_frames_in_time_range(
            min_time=min_time,
            max_time=max_time,
            time_mode=time_mode,
        )
    else:
        frames = list(all_frames)

    # 1. Retrieve candidates from the Parallel Representation.
    if image_query_emb:
        if not frames:
            return []
        # Use the provided image embedding for Spatial Retrieval.
        q_vec = torch.tensor(image_query_emb, dtype=torch.float32).flatten()
    else:
        # Use the text query for retrieval over frame and document stores.
        q_emb = await store.embed_texts(texts=[query])
        q_vec = q_emb[0]

    # Retrieve a slightly broader candidate pool first, then filter down.
    base_k = max(1, int(top_k))
    frame_k = base_k
    event_k = base_k
    summary_k = base_k
    candidates = store.topk_by_image_text(frames=frames, query_text_emb=q_vec, top_k=frame_k) if (frames and use_frame) else []
    if not candidates:
        candidates = []
    frame_lookup = {str(f.frame_id): f for f in all_frames}

    long_doc_hits: List[Dict[str, Any]] = []
    summary_doc_hits: List[Dict[str, Any]] = []
    pooled_doc_hits: List[Dict[str, Any]] = []
    if not image_query_emb:
        if use_event:
            long_doc_hits = await store.search_memory_documents(
                query=query,
                top_k=event_k,
                layer="long",
                kind="event",
                min_time=min_time,
                max_time=max_time,
                time_mode=time_mode,
            )
        if use_summary:
            summary_doc_hits = await store.search_memory_documents(
                query=query,
                top_k=summary_k,
                layer="summary",
                kind="summary",
                min_time=min_time,
                max_time=max_time,
                time_mode=time_mode,
                summary_filter=summary_filter,
            )
        pooled_doc_hits = summary_doc_hits + long_doc_hits
        for item in pooled_doc_hits:
            img = str(item.get("image") or "")
            fid = str(item.get("frame_id") or "")
            if (not img) and fid and fid in frame_lookup:
                item["image"] = frame_lookup[fid].image_data_uri

    scored: List[Dict[str, Any]] = []

    # 2. Score candidates before Visual Inspection.
    for frame, img_sim in candidates:
        if img_sim < 0.05:
            continue
        score = img_sim

        scored.append({
                "frame_id": frame.frame_id,
                "t": frame.t,
                "absolute_t": frame.absolute_t,
                "image": frame.image_data_uri,
                "caption": "",
                "score": score,
                "source": "frame",
                "layer": str(frame.memory_tier or ""),
                "img_sim": img_sim,
                "text_sim": 0.0,
            })

    merged: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    def _append_results(items: List[Dict[str, Any]]) -> None:
        for item in items:
            key = str(item.get("doc_id") or item.get("frame_id") or "")
            if not key or key in seen_keys:
                continue
            merged.append(item)
            seen_keys.add(key)

    combined = pooled_doc_hits + scored

    combined.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    _append_results(combined)

    eff_threshold = threshold * 0.8

    strong = [r for r in merged if float(r.get("score", 0.0)) >= eff_threshold]
    if strong:
        return strong[:top_k]

    return merged[:top_k]


async def search_generator(
    semantic: str,
    store_instance: Optional[FrameStore] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming Agentic Retrieval generator.
    Yields events:
    - {"type": "plan", "content": "..."}
    - {"type": "search_start", "queries": [...]}
    - {"type": "search_done", "query": "...", "hits": N, "results": [...]}
    - {"type": "inspection_start", "count": N, "query": "..."}
    - {"type": "inspection_done", "duration": 1.23}
    - {"type": "answer", "content": "..."}
    - {"type": "final", "result": {...}}
    """
    if torch is None:
        yield {"type": "error", "message": "torch is required"}
        return

    store = store_instance or default_store
    cfg = get_settings()
    planner = OpenRouterClient(model_id=cfg.openrouter_model_id_main or cfg.openrouter_model_id)

    t0 = time.time()
    goal = semantic
    context: List[Dict[str, Any]] = []
    try:
        max_turns = max(1, int(os.getenv("VAM_AGENT_MAX_TURNS", "20")))
    except Exception:
        max_turns = 20
    final_response = ""
    prev_unique_hits = 0
    stagnant_turns = 0
    chat_history_str = _build_chat_history_string(chat_history)

    def _build_fallback_answer() -> str:
        hits: List[Dict[str, Any]] = []
        for turn_item in context:
            for r in (turn_item.get("results") or []):
                hits.append(r)
        if not hits:
            return "No definitive match found."
        hits = sorted(hits, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        top = hits[:5]
        lines = ["Best matches:"]
        for r in top:
            t = float(r.get("t", 0.0) or 0.0)
            s = float(r.get("score", 0.0) or 0.0)
            desc = (r.get("inspection") or r.get("caption") or "").strip()
            if desc:
                lines.append(f"- {t:.1f}s (score {s:.2f}): {desc}")
            else:
                lines.append(f"- {t:.1f}s (score {s:.2f})")
        return "\n".join(lines)

    def _coerce_query_specs(raw: Any) -> List[Dict[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, dict):
            raw = [raw]
        out: List[Dict[str, Any]] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    q = item.strip()
                    if q:
                        out.append({"q": q})
                elif isinstance(item, dict):
                    q = str(item.get("q", item.get("query", "")) or "").strip()
                    if not q:
                        continue
                    spec: Dict[str, Any] = {"q": q}
                    if "top_k" in item:
                        spec["top_k"] = item.get("top_k")
                    if "inspect_k" in item:
                        spec["inspect_k"] = item.get("inspect_k")
                    if "threshold" in item:
                        spec["threshold"] = item.get("threshold")
                    out.append(spec)
        normalized: List[Dict[str, Any]] = []
        for spec in out:
            try:
                normalized.append(PlannerQuerySpec.model_validate(spec).model_dump())
            except Exception:
                continue
        return normalized

    def _safe_int(v: Any, default: int) -> int:
        try:
            i = int(v)
            return i
        except Exception:
            return int(default)

    def _safe_float(v: Any, default: float) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    def _parse_plan_text(raw: str) -> Dict[str, Any]:
        def _sanitize_payload(v: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(v)
            if isinstance(out.get("visual_ref"), dict) and not out.get("visual_ref"):
                out.pop("visual_ref", None)
            return out

        clean_text = (raw or "").strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        try:
            v = json.loads(clean_text)
            if isinstance(v, dict):
                return PlannerActionAdapter.parse_payload(_sanitize_payload(v)).model_dump(exclude_none=True)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", clean_text)
        if m:
            v = json.loads(m.group(0))
            if isinstance(v, dict):
                return PlannerActionAdapter.parse_payload(_sanitize_payload(v)).model_dump(exclude_none=True)
        raise ValueError("planner output is not valid JSON object")

    def _render_results_for_event(results: List[Dict[str, Any]], *, limit: int = 5) -> List[Dict[str, Any]]:
        rendered: List[Dict[str, Any]] = []
        for result in results[: max(1, int(limit))]:
            rendered.append(
                {
                    "frame_id": result.get("frame_id"),
                    "t": result.get("t"),
                    "absolute_t": result.get("absolute_t"),
                    "score": result.get("score"),
                    "image": result.get("image"),
                    "caption": result.get("caption", ""),
                    "source": result.get("source"),
                    "layer": result.get("layer"),
                    "kind": result.get("kind"),
                    "doc_id": result.get("doc_id"),
                    "frame_ids": result.get("frame_ids"),
                    "metadata": result.get("metadata"),
                    "time_range": result.get("time_range"),
                }
            )
        return rendered

    async def _resolve_requested_time_range(
        raw_time_range: Any,
    ) -> Tuple[float, float, str, Optional[Dict[str, Any]], List[Dict[str, Any]], bool]:
        time_range = raw_time_range if isinstance(raw_time_range, dict) else {}
        min_raw = time_range.get("min", 0.0)
        max_raw = time_range.get("max", None)
        min_t = _safe_float(min_raw, 0.0)
        max_t = float("inf") if max_raw in (None, "") else _safe_float(max_raw, float("inf"))
        requested_mode = _normalize_time_mode(time_range.get("mode", "auto"))
        events: List[Dict[str, Any]] = []

        async def _resolve_anchor_spec(
            anchor_spec: Any,
            *,
            role: str,
        ) -> Tuple[Optional[float], Optional[float], str, Optional[Dict[str, Any]], List[Dict[str, Any]], bool]:
            anchor = anchor_spec if isinstance(anchor_spec, dict) else {}
            local_events: List[Dict[str, Any]] = []
            anchor_result: Optional[Dict[str, Any]] = None

            anchor_ref = anchor.get("ref")
            if isinstance(anchor_ref, dict):
                anchor_result = _resolve_context_result_ref(context, anchor_ref)
                if anchor_result is not None:
                    local_events.append(
                        {
                            "type": "anchor_ref_resolved",
                            "role": role,
                            "ref": anchor_ref,
                            "result": {
                                "frame_id": anchor_result.get("frame_id"),
                                "t": anchor_result.get("t"),
                                "absolute_t": anchor_result.get("absolute_t"),
                                "score": anchor_result.get("score"),
                                "caption": anchor_result.get("caption", ""),
                                "time_range": anchor_result.get("time_range"),
                            },
                        }
                    )

            if anchor_result is None:
                anchor_queries = _anchor_query_list(anchor)
                if anchor_queries:
                    anchor_source_groups = _anchor_source_groups(anchor)
                    anchor_top_k = max(1, min(50, _safe_int(anchor.get("top_k", 5), 5)))
                    anchor_threshold = _safe_float(anchor.get("threshold", 0.0), 0.0)
                    local_events.append(
                        {
                            "type": "anchor_search_start",
                            "role": role,
                            "query": anchor_queries[0] if len(anchor_queries) == 1 else None,
                            "queries": anchor_queries,
                            "sources": anchor_source_groups,
                            "top_k": anchor_top_k,
                            "inspect_k": max(0, min(anchor_top_k, _safe_int(anchor.get("inspect_k", 3), 3))),
                            "threshold": anchor_threshold,
                        }
                    )
                    hit_lists = await asyncio.gather(
                        *[
                            hybrid_event_frame_search(
                                anchor_query,
                                top_k=anchor_top_k,
                                threshold=anchor_threshold,
                                store_instance=store,
                                min_time=min_t,
                                max_time=max_t,
                                time_mode=requested_mode,
                                sources=source_group,
                            )
                            for anchor_query in anchor_queries
                            for source_group in anchor_source_groups
                        ]
                    )
                    merged_by_key: Dict[str, Dict[str, Any]] = {}
                    combo_idx = 0
                    for anchor_query in anchor_queries:
                        for source_group in anchor_source_groups:
                            hits = hit_lists[combo_idx]
                            combo_idx += 1
                            for hit in hits:
                                candidate = dict(hit)
                                candidate["anchor_query"] = anchor_query
                                candidate["anchor_sources"] = list(source_group)
                                key = str(candidate.get("doc_id") or candidate.get("frame_id") or f"{anchor_query}:{candidate.get('t')}:{candidate.get('absolute_t')}")
                                existing = merged_by_key.get(key)
                                if existing is None or _anchor_candidate_rank(candidate) > _anchor_candidate_rank(existing):
                                    merged_by_key[key] = candidate
                    merged_candidates = sorted(
                        merged_by_key.values(),
                        key=_anchor_candidate_rank,
                        reverse=True,
                    )
                    local_events.append(
                        {
                            "type": "anchor_search_done",
                            "role": role,
                            "query": anchor_queries[0] if len(anchor_queries) == 1 else None,
                            "queries": anchor_queries,
                            "hits": len(merged_candidates),
                            "results": _render_results_for_event(merged_candidates, limit=5),
                        }
                    )
                    reviewed_candidates, review_events = await _adjudicate_anchor_candidates(
                        store=store,
                        candidates=merged_candidates,
                        anchor=anchor,
                        role=role,
                        time_mode=requested_mode,
                        default_target_event=anchor_queries[0],
                    )
                    local_events.extend(review_events)
                    occurrence_index = anchor.get("occurrence_index")
                    verified = [
                        candidate for candidate in reviewed_candidates
                        if isinstance(candidate.get("anchor_review"), dict) and bool(candidate["anchor_review"].get("match"))
                    ]
                    if occurrence_index is not None:
                        ordered = sorted(
                            verified,
                            key=lambda item: _result_time_sort_key(item, time_mode=requested_mode),
                        )
                        index = max(1, _safe_int(occurrence_index, 1))
                        if len(ordered) >= index:
                            anchor_result = ordered[index - 1]
                    elif verified:
                        anchor_result = max(
                            verified,
                            key=lambda item: (
                                float(item.get("anchor_review", {}).get("confidence", 0.0) or 0.0),
                                float(item.get("score", 0.0) or 0.0),
                            ),
                        )
                    elif reviewed_candidates and max(0, min(len(reviewed_candidates), _safe_int(anchor.get("inspect_k", 3), 3))) == 0:
                        anchor_result = reviewed_candidates[0]

            if anchor_result is None:
                local_events.append(
                    {
                        "type": "anchor_search_failed",
                        "role": role,
                        "anchor": {
                            "query": anchor.get("query"),
                            "query_variants": anchor.get("query_variants"),
                            "ref": anchor.get("ref"),
                            "occurrence_index": anchor.get("occurrence_index"),
                            "before_seconds": anchor.get("before_seconds", 0.0),
                            "after_seconds": anchor.get("after_seconds", 0.0),
                        },
                    }
                )
                return None, None, requested_mode, None, local_events, True

            anchor_start, anchor_end, resolved_mode = _extract_result_time_window(anchor_result, time_mode=requested_mode)
            if anchor_start is None or anchor_end is None:
                local_events.append(
                    {
                        "type": "anchor_search_failed",
                        "role": role,
                        "anchor": {
                            "query": anchor.get("query"),
                            "query_variants": anchor.get("query_variants"),
                            "ref": anchor.get("ref"),
                            "occurrence_index": anchor.get("occurrence_index"),
                        },
                        "message": "anchor_result_missing_time",
                    }
                )
                return None, None, requested_mode, None, local_events, True

            summary = {
                "role": role,
                "query": anchor.get("query"),
                "query_variants": anchor.get("query_variants"),
                "ref": anchor.get("ref"),
                "occurrence_index": anchor.get("occurrence_index"),
                "before_seconds": max(0.0, _safe_float(anchor.get("before_seconds", 0.0), 0.0)),
                "after_seconds": max(0.0, _safe_float(anchor.get("after_seconds", 0.0), 0.0)),
                "anchor_time_range": {
                    "min": anchor_start,
                    "max": anchor_end,
                    "mode": resolved_mode,
                },
                "matched_result": {
                    "frame_id": anchor_result.get("frame_id"),
                    "t": anchor_result.get("t"),
                    "absolute_t": anchor_result.get("absolute_t"),
                    "score": anchor_result.get("score"),
                    "caption": anchor_result.get("caption", ""),
                    "time_range": anchor_result.get("time_range"),
                    "anchor_query": anchor_result.get("anchor_query"),
                    "anchor_review": anchor_result.get("anchor_review"),
                },
            }
            return anchor_start, anchor_end, resolved_mode, summary, local_events, False

        single_anchor = time_range.get("anchor")
        start_anchor = time_range.get("start_anchor")
        end_anchor = time_range.get("end_anchor")

        if not isinstance(single_anchor, dict) and not (isinstance(start_anchor, dict) and isinstance(end_anchor, dict)):
            return min_t, max_t, requested_mode, None, events, False

        if isinstance(single_anchor, dict):
            anchor_start, anchor_end, resolved_mode, anchor_summary, anchor_events, anchor_failed = await _resolve_anchor_spec(
                single_anchor,
                role="anchor",
            )
            events.extend(anchor_events)
            if anchor_failed or anchor_start is None or anchor_end is None or anchor_summary is None:
                return min_t, max_t, requested_mode, None, events, True
            before_s = max(0.0, _safe_float(single_anchor.get("before_seconds", 0.0), 0.0))
            after_s = max(0.0, _safe_float(single_anchor.get("after_seconds", 0.0), 0.0))
            resolved_min = anchor_start - before_s
            resolved_max = anchor_end + after_s
            if resolved_mode != "absolute":
                resolved_min = max(0.0, resolved_min)
            if min_raw not in (None, ""):
                resolved_min = max(resolved_min, min_t)
            if max_raw not in (None, "") and math.isfinite(max_t):
                resolved_max = min(resolved_max, max_t)
            resolved_max = max(resolved_min, resolved_max)
            anchor_payload = {
                "mode": "single_anchor",
                **anchor_summary,
            }
            events.append(
                {
                    "type": "time_range_resolved",
                    "requested": time_range,
                    "resolved": {
                        "min": resolved_min,
                        "max": None if resolved_max == float("inf") else resolved_max,
                        "mode": resolved_mode,
                    },
                    "anchor": anchor_payload,
                }
            )
            return resolved_min, resolved_max, resolved_mode, anchor_payload, events, False

        start_min, start_max, start_mode, start_summary, start_events, start_failed = await _resolve_anchor_spec(
            start_anchor,
            role="start_anchor",
        )
        end_min, end_max, end_mode, end_summary, end_events, end_failed = await _resolve_anchor_spec(
            end_anchor,
            role="end_anchor",
        )
        events.extend(start_events)
        events.extend(end_events)
        if (
            start_failed
            or end_failed
            or start_min is None
            or start_max is None
            or end_min is None
            or end_max is None
            or start_summary is None
            or end_summary is None
        ):
            return min_t, max_t, requested_mode, None, events, True

        resolved_mode = start_mode if start_mode == end_mode else requested_mode
        start_before = max(0.0, _safe_float(start_anchor.get("before_seconds", 0.0), 0.0))
        start_after = max(0.0, _safe_float(start_anchor.get("after_seconds", 0.0), 0.0))
        end_before = max(0.0, _safe_float(end_anchor.get("before_seconds", 0.0), 0.0))
        end_after = max(0.0, _safe_float(end_anchor.get("after_seconds", 0.0), 0.0))
        resolved_min = float(start_max) - start_before + start_after
        resolved_max = float(end_min) - end_before + end_after
        if resolved_mode != "absolute":
            resolved_min = max(0.0, resolved_min)
            resolved_max = max(0.0, resolved_max)
        if min_raw not in (None, ""):
            resolved_min = max(resolved_min, min_t)
        if max_raw not in (None, "") and math.isfinite(max_t):
            resolved_max = min(resolved_max, max_t)
        if resolved_max < resolved_min:
            events.append(
                {
                    "type": "anchor_search_failed",
                    "role": "between_anchors",
                    "message": "resolved_anchor_interval_invalid",
                    "start_anchor": start_summary,
                    "end_anchor": end_summary,
                }
            )
            return min_t, max_t, requested_mode, None, events, True
        anchor_payload = {
            "mode": "between_anchors",
            "start_anchor": start_summary,
            "end_anchor": end_summary,
        }
        events.append(
            {
                "type": "time_range_resolved",
                "requested": time_range,
                "resolved": {
                    "min": resolved_min,
                    "max": None if resolved_max == float("inf") else resolved_max,
                    "mode": resolved_mode,
                },
                "anchor": anchor_payload,
            }
        )
        return resolved_min, resolved_max, resolved_mode, anchor_payload, events, False

    async def _resolve_inspection_target(
        *,
        raw_time_range: Any,
        raw_ref: Any,
        max_frames: int,
    ) -> Tuple[List[Any], Dict[str, Any], List[Dict[str, Any]], bool]:
        events: List[Dict[str, Any]] = []
        inspect_n = max(1, min(16, int(max_frames)))

        if isinstance(raw_ref, dict):
            ref_result = _resolve_context_result_ref(context, raw_ref)
            if ref_result is None:
                events.append(
                    {
                        "type": "inspect_target_failed",
                        "reason": "ref_not_found",
                        "ref": raw_ref,
                    }
                )
                return [], {}, events, True
            frames = await _frames_for_anchor_result(
                store=store,
                result=ref_result,
                time_mode="auto",
                max_frames=inspect_n,
            )
            start_t, end_t, resolved_mode = _extract_result_time_window(ref_result, time_mode="auto")
            payload = {
                "mode": "ref",
                "ref": raw_ref,
                "resolved_time_range": {
                    "min": start_t,
                    "max": end_t,
                    "mode": resolved_mode,
                },
                "matched_result": {
                    "frame_id": ref_result.get("frame_id"),
                    "doc_id": ref_result.get("doc_id"),
                    "t": ref_result.get("t"),
                    "absolute_t": ref_result.get("absolute_t"),
                    "time_range": ref_result.get("time_range"),
                },
            }
            if not frames:
                events.append(
                    {
                        "type": "inspect_target_failed",
                        "reason": "no_frames_for_ref",
                        "ref": raw_ref,
                    }
                )
                return [], payload, events, True
            events.append(
                {
                    "type": "inspect_target_resolved",
                    "target": payload,
                    "frame_count": len(frames),
                }
            )
            return frames, payload, events, False

        min_t, max_t, t_mode, anchor_summary, anchor_events, anchor_failed = await _resolve_requested_time_range(raw_time_range)
        events.extend(anchor_events)
        if anchor_failed:
            events.append(
                {
                    "type": "inspect_target_failed",
                    "reason": "time_range_unresolved",
                    "time_range": raw_time_range,
                }
            )
            return [], {"mode": "time_range", "time_range": raw_time_range, "anchor": anchor_summary}, events, True

        frames = await store.list_frames_in_time_range(
            min_time=min_t,
            max_time=max_t,
            time_mode=t_mode,
        )
        frames = _sample_frames_evenly(frames, inspect_n)
        payload = {
            "mode": "time_range",
            "time_range": {
                "min": min_t,
                "max": None if max_t == float("inf") else max_t,
                "mode": t_mode,
            },
            "anchor": anchor_summary,
        }
        if not frames:
            events.append(
                {
                    "type": "inspect_target_failed",
                    "reason": "no_frames_in_time_range",
                    "target": payload,
                }
            )
            return [], payload, events, True
        events.append(
            {
                "type": "inspect_target_resolved",
                "target": payload,
                "frame_count": len(frames),
            }
        )
        return frames, payload, events, False

    for turn in range(max_turns):
        frames_list = await store.list_frames()
        ctx_str = _build_context_string(context, frames_list)
        summary_structures = await store.list_summary_structures()
        summary_structures_str = _format_summary_structures(summary_structures)

        # 2. Call Planner
        yield {"type": "plan", "turn": turn + 1, "content": "Thinking..."}
        t_plan_start = time.time()

        messages = [
            {"role": "system", "content": prompts.agent_planner_system()},
            {
                "role": "user",
                "content": prompts.agent_planner_user(
                    goal=(
                        f"{goal}\n\n"
                        f"Recent chat history:\n{chat_history_str}"
                    ),
                    context=ctx_str,
                    remaining_turns=max_turns - turn,
                    available_summary_structures=summary_structures_str,
                ),
            },
        ]

        text, _ = await planner.chat(messages, temperature=0.1, max_tokens=500)
        yield {"type": "plan_done", "duration": time.time() - t_plan_start}

        # 3. Parse Action
        try:
            plan = _parse_plan_text(text)
        except Exception:
            logger.warning(
                "planner_invalid_json turn=%s goal=%s raw=%s",
                int(turn + 1),
                goal,
                (text or "").strip()[:1200],
            )
            try:
                retry_messages = [
                    *messages,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": "Your previous output was invalid. Output ONLY one valid JSON object (no code fences, no extra text) that matches the required schema.",
                    },
                ]
                text2, _ = await planner.chat(retry_messages, temperature=0.0, max_tokens=500)
                plan = _parse_plan_text(text2)
            except Exception:
                logger.error(
                    "planner_retry_failed turn=%s goal=%s raw2=%s",
                    int(turn + 1),
                    goal,
                    (locals().get("text2") or "").strip()[:1200],
                )
                yield {"type": "error", "message": "planner_output_invalid_json"}
            if turn == 0:
                plan = {"action": "search", "queries": [goal]}
            else:
                plan = {"action": "answer", "response": _build_fallback_answer()}

        action = plan.get("action")
        thought = plan.get("thought", "")
        # Use the planner-provided Visual Inspection prompt when available.
        inspection_prompt = plan.get("inspection_prompt", goal)

        yield {"type": "thought", "content": thought}

        if action == "answer":
            final_response = plan.get("response", "")
            best_frame = {"caption": final_response, "t": 0.0}

            best_ref = plan.get("best_ref")
            if isinstance(best_ref, dict):
                try:
                    t_idx = int(best_ref.get("turn_idx", 1)) - 1
                    r_idx = int(best_ref.get("result_idx", 0))
                    if 0 <= t_idx < len(context):
                        turn_res = context[t_idx].get("results") or []
                        if 0 <= r_idx < len(turn_res):
                            r = turn_res[r_idx]
                            best_frame = {
                                "caption": r.get("caption", "") or r.get("inspection", ""),
                                "t": r.get("t", 0.0),
                                "absolute_t": r.get("absolute_t"),
                                "image": r.get("image"),
                                "score": r.get("score", 0.0),
                            }
                            res = {
                                "found": _has_evidence(context),
                                "best": best_frame,
                                "answer": final_response,
                                "semantic_final": goal,
                                "turns": context,
                                "duration_s": time.time() - t0,
                            }
                            yield {"type": "answer", "content": final_response}
                            yield {"type": "final", "result": res}
                            return
                except Exception:
                    pass
            best_hit = _best_hit(context)
            if best_hit is not None:
                best_frame = {
                    "caption": best_hit.get("caption", "") or best_hit.get("inspection", ""),
                    "t": best_hit.get("t", 0.0),
                    "absolute_t": best_hit.get("absolute_t"),
                    "image": best_hit.get("image"),
                    "score": best_hit.get("score", 0.0),
                }
            res = {
                "found": _has_evidence(context),
                "best": best_frame,
                "answer": final_response,
                "semantic_final": goal,
                "turns": context,
                "duration_s": time.time() - t0,
            }
            yield {"type": "answer", "content": final_response}
            yield {"type": "final", "result": res}
            return

        elif action == "summarize":
            time_range = plan.get("time_range", {})
            min_t, max_t, t_mode, anchor_summary, anchor_events, anchor_failed = await _resolve_requested_time_range(time_range)
            for event in anchor_events:
                yield event
            granularity_s = max(5.0, _safe_float(plan.get("granularity_seconds", 60.0), 60.0))
            summary_structure = " ".join(str(plan.get("summary_structure") or "").split()).strip().lower() or None
            summary_prompt = str(plan.get("prompt") or goal).strip() or goal

            yield {
                "type": "tool_use",
                "tool": "summarize",
                "input": {
                    "time_range": {
                        "min": min_t,
                        "max": None if max_t == float("inf") else max_t,
                        "mode": t_mode,
                        "anchor": anchor_summary,
                    },
                    "granularity_seconds": granularity_s,
                    "prompt": summary_prompt,
                    "summary_structure": summary_structure,
                },
            }

            if anchor_failed:
                context.append(
                    {
                        "query": f"Summarize anchor unresolved: {summary_prompt}",
                        "results": [],
                        "notes": _summarize_anchor_events(anchor_events),
                    }
                )
                yield {
                    "type": "tool_result",
                    "tool": "summarize",
                    "result": {
                        "count": 0,
                        "granularity_seconds": granularity_s,
                        "summary_structure": summary_structure,
                        "results": [],
                        "error": "anchor_not_found",
                    },
                }
                continue

            created_docs = await store.summarize_time_range(
                min_time=min_t,
                max_time=max_t,
                time_mode=t_mode,
                granularity_s=granularity_s,
                summary_structure=summary_structure,
                prompt=summary_prompt,
            )
            summary_results: List[Dict[str, Any]] = []
            for doc in created_docs:
                frame = await store.get_frame_by_id(str(doc.representative_frame_id or ""))
                summary_results.append(
                    {
                        "frame_id": doc.representative_frame_id,
                        "t": float(doc.start_t),
                        "absolute_t": doc.absolute_start_t,
                        "image": frame.image_data_uri if frame is not None else "",
                        "caption": doc.text,
                        "score": 1.0,
                        "source": "memory_document",
                        "layer": doc.layer,
                        "kind": doc.kind,
                        "doc_id": doc.doc_id,
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

            context.append({
                "query": f"Summarize: {summary_prompt}",
                "results": summary_results,
                "notes": _summarize_anchor_events(anchor_events) if anchor_summary is not None else [],
            })
            yield {
                "type": "tool_result",
                "tool": "summarize",
                "result": {
                    "count": len(summary_results),
                    "granularity_seconds": granularity_s,
                    "summary_structure": summary_structure,
                    "results": summary_results,
                },
            }

        elif action == "inspect":
            inspect_prompt = str(plan.get("prompt") or goal).strip() or goal
            inspect_max_frames = _safe_int(plan.get("max_frames", 6), 6)
            inspect_ref = plan.get("ref")
            inspect_time_range = plan.get("time_range", {})
            frames_to_inspect, inspect_target, inspect_events, inspect_failed = await _resolve_inspection_target(
                raw_time_range=inspect_time_range,
                raw_ref=inspect_ref,
                max_frames=inspect_max_frames,
            )
            for event in inspect_events:
                yield event
            yield {
                "type": "tool_use",
                "tool": "inspect",
                "input": {
                    "prompt": inspect_prompt,
                    "ref": inspect_ref if isinstance(inspect_ref, dict) else None,
                    "time_range": inspect_target.get("time_range"),
                    "max_frames": inspect_max_frames,
                },
            }
            if inspect_failed or not frames_to_inspect:
                context.append(
                    {
                        "query": f"Inspect: {inspect_prompt}",
                        "results": [],
                        "notes": [
                            "Direct inspection could not resolve any frames from the requested time or reference."
                        ],
                    }
                )
                continue

            yield {
                "type": "inspection_start",
                "count": len(frames_to_inspect),
                "query": inspect_prompt,
                "joint": True,
            }
            t_insp_start = time.time()
            inspection_text = await store.inspect_frames(frames_to_inspect, inspect_prompt)
            dur = time.time() - t_insp_start
            yield {"type": "inspection_done", "duration": dur}

            first_frame = frames_to_inspect[0]
            last_frame = frames_to_inspect[-1]
            use_absolute = any(getattr(frame, "absolute_t", None) is not None for frame in frames_to_inspect)
            inspect_result = {
                "frame_id": first_frame.frame_id,
                "t": float(first_frame.t),
                "absolute_t": first_frame.absolute_t,
                "image": first_frame.image_data_uri,
                "caption": inspection_text,
                "inspection": inspection_text,
                "score": 1.0,
                "source": "inspection",
                "layer": "inspection",
                "kind": "time_window" if inspect_target.get("mode") == "time_range" else "result_ref",
                "doc_id": (inspect_target.get("matched_result") or {}).get("doc_id"),
                "frame_ids": [frame.frame_id for frame in frames_to_inspect],
                "metadata": {
                    "inspect_prompt": inspect_prompt,
                    "inspect_target": inspect_target,
                },
                "time_range": {
                    "start_t": float(first_frame.t),
                    "end_t": float(last_frame.t),
                    "absolute_start_t": first_frame.absolute_t if use_absolute else None,
                    "absolute_end_t": last_frame.absolute_t if use_absolute else None,
                },
            }
            context.append(
                {
                    "query": f"Inspect: {inspect_prompt}",
                    "results": [inspect_result],
                    "notes": _summarize_result_times(
                        [inspect_result],
                        time_mode="absolute" if use_absolute else "relative",
                        turn_idx=len(context) + 1,
                    ),
                }
            )
            yield {
                "type": "tool_result",
                "tool": "inspect",
                "result": inspect_result,
            }

        elif action == "search":
            query_specs = _coerce_query_specs(plan.get("queries", []))
            if not query_specs and goal:
                query_specs = [{"q": goal}]
            joint = bool(plan.get("joint_inspection", False))
            sources = [str(source).strip().lower() for source in (plan.get("sources") or []) if str(source).strip()]
            summary_filter = plan.get("summary_filter")

            # Parse time range and mode from plan
            time_range = plan.get("time_range", {})
            min_t, max_t, t_mode, anchor_summary, anchor_events, anchor_failed = await _resolve_requested_time_range(time_range)
            for event in anchor_events:
                yield event

            # Check for visual reference
            visual_ref = plan.get("visual_ref")
            ref_emb = None
            visual_label = None
            try:
                ref_emb, visual_label = await _resolve_visual_reference(
                    store=store,
                    context=context,
                    visual_ref=visual_ref,
                )
            except Exception as e:
                yield {"type": "error", "message": f"Failed to resolve visual ref: {e}"}

            yield {
                "type": "search_start",
                "queries": query_specs,
                "time_range": {
                    "min": min_t,
                    "max": None if max_t == float("inf") else max_t,
                    "mode": t_mode,
                    "anchor": anchor_summary,
                },
                "sources": sources or None,
                "summary_filter": summary_filter if isinstance(summary_filter, dict) else None,
                "visual_ref": visual_ref if isinstance(visual_ref, dict) else None,
                "retrieval_mode": "visual_ref" if ref_emb is not None else None,
            }

            if anchor_failed:
                for spec in query_specs:
                    context.append(
                        {
                            "query": f"{str(spec.get('q') or '')} [anchor unresolved]",
                            "results": [],
                            "notes": _summarize_anchor_events(anchor_events),
                        }
                    )
                continue

            # Execute searches in parallel
            t_search_start = time.time()
            tasks = []

            if ref_emb is not None:
                top_k = _safe_int(plan.get("top_k", query_specs[0].get("top_k") if query_specs else 5), 5)
                threshold = _safe_float(plan.get("threshold", query_specs[0].get("threshold") if query_specs else 0.65), 0.65)
                tasks.append(
                    hybrid_event_frame_search(
                        query="", # Ignored when image_query_emb is set
                        top_k=top_k,
                        threshold=threshold,
                        store_instance=store,
                        min_time=min_t,
                        max_time=max_t,
                        time_mode=t_mode,
                        image_query_emb=ref_emb,
                        sources=sources,
                        summary_filter=summary_filter if isinstance(summary_filter, dict) else None,
                    )
                )
            else:
                for spec in query_specs:
                    q = str(spec.get("q") or "").strip()
                    if not q:
                        continue
                    top_k = _safe_int(spec.get("top_k", plan.get("top_k", 5)), 5)
                    inspect_k = _safe_int(spec.get("inspect_k", plan.get("inspect_k", 5)), 5)
                    threshold = _safe_float(spec.get("threshold", plan.get("threshold", 0.65)), 0.65)
                    spec["top_k"] = max(1, min(200, top_k))
                    spec["inspect_k"] = max(0, min(spec["top_k"], min(50, inspect_k)))
                    spec["threshold"] = float(threshold)
                    tasks.append(
                        hybrid_event_frame_search(
                            q,
                            top_k=spec["top_k"],
                            threshold=spec["threshold"],
                            store_instance=store,
                            min_time=min_t,
                            max_time=max_t,
                            time_mode=t_mode,
                            sources=sources,
                            summary_filter=summary_filter if isinstance(summary_filter, dict) else None,
                        )
                    )

            if not tasks:
                continue

            results_list = await asyncio.gather(*tasks)
            yield {"type": "search_step_done", "duration": time.time() - t_search_start}

            # Update the Agentic Retrieval context with Visual Inspection results.
            current_specs = query_specs if ref_emb is None else [{
                "q": visual_label or "Visual Match",
                "top_k": len(results_list[0]) if results_list else 0,
                "inspect_k": min(5, len(results_list[0]) if results_list else 0),
            }]

            for spec, res in zip(current_specs, results_list):
                q = str(spec.get("q") or "")
                # Pass results (truncated) so frontend can display them
                display_results = []
                for r in res:
                    display_results.append({
                        "frame_id": r["frame_id"], # Ensure frame_id is passed
                        "t": r["t"],
                        "absolute_t": r.get("absolute_t"),
                        "score": r["score"],
                        "image": r["image"],  # Send image data
                        "caption": r.get("caption", ""),
                        "source": r.get("source"),
                        "layer": r.get("layer"),
                        "kind": r.get("kind"),
                        "doc_id": r.get("doc_id"),
                        "frame_ids": r.get("frame_ids"),
                        "metadata": r.get("metadata"),
                        "time_range": r.get("time_range"),
                    })
                yield {"type": "search_done", "query": q, "hits": len(res), "results": display_results, "top_k": spec.get("top_k"), "inspect_k": spec.get("inspect_k")}

                inspected_res = []
                inspect_k = _safe_int(spec.get("inspect_k", plan.get("inspect_k", 5)), 5)
                to_inspect = res[: max(0, min(len(res), inspect_k))]

                if to_inspect:
                    yield {"type": "inspection_start", "count": len(to_inspect), "query": q, "joint": joint}

                    t_insp_start = time.time()
                    inspected_res = await _inspect_results(
                        store=store,
                        results=res,
                        inspect_k=inspect_k,
                        prompt=inspection_prompt,
                        joint=joint,
                    )
                    dur = time.time() - t_insp_start
                    yield {"type": "inspection_done", "duration": dur}
                else:
                    inspected_res = res

                context_notes: List[str] = []
                if anchor_summary is not None:
                    context_notes.extend(_summarize_anchor_events(anchor_events))
                context_notes.extend(
                    _summarize_result_times(
                        inspected_res,
                        time_mode=t_mode,
                        turn_idx=len(context) + 1,
                    )
                )
                context.append(
                    {
                        "query": q,
                        "results": inspected_res,
                        "notes": context_notes,
                    }
                )

            unique_keys: set[str] = set()
            for turn_item in context:
                for r in (turn_item.get("results") or []):
                    key = str(r.get("doc_id") or r.get("frame_id") or "")
                    if key:
                        unique_keys.add(key)
            current_unique_hits = len(unique_keys)
            if current_unique_hits <= prev_unique_hits:
                stagnant_turns += 1
            else:
                stagnant_turns = 0
            prev_unique_hits = current_unique_hits

            if stagnant_turns >= max_turns:
                forced_answer = _build_fallback_answer()
                best_frame = {"caption": forced_answer, "t": 0.0}
                best_hit = _best_hit(context)
                if best_hit is not None:
                    best_frame = {
                        "caption": best_hit.get("caption", "") or best_hit.get("inspection", ""),
                        "t": best_hit.get("t", 0.0),
                        "absolute_t": best_hit.get("absolute_t"),
                        "image": best_hit.get("image"),
                        "score": float(best_hit.get("score", 0.0) or 0.0),
                    }
                res = {
                    "found": _has_evidence(context),
                    "best": best_frame,
                    "answer": forced_answer,
                    "semantic_final": goal,
                    "turns": context,
                    "duration_s": time.time() - t0,
                }
                yield {"type": "answer", "content": forced_answer}
                yield {"type": "final", "result": res}
                return

        else:
            yield {"type": "error", "message": f"unsupported_action: {action}"}
            break

    # Out of turns
    best_hit = _best_hit(context)
    best_payload = {"caption": "No definitive answer found.", "t": 0.0}
    if best_hit is not None:
        best_payload = {
            "caption": best_hit.get("caption", "") or best_hit.get("inspection", "") or best_payload["caption"],
            "t": best_hit.get("t", 0.0),
            "absolute_t": best_hit.get("absolute_t"),
            "image": best_hit.get("image"),
            "score": best_hit.get("score", 0.0),
        }
    res = {
        "found": _has_evidence(context),
        "best": best_payload,
        "answer": final_response or _build_fallback_answer(),
        "semantic_final": goal,
        "turns": context,
        "duration_s": time.time() - t0,
    }
    yield {"type": "answer", "content": res["answer"]}
    yield {"type": "final", "result": res}


async def search(
    semantic: str,
    store_instance: Optional[FrameStore] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Wrapper for backward compatibility.
    """
    final_res = {}
    async for event in search_generator(semantic, store_instance, chat_history):
        if event["type"] == "final":
            final_res = event["result"]
    return final_res
