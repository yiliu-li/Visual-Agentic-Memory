from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import TypeAdapter

from vam import prompts
from vam.config import get_settings
from vam.protocol import RetrieveToolArgs, SummarizeToolArgs, WsAgentDecision
from vam.llm.openrouter import OpenRouterClient
from vam.retrieval.agent_search import search_generator as retrieval_search_generator
from vam.retrieval.frame_store import resolve_memory_store


logger = logging.getLogger(__name__)
_ROUTER_DECISION_ADAPTER = TypeAdapter(WsAgentDecision)


def append_history(history: List[Dict[str, Any]], *, user_text: str, assistant_text: str) -> None:
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})


async def route_text_request(*, transcript: str) -> WsAgentDecision:
    cfg = get_settings()
    router_client = OpenRouterClient(
        model_id=cfg.openrouter_model_id_light or cfg.openrouter_model_id_main or cfg.openrouter_model_id
    )
    messages = [
        {"role": "system", "content": prompts.ws_agent_system()},
        {"role": "user", "content": transcript},
    ]
    text, _ = await router_client.chat(messages, temperature=0.0, max_tokens=300)
    raw = (text or "").strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    payload = json.loads(raw)
    decision = _ROUTER_DECISION_ADAPTER.validate_python(payload)
    return decision


async def stream_retrieve_tool_request(
    *,
    transcript: str,
    history: List[Dict[str, Any]],
) -> AsyncGenerator[Dict[str, Any], None]:
    store = resolve_memory_store()
    tool_args = RetrieveToolArgs(question=transcript)
    yield {"type": "tool_use", "tool": "retrieve", "input": tool_args.model_dump()}

    final_result: Optional[Dict[str, Any]] = None
    async for event in retrieval_search_generator(
        semantic=tool_args.question,
        store_instance=store,
        chat_history=history,
    ):
        if str(event.get("type") or "") == "final":
            maybe_result = event.get("result")
            final_result = maybe_result if isinstance(maybe_result, dict) else None
            continue
        yield event

    answer = ""
    if isinstance(final_result, dict):
        answer = str(final_result.get("answer") or "").strip()
        if not answer and final_result.get("found") and isinstance(final_result.get("best"), dict):
            best = final_result["best"]
            t_sec = float(best.get("t") or 0.0)
            detail = str(best.get("inspection") or best.get("caption") or "").strip()
            answer = f"Found relevant evidence at {t_sec:.1f}s: {detail}" if detail else f"Found relevant evidence at {t_sec:.1f}s."
    if not answer:
        answer = "No relevant evidence found."

    result = final_result or {"found": False, "answer": answer, "turns": []}
    yield {
        "type": "final",
        "text": answer,
        "result": result,
    }
    append_history(history, user_text=transcript, assistant_text=answer)


async def stream_summarize_tool_request(
    *,
    args: SummarizeToolArgs,
    history: List[Dict[str, Any]],
) -> AsyncGenerator[Dict[str, Any], None]:
    store = resolve_memory_store()
    payload = args.model_dump()
    yield {"type": "tool_use", "tool": "summarize", "input": payload}

    max_time = float("inf") if args.max_time is None else float(args.max_time)
    created_docs = await store.summarize_time_range(
        min_time=float(args.min_time),
        max_time=max_time,
        time_mode=str(args.time_mode),
        granularity_s=float(args.granularity_seconds),
        summary_structure=str(args.summary_structure) if args.summary_structure is not None else None,
        prompt=str(args.prompt),
    )

    results: List[Dict[str, Any]] = []
    for doc in created_docs:
        frame = await store.get_frame_by_id(str(doc.representative_frame_id or ""))
        results.append(
            {
                "doc_id": doc.doc_id,
                "layer": doc.layer,
                "kind": doc.kind,
                "text": doc.text,
                "representative_frame_id": doc.representative_frame_id,
                "frame_ids": list(doc.frame_ids or []),
                "metadata": dict(doc.metadata or {}),
                "time_range": {
                    "start_t": float(doc.start_t),
                    "end_t": float(doc.end_t),
                    "absolute_start_t": doc.absolute_start_t,
                    "absolute_end_t": doc.absolute_end_t,
                },
                "image": frame.image_data_uri if frame is not None else "",
            }
        )

    answer = f"Created {len(results)} summary documents in Hierarchical Memory."
    yield {
        "type": "final",
        "text": answer,
        "result": {
            "created": len(results),
            "documents": results,
            "answer": answer,
        },
    }
    append_history(
        history,
        user_text=f"summarize:{json.dumps(payload, ensure_ascii=False)}",
        assistant_text=answer,
    )


async def stream_agent_response(
    *,
    transcript: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    session_history = history if history is not None else []
    clean_transcript = str(transcript or "").strip()
    if not clean_transcript:
        return

    try:
        decision = await route_text_request(transcript=clean_transcript)
    except Exception as exc:
        logger.warning("top_level_router_failed; falling back to direct Agentic Retrieval: %s", exc)
        yield {"type": "user_text", "text": clean_transcript}
        async for event in stream_retrieve_tool_request(
            transcript=clean_transcript,
            history=session_history,
        ):
            yield event
        return

    if decision.type == "final":
        text = str(decision.text).strip()
        yield {"type": "final", "text": text, "result": {"answer": text}}
        append_history(session_history, user_text=clean_transcript, assistant_text=text)
        return

    if decision.name == "retrieve":
        routed_question = decision.args.question
        yield {"type": "user_text", "text": routed_question}
        async for event in stream_retrieve_tool_request(
            transcript=routed_question,
            history=session_history,
        ):
            yield event
        return

    if decision.name == "summarize":
        async for event in stream_summarize_tool_request(
            args=decision.args,
            history=session_history,
        ):
            yield event
        return

    raise ValueError(f"unsupported routed decision: {decision}")
