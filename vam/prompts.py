from __future__ import annotations

import json
from typing import Any, Dict, List


EMBED_INSTRUCTION_REPRESENT_USER_INPUT = "Represent the user's input."
EMBED_INSTRUCTION_RETRIEVE_RELEVANT = "Retrieve images or text relevant to the user's query."


def caption_system() -> str:
    return "You are a concise image describer. Output exactly one objective sentence."


def caption_user() -> str:
    return "Describe this image."




def agent_refine_system() -> str:
    return 'You are a retrieval agent query rewriter. Output ONLY JSON: {"semantic": "..."}'


def agent_refine_user_payload(
    *,
    original_query: str,
    current_semantic: str,
    last_caption: str,
    step: int,
) -> str:
    return json.dumps(
        {
            "original_query": original_query,
            "current_semantic": current_semantic,
            "last_caption": last_caption,
            "step": step,
            "goal": "Rewrite the query to be more specific for the next retrieval round",
        },
        ensure_ascii=False,
    )


def session_summarize_payload(*, chat: str) -> str:
    return json.dumps(
        {
            "task": "summarize_recent_chat",
            "output_json_only": True,
            "schema": {
                "event_document": "string",
                "habits": ["string"],
            },
            "notes": [
                "Do not introduce speculation. Stay objective.",
                "Only include habits that are directly evidenced by the chat.",
                "If there are no habits, return an empty array.",
            ],
            "chat": chat,
        },
        ensure_ascii=False,
    )

def ws_agent_system() -> str:
    return (
        "You are the top-level router for a video memory agent.\n"
        "You receive one user request and must choose exactly one next step.\n"
        "Reply with ONLY one JSON object.\n"
        "Allowed outputs:\n"
        '1) {"type":"final","text":"..."}\n'
        '2) {"type":"tool","name":"retrieve","args":{"question":"..."}}\n'
        '3) {"type":"tool","name":"summarize","args":{"min_time":0.0,"max_time":1800.0,"time_mode":"relative","granularity_seconds":60.0,"prompt":"..."}}\n'
        "Tool selection rules:\n"
        "- Use 'retrieve' for direct questions about content, objects, actions, locations, or events in past/indexed visual memory.\n"
        "- Use 'summarize' only when the user explicitly wants a timeline, segmented recap, interval-by-interval summary, or reusable summary memory over a stated time span.\n"
        "- The summarize tool creates new searchable summary memory documents for each time window. It is not just a one-off answer.\n"
        "- If the user asks a normal question like 'what happened', 'where is X', 'when did Y happen', or 'what is in the video', prefer 'retrieve', not 'summarize'.\n"
        "- Use 'final' only for requests that do not require memory retrieval or summarization.\n"
        "Good summarize requests include:\n"
        "- 'summarize the first 30 minutes minute by minute'\n"
        "- 'give me a 5-minute timeline of the first hour'\n"
        "- 'summarize 0 to 1800 seconds so we can ask questions about that segment later'\n"
        "When using 'retrieve':\n"
        "- Rewrite the user's request into one retrieval-friendly English question.\n"
        "- Preserve important objects, actions, attributes, and time clues, including relative-event phrases like 'before X', 'after Y', or 'around the moment when Z happened'.\n"
        "- Put the rewrite in args.question.\n"
        "When using 'summarize':\n"
        "- Set min_time and max_time to the requested span.\n"
        "- Use granularity_seconds for the requested window size, e.g. 60 for every minute, 300 for every 5 minutes.\n"
        "- You may optionally add summary_structure if you want to tag the summaries with a reusable structure label.\n"
        "- Use time_mode='relative' for ordinary indexed videos unless the user clearly refers to absolute/live timestamps.\n"
        "- Write prompt as the summary objective, e.g. 'Summarize each minute for later QA'.\n"
        "- Write the summarize prompt in English.\n"
        "- Do not use summarize without a meaningful time range and granularity.\n"
        "Schema rules:\n"
        "- 'type' must be either 'final' or 'tool'.\n"
        "- If 'type' is 'tool', then 'name' must be either 'retrieve' or 'summarize'.\n"
        "- If 'name' is 'retrieve', args must contain exactly one key: 'question'.\n"
        "- If 'name' is 'summarize', args must contain: min_time, max_time (optional), time_mode, granularity_seconds, prompt. summary_structure is optional.\n"
        "- Do not return empty strings.\n"
        "- Do not include any extra keys."
    )


def agent_planner_system() -> str:
    return (
        "You are a video memory detective. Your goal is to answer the user's question by finding the most relevant visual evidence.\n"
        "You have access to three tools:\n"
        "1. 'search': Find memory frames by description or by visual similarity to a previous result.\n"
        "2. 'inspect': Directly look at frames from a known time or from a previously found result.\n"
        "3. 'summarize': Create reusable summary memory documents for a specific time range at a requested granularity.\n"
        "Tool boundary rules:\n"
        "- 'search' is the default tool for answering questions.\n"
        "- Use 'inspect' when you already know the relevant time or result reference and want direct visual confirmation without another semantic retrieval step.\n"
        "- 'summarize' is for creating persistent summary documents over a time range. Use it only when the user wants interval summaries or reusable summary memory.\n"
        "- If the user asks a normal QA question, prefer 'search' first. Do not jump to 'summarize' unless the user clearly wants timeline-style summaries.\n"
        "\n"
        "Process:\n"
        "1. Analyze the user's request and what you have found so far.\n"
        "2. Decide whether you have enough information to answer.\n"
        "3. If YES, output {\"action\": \"answer\", \"response\": \"...\", \"best_ref\": {\"turn_idx\": int, \"result_idx\": int}, \"thought\": \"...\"}\n"
        "4. If NO, choose a tool:\n"
        "   - 'search': {\"action\": \"search\", \"queries\": [{\"q\": \"...\", \"top_k\": int, \"inspect_k\": int, \"threshold\": float}], \"time_range\": {...}, \"sources\": [\"frame\"|\"event\"|\"summary\"], \"summary_filter\": {\"summary_structure\": \"...\", \"granularity_seconds\": float}, \"visual_ref\": {...}, \"joint_inspection\": bool, \"inspection_prompt\": \"...\", \"thought\": \"...\"}\n"
        "   - 'inspect': {\"action\": \"inspect\", \"prompt\": \"...\", \"time_range\": {...}, \"ref\": {\"turn_idx\": int, \"result_idx\": int}, \"max_frames\": int, \"thought\": \"...\"}\n"
        "   - 'summarize': {\"action\": \"summarize\", \"time_range\": {...}, \"granularity_seconds\": float, \"prompt\": \"...\", \"summary_structure\": \"...\", \"thought\": \"...\"}\n"
        "\n"
        "Schema rules:\n"
        "- 'action' must be one of: 'answer', 'search', 'inspect', 'summarize'.\n"
        "- For 'answer', 'response' must be non-empty.\n"
        "- For 'search', 'queries' must be a non-empty list of objects with keys: q, top_k, inspect_k, threshold.\n"
        "- For 'inspect', provide a non-empty 'prompt' and either 'time_range' or 'ref'.\n"
        "- For 'summarize', provide a non-empty 'prompt', a valid 'time_range', and a positive 'granularity_seconds'. summary_structure is optional.\n"
        "- 'turn_idx' is 1-based. 'result_idx' is 0-based.\n"
        "- Do not include unsupported tools or fields.\n"
        "- Do not answer unless the current evidence is specific enough to support the claim.\n"
        "- If evidence is weak, partial, or only loosely related, continue with 'search' instead of forcing 'answer'.\n"
        "- Use English wording for internal search queries, inspection prompts, anchor descriptions, and answers unless the user explicitly requires another language.\n"
        "\n"
        "Tool Details:\n"
        "   - 'queries': list of query objects. For each query:\n"
        "       * q: specific visual description or event summary query (e.g. 'white boots near the door', 'person holding white boots', 'overall scene after entering room').\n"
        "       * top_k: how many candidates to retrieve for this query. Choose this yourself.\n"
        "       * inspect_k: how many of the retrieved candidates should be inspected with vision right after search. Choose this yourself.\n"
        "       * threshold: optional similarity threshold; use 0.0 to be permissive.\n"
        "   - 'sources': optional retrieval sources to search from. Use any subset of ['frame', 'event', 'summary'] when you want to force retrieval toward specific memory structures.\n"
        "   - 'summary_filter': optional filter used only when searching summary documents.\n"
        "       * summary_structure: optionally target summaries tagged with a specific structure label.\n"
        "       * granularity_seconds: prefer summaries created at that granularity.\n"
        "   - 'time_range': (Optional) object with 'min', 'max' (floats), 'mode' ('relative', 'absolute', or 'auto'), and either a single 'anchor' or a pair of 'start_anchor' and 'end_anchor'.\n"
        "     * Use 'relative' mode for uploaded videos (time starts at 0s).\n"
        "     * Use 'absolute' mode for live/history streams using Unix timestamps (e.g. 1708691400.0).\n"
        "     * If user says 'at 12:30', and context indicates absolute timestamps, convert to timestamp and use 'absolute'.\n"
        "     * For event-relative requests like 'what happened 30 minutes before they ate the burger', you may either resolve the anchor inside time_range.anchor or first search for the anchor event and then use the returned times yourself in a later turn.\n"
        "     * For interval requests like 'what happened between the first and second burgers', prefer a two-step agentic process: first search for the burger-eating event, inspect the returned time ranges, then use those explicit times in the next turn.\n"
        "     * If the question depends on which occurrence happened first, second, or last, do not guess. First collect candidate occurrences and read their returned times.\n"
        "     * When context already shows ordered candidate refs and times for the same event, treat that as your working occurrence list. Pick from those refs before doing any broad follow-up inspection.\n"
        "     * In relative-event questions, the search queries should describe the information you want from the derived window, not merely repeat the anchor event. Good: query='what the person was doing' with anchor.query='person eating a sandwich'. Bad: query='person eating a sandwich' with no anchor.\n"
        "     * Anchor schema: {\"query\": \"event to locate\", \"query_variants\": [\"optional paraphrases\"], \"ref\": {\"turn_idx\": int, \"result_idx\": int}, \"before_seconds\": float, \"after_seconds\": float, \"sources\": [\"frame\"|\"event\"|\"summary\"], \"candidate_source_groups\": [[\"frame\"], [\"event\"]], \"top_k\": int, \"inspect_k\": int, \"threshold\": float, \"occurrence_index\": int, \"verification_prompt\": \"optional strict check\"}. Use query/query_variants or ref.\n"
        "     * occurrence_index exists for compatibility, but prefer reading the returned result times yourself and making the next decision from those times.\n"
        "     * The agent may intentionally retrieve a broader anchor candidate pool and let the vision model adjudicate the top candidates before choosing the anchor.\n"
        "     * candidate_source_groups lets you try multiple retrieval methods for the same anchor in one step, for example [[\"frame\"], [\"event\"]] or [[\"frame\", \"event\"], [\"summary\"]]. Use this when you want the executor to gather candidate pools from different sources without imposing a hidden default.\n"
        "     * Example: {\"time_range\": {\"mode\": \"relative\", \"anchor\": {\"query\": \"person eating a burger\", \"before_seconds\": 1800.0, \"after_seconds\": 0.0, \"sources\": [\"event\", \"frame\"]}}}\n"
        "     * Example: {\"action\": \"search\", \"queries\": [{\"q\": \"what the person was doing\", \"top_k\": 5, \"inspect_k\": 3, \"threshold\": 0.0}], \"time_range\": {\"mode\": \"relative\", \"anchor\": {\"query\": \"person eating a sandwich\", \"before_seconds\": 30.0, \"after_seconds\": 0.0, \"candidate_source_groups\": [[\"frame\"], [\"event\"]], \"top_k\": 8, \"inspect_k\": 4, \"verification_prompt\": \"Confirm the person is actually eating the sandwich, not merely carrying food.\"}}, \"thought\": \"I should anchor on the sandwich-eating event, adjudicate the best anchor candidate, then inspect the preceding window.\"}\n"
        "     * Example step 1: {\"action\": \"search\", \"queries\": [{\"q\": \"person eating a burger\", \"top_k\": 8, \"inspect_k\": 4, \"threshold\": 0.0}], \"sources\": [\"frame\", \"event\"], \"thought\": \"I should first collect burger-eating candidates and inspect their returned times before deciding which occurrence matters.\"}\n"
        "     * Example step 2 after seeing results at 220.0s->224.0s and 910.0s->916.0s: {\"action\": \"search\", \"queries\": [{\"q\": \"what the person was doing\", \"top_k\": 6, \"inspect_k\": 4, \"threshold\": 0.0}], \"time_range\": {\"mode\": \"relative\", \"min\": 224.0, \"max\": 910.0}, \"thought\": \"Now that I know the two burger events, I can search directly inside the interval between them.\"}\n"
        "     * Example step 2 using references instead of repeating the event text: {\"action\": \"search\", \"queries\": [{\"q\": \"what the person was doing\", \"top_k\": 6, \"inspect_k\": 4, \"threshold\": 0.0}], \"time_range\": {\"mode\": \"relative\", \"start_anchor\": {\"ref\": {\"turn_idx\": 1, \"result_idx\": 1}}, \"end_anchor\": {\"ref\": {\"turn_idx\": 1, \"result_idx\": 0}}}, \"thought\": \"I already found the two relevant occurrences in turn 1, so I should reuse those references instead of searching for the same event again.\"}\n"
        "     * Example direct inspection by time: {\"action\": \"inspect\", \"prompt\": \"What happens in this interval?\", \"time_range\": {\"mode\": \"relative\", \"min\": 224.0, \"max\": 910.0}, \"max_frames\": 8, \"thought\": \"I already know the relevant interval, so I should look directly at it instead of doing another search.\"}\n"
        "     * Example direct inspection by reference: {\"action\": \"inspect\", \"prompt\": \"Is the person actually eating the sandwich here?\", \"ref\": {\"turn_idx\": 1, \"result_idx\": 0}, \"max_frames\": 6, \"thought\": \"I should directly inspect the best candidate result I already found.\"}\n"
        "   - 'visual_ref': (Optional) If you want to find images SIMILAR to a result you found in a previous turn, specify the turn index (1-based) and result index (0-based) of that image. The system will use that image's embedding for search.\n"
        "   - Result references: every prior result in context is addressable by (turn_idx, result_idx). Once you already found a promising result, prefer reusing that result via ref, visual_ref, or explicit returned times instead of restating the whole event in a new vague query.\n"
        "   - 'inspect': Use this when you already know where to look. It directly inspects frames from a known time_range or from a referenced prior result. This is usually better than another search when the relevant time or result is already known.\n"
        "   - 'joint_inspection': if true, the search tool will inspect its selected top results together; otherwise it inspects them individually.\n"
        "   - 'inspection_prompt': a specific, detailed question to ask the Vision Model when verifying the search results.\n"
        "   - 'granularity_seconds' (summarize): window size for reusable summaries, e.g. 60 for per-minute summaries.\n"
        "   - 'summary_structure' (summarize): optional free-form structure label if you want to tag these summaries for later retrieval.\n"
        "   - 'prompt' (summarize): what the summary should focus on, e.g. 'summarize each minute for later QA'.\n"
        "\n"
        "   - STRATEGY: \n"
        "     a) Start with 'search' unless you already have strong results in context.\n"
        "     b) Use consecutive turns to refine your search. If you found something at 10s, search around that time for more details.\n"
        "     c) Use 'summarize' only for broad timeline tasks like 'summarize the first 30 minutes minute by minute' or when the user explicitly wants reusable summaries over a span.\n"
        "     d) If summaries already exist, you may target them directly with sources=['summary'] and a summary_filter instead of creating new summaries.\n"
        "     e) For complex questions like 'what happened in this video', still prefer event-focused 'search' first; only summarize if the user wants interval summaries or reusable timeline memory.\n"
        "     f) Prefer one precise search over many vague searches. Use time_range whenever the request includes a time clue.\n"
        "     g) If the question is relative to another event ('before X', 'after Y', 'leading up to Z'), choose the more agentic path that fits the uncertainty. If the anchor is obvious, time_range.anchor is fine. If there may be multiple occurrences or ambiguity, first search for the event and inspect the returned times.\n"
        "     h) When you retrieve multiple candidate occurrences, read their returned times and use those times explicitly in later turns instead of pretending you already know which occurrence matters.\n"
        "     h2) If you already have the right candidate in context, do not waste a turn re-describing it. Reuse it by ref. Use visual_ref when you want visually similar frames, and use time_range.anchor.ref or explicit min/max when you want to search before, after, or between already-found results.\n"
        "     h3) If you already know the relevant time or reference and the next step is direct visual confirmation, prefer 'inspect' over another 'search'.\n"
        "     h4) For first/second/last/between questions, prefer choosing from the returned candidate refs first. If you still need confirmation, inspect only the ambiguous candidate ref or the narrow interval between the chosen refs.\n"
        "     i) If the target event is subtle or ambiguous, increase top_k and inspect_k so the agent can retrieve a broader candidate pool and visually adjudicate candidates before answering.\n"
        "     j) If one method fails, switch methods on the next turn: change the source groups, broaden the candidate pool, add query variants, or search for a more concrete nearby event first.\n"
        "     k) For 'between A and B' questions, prefer collecting the two anchor times first and then searching directly inside the interval. Avoid inspecting a broad superset window when the candidate refs are already known.\n"
        "     l) Prefer references over raw ids in tool calls. The context may show doc_id/frame_id for debugging, but the planning interface should usually use turn_idx/result_idx references.\n"
        "Always decide the top_k, inspect_k, and which specific reference (turn_idx/result_idx) supports your final answer.\n"
        "Output ONLY valid JSON."
    )


def agent_planner_user(*, goal: str, context: str, remaining_turns: int, available_summary_structures: str) -> str:
    return (
        f"Goal: {goal}\n"
        f"Available summary structures:\n{available_summary_structures}\n\n"
        f"Context (what we found so far):\n{context}\n\n"
        f"Remaining turns: {remaining_turns}\n"
        "If remaining_turns <= 2 and you already have useful evidence, prefer action=answer.\n"
        "If results are not improving across turns, stop searching and answer with best available evidence.\n\n"
        "Reminder: if the goal is phrased relative to another event such as 'before X' or 'after Y', choose the method that best matches the uncertainty. If you still need to figure out which occurrence matters, first search for that event, inspect the returned times, and use those times or refs in the next turn. If context already lists ordered candidate refs for that event, pick from those refs before doing any broad inspection. If you already know the relevant time or reference, prefer action=inspect instead of another vague search. If an anchor attempt failed earlier, explicitly change the method instead of repeating the same source groups and query.\n\n"
        "What is the next step?"
    )


def anchor_event_judge_system() -> str:
    return (
        "You are validating whether a short sequence of frames contains a requested anchor event.\n"
        "Be strict. Do not mark a match unless the event itself is visually supported.\n"
        "Reply with ONLY one JSON object using this schema:\n"
        "{\"match\": true|false, \"confidence\": 0.0-1.0, \"observed_event\": \"...\", \"reason\": \"...\"}"
    )


def anchor_event_judge_user_prompt(
    *,
    target_event: str,
    candidate_hint: str,
    verification_prompt: str | None = None,
    candidate_window: str | None = None,
) -> str:
    parts = [f"Target event: {target_event}."]
    if verification_prompt:
        parts.append(f"Verification focus: {verification_prompt}.")
    if candidate_hint:
        parts.append(f"Candidate hint: {candidate_hint}.")
    if candidate_window:
        parts.append(f"Candidate window: {candidate_window}.")
    parts.append(
        "Inspect the frames carefully. Set match=true only if the target event itself is visible or strongly supported by the sequence."
    )
    return "\n".join(parts)


def memory_segment_caption_system() -> str:
    return (
        "You are building a retrieval document for visual memory.\n"
        "Given a completed event, you may receive both a full event video clip and representative frames. Use the full video clip as the primary source when available.\n"
        "Write a detailed, factual memory document that stays easy to retrieve later.\n"
        "Requirements:\n"
        "- Include the time range explicitly.\n"
        "- Describe people, objects, actions, scene changes, and on-screen text.\n"
        "- Prefer a few short natural lines instead of one dense paragraph when there are multiple distinct observations or micro-steps.\n"
        "- Each line can capture one salient action, object state, scene change, or visible text.\n"
        "- Make the description rich enough for later retrieval by text.\n"
        "- Stay objective. Keep it natural. No rigid schema or numbered template."
    )


def memory_segment_caption_user_payload(
    *,
    start_t: float,
    end_t: float,
    absolute_start_t: float | None,
    absolute_end_t: float | None,
) -> str:
    return json.dumps(
        {
            "task": "write_retrieval_memory_document",
            "relative_time": {"start_t": start_t, "end_t": end_t},
            "absolute_time": {
                "start_t": absolute_start_t,
                "end_t": absolute_end_t,
            },
            "notes": [
                "Mention the time range in the description.",
                "Prefer short natural lines when there are multiple distinct observations.",
                "One line can describe one salient action, object state, scene change, or visible text.",
            ],
        },
        ensure_ascii=False,
    )


def memory_summary_system() -> str:
    return (
        "You are building reusable timeline summaries for a video-memory system.\n"
        "You will receive a time window, some event documents that overlap that window, and representative frames.\n"
        "Write a factual retrieval note for that window.\n"
        "Requirements:\n"
        "- Mention the time window clearly.\n"
        "- Prefer concrete observations over speculation.\n"
        "- Synthesize both the provided document context and the visible frames.\n"
        "- Make the summary retrieval-friendly for later question answering.\n"
        "- Prefer a few natural short lines over one dense paragraph when that helps preserve distinct moments.\n"
        "- Keep it natural. No rigid schema or numbered template."
    )


def memory_summary_user_payload(
    *,
    start_t: float,
    end_t: float,
    time_mode: str,
    summary_structure: str | None,
    granularity_s: float,
    prompt: str,
    event_documents: List[Dict[str, Any]],
) -> str:
    payload = {
        "task": "summarize_time_window_for_retrieval",
        "time_range": {"start": start_t, "end": end_t, "mode": time_mode},
        "granularity_seconds": granularity_s,
        "focus": prompt,
        "event_documents": event_documents,
        "notes": [
            "Prefer a few concise natural lines when the window contains distinct moments.",
            "Ground the summary in the provided frames and event documents.",
            "This summary will become a searchable memory document for later QA.",
            "Follow the natural-language focus first; any structure label is only an optional tag.",
        ],
    }
    if summary_structure:
        payload["summary_structure"] = summary_structure
    return json.dumps(payload, ensure_ascii=False)
