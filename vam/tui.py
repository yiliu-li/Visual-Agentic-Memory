from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ModuleNotFoundError:
    box = Align = Group = Live = Panel = Confirm = FloatPrompt = IntPrompt = Prompt = Rule = Table = Text = None
    Console = None
    RICH_AVAILABLE = False


def _stream_is_utf8(stream: Any) -> bool:
    encoding = str(getattr(stream, "encoding", "") or "").lower()
    return "utf" in encoding


def _prepare_stdio_for_rich() -> bool:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    return _stream_is_utf8(sys.stdout) and _stream_is_utf8(sys.stderr)


if RICH_AVAILABLE and not _prepare_stdio_for_rich():
    RICH_AVAILABLE = False

from vam.video import index_video_async, parse_absolute_time
from vam.config import get_settings
from vam.protocol import SummarizeToolArgs
from vam.agent import stream_agent_response, stream_summarize_tool_request
from vam.jobs import manager as job_manager
from vam.retrieval.frame_store import FrameRecord, MemoryDocument, resolve_memory_store


def _truncate(text: str, limit: int = 110) -> str:
    clean = " ".join(str(text or "").split()).strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."

def _format_rel_abs(rel_t: float, abs_t: Optional[float]) -> str:
    rel_part = f"{float(rel_t):.1f}s"
    if abs_t is None:
        return rel_part
    return f"{rel_part} | abs {abs_t:.0f}"


def _format_doc_range(doc: MemoryDocument) -> str:
    return _format_rel_abs(float(doc.start_t), doc.absolute_start_t) + " -> " + _format_rel_abs(
        float(doc.end_t), doc.absolute_end_t
    )


def _job_percent(job: Any) -> float:
    total = int(getattr(job, "total", 0) or 0)
    current = int(getattr(job, "current", 0) or 0)
    if total <= 0:
        return 0.0
    return max(0.0, min(100.0, (float(current) / float(total)) * 100.0))


class AgentTUI:
    def __init__(self) -> None:
        self.console = Console()
        self.store = resolve_memory_store()
        self.history: List[Dict[str, Any]] = []

    async def _build_overview(self) -> Dict[str, Any]:
        frames = await self.store.list_frames()
        docs = await self.store.list_memory_documents()
        summary_structures = await self.store.list_summary_structures()
        latest_filters = await self.store.get_latest_filters()

        tiers = {"recent": 0, "mid": 0, "long": 0, "unknown": 0}
        for frame in frames:
            tier = str(frame.memory_tier or "unknown")
            tiers[tier] = int(tiers.get(tier, 0)) + 1

        by_layer: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}
        for doc in docs:
            by_layer[str(doc.layer)] = int(by_layer.get(str(doc.layer), 0)) + 1
            by_kind[str(doc.kind)] = int(by_kind.get(str(doc.kind), 0)) + 1

        latest_event = None
        long_events = [doc for doc in docs if str(doc.layer) == "long" and str(doc.kind) == "event"]
        if long_events:
            latest_event = max(long_events, key=lambda doc: (float(doc.end_t), float(doc.start_t), str(doc.doc_id)))

        return {
            "frames": frames,
            "docs": docs,
            "tiers": tiers,
            "by_layer": by_layer,
            "by_kind": by_kind,
            "latest_event": latest_event,
            "latest_filters": latest_filters,
            "summary_structures": summary_structures,
        }

    def _render_home(self, overview: Dict[str, Any]) -> Group:
        stats = Table.grid(expand=True)
        stats.add_column(justify="left")
        stats.add_column(justify="left")
        stats.add_row(
            f"[bold]Frames[/] {len(overview['frames'])}",
            f"[bold]Docs[/] {len(overview['docs'])}",
        )
        stats.add_row(
            f"[bold]Events[/] {overview['by_kind'].get('event', 0)}",
            f"[bold]Summaries[/] {overview['by_kind'].get('summary', 0)}",
        )
        stats.add_row(
            f"[bold]Tiers[/] recent={overview['tiers'].get('recent', 0)} mid={overview['tiers'].get('mid', 0)} long={overview['tiers'].get('long', 0)}",
            f"[bold]Chat Turns[/] {len(self.history) // 2}",
        )

        latest_event = overview.get("latest_event")
        latest_text = "No event memory yet."
        if latest_event is not None:
            latest_text = f"{str(latest_event.doc_id)[:10]} | {_format_doc_range(latest_event)} | {_truncate(latest_event.text, 120)}"

        menu = Table(box=box.SIMPLE_HEAVY, expand=True)
        menu.add_column("Key", style="bold cyan", width=6)
        menu.add_column("Action", style="white")
        menu.add_row("1", "Overview")
        menu.add_row("2", "Index Video")
        menu.add_row("3", "Ask Agent")
        menu.add_row("4", "Summarize Range")
        menu.add_row("5", "Browse Memory")
        menu.add_row("6", "Reset Chat")
        menu.add_row("7", "Clear Memory")
        menu.add_row("8", "Quit")

        filters_text = str(overview.get("latest_filters") or {})
        body = Group(
            Panel(
                Group(
                    Align.center(Text("Visual Agentic Memory TUI", style="bold")),
                    Rule(style="dim"),
                    stats,
                ),
                title="Dashboard",
                border_style="cyan",
            ),
            Panel(latest_text, title="Latest Event", border_style="blue"),
            Panel(_truncate(filters_text, 240), title="Latest Filters", border_style="magenta"),
            Panel(menu, title="Main Menu", border_style="green"),
        )
        return body

    def _pause(self) -> None:
        self.console.input("\n[dim]Press Enter to return to the menu...[/]")

    def _event_detail(self, event: Dict[str, Any]) -> str:
        event_type = str(event.get("type") or "")
        if event_type == "user_text":
            return str(event.get("text") or "")
        if event_type in {"plan", "thought", "answer", "info", "error"}:
            return str(event.get("content") or event.get("message") or "")
        if event_type == "tool_use":
            return f"{event.get('tool')} {event.get('input')}"
        if event_type == "tool_result":
            tool = str(event.get("tool") or "")
            result = event.get("result") or {}
            if tool == "inspect" and isinstance(result, dict):
                frame_ids = result.get("frame_ids") or []
                target = (result.get("metadata") or {}).get("inspect_target") or {}
                time_range = result.get("time_range") or target.get("time_range") or target.get("resolved_time_range")
                return f"inspect frames={len(frame_ids)} time={time_range}"
            if isinstance(result, dict):
                count = result.get("count")
                if count is not None:
                    return f"{tool} count={count}"
            return tool or "tool_result"
        if event_type == "search_start":
            queries = event.get("queries") or []
            return f"queries={len(queries)} time={event.get('time_range')} sources={event.get('sources')}"
        if event_type == "search_done":
            return f"{event.get('query')} | hits={event.get('hits')}"
        if event_type == "inspection_start":
            return f"query={event.get('query')} count={event.get('count')} joint={event.get('joint')}"
        if event_type == "final":
            result = event.get("result") or {}
            return str(event.get("text") or result.get("answer") or "")
        if event_type.endswith("_done"):
            duration = event.get("duration")
            if duration is None:
                return ""
            return f"duration={float(duration):.2f}s"
        return _truncate(str(event), 120)

    def _render_agent_live(self, *, question: str, events: List[Dict[str, Any]]) -> Group:
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("#", width=4, style="dim")
        table.add_column("Type", width=18, style="cyan")
        table.add_column("Detail", style="white")
        for idx, event in enumerate(events[-14:], start=max(1, len(events) - 13)):
            table.add_row(str(idx), str(event.get("type") or ""), _truncate(self._event_detail(event), 180))

        final_event = next((item for item in reversed(events) if str(item.get("type") or "") == "final"), None)
        answer = ""
        if isinstance(final_event, dict):
            result = final_event.get("result") or {}
            answer = str(final_event.get("text") or result.get("answer") or "").strip()

        status_text = "Running..."
        if events:
            status_text = self._event_detail(events[-1]) or str(events[-1].get("type") or "running")
        if final_event is not None:
            status_text = "Completed"

        panels: List[Any] = [
            Panel(question, title="Question", border_style="cyan"),
            Panel(status_text, title="Status", border_style="yellow"),
            Panel(table, title="Event Stream", border_style="blue"),
        ]
        if answer:
            panels.append(Panel(answer, title="Final Answer", border_style="green"))
        return Group(*panels)

    def _render_job_live(self, *, job_id: str, job: Any) -> Group:
        if job is None:
            return Group(Panel(f"Waiting for job {job_id}", title="Index Job", border_style="yellow"))

        status = f"status={job.status} phase={job.phase} progress={job.current}/{job.total} ({_job_percent(job):.1f}%)"
        logs = "\n".join(job.logs[-12:]) if job.logs else "(no logs yet)"

        panels: List[Any] = [
            Panel(status, title=f"Index Job {job_id[:8]}", border_style="cyan"),
            Panel(logs, title="Logs", border_style="blue"),
        ]

        if job.result:
            result = job.result
            summary = (
                f"count={result.get('count', 0)} | "
                f"source_fps={result.get('source_fps')} | "
                f"truncated={result.get('truncated')}"
            )
            panels.append(Panel(summary, title="Result", border_style="green"))

        if job.error:
            panels.append(Panel(job.error, title="Error", border_style="red"))

        return Group(*panels)

    def _render_docs_table(self, docs: List[MemoryDocument], *, title: str) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Doc", style="cyan", width=12)
        table.add_column("Layer", width=10)
        table.add_column("Kind", width=10)
        table.add_column("Time", width=28)
        table.add_column("Text", style="white")
        for doc in docs:
            table.add_row(
                str(doc.doc_id)[:12],
                str(doc.layer),
                str(doc.kind),
                _format_doc_range(doc),
                _truncate(doc.text, 140),
            )
        return Panel(table, title=title, border_style="blue")

    def _render_frames_table(self, frames: List[FrameRecord], *, title: str) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Frame", style="cyan", width=14)
        table.add_column("Tier", width=10)
        table.add_column("Time", width=22)
        table.add_column("Event", width=14)
        for frame in frames:
            table.add_row(
                str(frame.frame_id)[:14],
                str(frame.memory_tier or "-"),
                _format_rel_abs(float(frame.t), frame.absolute_t),
                str(frame.event_id or "-")[:14],
            )
        return Panel(table, title=title, border_style="blue")

    async def action_overview(self) -> None:
        overview = await self._build_overview()
        self.console.clear()
        self.console.print(self._render_home(overview))

        summary_structures = overview.get("summary_structures") or []
        if summary_structures:
            table = Table(box=box.SIMPLE_HEAVY, expand=True)
            table.add_column("Structure", style="cyan")
            table.add_column("Granularity", width=14)
            table.add_column("Windows", width=10)
            table.add_column("Focus")
            for item in summary_structures[:12]:
                table.add_row(
                    str(item.get("summary_structure") or "-"),
                    f"{float(item.get('granularity_seconds') or 0.0):.1f}s",
                    str(item.get("count") or 0),
                    _truncate(str(item.get("focus") or ""), 80),
                )
            self.console.print(Panel(table, title="Summary Structures", border_style="magenta"))

        self._pause()

    async def action_index_video(self) -> None:
        self.console.clear()
        self.console.print(Panel("Index a local video into the memory store.", title="Index Video", border_style="cyan"))

        video_path = Prompt.ask("Video path").strip()
        if not video_path:
            return

        settings = get_settings()
        fps = FloatPrompt.ask("FPS", default=float(settings.video_fps))
        max_frames_raw = Prompt.ask("Max frames (blank = no limit)", default="").strip()
        start_time_raw = Prompt.ask("Absolute video start time (blank = relative only)", default="").strip()

        job_id = await index_video_async(
            tmp_path=video_path,
            cleanup_temp=False,
            fps=float(fps),
            max_frames=int(max_frames_raw) if max_frames_raw else None,
            laplacian_min=float(settings.video_laplacian_min),
            diff_threshold=float(settings.video_diff_threshold),
            ssim_threshold=float(settings.video_ssim_threshold),
            hist_threshold=float(settings.video_hist_threshold),
            similarity_threshold=float(settings.video_similarity_threshold),
            frame_id_prefix="v",
            video_absolute_start_time=parse_absolute_time(start_time_raw or None),
            debug=False,
        )

        self.console.print("\n[bold yellow]Indexing started.[/] You can wait here or background this task to start chatting immediately.")
        choice = Prompt.ask("Action", choices=["wait", "background"], default="wait")

        if choice == "background":
            self.console.print(f"[green]Job {job_id} is running in the background.[/] You can check progress in 'Overview'.")
            self._pause()
            return

        with Live(self._render_job_live(job_id=job_id, job=job_manager.get(job_id)), console=self.console, refresh_per_second=5) as live:
            while True:
                job = job_manager.get(job_id)
                live.update(self._render_job_live(job_id=job_id, job=job))
                if job is not None and str(job.status) in {"done", "error", "failed"}:
                    break
                await asyncio.sleep(0.25)

        if job and job.status == "done":
            self.console.print("\n[bold green]Indexing complete![/]")
        elif job and job.status in {"error", "failed"}:
            self.console.print(f"\n[bold red]Indexing failed: {job.error}[/]")

        self._pause()

    async def action_ask_agent(self) -> None:
        self.console.clear()
        self.console.print(Panel("Ask retrieval or summary questions in the terminal.", title="Ask Agent", border_style="cyan"))
        question = Prompt.ask("Question").strip()
        if not question:
            return

        events: List[Dict[str, Any]] = []
        with Live(self._render_agent_live(question=question, events=events), console=self.console, refresh_per_second=5) as live:
            async for event in stream_agent_response(transcript=question, history=self.history):
                events.append(event)
                live.update(self._render_agent_live(question=question, events=events))

        self._pause()

    async def action_summarize_range(self) -> None:
        self.console.clear()
        self.console.print(Panel("Create reusable summary documents in Hierarchical Memory over a time range.", title="Summarize Range", border_style="cyan"))

        min_time = FloatPrompt.ask("Start time", default=0.0)
        max_time_raw = Prompt.ask("End time (blank = to the end)", default="").strip()
        granularity = FloatPrompt.ask("Granularity seconds", default=60.0)
        time_mode = Prompt.ask("Time mode", choices=["auto", "relative", "absolute"], default="auto")
        summary_structure = Prompt.ask("Summary structure (blank = none)", default="").strip()
        prompt_text = Prompt.ask("Prompt", default="Summarize the important events in this window.").strip()

        args = SummarizeToolArgs(
            min_time=float(min_time),
            max_time=float(max_time_raw) if max_time_raw else None,
            time_mode=time_mode,
            granularity_seconds=float(granularity),
            summary_structure=summary_structure or None,
            prompt=prompt_text,
        )

        events: List[Dict[str, Any]] = []
        question = f"Summarize {args.min_time} -> {args.max_time if args.max_time is not None else 'end'}"
        with Live(self._render_agent_live(question=question, events=events), console=self.console, refresh_per_second=5) as live:
            async for event in stream_summarize_tool_request(args=args, history=self.history):
                events.append(event)
                live.update(self._render_agent_live(question=question, events=events))

        final_event = next((item for item in reversed(events) if str(item.get("type") or "") == "final"), None)
        result = final_event.get("result") if isinstance(final_event, dict) else None
        documents = list((result or {}).get("documents") or [])
        if documents:
            table = Table(box=box.SIMPLE_HEAVY, expand=True)
            table.add_column("Doc", width=12, style="cyan")
            table.add_column("Range", width=24)
            table.add_column("Text")
            for item in documents[:10]:
                time_range = item.get("time_range") or {}
                table.add_row(
                    str(item.get("doc_id") or "")[:12],
                    f"{float(time_range.get('start_t') or 0.0):.1f}s -> {float(time_range.get('end_t') or 0.0):.1f}s",
                    _truncate(str(item.get("text") or ""), 140),
                )
            self.console.print(Panel(table, title="Created Summaries", border_style="green"))

        self._pause()

    async def action_browse_memory(self) -> None:
        self.console.clear()
        self.console.print(Panel("Browse frames, events, or summaries in the memory store.", title="Browse Memory", border_style="cyan"))
        mode = Prompt.ask("Browse mode", choices=["frames", "events", "summaries", "docs"], default="events")
        limit = IntPrompt.ask("Rows", default=10)

        if mode == "frames":
            frames = await self.store.list_frames()
            frames = sorted(frames, key=lambda item: (float(item.t), str(item.frame_id)), reverse=True)[:limit]
            self.console.print(self._render_frames_table(frames, title="Frames"))
            self._pause()
            return

        docs = await self.store.list_memory_documents()
        docs = sorted(docs, key=lambda item: (float(item.end_t), float(item.start_t), str(item.doc_id)), reverse=True)
        if mode == "events":
            docs = [doc for doc in docs if str(doc.kind) == "event"]
        elif mode == "summaries":
            docs = [doc for doc in docs if str(doc.kind) == "summary"]
        docs = docs[:limit]
        self.console.print(self._render_docs_table(docs, title=f"Memory: {mode}"))
        self._pause()

    async def action_reset_chat(self) -> None:
        self.history.clear()
        self.console.clear()
        self.console.print(Panel("Chat history reset.", title="Reset Chat", border_style="green"))
        self._pause()

    async def action_clear_memory(self) -> None:
        self.console.clear()
        if not Confirm.ask("Clear all indexed memory and reset chat history?", default=False):
            return
        stats = await self.store.clear_with_stats()
        self.history.clear()
        self.console.print(
            Panel(
                f"Cleared {int(stats['frames'])} frames and {int(stats['documents'])} memory documents; chat history was reset.",
                title="Clear Memory",
                border_style="red",
            )
        )
        self._pause()

    async def run(self) -> None:
        while True:
            overview = await self._build_overview()
            self.console.clear()
            self.console.print(self._render_home(overview))
            choice = Prompt.ask("Select action", choices=["1", "2", "3", "4", "5", "6", "7", "8"], default="1")

            try:
                if choice == "1":
                    await self.action_overview()
                elif choice == "2":
                    await self.action_index_video()
                elif choice == "3":
                    await self.action_ask_agent()
                elif choice == "4":
                    await self.action_summarize_range()
                elif choice == "5":
                    await self.action_browse_memory()
                elif choice == "6":
                    await self.action_reset_chat()
                elif choice == "7":
                    await self.action_clear_memory()
                elif choice == "8":
                    self.console.clear()
                    self.console.print(Panel("Goodbye.", border_style="cyan"))
                    return
            except Exception as exc:
                self.console.print(Panel(str(exc), title="Error", border_style="red"))
                self._pause()


class PlainAgentCLI:
    def __init__(self) -> None:
        self.store = resolve_memory_store()
        self.history: List[Dict[str, Any]] = []

    def _ask(self, label: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        value = input(f"{label}{suffix}: ").strip()
        return value or default

    def _pause(self) -> None:
        input("\nPress Enter to continue...")

    async def _show_overview(self) -> None:
        frames = await self.store.list_frames()
        docs = await self.store.list_memory_documents()
        latest_event = None
        events = [doc for doc in docs if str(doc.kind) == "event"]
        if events:
            latest_event = max(events, key=lambda doc: (float(doc.end_t), float(doc.start_t), str(doc.doc_id)))

        print("\n== Overview ==")
        print(f"Frames: {len(frames)}")
        print(f"Docs: {len(docs)}")
        print(f"Events: {sum(1 for doc in docs if str(doc.kind) == 'event')}")
        print(f"Summaries: {sum(1 for doc in docs if str(doc.kind) == 'summary')}")
        print(f"Chat turns: {len(self.history) // 2}")
        if latest_event is not None:
            print(f"Latest event: {latest_event.doc_id[:12]} | {_format_doc_range(latest_event)}")
            print(_truncate(latest_event.text, 180))
        self._pause()

    async def _index_video(self) -> None:
        print("\n== Index Video ==")
        video_path = self._ask("Video path")
        if not video_path:
            return
        settings = get_settings()
        fps = float(self._ask("FPS", str(settings.video_fps)))
        max_frames_raw = self._ask("Max frames (blank = no limit)")
        start_time_raw = self._ask("Absolute start time (blank = relative only)")

        job_id = await index_video_async(
            tmp_path=video_path,
            cleanup_temp=False,
            fps=float(fps),
            max_frames=int(max_frames_raw) if max_frames_raw else None,
            laplacian_min=float(settings.video_laplacian_min),
            diff_threshold=float(settings.video_diff_threshold),
            ssim_threshold=float(settings.video_ssim_threshold),
            hist_threshold=float(settings.video_hist_threshold),
            similarity_threshold=float(settings.video_similarity_threshold),
            frame_id_prefix="v",
            video_absolute_start_time=parse_absolute_time(start_time_raw or None),
            debug=False,
        )
        background = self._ask("Wait for completion or background? (wait/background)", "wait").lower()
        if background == "background":
            print(f"Job {job_id} is running in the background.")
            self._pause()
            return

        last_status = ""
        last_log_count = -1
        while True:
            job = job_manager.get(job_id)
            if job is None:
                print("Job disappeared.")
                break
            status_line = f"{job.status} | {job.phase} | {job.current}/{job.total} ({_job_percent(job):.1f}%)"
            if status_line != last_status:
                print(status_line)
                last_status = status_line
            if len(job.logs) != last_log_count:
                for log in job.logs[last_log_count + 1 :]:
                    print(f"  log: {log}")
                last_log_count = len(job.logs)
            if str(job.status) in {"done", "error", "failed"}:
                if job.result:
                    print(f"Result: count={job.result.get('count', 0)} source_fps={job.result.get('source_fps')}")
                if job.error:
                    print(f"Error: {job.error}")
                break
            await asyncio.sleep(0.25)
        self._pause()

    async def _ask_agent(self) -> None:
        print("\n== Ask Agent ==")
        question = self._ask("Question")
        if not question:
            return
        async for event in stream_agent_response(transcript=question, history=self.history):
            detail = _truncate(AgentTUI._event_detail(self, event), 180)  # type: ignore[misc]
            print(f"[{event.get('type')}] {detail}")
        self._pause()

    async def _summarize(self) -> None:
        print("\n== Summarize Range ==")
        min_time = float(self._ask("Start time", "0"))
        max_time_raw = self._ask("End time (blank = to the end)")
        granularity = float(self._ask("Granularity seconds", "60"))
        time_mode = self._ask("Time mode", "auto")
        summary_structure = self._ask("Summary structure")
        prompt_text = self._ask("Prompt", "Summarize the important events in this window.")

        args = SummarizeToolArgs(
            min_time=min_time,
            max_time=float(max_time_raw) if max_time_raw else None,
            time_mode=time_mode,
            granularity_seconds=granularity,
            summary_structure=summary_structure or None,
            prompt=prompt_text,
        )
        async for event in stream_summarize_tool_request(args=args, history=self.history):
            detail = _truncate(AgentTUI._event_detail(self, event), 180)  # type: ignore[misc]
            print(f"[{event.get('type')}] {detail}")
        self._pause()

    async def _browse_memory(self) -> None:
        print("\n== Browse Memory ==")
        mode = self._ask("Mode (frames/events/summaries/docs)", "events")
        limit = int(self._ask("Rows", "10"))
        if mode == "frames":
            frames = await self.store.list_frames()
            frames = sorted(frames, key=lambda item: (float(item.t), str(item.frame_id)), reverse=True)[:limit]
            for frame in frames:
                print(f"{frame.frame_id[:14]} | {frame.memory_tier or '-':<6} | {_format_rel_abs(float(frame.t), frame.absolute_t)}")
            self._pause()
            return

        docs = await self.store.list_memory_documents()
        docs = sorted(docs, key=lambda item: (float(item.end_t), float(item.start_t), str(item.doc_id)), reverse=True)
        if mode == "events":
            docs = [doc for doc in docs if str(doc.kind) == "event"]
        elif mode == "summaries":
            docs = [doc for doc in docs if str(doc.kind) == "summary"]
        for doc in docs[:limit]:
            print(f"{doc.doc_id[:12]} | {doc.layer}/{doc.kind} | {_format_doc_range(doc)}")
            print(f"  {_truncate(doc.text, 180)}")
        self._pause()

    async def _reset_chat(self) -> None:
        self.history.clear()
        print("Chat history reset.")
        self._pause()

    async def _clear_memory(self) -> None:
        confirm = self._ask("Type YES to clear memory and chat", "").upper()
        if confirm != "YES":
            return
        stats = await self.store.clear_with_stats()
        self.history.clear()
        print(f"Cleared {int(stats['frames'])} frames and {int(stats['documents'])} memory documents; chat history was reset.")
        self._pause()

    async def run(self) -> None:
        while True:
            print("\n== Visual Agentic Memory CLI ==")
            print("1. Overview")
            print("2. Index Video")
            print("3. Ask Agent")
            print("4. Summarize Range")
            print("5. Browse Memory")
            print("6. Reset Chat")
            print("7. Clear Memory")
            print("8. Quit")
            choice = self._ask("Select action", "1")
            if choice == "1":
                await self._show_overview()
            elif choice == "2":
                await self._index_video()
            elif choice == "3":
                await self._ask_agent()
            elif choice == "4":
                await self._summarize()
            elif choice == "5":
                await self._browse_memory()
            elif choice == "6":
                await self._reset_chat()
            elif choice == "7":
                await self._clear_memory()
            elif choice == "8":
                print("Goodbye.")
                return
            else:
                print("Unknown choice.")


async def main() -> None:
    ui = AgentTUI() if RICH_AVAILABLE else PlainAgentCLI()
    await ui.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        if RICH_AVAILABLE and Console is not None:
            Console().print("\n[dim]Interrupted.[/]")
        else:
            print("\nInterrupted.")
