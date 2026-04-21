from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class StrictSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrieveToolArgs(StrictSchemaModel):
    question: str = Field(..., min_length=1, description="Canonical retrieval question for memory search.")

    @field_validator("question")
    @classmethod
    def _normalize_question(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("question must not be empty")
        return text


class RetrieveToolCall(StrictSchemaModel):
    type: Literal["tool"]
    name: Literal["retrieve"]
    args: RetrieveToolArgs


class SummarizeToolArgs(StrictSchemaModel):
    min_time: float = Field(..., ge=0.0)
    max_time: Optional[float] = Field(None, ge=0.0)
    time_mode: Literal["relative", "absolute", "auto"] = "auto"
    granularity_seconds: float = Field(..., gt=0.0, le=86400.0)
    summary_structure: Optional[str] = Field(None, min_length=1, max_length=64)
    prompt: str = Field(..., min_length=1)

    @field_validator("summary_structure")
    @classmethod
    def _normalize_summary_structure(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip().lower()
        if not text:
            raise ValueError("summary_structure must not be empty")
        return text

    @field_validator("prompt")
    @classmethod
    def _normalize_prompt(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("prompt must not be empty")
        return text

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SummarizeToolArgs":
        if self.max_time is not None and self.max_time < self.min_time:
            raise ValueError("max_time must be >= min_time")
        return self


class SummarizeToolCall(StrictSchemaModel):
    type: Literal["tool"]
    name: Literal["summarize"]
    args: SummarizeToolArgs


class FinalAnswerCall(StrictSchemaModel):
    type: Literal["final"]
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("text must not be empty")
        return text


WsAgentDecision = Union[RetrieveToolCall, SummarizeToolCall, FinalAnswerCall]


class PlannerBestRef(StrictSchemaModel):
    turn_idx: int = Field(..., ge=1)
    result_idx: int = Field(..., ge=0)


class PlannerTimeRange(StrictSchemaModel):
    min: Optional[float] = None
    max: Optional[float] = None
    mode: Literal["relative", "absolute", "auto"] = "auto"
    anchor: Optional["PlannerTimeAnchor"] = None
    start_anchor: Optional["PlannerTimeAnchor"] = None
    end_anchor: Optional["PlannerTimeAnchor"] = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "PlannerTimeRange":
        if self.min is not None and self.max is not None and self.max < self.min:
            raise ValueError("time_range.max must be >= time_range.min")
        has_single_anchor = self.anchor is not None
        has_range_anchor = self.start_anchor is not None or self.end_anchor is not None
        if has_single_anchor and has_range_anchor:
            raise ValueError("time_range cannot mix anchor with start_anchor/end_anchor")
        if (self.start_anchor is None) != (self.end_anchor is None):
            raise ValueError("time_range requires both start_anchor and end_anchor when using anchored ranges")
        return self


class PlannerVisualRef(StrictSchemaModel):
    turn_idx: int = Field(..., ge=1)
    result_idx: int = Field(..., ge=0)


class PlannerTimeAnchor(StrictSchemaModel):
    query: Optional[str] = Field(None, min_length=1)
    query_variants: Optional[List[str]] = None
    ref: Optional[PlannerBestRef] = None
    before_seconds: float = Field(0.0, ge=0.0, le=604800.0)
    after_seconds: float = Field(0.0, ge=0.0, le=604800.0)
    sources: Optional[List[Literal["frame", "event", "summary"]]] = None
    candidate_source_groups: Optional[List[List[Literal["frame", "event", "summary"]]]] = None
    top_k: int = Field(5, ge=1, le=50)
    inspect_k: int = Field(3, ge=0, le=20)
    threshold: float = Field(0.0, ge=0.0)
    occurrence_index: Optional[int] = Field(None, ge=1, le=100)
    verification_prompt: Optional[str] = Field(None, min_length=1, max_length=400)

    @field_validator("query")
    @classmethod
    def _normalize_query(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("anchor query must not be empty")
        return text

    @field_validator("query_variants")
    @classmethod
    def _normalize_query_variants(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        normalized: List[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            normalized.append(text)
            seen.add(key)
        if not normalized:
            raise ValueError("query_variants must contain at least one non-empty string")
        return normalized

    @field_validator("candidate_source_groups")
    @classmethod
    def _normalize_candidate_source_groups(
        cls,
        value: Optional[List[List[Literal["frame", "event", "summary"]]]],
    ) -> Optional[List[List[Literal["frame", "event", "summary"]]]]:
        if value is None:
            return None
        normalized: List[List[Literal["frame", "event", "summary"]]] = []
        seen: set[tuple[str, ...]] = set()
        for group in value:
            cleaned = [str(item).strip().lower() for item in (group or []) if str(item).strip()]
            if not cleaned:
                continue
            deduped = list(dict.fromkeys(cleaned))
            key = tuple(deduped)
            if key in seen:
                continue
            normalized.append(deduped)  # type: ignore[list-item]
            seen.add(key)
        if not normalized:
            raise ValueError("candidate_source_groups must contain at least one non-empty group")
        return normalized

    @field_validator("verification_prompt")
    @classmethod
    def _normalize_verification_prompt(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("verification_prompt must not be empty")
        return text

    @model_validator(mode="after")
    def _validate_anchor(self) -> "PlannerTimeAnchor":
        if self.inspect_k > self.top_k:
            self.inspect_k = self.top_k
        has_query = self.query is not None or bool(self.query_variants)
        if not has_query and self.ref is None:
            raise ValueError("time_range.anchor requires query, query_variants, or ref")
        return self


class PlannerQuerySpec(StrictSchemaModel):
    q: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=200)
    inspect_k: int = Field(5, ge=0, le=50)
    threshold: float = Field(0.65, ge=0.0)

    @field_validator("q")
    @classmethod
    def _normalize_q(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("query text must not be empty")
        return text

    @model_validator(mode="after")
    def _validate_inspect_k(self) -> "PlannerQuerySpec":
        if self.inspect_k > self.top_k:
            self.inspect_k = self.top_k
        return self


class PlannerSummaryFilter(StrictSchemaModel):
    summary_structure: Optional[str] = Field(None, min_length=1, max_length=64)
    granularity_seconds: Optional[float] = Field(None, gt=0.0, le=86400.0)

    @field_validator("summary_structure")
    @classmethod
    def _normalize_summary_structure(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip().lower()
        if not text:
            raise ValueError("summary_structure must not be empty")
        return text


class PlannerAnswerAction(StrictSchemaModel):
    action: Literal["answer"]
    response: str = Field(..., min_length=1)
    best_ref: Optional[PlannerBestRef] = None
    thought: str = ""

    @field_validator("response")
    @classmethod
    def _normalize_response(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("response must not be empty")
        return text


class PlannerSearchAction(StrictSchemaModel):
    action: Literal["search"]
    queries: List[PlannerQuerySpec] = Field(..., min_length=1)
    time_range: Optional[PlannerTimeRange] = None
    sources: Optional[List[Literal["frame", "event", "summary"]]] = None
    summary_filter: Optional[PlannerSummaryFilter] = None
    visual_ref: Optional[PlannerVisualRef] = None
    joint_inspection: bool = False
    inspection_prompt: str = ""
    thought: str = ""


class PlannerInspectAction(StrictSchemaModel):
    action: Literal["inspect"]
    prompt: str = Field(..., min_length=1)
    time_range: Optional[PlannerTimeRange] = None
    ref: Optional[PlannerBestRef] = None
    max_frames: int = Field(6, ge=1, le=16)
    thought: str = ""

    @field_validator("prompt")
    @classmethod
    def _normalize_prompt(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("prompt must not be empty")
        return text

    @model_validator(mode="after")
    def _validate_target(self) -> "PlannerInspectAction":
        if self.time_range is None and self.ref is None:
            raise ValueError("inspect requires time_range or ref")
        return self


class PlannerSummarizeAction(StrictSchemaModel):
    action: Literal["summarize"]
    time_range: PlannerTimeRange
    granularity_seconds: float = Field(..., gt=0.0, le=86400.0)
    summary_structure: Optional[str] = Field(None, min_length=1, max_length=64)
    prompt: str = Field(..., min_length=1)
    thought: str = ""

    @field_validator("summary_structure")
    @classmethod
    def _normalize_summary_structure(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip().lower()
        if not text:
            raise ValueError("summary_structure must not be empty")
        return text

    @field_validator("prompt")
    @classmethod
    def _normalize_prompt(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("prompt must not be empty")
        return text


PlannerAction = Union[PlannerAnswerAction, PlannerSearchAction, PlannerInspectAction, PlannerSummarizeAction]


class PlannerActionAdapter(StrictSchemaModel):
    payload: PlannerAction

    @classmethod
    def parse_payload(cls, raw: Dict[str, Any]) -> PlannerAction:
        return cls.model_validate({"payload": raw}).payload
