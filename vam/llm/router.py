from typing import Any, Dict, List, Optional, Tuple

from vam.config import get_settings
from vam.llm.openrouter import OpenRouterClient


class LLMRouter:
    """
    Simple dual-LLM router with 'main' and 'light' routes.

    Usage:
        router = LLMRouter()
        text, raw = await router.chat(messages, route="main", temperature=0.2)
    """

    def __init__(self) -> None:
        cfg = get_settings()
        # Instantiate clients; fall back to single-model id when specific ids are unset
        main_id = cfg.openrouter_model_id_main or cfg.openrouter_model_id_light or cfg.openrouter_model_id
        light_id = cfg.openrouter_model_id_light or cfg.openrouter_model_id
        self.main = OpenRouterClient(model_id=main_id)
        self.light = OpenRouterClient(model_id=light_id)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        route: str = "main",  # 'main' | 'light'
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        client = self.main if route == "main" else self.light
        return await client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            extra_params=extra_params,
        )

    # Placeholder policy hook — expand later if needed
    def pick_route(
        self,
        *,
        input_length: Optional[int] = None,
        has_images: bool = False,
        latency_budget_ms: Optional[int] = None,
    ) -> str:
        """Decide 'main' vs 'light' based on simple heuristics (stub)."""
        if input_length and input_length > 2000:
            return "main"
        if has_images:
            return "main"
        if latency_budget_ms and latency_budget_ms < 300:
            return "light"
        return "main"
