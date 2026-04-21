from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
import httpx
import asyncio
import base64
import io
import os
import time
from PIL import Image

from vam.config import get_settings


def _ensure_image_limit(data_uri: str) -> str:
    """
    Ensures the image in the data URI doesn't exceed the configured LLM limits.
    Resizes if pixels exceed max, or compresses if size exceeds max.
    """
    if not data_uri.startswith("data:image/"):
        return data_uri

    cfg = get_settings()
    max_pixels = cfg.llm_max_image_pixels
    max_bytes = int(cfg.llm_max_image_size_mb * 1024 * 1024)

    try:
        # 1. Parse Data URI
        header, encoded = data_uri.split(",", 1)
        image_data = base64.b64decode(encoded)
        
        # 2. Check size initially
        if len(image_data) <= max_bytes / 2 and "jpeg" in header:
            # Already small enough and is JPEG, skip processing for speed
            # (Heuristic: base64 is ~33% larger than raw bytes, so check max_bytes/2)
            pass

        # 3. Open image
        img = Image.open(io.BytesIO(image_data))
        orig_w, orig_h = img.size
        
        # 4. Resize if needed
        if orig_w * orig_h > max_pixels:
            scale = (max_pixels / (orig_w * orig_h)) ** 0.5
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 5. Compress/Convert to JPEG
        # Convert to RGB if needed (e.g. for PNG with alpha)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        output = io.BytesIO()
        quality = 85
        img.save(output, format="JPEG", quality=quality, optimize=True)
        
        # 6. Iterative compression if still too large
        while output.tell() > max_bytes and quality > 30:
            quality -= 10
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=quality, optimize=True)
            
        new_encoded = base64.b64encode(output.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{new_encoded}"
        
    except Exception as e:
        print(f"[OpenRouter] Image processing warning: {e}")
        return data_uri


def build_mm_user_content(
    text: str,
    images: Optional[List[str]] = None,
    videos: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for url in images or []:
        processed_url = _ensure_image_limit(url)
        content.append({"type": "image_url", "image_url": {"url": processed_url}})
    for url in videos or []:
        content.append({"type": "video_url", "video_url": {"url": url}})
    return content


class TokenUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.reasoning_tokens = 0
        self.request_count = 0

    def add(self, usage: Optional[Dict[str, Any]]) -> None:
        payload = usage or {}
        self.prompt_tokens += int(payload.get("prompt_tokens") or 0)
        self.completion_tokens += int(payload.get("completion_tokens") or 0)
        self.total_tokens += int(payload.get("total_tokens") or 0)
        self.reasoning_tokens += int(
            payload.get("reasoning_tokens")
            or payload.get("reasoningTokens")
            or 0
        )
        self.request_count += 1

    def to_dict(self) -> Dict[str, int]:
        return {
            "request_count": self.request_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }

    def __repr__(self) -> str:
        data = self.to_dict()
        return (
            "TokenUsage("
            f"requests={data['request_count']}, "
            f"prompt={data['prompt_tokens']}, "
            f"completion={data['completion_tokens']}, "
            f"reasoning={data['reasoning_tokens']}, "
            f"total={data['total_tokens']}"
            ")"
        )


class UsageLedger:
    def __init__(self) -> None:
        self.total = TokenUsage()
        self.by_model: Dict[str, TokenUsage] = {}

    def add(self, *, model_id: str, usage: Optional[Dict[str, Any]]) -> None:
        self.total.add(usage)
        key = (model_id or "unknown").strip() or "unknown"
        entry = self.by_model.get(key)
        if entry is None:
            entry = TokenUsage()
            self.by_model[key] = entry
        entry.add(usage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total.to_dict(),
            "by_model": {k: v.to_dict() for k, v in self.by_model.items()},
        }

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


global_usage = UsageLedger()


def get_global_usage_snapshot() -> Dict[str, Any]:
    return global_usage.to_dict()


class OpenRouterClient:
    def __init__(self, *, model_id: Optional[str] = None) -> None:
        cfg = get_settings()
        # Priority: explicit param > openrouter_model_id (env) > llm_model (env/default)
        self.model_id = model_id or cfg.openrouter_model_id or cfg.llm_model

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        text, raw = await self._chat_impl(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            extra_params=extra_params,
        )
        if raw and "usage" in raw:
            model_name = str(raw.get("model") or self.model_id or "unknown")
            global_usage.add(model_id=model_name, usage=raw["usage"])
            total = global_usage.total
            if total.request_count % 10 == 0:
                print(
                    f"[TokenStats] Requests={total.request_count} "
                    f"Prompt={total.prompt_tokens} "
                    f"Completion={total.completion_tokens} "
                    f"Reasoning={total.reasoning_tokens} "
                    f"Total={total.total_tokens}"
                )
        return text, raw

    async def _chat_impl(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        cfg = get_settings()
        if not cfg.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")

        base_url = (cfg.openrouter_base_url or "https://openrouter.ai/api/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": cfg.app_name,
        }

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stream:
            payload["stream"] = True
        if extra_params:
            payload.update(extra_params)

        image_count = 0
        video_count = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "image_url":
                        image_count += 1
                    if part.get("type") == "video_url":
                        video_count += 1

        try:
            request_timeout_s = float(os.getenv("OPENROUTER_REQUEST_TIMEOUT_S", "240"))
        except Exception:
            request_timeout_s = 240.0
        try:
            connect_timeout_s = float(os.getenv("OPENROUTER_CONNECT_TIMEOUT_S", "60"))
        except Exception:
            connect_timeout_s = 60.0
        try:
            max_retries = max(1, int(os.getenv("OPENROUTER_MAX_RETRIES", "10")))
        except Exception:
            max_retries = 10

        raw: Dict[str, Any]
        timeout = httpx.Timeout(request_timeout_s, connect=connect_timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            last_err: Optional[Exception] = None
            for i in range(max_retries):
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    raw = resp.json()
                    break
                except httpx.HTTPStatusError as e:
                    print(f"[OpenRouter] Error {e.response.status_code}: {e.response.text}")
                    last_err = e
                    if e.response.status_code not in {429, 500, 502, 503, 504}:
                        raise
                except httpx.HTTPError as e:
                    last_err = e
                if i < max_retries - 1:
                    await asyncio.sleep(1.0 * (i + 1))
            else:
                raise last_err if last_err is not None else RuntimeError("OpenRouter request failed")

        text: Optional[str] = None
        try:
            text = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content")
        except Exception:
            text = None
        if text is not None and not isinstance(text, str):
            try:
                text = json.dumps(text, ensure_ascii=False)
            except Exception:
                text = str(text)

        return text, raw


class OpenRouterEmbeddingsClient:
    def __init__(self, *, model_id: Optional[str] = None) -> None:
        cfg = get_settings()
        self.model_id = model_id or cfg.openrouter_embedding_model or "google/gemini-embedding-2-preview"

    def _headers(self) -> Dict[str, str]:
        cfg = get_settings()
        if not cfg.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        return {
            "Authorization": f"Bearer {cfg.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": cfg.app_name,
        }

    def _base_payload(self, *, input_type: Optional[str] = None) -> Dict[str, Any]:
        cfg = get_settings()
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "encoding_format": "float",
        }
        if input_type:
            payload["input_type"] = input_type
        if cfg.openrouter_embedding_dimensions is not None and int(cfg.openrouter_embedding_dimensions) > 0:
            payload["dimensions"] = int(cfg.openrouter_embedding_dimensions)
        return payload

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cfg = get_settings()
        base_url = (cfg.openrouter_base_url or "https://openrouter.ai/api/v1").rstrip("/")
        url = f"{base_url}/embeddings"
        last_err: Optional[Exception] = None
        for i in range(4):
            try:
                response = httpx.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=httpx.Timeout(60.0, connect=10.0),
                )
                response.raise_for_status()
                raw = response.json()
                if not isinstance(raw, dict):
                    raise RuntimeError(f"unexpected OpenRouter embedding response: {raw}")
                return raw
            except httpx.HTTPStatusError as exc:
                last_err = exc
                if exc.response.status_code not in {429, 500, 502, 503, 504} or i >= 3:
                    raise
            except httpx.HTTPError as exc:
                last_err = exc
                if i >= 3:
                    raise
            time.sleep(1.0 * (i + 1))
        raise last_err if last_err is not None else RuntimeError("OpenRouter embedding request failed")

    def _extract_embeddings(self, raw: Dict[str, Any]) -> Tuple[List[List[float]], int]:
        data = raw.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"unexpected OpenRouter embedding response: {raw}")
        vectors: List[List[float]] = []
        for item in data:
            emb = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(emb, list):
                raise RuntimeError(f"missing embedding in response item: {item}")
            vectors.append([float(v) for v in emb])
        return vectors, len(vectors[0])

    def generate_text_embeddings(
        self,
        texts: List[str],
        *,
        input_type: Optional[str] = None,
    ) -> Tuple[List[List[float]], int]:
        payload = self._base_payload(input_type=input_type)
        payload["input"] = list(texts)
        return self._extract_embeddings(self._post(payload))

    def generate_image_embeddings(
        self,
        image_urls: List[str],
        *,
        input_type: Optional[str] = None,
    ) -> Tuple[List[List[float]], int]:
        payload = self._base_payload(input_type=input_type)
        payload["input"] = [
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                ]
            }
            for url in image_urls
        ]
        return self._extract_embeddings(self._post(payload))
