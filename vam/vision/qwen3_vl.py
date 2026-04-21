from __future__ import annotations

import base64
import io
import os
import time
from functools import lru_cache
from typing import List, Tuple

import httpx

try:
    from PIL import Image
except Exception:
    Image = None

from vam.config import get_settings
from vam import prompts


def _b64decode(value: str) -> bytes:
    v = "".join((value or "").split())
    try:
        return base64.b64decode(v, validate=True)
    except Exception:
        pad = (-len(v)) % 4
        v2 = v + ("=" * pad)
        return base64.urlsafe_b64decode(v2)


def _downscale_image(img: Image.Image, *, max_side: int = 448) -> Image.Image:
    try:
        w, h = img.size
    except Exception:
        return img
    m = int(max_side)
    if m <= 0 or max(w, h) <= m:
        return img
    out = img.copy()
    out.thumbnail((m, m))
    return out


def _coerce_image_str_to_data_uri(value: str) -> str:
    if Image is None:
        raise RuntimeError("Pillow is required")
    v = (value or "").strip()
    if not v:
        raise ValueError("image input is empty")
    if v.startswith("data:"):
        return v
    if v.startswith(("http://", "https://")):
        resp = httpx.get(v, timeout=30.0)
        resp.raise_for_status()
        mime = resp.headers.get("content-type") or "image/png"
        return f"data:{mime};base64,{base64.b64encode(resp.content).decode('ascii')}"
    try:
        raw = _b64decode(v)
        return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
    except Exception:
        pass
    img = _downscale_image(Image.open(v).convert("RGB"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


class _QwenEmbeddingHttpClient:
    def __init__(self) -> None:
        cfg = get_settings()
        self.api = str(cfg.self_qwen_embedding_api or "").strip()
        self.api_key = str(cfg.self_qwen_embedding_key or "").strip()
        self.model = str(cfg.qwen3_vl_embedding_model or "").strip()
        if not self.api:
            raise RuntimeError("SELF_QWEN_EMBEDDING_API is required for qwen embeddings")
        if not self.api_key:
            raise RuntimeError("SELF_QWEN_EMBEDDING_KEY is required for qwen embeddings")
        if not self.model:
            raise RuntimeError("QWEN3_VL_EMBEDDING_MODEL is required for qwen embeddings")

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def embed_inputs(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        if not inputs:
            raise ValueError("inputs is empty")
        payload = {
            "model": self.model,
            "input": inputs,
        }
        try:
            request_timeout_s = float(os.getenv("QWEN_EMBEDDING_REQUEST_TIMEOUT_S", "240"))
        except Exception:
            request_timeout_s = 240.0
        try:
            connect_timeout_s = float(os.getenv("QWEN_EMBEDDING_CONNECT_TIMEOUT_S", "60"))
        except Exception:
            connect_timeout_s = 60.0
        try:
            max_retries = max(1, int(os.getenv("QWEN_EMBEDDING_MAX_RETRIES", "10")))
        except Exception:
            max_retries = 10

        timeout = httpx.Timeout(request_timeout_s, connect=connect_timeout_s)
        last_exc: Exception | None = None
        with httpx.Client(timeout=timeout) as client:
            for attempt in range(1, max_retries + 1):
                try:
                    resp = client.post(self.api, headers=self._headers(), json=payload)
                    resp.raise_for_status()
                    break
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    if exc.response.status_code not in {429, 500, 502, 503, 504} or attempt >= max_retries:
                        raise
                except httpx.HTTPError as exc:
                    last_exc = exc
                    if attempt >= max_retries:
                        raise
                time.sleep(min(10.0, float(attempt)))
            else:
                raise last_exc if last_exc is not None else RuntimeError("qwen embedding request failed")
        body = resp.json()
        data = body.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"unexpected qwen embedding response: {body}")
        vectors: List[List[float]] = []
        for item in data:
            emb = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(emb, list):
                raise RuntimeError(f"missing embedding in response item: {item}")
            vectors.append(emb)
        return vectors, len(vectors[0])


@lru_cache(maxsize=1)
def _client() -> _QwenEmbeddingHttpClient:
    return _QwenEmbeddingHttpClient()


def embed_text(text: str, *, instruction: str = prompts.EMBED_INSTRUCTION_RETRIEVE_RELEVANT) -> Tuple[List[float], int]:
    merged = f"{instruction}\n\n{text or 'NULL'}".strip()
    vectors, dim = _client().embed_inputs([merged])
    return vectors[0], dim


def embed_texts_batch(
    texts: List[str], *, instruction: str = prompts.EMBED_INSTRUCTION_RETRIEVE_RELEVANT, batch_size: int = 8
) -> Tuple[List[List[float]], int]:
    del batch_size
    merged = [f"{instruction}\n\n{(text or 'NULL')}".strip() for text in (texts or [])]
    return _client().embed_inputs(merged)


def embed_image_base64(
    image_base64_or_data_uri: str, *, instruction: str = prompts.EMBED_INSTRUCTION_REPRESENT_USER_INPUT
) -> Tuple[List[float], int]:
    del instruction
    data_uri = _coerce_image_str_to_data_uri(image_base64_or_data_uri)
    vectors, dim = _client().embed_inputs([data_uri])
    return vectors[0], dim


def embed_images_base64_batch(
    images_base64_or_data_uris: List[str], *, instruction: str = prompts.EMBED_INSTRUCTION_REPRESENT_USER_INPUT, batch_size: int = 8
) -> Tuple[List[List[float]], int]:
    del instruction, batch_size
    data_uris = [_coerce_image_str_to_data_uri(v) for v in (images_base64_or_data_uris or [])]
    return _client().embed_inputs(data_uris)
