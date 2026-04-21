from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

try:
    import torch
except Exception:
    torch = None

from vam.config import get_settings
from vam import prompts


@lru_cache(maxsize=1)
def _backend_instance():
    from .backends import get_backend
    cfg = get_settings()
    return get_backend(cfg.vision_embedding_backend)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required")


def embed_text(text: str, *, instruction: str = prompts.EMBED_INSTRUCTION_RETRIEVE_RELEVANT) -> Tuple[List[float], int]:
    return _backend_instance().embed_text(text, instruction=instruction)


def embed_texts_batch(
    texts: List[str],
    *,
    instruction: str = prompts.EMBED_INSTRUCTION_RETRIEVE_RELEVANT,
    batch_size: int = 8,
) -> Tuple[List[List[float]], int]:
    return _backend_instance().embed_texts_batch(texts, instruction=instruction, batch_size=batch_size)


def embed_image_base64(
    image_base64_or_data_uri: str, *, instruction: str = prompts.EMBED_INSTRUCTION_REPRESENT_USER_INPUT
) -> Tuple[List[float], int]:
    return _backend_instance().embed_image_base64(image_base64_or_data_uri, instruction=instruction)


def embed_images_base64_batch(
    images_base64_or_data_uris: List[str],
    *,
    instruction: str = prompts.EMBED_INSTRUCTION_REPRESENT_USER_INPUT,
    batch_size: int = 8,
) -> Tuple[List[List[float]], int]:
    return _backend_instance().embed_images_base64_batch(images_base64_or_data_uris, instruction=instruction, batch_size=batch_size)


def backend_name() -> str:
    return _backend_instance().name
