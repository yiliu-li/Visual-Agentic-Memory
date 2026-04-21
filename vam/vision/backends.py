from __future__ import annotations

from typing import Callable, List, Optional, Protocol, Tuple


from vam.config import get_settings


class VisionBackend(Protocol):
    name: str

    def embed_text(self, text: str, *, instruction: str) -> Tuple[List[float], int]: ...
    def embed_texts_batch(
        self, texts: List[str], *, instruction: str, batch_size: int
    ) -> Tuple[List[List[float]], int]: ...
    def embed_image_base64(self, image_base64_or_data_uri: str, *, instruction: str) -> Tuple[List[float], int]: ...
    def embed_images_base64_batch(
        self, images_base64_or_data_uris: List[str], *, instruction: str, batch_size: int
    ) -> Tuple[List[List[float]], int]: ...


class Qwen3VLBackend:
    name = "qwen3_vl"

    def embed_text(self, text: str, *, instruction: str) -> Tuple[List[float], int]:
        from .qwen3_vl import embed_text

        return embed_text(text, instruction=instruction)

    def embed_texts_batch(self, texts: List[str], *, instruction: str, batch_size: int) -> Tuple[List[List[float]], int]:
        from .qwen3_vl import embed_texts_batch

        return embed_texts_batch(texts, instruction=instruction, batch_size=batch_size)

    def embed_image_base64(self, image_base64_or_data_uri: str, *, instruction: str) -> Tuple[List[float], int]:
        from .qwen3_vl import embed_image_base64

        return embed_image_base64(image_base64_or_data_uri, instruction=instruction)

    def embed_images_base64_batch(
        self, images_base64_or_data_uris: List[str], *, instruction: str, batch_size: int
    ) -> Tuple[List[List[float]], int]:
        from .qwen3_vl import embed_images_base64_batch

        return embed_images_base64_batch(images_base64_or_data_uris, instruction=instruction, batch_size=batch_size)


class OpenRouterGeminiBackend:
    name = "openrouter_gemini"

    def embed_text(self, text: str, *, instruction: str) -> Tuple[List[float], int]:
        from vam.llm.openrouter import OpenRouterEmbeddingsClient
        client = OpenRouterEmbeddingsClient()
        return client.generate_text_embeddings([text], input_type="search_query" if "query" in instruction.lower() else "search_document")

    def embed_texts_batch(self, texts: List[str], *, instruction: str, batch_size: int) -> Tuple[List[List[float]], int]:
        from ..llm.openrouter import OpenRouterEmbeddingsClient
        client = OpenRouterEmbeddingsClient()
        return client.generate_text_embeddings(texts, input_type="search_query" if "query" in instruction.lower() else "search_document")

    def embed_image_base64(self, image_base64_or_data_uri: str, *, instruction: str) -> Tuple[List[float], int]:
        from ..llm.openrouter import OpenRouterEmbeddingsClient
        client = OpenRouterEmbeddingsClient()
        return client.generate_image_embeddings([image_base64_or_data_uri])

    def embed_images_base64_batch(
        self, images_base64_or_data_uris: List[str], *, instruction: str, batch_size: int
    ) -> Tuple[List[List[float]], int]:
        from ..llm.openrouter import OpenRouterEmbeddingsClient
        client = OpenRouterEmbeddingsClient()
        return client.generate_image_embeddings(images_base64_or_data_uris)


class DummyBackend:
    name = "dummy"

    def embed_text(self, text: str, *, instruction: str) -> Tuple[List[float], int]:
        return [0.0] * 128, 128

    def embed_texts_batch(self, texts: List[str], *, instruction: str, batch_size: int) -> Tuple[List[List[float]], int]:
        n = len(texts)
        return [[0.0] * 128 for _ in range(n)], 128

    def embed_image_base64(self, image_base64_or_data_uri: str, *, instruction: str) -> Tuple[List[float], int]:
        return [0.0] * 128, 128

    def embed_images_base64_batch(
        self, images_base64_or_data_uris: List[str], *, instruction: str, batch_size: int
    ) -> Tuple[List[List[float]], int]:
        n = len(images_base64_or_data_uris)
        return [[0.0] * 128 for _ in range(n)], 128


def normalize_backend(name: Optional[str]) -> str:
    v = (name or "").strip().lower()
    if not v:
        return "openrouter_gemini"
    if v in ("qwen", "qwen3", "qwen3vl", "qwen3_vl"):
        return "qwen3_vl"
    if v in ("gemini", "google", "openrouter_gemini", "gemini_openrouter", "openrouter-gemini"):
        return "openrouter_gemini"
    return v


def get_backend(name: Optional[str]) -> VisionBackend:
    if not get_settings().enable_retrieval:
        return DummyBackend()
    v = normalize_backend(name)
    if v == "qwen3_vl":
        return Qwen3VLBackend()
    if v == "openrouter_gemini":
        return OpenRouterGeminiBackend()
    raise RuntimeError(f"unknown backend: {v}")
