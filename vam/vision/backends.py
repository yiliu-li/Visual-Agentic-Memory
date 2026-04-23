from __future__ import annotations

from typing import Callable, List, Protocol, Tuple


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


class OpenRouterGeminiBackend:
    name = "gemini"

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


def get_backend() -> VisionBackend:
    if not get_settings().enable_retrieval:
        return DummyBackend()
    return OpenRouterGeminiBackend()
