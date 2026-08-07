from __future__ import annotations

import base64
import io
import math
from typing import Tuple

from PIL import Image, ImageOps

from vam.config import get_settings


def _decode_data_uri(data_uri: str) -> Tuple[str, bytes]:
    header, encoded = data_uri.split(",", 1)
    if ";base64" not in header.lower():
        raise ValueError("image data URI must use base64 encoding")
    return header, base64.b64decode(encoded, validate=True)


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def ensure_image_limit(
    image_url: str,
    *,
    max_size_mb: float | None = None,
    max_pixels: int | None = None,
) -> str:
    """Compress an inline image before upload so it fits the configured limits."""
    if not image_url.startswith("data:image/"):
        return image_url

    settings = get_settings()
    size_mb = settings.llm_max_image_size_mb if max_size_mb is None else max_size_mb
    pixel_limit = settings.llm_max_image_pixels if max_pixels is None else max_pixels
    max_bytes = int(float(size_mb) * 1024 * 1024)
    if max_bytes <= 0 or int(pixel_limit) <= 0:
        raise ValueError("image size and pixel limits must be positive")

    header, image_data = _decode_data_uri(image_url)
    with Image.open(io.BytesIO(image_data)) as source:
        image = ImageOps.exif_transpose(source)
        width, height = image.size

        if len(image_data) <= max_bytes and width * height <= int(pixel_limit):
            return image_url

        if width * height > int(pixel_limit):
            scale = math.sqrt(int(pixel_limit) / float(width * height))
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )

        if image.mode != "RGB":
            background = Image.new("RGB", image.size, "white")
            if "A" in image.getbands():
                background.paste(image, mask=image.getchannel("A"))
            else:
                background.paste(image.convert("RGB"))
            image = background

        while True:
            for quality in (90, 80, 70, 60, 50, 40, 30, 20):
                compressed = _encode_jpeg(image, quality)
                if len(compressed) <= max_bytes:
                    encoded = base64.b64encode(compressed).decode("ascii")
                    return f"data:image/jpeg;base64,{encoded}"

            width, height = image.size
            if width == 1 and height == 1:
                raise ValueError(
                    f"could not compress image below {max_bytes} bytes"
                )
            scale = min(0.85, math.sqrt(max_bytes / float(len(compressed))) * 0.95)
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
