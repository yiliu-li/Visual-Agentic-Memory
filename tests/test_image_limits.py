import base64
import io
import unittest

import numpy as np
from PIL import Image

from vam.vision.image_limits import ensure_image_limit


def _data_uri(image: Image.Image, image_format: str) -> str:
    output = io.BytesIO()
    image.save(output, format=image_format)
    payload = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{payload}"


class ImageLimitTests(unittest.TestCase):
    def test_compresses_under_byte_and_pixel_limits(self):
        pixels = np.random.default_rng(7).integers(
            0, 256, size=(1200, 1600, 3), dtype=np.uint8
        )
        source = _data_uri(Image.fromarray(pixels), "PNG")

        result = ensure_image_limit(source, max_size_mb=0.08, max_pixels=300_000)
        raw = base64.b64decode(result.split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            self.assertLessEqual(image.width * image.height, 300_000)
        self.assertLessEqual(len(raw), int(0.08 * 1024 * 1024))

    def test_leaves_compliant_image_unchanged(self):
        source = _data_uri(Image.new("RGB", (32, 32), "red"), "JPEG")
        self.assertEqual(
            ensure_image_limit(source, max_size_mb=1, max_pixels=10_000), source
        )

    def test_rejects_non_base64_data_uri(self):
        with self.assertRaises(ValueError):
            ensure_image_limit(
                "data:image/png,not-base64", max_size_mb=1, max_pixels=10_000
            )


if __name__ == "__main__":
    unittest.main()
