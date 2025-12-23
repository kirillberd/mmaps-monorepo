from __future__ import annotations

import io

import numpy as np
from PIL import Image


class ImagePreprocessingError(ValueError):
    """Raised when an uploaded file cannot be decoded as an image."""


class ImagePreprocessor:
    """Preprocesses image bytes into an NCHW float32 tensor.

    Equivalent to the reference torchvision pipeline:
    - Resize((image_size, image_size))
    - ToTensor() -> float32 [0, 1], CHW
    - Normalize(mean, std)

    Output shape: [1, 3, image_size, image_size]
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: list[float] | tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: list[float] | tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_size = int(image_size)
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean/std must have length 3 for RGB")
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Returns float32 tensor with shape [1, 3, H, W]."""

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.convert("RGB")
                # Pillow expects size as (W, H)
                resample = getattr(Image, "Resampling", Image).BILINEAR
                img = img.resize((self.image_size, self.image_size), resample=resample)
                arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        except Exception as e:  # noqa: BLE001
            raise ImagePreprocessingError("Invalid or unsupported image") from e

        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        # Normalize in CHW
        arr = (arr - self.mean) / self.std
        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)
        return np.ascontiguousarray(arr, dtype=np.float32)
