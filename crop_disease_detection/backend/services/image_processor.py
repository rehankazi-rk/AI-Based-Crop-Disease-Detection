import io
import os
import warnings
from typing import Tuple

from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MIN_WIDTH = 32
MIN_HEIGHT = 32
MAX_IMAGE_PIXELS = 25_000_000


def validate_image_file(file_name: str, file_content: bytes) -> Tuple[bool, str | None, tuple[int, int] | None]:
    """Validate filename, file size, readability, and image dimensions."""
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, "Unsupported file type. Use JPG, JPEG, or PNG.", None

    if len(file_content) == 0:
        return False, "Uploaded file is empty.", None

    if len(file_content) > MAX_IMAGE_BYTES:
        return False, "Image is too large. Maximum size is 5MB.", None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(file_content))
            image.verify()
            image = Image.open(io.BytesIO(file_content))
            width, height = image.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return False, "File could not be read as a valid image.", None

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return False, f"Image dimensions are too small. Minimum is {MIN_WIDTH}x{MIN_HEIGHT}px.", None
    if width * height > MAX_IMAGE_PIXELS:
        return False, "Image dimensions are too large to process safely.", None

    return True, None, (width, height)


def preprocess_image(file_content: bytes, target_size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Open a PIL image and return a normalized tensor-ready array shape."""
    image = Image.open(io.BytesIO(file_content)).convert("RGB")
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    return image
