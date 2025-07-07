"""Image preprocessing utilities for Llama-3.2-Vision package."""

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps

from ..utils import setup_logging

try:
    import requests
except ImportError:
    requests = None


def preprocess_image_for_llama(
    image_path: str,
    max_size: int = 1024,
    maintain_aspect_ratio: bool = True,
    log_level: str = "INFO",
) -> Image.Image:
    """Preprocess image for Llama-3.2-Vision compatibility.

    Args:
        image_path: Path to image file or HTTP URL
        max_size: Maximum dimension size
        maintain_aspect_ratio: Whether to maintain aspect ratio when resizing
        log_level: Logging level

    Returns:
        Preprocessed PIL Image
    """
    logger = setup_logging(log_level)

    # Load image
    if image_path.startswith("http"):
        if requests is None:
            raise ImportError("requests library not available for HTTP image loading")

        logger.info(f"Loading image from URL: {image_path}")
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
    else:
        logger.info(f"Loading image from file: {image_path}")
        image = Image.open(image_path)

    original_size = image.size
    logger.debug(f"Original image size: {original_size}")

    # Convert to RGB if needed
    if image.mode != "RGB":
        logger.debug(f"Converting image from {image.mode} to RGB")
        image = image.convert("RGB")

    # Fix image orientation based on EXIF data
    image = ImageOps.exif_transpose(image)

    # Resize if too large
    if max(image.size) > max_size:
        if maintain_aspect_ratio:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        else:
            image = image.resize((max_size, max_size), Image.Resampling.LANCZOS)

        logger.info(f"Image resized from {original_size} to {image.size}")

    return image


def validate_image(image_path: str) -> Tuple[bool, Optional[str]]:
    """Validate if image can be processed.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if file exists (for local files)
        if not image_path.startswith("http"):
            if not Path(image_path).exists():
                return False, f"Image file does not exist: {image_path}"

        # Try to open and validate image
        if image_path.startswith("http"):
            if requests is None:
                return False, "requests library not available for HTTP image loading"

            response = requests.head(image_path)
            if response.status_code != 200:
                return False, f"HTTP error {response.status_code} for URL: {image_path}"

            # Check content type
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return False, f"URL does not point to an image: {content_type}"

        # Try to open the image
        with Image.open(image_path) as img:
            # Verify image can be loaded
            img.verify()

        return True, None

    except Exception as e:
        return False, str(e)


def get_image_info(image_path: str) -> dict:
    """Get information about an image file.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with image information
    """
    try:
        with Image.open(image_path) as img:
            info = {
                "path": image_path,
                "size": img.size,
                "mode": img.mode,
                "format": img.format,
                "has_exif": hasattr(img, "_getexif") and img._getexif() is not None,
            }

            # Calculate file size for local files
            if not image_path.startswith("http"):
                file_size = Path(image_path).stat().st_size
                info["file_size_bytes"] = file_size
                info["file_size_mb"] = file_size / (1024 * 1024)

            return info

    except Exception as e:
        return {"path": image_path, "error": str(e), "valid": False}


def optimize_image_for_inference(
    image: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
    _quality: int = 95,
) -> Image.Image:
    """Optimize image for faster inference while maintaining quality.

    Args:
        image: PIL Image to optimize
        target_size: Target size tuple (width, height)
        quality: JPEG quality for compression (if applicable)

    Returns:
        Optimized PIL Image
    """
    optimized = image.copy()

    # Resize to target size if specified
    if target_size:
        optimized = optimized.resize(target_size, Image.Resampling.LANCZOS)

    # Apply mild sharpening for better OCR (optional)
    # This can help with text recognition in some cases
    try:
        from PIL import ImageFilter

        optimized = optimized.filter(
            ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3)
        )
    except ImportError:
        pass  # Skip sharpening if ImageFilter not available

    return optimized
