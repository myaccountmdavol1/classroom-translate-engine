"""Wrap a validated image in a PDF for the existing page pipeline."""
from io import BytesIO
import math
import fitz
from PIL import Image, ImageOps

MAX_PIXELS = 25_000_000


def image_to_pdf(data: bytes) -> tuple[bytes, str]:
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in ("PNG", "JPEG"):
                raise ValueError("Use a PNG or JPEG page image.")
            mime = "image/png" if image.format == "PNG" else "image/jpeg"
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("That image is too large. Use an image under 25 megapixels.")
            image.load()
            dpi = image.info.get("dpi", (150, 150))
            density = float(dpi[0]) if isinstance(dpi, tuple) else 150.0
            if not math.isfinite(density) or not 72 <= density <= 600:
                density = 150.0
            rgba = ImageOps.exif_transpose(image).convert("RGBA")
            upright = Image.new("RGBA", rgba.size, "white")
            upright.alpha_composite(rgba)
            upright = upright.convert("RGB")
            payload = BytesIO()
            upright.save(payload, format="PNG")
            width, height = upright.size
        with fitz.open() as pdf:
            page = pdf.new_page(width=width * 72 / density, height=height * 72 / density)
            page.insert_image(page.rect, stream=payload.getvalue())
            return pdf.tobytes(deflate=True), mime
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("That page image could not be read. Try another PNG or JPEG.") from exc
