"""Photo Geolocator package init.

Register the HEIF/HEIC opener with Pillow as early as possible, so every engine
that opens an image (GeoCLIP, EXIF evidence, OCR) can read iPhone .HEIC files.
If pillow-heif isn't installed, we degrade quietly — JPEG/PNG still work, and
HEIC uploads will surface a readable error instead of crashing import.
"""

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass
