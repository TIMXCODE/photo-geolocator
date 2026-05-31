"""EXIF evidence extractor.

In real casework metadata is preserved as evidence, not discarded — but per the
product design it must NOT drive the location answer. So we extract it, label
every clue with drives_answer=False, and keep it in a separate bucket on the
result. The candidate engine never sees this.
"""

from __future__ import annotations

from app.models import Clue, ClueType

try:
    from PIL import Image
    from PIL.ExifTags import GPSTAGS, TAGS
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False


def extract_exif_evidence(image_path: str) -> list[Clue]:
    if not _PIL_OK:
        return []

    try:
        img = Image.open(image_path)
        exif = img._getexif()  # type: ignore[attr-defined]
    except Exception:
        return []

    if not exif:
        return []

    clues: list[Clue] = []

    # Decode the raw EXIF tag map into readable names.
    decoded = {TAGS.get(tag, tag): val for tag, val in exif.items()}

    gps_block = decoded.get("GPSInfo")
    if gps_block:
        coords = _gps_to_decimal(gps_block)
        if coords:
            lat, lon = coords
            clues.append(
                Clue(
                    type=ClueType.EXIF_GPS,
                    value=f"{lat:.6f}, {lon:.6f}",
                    detail="GPS coordinates embedded in EXIF (evidence only).",
                    drives_answer=False,
                )
            )

    # A couple of useful non-GPS evidence fields for the case record.
    for field in ("Make", "Model", "DateTimeOriginal"):
        if field in decoded and decoded[field]:
            clues.append(
                Clue(
                    type=ClueType.EXIF_GPS,
                    value=str(decoded[field]),
                    detail=f"EXIF {field} (evidence only).",
                    drives_answer=False,
                )
            )

    return clues


def _gps_to_decimal(gps_block) -> tuple[float, float] | None:
    try:
        gps = {GPSTAGS.get(t, t): v for t, v in gps_block.items()}
        lat = _dms_to_dd(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        lon = _dms_to_dd(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        return lat, lon
    except Exception:
        return None


def _dms_to_dd(dms, ref) -> float:
    deg, minutes, sec = (float(x) for x in dms)
    dd = deg + minutes / 60.0 + sec / 3600.0
    if ref in ("S", "W"):
        dd = -dd
    return dd
