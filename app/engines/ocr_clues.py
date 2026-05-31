"""OCR corroboration engine.

Pulls readable text out of the image — shop names, street signs, license-plate
formats, language/script — which an analyst uses to confirm and narrow the
GeoCLIP candidates. Uses easyocr, which (like GeoCLIP) pulls torch and weights,
so it is lazy-loaded with the same graceful-degradation contract.
"""

from __future__ import annotations

import os
import threading

from app.models import Clue, ClueType

def _enabled() -> bool:
    return os.getenv("ENABLE_OCR", "true").lower() not in ("0", "false", "no")


def _langs() -> list[str]:
    return [s.strip() for s in os.getenv("OCR_LANGS", "en").split(",") if s.strip()]


_reader = None
_load_error: str | None = None
_lock = threading.Lock()


def _load_reader():
    global _reader, _load_error
    if _reader is not None or _load_error is not None:
        return
    with _lock:
        if _reader is not None or _load_error is not None:
            return
        try:
            import easyocr  # type: ignore

            _reader = easyocr.Reader(_langs(), gpu=False)
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"


def extract_text_clues(image_path: str, max_clues: int = 12) -> tuple[list[Clue], list[str]]:
    notes: list[str] = []

    if not _enabled():
        notes.append("OCR disabled via ENABLE_OCR=false (lite mode).")
        return [], notes

    _load_reader()
    if _reader is None:
        notes.append(
            f"OCR unavailable: {_load_error or 'reader not loaded'}. "
            "Text corroboration needs easyocr + torch installed."
        )
        return [], notes

    try:
        # Route through PIL (which now has the HEIF opener registered) so easyocr
        # can handle iPhone .HEIC too, not just the formats its own loader knows.
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        # easyocr returns [(bbox, text, confidence), ...]
        results = _reader.readtext(np.array(img))
    except Exception as exc:
        notes.append(f"OCR failed: {type(exc).__name__}: {exc}")
        return [], notes

    clues: list[Clue] = []
    for _bbox, text, conf in sorted(results, key=lambda r: r[2], reverse=True)[:max_clues]:
        text = text.strip()
        if not text:
            continue
        clues.append(
            Clue(
                type=ClueType.OCR_TEXT,
                value=text,
                detail="Text detected in image (signage / labels / plates).",
                confidence=round(float(conf), 3),
            )
        )
    return clues, notes
