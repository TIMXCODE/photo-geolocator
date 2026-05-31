"""Visual candidate engine.

Wraps GeoCLIP (https://github.com/VicenteVivan/geo-clip): a model trained on
~1.2M image/location pairs that does zero-shot image -> GPS retrieval. Given a
photo it returns the top-K most probable coordinates with confidence scores.

This is the *answer* engine. It looks only at pixels — terrain, architecture,
vegetation, road furniture, signage style — and never touches metadata.

Design notes for deployment:
- The model + PyTorch weights are heavy (hundreds of MB) and download from
  Hugging Face on first run. We therefore load it LAZILY on first request and
  cache the instance, so a small instance can still boot the web service.
- If torch / geoclip aren't installed, or weights can't be fetched, we degrade
  gracefully: no candidates, plus an engine note explaining why. The rest of
  the pipeline (EXIF evidence, OCR clues, analyst workflow) still works.
"""

from __future__ import annotations

import os
import threading

from app.models import Candidate

def _enabled() -> bool:
    return os.getenv("ENABLE_GEOCLIP", "true").lower() not in ("0", "false", "no")


_model = None
_load_error: str | None = None
_lock = threading.Lock()


def _load_model():
    """Import + instantiate GeoCLIP exactly once, behind a lock."""
    global _model, _load_error
    if _model is not None or _load_error is not None:
        return
    with _lock:
        if _model is not None or _load_error is not None:
            return
        try:
            from geoclip import GeoCLIP  # type: ignore

            _model = GeoCLIP()
        except Exception as exc:  # ImportError, weight download failure, OOM, ...
            _load_error = f"{type(exc).__name__}: {exc}"


def available() -> bool:
    return _enabled()


def predict_candidates(image_path: str, top_k: int = 5) -> tuple[list[Candidate], list[str]]:
    """Return ranked location candidates and any engine notes for the UI."""
    notes: list[str] = []

    if not _enabled():
        notes.append("GeoCLIP disabled via ENABLE_GEOCLIP=false (lite mode).")
        return [], notes

    _load_model()
    if _model is None:
        notes.append(
            "GeoCLIP unavailable: "
            f"{_load_error or 'model not loaded'}. "
            "Full visual geolocation needs a 2 GB+ instance with torch + weights."
        )
        return [], notes

    try:
        # GeoCLIP returns (gps_coords, confidence_scores), best first.
        gps, scores = _model.predict(image_path, top_k=top_k)
    except Exception as exc:
        notes.append(f"GeoCLIP inference failed: {type(exc).__name__}: {exc}")
        return [], notes

    candidates: list[Candidate] = []
    for i, ((lat, lon), score) in enumerate(zip(gps, scores)):
        candidates.append(
            Candidate(
                rank=i + 1,
                latitude=float(lat),
                longitude=float(lon),
                confidence=_normalise(float(score)),
            )
        )
    return candidates, notes


def _normalise(score: float) -> float:
    """GeoCLIP scores aren't bounded 0-1; clamp defensively for the schema."""
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
