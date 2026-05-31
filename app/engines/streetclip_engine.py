"""StreetCLIP region classifier (corroboration).

StreetCLIP (geolocal/StreetCLIP on Hugging Face) is a zero-shot
image-geolocalization model. It does NOT output coordinates — instead you give
it a list of candidate place names and it scores how well the photo matches each
one. We use it as a *corroboration* signal: rank a fixed list of world regions
and surface the top matches as clues for the analyst, independent of GeoCLIP.

LICENSE WARNING: StreetCLIP is CC-BY-NC-4.0 (non-commercial). Fine for research,
testing, and internal evaluation. Do NOT ship it in a product you sell without
clearing the license. This engine is OFF by default for that reason — enable it
explicitly with ENABLE_STREETCLIP=true.

Same contract as the other heavy engines: lazy-load, cache, degrade gracefully.
Runs on the `transformers` + `torch` you already have.
"""

from __future__ import annotations

import os
import threading

from app.models import Clue, ClueType

_model = None
_processor = None
_load_error: str | None = None
_lock = threading.Lock()

# A coarse world-region list to score against. Edit freely; more specific lists
# (e.g. US states, or cities within a known country) give sharper corroboration.
_DEFAULT_REGIONS = [
    "the United States", "Canada", "Mexico", "Brazil", "Argentina",
    "the United Kingdom", "France", "Germany", "Italy", "Spain", "Portugal",
    "Greece", "the Netherlands", "Scandinavia", "Eastern Europe", "Russia",
    "Turkey", "the Middle East", "North Africa", "Sub-Saharan Africa",
    "India", "China", "Japan", "South Korea", "Southeast Asia",
    "Australia", "New Zealand",
]


def _enabled() -> bool:
    # OFF by default because of the non-commercial license.
    return os.getenv("ENABLE_STREETCLIP", "false").lower() in ("1", "true", "yes")


def _regions() -> list[str]:
    raw = os.getenv("STREETCLIP_REGIONS", "")
    if raw.strip():
        return [r.strip() for r in raw.split(",") if r.strip()]
    return _DEFAULT_REGIONS


def _load_model():
    global _model, _processor, _load_error
    if _model is not None or _load_error is not None:
        return
    with _lock:
        if _model is not None or _load_error is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            _model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
            _processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"


def classify_region(image_path: str, top_k: int = 3) -> tuple[list[Clue], list[str]]:
    notes: list[str] = []

    if not _enabled():
        notes.append("StreetCLIP disabled (set ENABLE_STREETCLIP=true; note: non-commercial license).")
        return [], notes

    _load_model()
    if _model is None:
        notes.append(
            f"StreetCLIP unavailable: {_load_error or 'model not loaded'}. "
            "Needs transformers + torch (already in full mode)."
        )
        return [], notes

    try:
        import torch
        from PIL import Image

        regions = _regions()
        img = Image.open(image_path).convert("RGB")
        inputs = _processor(text=regions, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = _model(**inputs)
        # logits_per_image: [1, num_regions] -> softmax to probabilities
        probs = out.logits_per_image.softmax(dim=1)[0].tolist()
    except Exception as exc:
        notes.append(f"StreetCLIP inference failed: {type(exc).__name__}: {exc}")
        return [], notes

    ranked = sorted(zip(regions, probs), key=lambda x: x[1], reverse=True)[:top_k]
    clues: list[Clue] = []
    for region, p in ranked:
        clues.append(
            Clue(
                type=ClueType.LANDMARK,  # reuse a region-ish clue type
                value=f"StreetCLIP: {region}",
                detail="Independent region match (second geolocation model).",
                confidence=round(float(p), 3),
            )
        )
    return clues, notes
