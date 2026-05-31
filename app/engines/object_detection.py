"""Object-detection corroboration engine (YOLO via ultralytics).

Detects everyday objects in the photo — cars, buses, traffic lights, stop signs,
fire hydrants, benches, etc. On their own these don't locate anything, but they
are corroboration fodder for an analyst: a left-hand-drive bus, a North-American
fire hydrant, a European number plate shape all narrow the search. Each detected
object becomes a Clue the analyst can reason about.

Same contract as the other heavy engines: lazy-loaded on first use, cached, and
degrades to an honest engine note if ultralytics/torch isn't installed.

Model: ultralytics ships YOLO weights that auto-download on first run (a small
file, ~6 MB for yolov8n). Override with YOLO_MODEL, e.g. "yolov8s.pt".
"""

from __future__ import annotations

import os
import threading
from collections import Counter

from app.models import Clue, ClueType

_model = None
_load_error: str | None = None
_lock = threading.Lock()


def _enabled() -> bool:
    return os.getenv("ENABLE_OBJECTS", "true").lower() not in ("0", "false", "no")


def _model_name() -> str:
    # Bigger weights = more accurate detection. yolov8x is the largest/most
    # accurate; override with YOLO_MODEL=yolov8n.pt for speed on weak machines.
    return os.getenv("YOLO_MODEL", "yolov8x.pt")


def _load_model():
    global _model, _load_error
    if _model is not None or _load_error is not None:
        return
    with _lock:
        if _model is not None or _load_error is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore

            _model = YOLO(_model_name())
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"


def detect_objects(image_path: str, conf: float = 0.35) -> tuple[list[Clue], list[str]]:
    notes: list[str] = []

    if not _enabled():
        notes.append("Object detection disabled via ENABLE_OBJECTS=false.")
        return [], notes

    _load_model()
    if _model is None:
        notes.append(
            f"Object detection unavailable: {_load_error or 'model not loaded'}. "
            "Needs `pip install ultralytics`."
        )
        return [], notes

    try:
        # Load via PIL so HEIC (registered in app/__init__) works here too.
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        results = _model.predict(img, conf=conf, verbose=False)
    except Exception as exc:
        notes.append(f"Object detection failed: {type(exc).__name__}: {exc}")
        return [], notes

    # Tally detections by label, keep the best confidence seen for each.
    counts: Counter[str] = Counter()
    best_conf: dict[str, float] = {}
    for r in results:
        names = r.names  # class-id -> label
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for cls_id, c in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            label = names.get(int(cls_id), str(int(cls_id)))
            counts[label] += 1
            best_conf[label] = max(best_conf.get(label, 0.0), float(c))

    clues: list[Clue] = []
    for label, n in counts.most_common():
        value = f"{label} x{n}" if n > 1 else label
        clues.append(
            Clue(
                type=ClueType.OBJECT,
                value=value,
                detail="Object detected in image (region/era corroboration).",
                confidence=round(best_conf.get(label, 0.0), 3),
            )
        )
    return clues, notes
