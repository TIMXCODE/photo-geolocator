"""Pipeline orchestrator.

Ties the engines into the two-stage flow:

  STAGE 1 — generate candidates (visual only)
      GeoCLIP looks at the pixels and proposes ranked probable locations.

  STAGE 2 — corroborate (visual only)
      OCR text + sun-position clues are extracted to help an analyst confirm
      and narrow those candidates.

  EVIDENCE (does not drive the answer)
      EXIF/metadata is logged in a separate bucket for the case record.

Nothing is ever marked CONFIRMED automatically — a human analyst commits the
final pin via confirm_pin(). That human-in-the-loop step is what makes the
output defensible in an investigation.
"""

from __future__ import annotations

import uuid

from app.engines import (
    exif_evidence,
    geoclip_engine,
    object_detection,
    ocr_clues,
    streetclip_engine,
    sun_position,
    vision_llm,
)
from app.models import (
    CaseStatus,
    ConfirmedPin,
    GeolocationResult,
)


def analyze_image(image_path: str, image_name: str) -> GeolocationResult:
    notes: list[str] = []

    # Stage 1: visual candidates
    candidates, n1 = geoclip_engine.predict_candidates(image_path, top_k=5)
    notes += n1

    # Stage 2: visual corroboration clues
    ocr, n2 = ocr_clues.extract_text_clues(image_path)
    sun, n3 = sun_position.analyze_sun_position(image_path)
    objs, n4 = object_detection.detect_objects(image_path)
    street, n5 = streetclip_engine.classify_region(image_path)
    vllm, n6 = vision_llm.analyze(image_path)
    notes += n2 + n3 + n4 + n5 + n6
    clues = vllm + ocr + objs + street + sun

    # Evidence only — logged, never fed into candidates above.
    exif = exif_evidence.extract_exif_evidence(image_path)

    return GeolocationResult(
        case_id=uuid.uuid4().hex[:12],
        image_name=image_name,
        candidates=candidates,
        clues=clues,
        exif_evidence=exif,
        engine_notes=notes,
        status=CaseStatus.PENDING_REVIEW,
    )


def confirm_pin(
    result: GeolocationResult,
    latitude: float,
    longitude: float,
    note: str | None = None,
    analyst: str | None = None,
) -> GeolocationResult:
    result.confirmed_pin = ConfirmedPin(
        latitude=latitude,
        longitude=longitude,
        note=note,
        confirmed_by=analyst,
    )
    result.status = CaseStatus.CONFIRMED
    return result


def mark_inconclusive(result: GeolocationResult) -> GeolocationResult:
    result.status = CaseStatus.INCONCLUSIVE
    return result
