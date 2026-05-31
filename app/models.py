"""Data models for the geolocation evidence trail.

The whole point of this structure is defensibility: every result carries a
chain of *candidate -> corroborating clues -> human-confirmed pin*, so an
analyst can show their work rather than presenting a black-box guess.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClueType(str, Enum):
    OCR_TEXT = "ocr_text"          # signage, license plates, business names
    LANDMARK = "landmark"          # recognised building / monument (stub hook)
    SUN_POSITION = "sun_position"  # shadow angle -> latitude band (stub hook)
    OBJECT = "object"              # detected items (cars, signs, poles) via YOLO
    EXIF_GPS = "exif_gps"          # metadata, logged as evidence ONLY


class CaseStatus(str, Enum):
    PENDING_REVIEW = "pending_review"  # engine has proposed, analyst hasn't confirmed
    CONFIRMED = "confirmed"            # analyst picked the final pin
    INCONCLUSIVE = "inconclusive"      # analyst reviewed, no confident answer


class Candidate(BaseModel):
    """A probable location proposed by the visual engine (e.g. GeoCLIP)."""
    rank: int
    latitude: float
    longitude: float
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "geoclip"


class Clue(BaseModel):
    """A corroborating signal pulled from the pixels to narrow / confirm."""
    type: ClueType
    value: str
    detail: Optional[str] = None
    confidence: Optional[float] = None
    # EXIF clues are flagged so the UI/analyst can see they are evidence-only
    # and were NOT used to produce the candidate list.
    drives_answer: bool = True


class ConfirmedPin(BaseModel):
    latitude: float
    longitude: float
    note: Optional[str] = None
    confirmed_by: Optional[str] = None
    confirmed_at: str = Field(default_factory=_now)


class GeolocationResult(BaseModel):
    case_id: str
    image_name: str
    created_at: str = Field(default_factory=_now)
    status: CaseStatus = CaseStatus.PENDING_REVIEW

    # Visual-only outputs — these are the actual "answer engine".
    candidates: list[Candidate] = Field(default_factory=list)
    clues: list[Clue] = Field(default_factory=list)

    # Logged for the record, never feeds the candidate list.
    exif_evidence: list[Clue] = Field(default_factory=list)

    # Filled once a human analyst commits to a location.
    confirmed_pin: Optional[ConfirmedPin] = None

    # Surface engine availability so the UI can be honest about lite vs full mode.
    engine_notes: list[str] = Field(default_factory=list)
