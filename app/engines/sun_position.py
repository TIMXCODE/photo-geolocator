"""Sun / shadow position corroboration — STUB HOOK.

Intended behaviour: detect shadow direction + length and, combined with an EXIF
or analyst-supplied timestamp, narrow the plausible latitude band (and azimuth).
This is a documented extension point, not yet implemented, so it returns nothing
but advertises itself in the engine notes.

A real implementation would likely:
  1. estimate shadow vectors from detected objects/people,
  2. solve for solar elevation/azimuth at the given timestamp,
  3. back out a latitude band consistent with that geometry.
"""

from __future__ import annotations

from app.models import Clue  # noqa: F401  (kept for the eventual real return type)


def analyze_sun_position(image_path: str) -> tuple[list[Clue], list[str]]:
    return [], ["Sun-position analysis not yet implemented (stub hook)."]
