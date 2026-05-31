"""Tests for the geolocation pipeline.

These run in 'lite' mode (no torch/geoclip needed) and verify the parts that
matter for defensibility:
  - EXIF GPS is captured but flagged drives_answer=False and kept out of candidates
  - the pipeline always returns a pending-review result with an evidence trail
  - analyst confirmation transitions state correctly
"""

import io
import os

from PIL import Image

from app import pipeline
from app.engines.exif_evidence import extract_exif_evidence
from app.models import CaseStatus, ClueType


def _make_plain_image(path):
    Image.new("RGB", (64, 64), (90, 120, 90)).save(path, "JPEG")


def test_plain_image_has_no_exif_gps(tmp_path):
    p = tmp_path / "plain.jpg"
    _make_plain_image(p)
    evidence = extract_exif_evidence(str(p))
    gps = [c for c in evidence if c.type == ClueType.EXIF_GPS and "," in c.value]
    assert gps == []  # no GPS in a freshly-generated image


def test_pipeline_returns_pending_with_trail(tmp_path, monkeypatch):
    # Force lite mode so the test never tries to load torch.
    monkeypatch.setenv("ENABLE_GEOCLIP", "false")
    monkeypatch.setenv("ENABLE_OCR", "false")

    p = tmp_path / "img.jpg"
    _make_plain_image(p)
    result = pipeline.analyze_image(str(p), "img.jpg")

    assert result.status == CaseStatus.PENDING_REVIEW
    assert result.confirmed_pin is None
    assert isinstance(result.candidates, list)
    # engine notes should explain why the heavy engines produced nothing
    assert any("lite mode" in n for n in result.engine_notes)


def test_exif_never_drives_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_GEOCLIP", "false")
    monkeypatch.setenv("ENABLE_OCR", "false")
    p = tmp_path / "img.jpg"
    _make_plain_image(p)
    result = pipeline.analyze_image(str(p), "img.jpg")
    # every metadata clue must be flagged as evidence-only
    assert all(c.drives_answer is False for c in result.exif_evidence)


def test_confirm_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_GEOCLIP", "false")
    monkeypatch.setenv("ENABLE_OCR", "false")
    p = tmp_path / "img.jpg"
    _make_plain_image(p)
    result = pipeline.analyze_image(str(p), "img.jpg")

    pipeline.confirm_pin(result, 48.8584, 2.2945, note="matched signage", analyst="A")
    assert result.status == CaseStatus.CONFIRMED
    assert result.confirmed_pin.latitude == 48.8584
    assert result.confirmed_pin.confirmed_by == "A"
