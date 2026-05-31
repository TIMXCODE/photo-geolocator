"""FastAPI service for the photo geolocation analyst tool.

Endpoints
---------
GET  /                     -> analyst web UI
GET  /healthz              -> liveness probe (used by Render)
POST /api/analyze          -> upload an image, get candidates + clues + evidence
GET  /api/case/{case_id}   -> fetch a stored case
POST /api/confirm/{case_id}-> analyst commits the final pin
POST /api/inconclusive/{case_id}

Storage is in-memory. On Render's free/standard tiers the filesystem and process
memory are ephemeral (wiped on redeploy/restart), so this is fine for a demo and
for short-lived analysis sessions. For durable case files, swap _CASES for a DB.
"""

from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import pipeline
from app.models import GeolocationResult

app = FastAPI(title="Photo Geolocator", version="0.1.0")

# In-memory case store: case_id -> GeolocationResult
_CASES: dict[str, GeolocationResult] = {}

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "15")) * 1024 * 1024
_ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/heic"}


class ConfirmRequest(BaseModel):
    latitude: float
    longitude: float
    note: str | None = None
    analyst: str | None = None


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=GeolocationResult)
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in _ALLOWED:
        raise HTTPException(415, f"Unsupported type {file.content_type!r}.")

    data = await file.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(413, f"File exceeds {_MAX_BYTES // (1024*1024)} MB limit.")

    suffix = os.path.splitext(file.filename or "upload")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = pipeline.analyze_image(tmp_path, file.filename or "upload")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    _CASES[result.case_id] = result
    return result


@app.get("/api/case/{case_id}", response_model=GeolocationResult)
def get_case(case_id: str):
    result = _CASES.get(case_id)
    if not result:
        raise HTTPException(404, "Case not found.")
    return result


@app.post("/api/confirm/{case_id}", response_model=GeolocationResult)
def confirm(case_id: str, body: ConfirmRequest):
    result = _CASES.get(case_id)
    if not result:
        raise HTTPException(404, "Case not found.")
    pipeline.confirm_pin(
        result, body.latitude, body.longitude, body.note, body.analyst
    )
    return result


@app.post("/api/inconclusive/{case_id}", response_model=GeolocationResult)
def inconclusive(case_id: str):
    result = _CASES.get(case_id)
    if not result:
        raise HTTPException(404, "Case not found.")
    pipeline.mark_inconclusive(result)
    return result


@app.get("/")
def index():
    path = os.path.join(_STATIC_DIR, "index.html")
    if not os.path.exists(path):
        return JSONResponse({"detail": "UI not found"}, status_code=404)
    return FileResponse(path)


# Serve any other static assets (none required today, but future-proof).
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
