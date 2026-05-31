# Photo Geolocator

A visual geolocation tool for investigations. Upload a photo and the service
estimates **where it was taken from the image content itself** — terrain,
architecture, vegetation, signage — then lets a human analyst confirm the final
location. Built as a web service so the code is reviewable on GitHub and
deployable on Render.

> **What this is honestly capable of.** No tool can take an arbitrary photo and
> output the exact spot purely from pixels — that's an unsolved research
> problem. This tool does what real investigators (e.g. Bellingcat-style
> geolocation) actually rely on: a **two-stage** flow.
>
> 1. **Generate candidates** — [GeoCLIP](https://github.com/VicenteVivan/geo-clip)
>    proposes a ranked list of probable coordinates from the pixels (often the
>    right region/city, sometimes closer).
> 2. **Corroborate & confirm** — OCR pulls signage/labels, a sun-position hook
>    narrows latitude, and a **human analyst** commits the final pin against the
>    candidates.
>
> The output is an **evidence trail** (candidate → clues → confirmed pin), which
> is far more defensible in casework than a black-box one-click guess.

## Design principles

- **Visual-only answer.** The location estimate comes from pixels. Metadata
  never drives it.
- **Metadata logged as evidence, not used.** Real casework preserves EXIF/GPS as
  evidence. We extract and display it, flagged `drives_answer: false`, in a
  separate bucket — it never enters the candidate list.
- **Human in the loop.** Nothing is `confirmed` until an analyst commits a pin.
- **Honest degradation.** The heavy ML loads lazily; if it isn't available, the
  service still runs and tells you why (engine notes) instead of faking results.

## Architecture

```
app/
  main.py                 FastAPI service + analyst UI
  pipeline.py             orchestrator: candidates -> clues -> evidence -> review
  models.py               evidence-trail data models (pydantic)
  engines/
    geoclip_engine.py     visual candidate engine (GeoCLIP, lazy)   [the answer]
    ocr_clues.py          text corroboration (easyocr, lazy)        [corroborate]
    sun_position.py       shadow-angle latitude narrowing           [stub hook]
    exif_evidence.py      EXIF/GPS extractor                        [evidence only]
  static/index.html       dark analyst console (Leaflet map)
tests/test_pipeline.py    proves EXIF isolation + confirm flow
render.yaml               Render Blueprint
requirements.txt          lite deps (always deploys)
requirements-full.txt     + torch/geoclip/easyocr (full mode)
```

### Lite mode vs full mode

The visual engines (GeoCLIP, easyocr) pull PyTorch and download weights — too
heavy for a small box. So the service has two modes:

| Mode  | Deps                                    | RAM    | Behaviour |
|-------|-----------------------------------------|--------|-----------|
| Lite  | `requirements.txt`                      | ~512MB | UI, EXIF evidence logging, analyst workflow. Candidates/OCR return empty with an honest engine note. **Deploys on Render free tier.** |
| Full  | `+ requirements-full.txt`, env flags on | ~2GB+  | GeoCLIP candidates + OCR clues fully live. |

## Run locally

```bash
git clone <your-repo-url> && cd photo-geolocator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000
```

Full mode locally:

```bash
pip install -r requirements.txt -r requirements-full.txt
ENABLE_GEOCLIP=true ENABLE_OCR=true uvicorn app.main:app --reload
# first request downloads GeoCLIP weights from Hugging Face
```

Run the tests:

```bash
pip install pytest && pytest -q
```

## Put it on GitHub

```bash
cd photo-geolocator
git init
git add .
git commit -m "Photo geolocator: visual candidates + analyst confirmation"
git branch -M main
git remote add origin https://github.com/<you>/photo-geolocator.git
git push -u origin main
```

## Deploy on Render

**Lite mode (free, one click):** commit `render.yaml`, then in Render choose
**New + → Blueprint**, pick the repo, and apply. It builds, starts
`uvicorn app.main:app`, and health-checks `/healthz`. You get the full UI and
the analyst workflow immediately.

**Full mode (GeoCLIP live):** the free tier can't hold torch + weights. On a
plan with **≥ 2 GB RAM** (Render *Standard* or larger), change two things:

- `buildCommand:` → `pip install -r requirements.txt -r requirements-full.txt`
- env vars `ENABLE_GEOCLIP` and `ENABLE_OCR` → `true`

Also add a Render **persistent disk** mounted at `/root/.cache` (a few GB) so the
GeoCLIP/easyocr weights survive restarts instead of re-downloading every cold
start.

> Note: in-memory case storage is wiped on restart/redeploy (fine for live
> analysis sessions and demos). For durable case files, replace the `_CASES`
> dict in `app/main.py` with a database.

## API

| Method | Path                        | Purpose |
|--------|-----------------------------|---------|
| GET    | `/`                         | analyst UI |
| GET    | `/healthz`                  | liveness probe |
| POST   | `/api/analyze`              | multipart image upload → result JSON |
| GET    | `/api/case/{id}`            | fetch a stored case |
| POST   | `/api/confirm/{id}`         | `{latitude, longitude, note?, analyst?}` → confirmed |
| POST   | `/api/inconclusive/{id}`    | mark reviewed, no confident answer |

Interactive API docs are auto-served at `/docs`.

## Configuration (env vars)

| Var              | Default | Meaning |
|------------------|---------|---------|
| `ENABLE_GEOCLIP` | `true`  | run the visual candidate engine |
| `ENABLE_OCR`     | `true`  | run OCR corroboration |
| `OCR_LANGS`      | `en`    | comma-separated easyocr languages, e.g. `en,fr,ar` |
| `MAX_UPLOAD_MB`  | `15`    | upload size cap |

## Scope & ethics

This is built for legitimate enterprise investigation work — OSINT verification,
fraud investigation, journalism — where determining where imagery was captured is
the task, and where an auditable evidence trail matters. It is a decision-support
tool: it proposes and corroborates, a human confirms.
