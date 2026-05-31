"""Vision-LLM geolocation reasoning engine (Claude).

This is the "analyst brain": instead of pattern-matching like GeoCLIP/StreetCLIP,
it sends the photo to Claude's vision model and asks it to reason about WHERE the
photo was taken and WHY — reading subtle cues (architecture, vegetation, signage,
road furniture, license-plate style, climate) the way a human OSINT analyst would.

It returns its best region/location guess plus a short rationale, surfaced as a
corroboration clue. It does NOT place map pins — it explains, to help the analyst
weigh the GeoCLIP/StreetCLIP candidates.

SECURITY — API KEY HANDLING:
The Anthropic API key is read from the ANTHROPIC_API_KEY environment variable.
It is NEVER hardcoded in this file. You set it in your own shell, e.g.:

    export ANTHROPIC_API_KEY="sk-ant-...your-new-key..."

then start the server in that same shell. The key stays on your machine; it is
never committed to the repo or sent anywhere except Anthropic's API over HTTPS.

Enable with ENABLE_VISION_LLM=true. Pay-per-use (a few cents per photo).
Requires:  pip install anthropic
"""

from __future__ import annotations

import base64
import os

from app.models import Clue, ClueType

_MODEL = os.getenv("VISION_LLM_MODEL", "claude-opus-4-8")

_PROMPT = (
    "You are an OSINT geolocation analyst. Examine this photo and infer where it "
    "was most likely taken, reasoning ONLY from visible content: architecture, "
    "building materials, vegetation, terrain, signage and language, road markings, "
    "vehicles and license-plate styles, utility poles, climate and light. "
    "Do not guess wildly. Give your single best region/country (and city if the "
    "evidence supports it), a rough confidence 0-100, and 1-2 sentences naming the "
    "specific visual cues that drove your answer. Be concise. If there isn't enough "
    "information, say so rather than inventing a location."
)


def _enabled() -> bool:
    return os.getenv("ENABLE_VISION_LLM", "false").lower() in ("1", "true", "yes")


def _media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }.get(ext, "image/jpeg")


def analyze(image_path: str) -> tuple[list[Clue], list[str]]:
    notes: list[str] = []

    if not _enabled():
        notes.append("Vision-LLM disabled (set ENABLE_VISION_LLM=true).")
        return [], notes

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        notes.append(
            "Vision-LLM unavailable: ANTHROPIC_API_KEY not set. "
            'Run  export ANTHROPIC_API_KEY="sk-ant-..."  in the same shell, then restart.'
        )
        return [], notes

    try:
        import anthropic
    except Exception as exc:
        notes.append(f"Vision-LLM unavailable: {type(exc).__name__} ({exc}). Needs `pip install anthropic`.")
        return [], notes

    try:
        # Re-encode through PIL to JPEG so HEIC etc. are handled and size is sane.
        import io

        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": b64,
                    }},
                    {"type": "text", "text": _PROMPT},
                ],
            }],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
    except Exception as exc:
        notes.append(f"Vision-LLM call failed: {type(exc).__name__}: {exc}")
        return [], notes

    if not text:
        notes.append("Vision-LLM returned no text.")
        return [], notes

    return [Clue(
        type=ClueType.LANDMARK,
        value=f"Vision-LLM: {text}",
        detail="Analyst-style reasoning from image content (Claude).",
    )], notes
