"""Sample image routes — serves local files (no internet required)."""
from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from backend.app.config import BASE_DIR, SAMPLE_IMAGES

router = APIRouter(prefix="/api", tags=["samples"])

SAMPLES_DIR = BASE_DIR / "frontend" / "assets" / "samples"
TIMEOUT = 30.0


@router.get("/samples")
async def list_samples():
    """Return sample images with expected labels for accuracy testing."""
    return {"samples": SAMPLE_IMAGES}


@router.get("/samples/{scan_type}/{sample_id}")
async def get_sample(scan_type: str, sample_id: str):
    """Serve a sample image from local assets, falling back to remote URL."""
    samples = SAMPLE_IMAGES.get(scan_type, [])
    match = next((s for s in samples if s["id"] == sample_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Sample not found")

    local = SAMPLES_DIR / scan_type / f"{sample_id}.jpg"
    if local.is_file():
        return FileResponse(str(local), media_type="image/jpeg")

    # Remote fallback (may fail with 403 on some networks)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(
                match["url"],
                headers={"User-Agent": "PulmoScanAI/2.0 (research)"},
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Sample not on disk. Run: python3 scripts/setup_samples.py — ({exc})",
        ) from exc

    content_type = resp.headers.get("content-type", "image/jpeg")
    return Response(content=resp.content, media_type=content_type)
