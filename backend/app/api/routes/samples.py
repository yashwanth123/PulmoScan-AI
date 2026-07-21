"""Sample image routes for demo and testing."""
from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.app.config import SAMPLE_IMAGES

router = APIRouter(prefix="/api", tags=["samples"])

TIMEOUT = 30.0


@router.get("/samples")
async def list_samples():
    """Return curated sample X-ray and CT images for testing."""
    return {"samples": SAMPLE_IMAGES}


@router.get("/samples/{scan_type}/{sample_id}")
async def proxy_sample(scan_type: str, sample_id: str):
    """Proxy-fetch a sample image (avoids CORS issues in browser)."""
    samples = SAMPLE_IMAGES.get(scan_type, [])
    match = next((s for s in samples if s["id"] == sample_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Sample not found")

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(match["url"])
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch sample: {exc}") from exc

    content_type = resp.headers.get("content-type", "image/jpeg")
    return Response(content=resp.content, media_type=content_type)
