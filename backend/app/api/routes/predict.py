"""Prediction API routes."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.config import UPLOADS_DIR
from backend.app.ml.predictor import get_history, get_stats, predict
from backend.app.models.schemas import PredictionResponse, StatsResponse

router = APIRouter(prefix="/api", tags=["predictions"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/predict", response_model=PredictionResponse)
async def predict_scan(
    file: UploadFile = File(...),
    scan_type: str = Form(default="chest_xray"),
    include_gradcam: bool = Form(default=True),
):
    """Analyze uploaded lung scan and return diagnosis with explainability."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 20 MB limit")
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="File appears empty or corrupt")

    # Persist upload for audit trail
    save_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
    save_path.write_bytes(contents)

    try:
        result = predict(contents, scan_type=scan_type, include_gradcam=include_gradcam)
        return PredictionResponse(**result.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@router.get("/stats", response_model=StatsResponse)
async def dashboard_stats():
    """Return platform usage statistics."""
    return StatsResponse(**get_stats())


@router.get("/history")
async def prediction_history(limit: int = 20):
    """Return recent predictions."""
    return {"history": get_history(limit=min(limit, 50))}
