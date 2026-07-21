"""Health and model info routes."""
from __future__ import annotations

from fastapi import APIRouter

from backend.app.config import SCAN_TYPES
from backend.app.ml.metrics_loader import get_model_performance_summary
from backend.app.ml.predictor import get_model_status
from backend.app.models.schemas import (
    HealthResponse,
    ModelInfoResponse,
    ScanTypeInfo,
    ScanTypesResponse,
)

router = APIRouter(tags=["system"])


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()


@router.get("/api/model", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(**get_model_status())


@router.get("/api/model/metrics")
async def model_metrics():
    """Return training evaluation metrics and how to verify prediction accuracy."""
    return get_model_performance_summary()


@router.get("/api/scan-types", response_model=ScanTypesResponse)
async def scan_types():
    items = [
        ScanTypeInfo(key=k, **v)
        for k, v in SCAN_TYPES.items()
    ]
    return ScanTypesResponse(scan_types=items)
