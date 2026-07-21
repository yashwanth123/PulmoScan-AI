"""Pydantic schemas for API requests/responses."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "PulmoScan AI"
    version: str = "2.0.0"


class ModelInfoResponse(BaseModel):
    model_path: str
    model_exists: bool
    classes: list[str]
    architecture: str
    input_size: list[int]


class PredictionResponse(BaseModel):
    diagnosis: str
    confidence: float
    probabilities: dict[str, float]
    scan_type: str
    risk_level: str
    recommendation: str
    gradcam_image: str | None = None
    model_loaded: bool = True
    timestamp: str


class StatsResponse(BaseModel):
    total_scans: int
    by_class: dict[str, int]
    by_scan_type: dict[str, int]
    classes: list[str]
    scan_types: dict[str, Any]
    model_trained: bool


class ScanTypeInfo(BaseModel):
    key: str
    name: str
    icon: str
    description: str
    supported: bool


class ScanTypesResponse(BaseModel):
    scan_types: list[ScanTypeInfo]
