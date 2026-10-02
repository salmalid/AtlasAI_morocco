"""Request parsing and shared response shapes."""

from __future__ import annotations

import datetime as dt

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from app.ai.base import AnalysisRequest
from app.core.geo import MOROCCO_BBOX, BBox


class AnalysisParams(BaseModel):
    """Query parameters shared by every analysis endpoint."""

    bbox: str = Field(
        description="Area of interest as west,south,east,north in EPSG:4326.",
        examples=["-7.75,32.45,-7.50,32.62"],
    )
    start: dt.date = Field(description="Start of the observation window.")
    end: dt.date = Field(description="End of the observation window.")
    baseline_start: dt.date | None = Field(
        default=None, description="Start of the comparison window, for change detection."
    )
    baseline_end: dt.date | None = Field(
        default=None, description="End of the comparison window."
    )
    resolution: float | None = Field(
        default=None, ge=10, le=1000,
        description="Target ground resolution in metres. Coarser is faster.",
    )
    max_cloud: float = Field(
        default=30.0, ge=0, le=100,
        description="Reject optical scenes cloudier than this percentage.",
    )
    max_items: int = Field(
        default=8, ge=1, le=40,
        description="Maximum scenes to composite. More is cleaner but slower.",
    )


def parse_analysis(
    bbox: str = Query(..., description="west,south,east,north in EPSG:4326"),
    start: dt.date = Query(..., description="Start of the observation window"),
    end: dt.date = Query(..., description="End of the observation window"),
    baseline_start: dt.date | None = Query(None, description="Comparison window start"),
    baseline_end: dt.date | None = Query(None, description="Comparison window end"),
    resolution: float | None = Query(None, ge=10, le=1000, description="Metres per pixel"),
    max_cloud: float = Query(30.0, ge=0, le=100, description="Max scene cloud cover %"),
    # Each extra scene is roughly one more windowed read per band, so this is
    # the dominant cost knob. Four is enough for a clean median composite;
    # eight doubled request time for a marginal gain in cloud rejection.
    max_items: int = Query(4, ge=1, le=40, description="Max scenes to composite"),
) -> AnalysisRequest:
    """FastAPI dependency turning query parameters into an `AnalysisRequest`."""
    try:
        area = BBox.parse(bbox)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid bbox: {exc}") from exc

    if end < start:
        raise HTTPException(status_code=422, detail="`end` must not precede `start`.")
    if (baseline_start is None) != (baseline_end is None):
        raise HTTPException(
            status_code=422,
            detail="Provide both `baseline_start` and `baseline_end`, or neither.",
        )
    if baseline_start and baseline_end and baseline_end < baseline_start:
        raise HTTPException(
            status_code=422, detail="`baseline_end` must not precede `baseline_start`."
        )

    # A soft guard, not a hard one: analysing a neighbouring country is a
    # reasonable thing to do, the platform is simply not tuned for it.
    morocco = BBox(*MOROCCO_BBOX)
    if not area.intersects(morocco):
        raise HTTPException(
            status_code=422,
            detail=(
                "This area lies entirely outside Morocco. The platform's reference data "
                "and calibration are Morocco-specific."
            ),
        )

    if area.area_km2 > 250_000:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Area of {area.area_km2:,.0f} km² is too large for a single analysis. "
                "Request a region or city, or raise `resolution` to coarsen the grid."
            ),
        )

    return AnalysisRequest(
        bbox=area,
        start=start,
        end=end,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        resolution_m=resolution,
        max_cloud=max_cloud,
        max_items=max_items,
    )


class ErrorResponse(BaseModel):
    error: str
    detail: str
    hint: str | None = None


class AssistantQuery(BaseModel):
    question: str = Field(
        min_length=3, max_length=500,
        examples=["Which reservoirs are losing water?", "Show new urban areas in Marrakech"],
    )
    bbox: str | None = Field(
        default=None, description="Optional area override; otherwise inferred from the question."
    )
    start: dt.date | None = None
    end: dt.date | None = None
