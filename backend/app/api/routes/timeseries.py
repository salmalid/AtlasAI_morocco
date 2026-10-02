"""Time-series endpoint — the data behind the timeline chart and the time slider.

Builds one composite per interval and reduces it to statistics. This is
genuinely expensive (one imagery search and read per step), so the endpoint
defaults to a coarse resolution and caps the number of steps: the shape of a
five-year NDVI curve does not need 10 m pixels.
"""

from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Query

from app.ai import sensing
from app.ai.base import AnalysisRequest
from app.core.geo import BBox
from app.eo import indices
from app.eo.catalog import NoDataError
from app.eo.raster import Grid
from app.services import cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/timeseries", tags=["timeseries"])

MAX_STEPS = 40


@router.get("", summary="Index statistics over time for an area")
def timeseries(
    bbox: str = Query(..., description="west,south,east,north"),
    index: str = Query("ndvi", description=f"One of: {', '.join(sorted(indices.REGISTRY))}"),
    start: dt.date = Query(...),
    end: dt.date = Query(...),
    interval: str = Query("month", pattern="^(month|quarter|year)$"),
    resolution: float = Query(100, ge=20, le=1000, description="Metres per pixel"),
    max_cloud: float = Query(40, ge=0, le=100),
) -> dict:
    if index not in indices.REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown index '{index}'. Available: {', '.join(sorted(indices.REGISTRY))}",
        )
    try:
        area = BBox.parse(bbox)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid bbox: {exc}") from exc
    if end < start:
        raise HTTPException(status_code=422, detail="`end` must not precede `start`.")

    windows = _windows(start, end, interval)
    if len(windows) > MAX_STEPS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{len(windows)} steps requested but the limit is {MAX_STEPS}. "
                "Use a coarser interval or a shorter range."
            ),
        )

    key = cache.token("timeseries", bbox, index, start, end, interval, resolution, max_cloud)
    cached = cache.DATA.get(key)
    if cached is not None:
        return cached

    definition = indices.REGISTRY[index]
    bands = list(definition.requires)
    # One grid for every step, so the series compares like with like.
    grid = Grid.from_bbox(area, resolution)

    def _step(window: tuple[dt.date, dt.date]) -> dict:
        window_start, window_end = window
        request = AnalysisRequest(
            bbox=area, start=window_start, end=window_end,
            resolution_m=resolution, max_cloud=max_cloud, max_items=6,
        )
        entry: dict = {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "label": _label(window_start, interval),
        }
        try:
            scene = sensing.load_with_fallback(request, bands, grid=grid)
        except (NoDataError, ValueError) as exc:
            # A gap is information: it means cloud or no overpass, not zero.
            entry.update({"status": "no_data", "detail": str(exc), "value": None})
            return entry

        values = scene.index(index)
        entry.update(
            {
                "status": "ok",
                "value": indices.stats(values)["mean"],
                "stats": indices.stats(values),
                "scene_count": len(scene.items),
                "dates": scene.dates(),
            }
        )
        return entry

    with ThreadPoolExecutor(max_workers=6) as pool:
        points = list(pool.map(_step, windows))

    observed = [p["value"] for p in points if p.get("value") is not None]
    payload = {
        "index": {
            "key": definition.key,
            "name": definition.name,
            "description": definition.description,
            "reference": definition.reference,
            "vmin": definition.vmin,
            "vmax": definition.vmax,
        },
        "bbox": list(area.as_tuple()),
        "interval": interval,
        "resolution_m": round(grid.resolution_m, 1),
        "points": points,
        "summary": {
            "steps": len(points),
            "steps_with_data": len(observed),
            "mean": round(sum(observed) / len(observed), 4) if observed else None,
            "min": min(observed) if observed else None,
            "max": max(observed) if observed else None,
            "trend": _trend([p["value"] for p in points]),
        },
        "caveats": [
            "Each point is a cloud-free composite over its interval, not a single date.",
            "Gaps are intervals where no scene met the cloud filter — they are missing "
            "observations, not zero values, and are excluded from the summary statistics.",
        ],
    }
    cache.DATA.set(key, payload)
    return payload


def _windows(start: dt.date, end: dt.date, interval: str) -> list[tuple[dt.date, dt.date]]:
    out: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= end:
        if interval == "month":
            nxt = _add_months(cursor, 1)
        elif interval == "quarter":
            nxt = _add_months(cursor, 3)
        else:
            nxt = _add_months(cursor, 12)
        out.append((cursor, min(nxt - dt.timedelta(days=1), end)))
        cursor = nxt
    return out


def _add_months(date: dt.date, months: int) -> dt.date:
    month_index = date.month - 1 + months
    year = date.year + month_index // 12
    month = month_index % 12 + 1
    # Clamp for short months so 31 January + 1 month lands on 28/29 February.
    import calendar

    day = min(date.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)


def _label(date: dt.date, interval: str) -> str:
    if interval == "year":
        return str(date.year)
    if interval == "quarter":
        return f"{date.year} Q{(date.month - 1) // 3 + 1}"
    return date.strftime("%Y-%m")


def _trend(values: list[float | None]) -> str | None:
    """Sign of a least-squares slope over the observed points."""
    observed = [(i, value) for i, value in enumerate(values) if value is not None]
    if len(observed) < 3:
        return None
    n = len(observed)
    mean_x = sum(i for i, _ in observed) / n
    mean_y = sum(v for _, v in observed) / n
    numerator = sum((i - mean_x) * (v - mean_y) for i, v in observed)
    denominator = sum((i - mean_x) ** 2 for i, _ in observed)
    if denominator == 0:
        return None
    slope = numerator / denominator
    if abs(slope) < 0.002:
        return "stable"
    return "increasing" if slope > 0 else "decreasing"
