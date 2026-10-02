"""Capability discovery — what the platform can analyse, where, and with what data."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, HTTPException, Query

from app.ai import backends, registry
from app.core.config import settings
from app.core.geo import MOROCCO_BBOX, BBox
from app.eo import indices
from app.eo.catalog import COLLECTIONS, NoDataError
from app.eo import catalog as eo_catalog
from app.services import cache, regions

router = APIRouter(tags=["catalog"])


@router.get("/modules", summary="Available analysis modules")
def modules() -> dict:
    # Each module reports which implementation would actually run, so a caller
    # is never left guessing whether a result came from a trained network or
    # from the spectral-physics baseline.
    entries = []
    for entry in registry.catalog():
        entry = dict(entry)
        entry["backend"] = backends.describe(entry["id"])
        entry["implementation"] = backends.implementation_note(entry["id"])
        entries.append(entry)

    return {
        "modules": entries,
        "point_report_modules": list(registry.POINT_REPORT_MODULES),
        "note": (
            "Modules run their analytic (spectral-index) implementation unless trained "
            "weights are installed. No module fabricates a neural prediction it cannot make."
        ),
    }


@router.get("/indices", summary="Available spectral indices")
def spectral_indices() -> dict:
    return {
        "indices": [
            {
                "key": index.key,
                "name": index.name,
                "description": index.description,
                "requires": list(index.requires),
                "vmin": index.vmin,
                "vmax": index.vmax,
                "colormap": index.colormap,
                "reference": index.reference,
            }
            for index in indices.REGISTRY.values()
        ]
    }


@router.get("/collections", summary="Satellite data sources in use")
def collections() -> dict:
    return {
        "collections": [
            {
                "id": collection.id,
                "stac_id": collection.stac_id,
                "title": collection.title,
                "resolution_m": collection.resolution_m,
                "revisit_days": collection.revisit_days,
                "bands": list(collection.bands),
                "has_cloud_cover": collection.has_cloud_cover,
            }
            for collection in COLLECTIONS.values()
        ],
        "note": (
            "Revisit intervals are best case under clear skies. This platform provides "
            "near-real-time monitoring, not live imagery: a given point is imaged every "
            "few days at best, and cloud can extend that to weeks."
        ),
    }


@router.get("/places", summary="Known regions, cities and reservoirs")
def places(
    kind: str | None = Query(None, pattern="^(region|city|reservoir)$"),
    search: str | None = Query(None, min_length=2),
) -> dict:
    if search:
        found = regions.search(search)
    else:
        found = list(regions.ALL_PLACES)
    if kind:
        found = [place for place in found if place.kind == kind]
    return {
        "places": [place.to_dict() for place in found],
        "morocco_bbox": list(MOROCCO_BBOX),
        "note": "Bounding boxes are approximate envelopes, not official boundaries.",
    }


@router.get("/scenes", summary="Imagery available for an area and window")
def scenes(
    bbox: str = Query(..., description="west,south,east,north"),
    sensor: str = Query("sentinel-2"),
    start: dt.date = Query(...),
    end: dt.date = Query(...),
    max_cloud: float = Query(100, ge=0, le=100),
    limit: int = Query(30, ge=1, le=100),
) -> dict:
    if sensor not in COLLECTIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown sensor '{sensor}'. Available: {', '.join(COLLECTIONS)}",
        )
    if end < start:
        raise HTTPException(status_code=422, detail="`end` must not precede `start`.")
    try:
        area = BBox.parse(bbox)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid bbox: {exc}") from exc

    try:
        items = eo_catalog.search(
            sensor, area, start, end,  # type: ignore[arg-type]
            max_cloud=max_cloud if COLLECTIONS[sensor].has_cloud_cover else None,
            limit=limit,
        )
    except NoDataError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "no_data", "detail": str(exc), "hint": exc.hint},
        ) from exc

    return {
        "sensor": sensor,
        "count": len(items),
        "scenes": eo_catalog.describe(items),
    }


@router.get("/health", summary="Service health and subsystem availability")
def health() -> dict:
    return {
        "status": "ok",
        "modules": len(registry.REGISTRY),
        "subsystems": {
            "stac": settings.stac_url,
            "planetary_computer_key": bool(settings.pc_subscription_key),
            "database": settings.has_database,
            "llm_assistant": settings.has_llm,
        },
        "cache": {"results": cache.RESULTS.stats(), "data": cache.DATA.stats()},
        "limits": {
            "max_pixels": settings.max_pixels,
            "default_resolution_m": settings.default_resolution,
        },
    }
