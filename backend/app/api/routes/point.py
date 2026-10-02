"""Point inspection — everything the platform knows about one clicked location.

Clicking the map should answer "what is here?" in one round trip. This runs the
cheap modules over a small window around the point and reports both the value
*at* the pixel and the context *around* it, because a single 20 m pixel is a
noisy thing to make a claim from.
"""

from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.ai import registry
from app.ai.base import AnalysisRequest
from app.api.routes.analysis import run_module
from app.core.geo import BBox, in_morocco
from app.eo.catalog import NoDataError
from app.services import regions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/point", tags=["point"])


@router.get("", summary="Full report for a single location")
def point_report(
    lon: float = Query(..., ge=-180, le=180),
    lat: float = Query(..., ge=-90, le=90),
    start: dt.date | None = Query(None, description="Defaults to the last 90 days"),
    end: dt.date | None = Query(None),
    radius_m: float = Query(1500, ge=200, le=20000, description="Context radius around the point"),
    modules: str | None = Query(
        None, description="Comma-separated module ids; defaults to the standard set"
    ),
) -> dict:
    if not in_morocco(lon, lat):
        raise HTTPException(
            status_code=422,
            detail="Point lies outside Morocco, which is the platform's calibrated area.",
        )

    end = end or dt.date.today()
    start = start or (end - dt.timedelta(days=90))
    bbox = BBox.around(lon, lat, radius_m)

    requested = (
        [m.strip() for m in modules.split(",") if m.strip()]
        if modules
        else list(registry.POINT_REPORT_MODULES)
    )
    unknown = [m for m in requested if m not in registry.REGISTRY]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown module(s): {', '.join(unknown)}")

    request = AnalysisRequest(
        bbox=bbox,
        start=start,
        end=end,
        # Coarser than the map default: this is a context read, and the point
        # should feel responsive.
        resolution_m=20,
        max_cloud=40,
        max_items=6,
    )

    def _run(module_id: str) -> tuple[str, dict]:
        try:
            key, result = run_module(module_id, request)
        except NoDataError as exc:
            return module_id, {"status": "no_data", "detail": str(exc), "hint": exc.hint}
        except Exception as exc:  # noqa: BLE001 - one failing module must not sink the report
            logger.exception("point report: %s failed", module_id)
            return module_id, {"status": "error", "detail": str(exc)}

        pixel = result.grid.index_of(lon, lat)
        values: dict[str, object] = {}
        for layer in result.layers:
            if layer.kind == "rgb":
                continue
            at_point = None
            if pixel is not None:
                raw = layer.array[pixel[0], pixel[1]]
                at_point = None if not np.isfinite(raw) else round(float(raw), 4)
            if layer.kind == "categorical" and at_point is not None:
                label = layer.palette.get(int(at_point), ("unclassified", "#000"))[0]
                values[layer.key] = {"value": int(at_point), "label": label}
            else:
                values[layer.key] = {"value": at_point, "unit": layer.unit}

        return module_id, {
            "status": "ok",
            "title": result.title,
            "summary": result.summary,
            "at_point": values,
            "area_metrics": result.metrics,
            "breakdown": [c.to_dict() for c in result.breakdown],
            "confidence": round(result.confidence, 3),
            "caveats": result.caveats,
            "observed_dates": result.observed_dates,
            "result_key": key,
        }

    # Modules are network-bound and independent; running them together turns a
    # ~30 s serial report into a ~10 s one.
    with ThreadPoolExecutor(max_workers=min(5, len(requested))) as pool:
        outputs = dict(pool.map(_run, requested))

    nearest = _nearest_place(lon, lat)
    succeeded = [m for m, r in outputs.items() if r.get("status") == "ok"]

    return {
        "location": {
            "lon": round(lon, 6),
            "lat": round(lat, 6),
            "radius_m": radius_m,
            "bbox": list(bbox.as_tuple()),
            "nearest_place": nearest,
        },
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "modules": outputs,
        "modules_succeeded": succeeded,
        "narrative": _narrative(outputs, nearest),
    }


def _nearest_place(lon: float, lat: float) -> dict | None:
    """Closest named place, for orientation in the UI."""
    best, best_distance = None, float("inf")
    for place in regions.ALL_PLACES:
        if place.kind == "region":
            continue
        distance = ((place.lon - lon) ** 2 + (place.lat - lat) ** 2) ** 0.5
        if distance < best_distance:
            best, best_distance = place, distance
    if best is None:
        return None
    payload = best.to_dict()
    # 1 degree ~ 111 km; good enough for a "near X" label.
    payload["distance_km"] = round(best_distance * 111.0, 1)
    return payload


def _narrative(outputs: dict[str, dict], nearest: dict | None) -> str:
    """One-paragraph plain-language summary of the point."""
    parts: list[str] = []
    if nearest:
        parts.append(f"About {nearest['distance_km']} km from {nearest['name']}.")

    landcover = outputs.get("landcover", {})
    if landcover.get("status") == "ok":
        label = (landcover.get("at_point") or {}).get("landcover", {})
        if isinstance(label, dict) and label.get("label"):
            parts.append(f"Land cover at this pixel is {str(label['label']).lower()}.")

    vegetation = outputs.get("vegetation", {})
    if vegetation.get("status") == "ok":
        ndvi = (vegetation.get("at_point") or {}).get("ndvi", {}).get("value")
        if ndvi is not None:
            descriptor = (
                "bare" if ndvi < 0.1 else "sparsely vegetated" if ndvi < 0.25
                else "moderately vegetated" if ndvi < 0.45 else "densely vegetated"
            )
            parts.append(f"It is {descriptor} (NDVI {ndvi}).")

    drought = outputs.get("drought", {})
    if drought.get("status") == "ok":
        severity = (drought.get("at_point") or {}).get("severity", {})
        if isinstance(severity, dict) and severity.get("label"):
            parts.append(f"Drought status: {str(severity['label']).lower()}.")

    terrain = outputs.get("terrain", {})
    if terrain.get("status") == "ok":
        elevation = (terrain.get("at_point") or {}).get("elevation", {}).get("value")
        slope = (terrain.get("at_point") or {}).get("slope", {}).get("value")
        if elevation is not None:
            parts.append(f"Elevation {elevation:.0f} m, slope {slope:.1f}°." if slope is not None
                         else f"Elevation {elevation:.0f} m.")

    water = outputs.get("water", {})
    if water.get("status") == "ok":
        area = (water.get("area_metrics") or {}).get("water_area_km2")
        if area:
            parts.append(f"{area} km² of surface water within the surrounding area.")

    if not parts:
        return "No modules returned usable data for this location and time window."
    return " ".join(parts)
