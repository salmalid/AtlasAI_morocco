"""Analysis endpoints — run a module, then stream its layers as PNG."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response

from app.ai import registry
from app.ai.base import AnalysisRequest, AnalysisResult
from app.api.schemas import parse_analysis
from app.core.config import settings
from app.eo import render
from app.eo.catalog import NoDataError
from app.services import cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["analysis"])


def run_module(module_id: str, request: AnalysisRequest) -> tuple[str, AnalysisResult]:
    """Execute a module, memoised on the exact request parameters."""
    key = cache.token(
        module_id,
        request.bbox.as_tuple(),
        request.start,
        request.end,
        request.baseline_start,
        request.baseline_end,
        request.resolution_m,
        request.max_cloud,
        request.max_items,
    )
    cached = cache.RESULTS.get(key)
    if cached is not None:
        return key, cached

    module = registry.get(module_id)
    result = module.run(request)
    cache.RESULTS.set(key, result)
    return key, result


@router.get(
    "/{module_id}",
    summary="Run an analysis module over an area and time window",
    response_description="Metrics, class areas, layer descriptors and provenance",
)
def analyse(
    module_id: str = Path(description="Module id, e.g. drought, water, urban"),
    request: AnalysisRequest = Depends(parse_analysis),
) -> dict:
    try:
        registry.get(module_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        key, result = run_module(module_id, request)
    except NoDataError as exc:
        # 404 rather than 500: the pipeline worked, the sky was simply empty.
        raise HTTPException(
            status_code=404,
            detail={"error": "no_data", "detail": str(exc), "hint": exc.hint},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result.to_dict(
        layer_url=f"/api/v1/analysis/result/{key}/layer/{{layer}}.png"
    )


@router.get(
    "/result/{key}/layer/{layer}.png",
    summary="Render a computed layer as a map overlay",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def layer_png(
    key: str = Path(description="Result key returned by an analysis call"),
    layer: str = Path(description="Layer key from the analysis response"),
    opacity: float = Query(default=None, ge=0, le=1, description="Override layer opacity"),
) -> Response:
    result: AnalysisResult | None = cache.RESULTS.get(key)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Result expired or unknown — re-run the analysis to regenerate it.",
        )

    target = result.layer(layer)
    if target is None:
        available = ", ".join(item.key for item in result.layers)
        raise HTTPException(status_code=404, detail=f"No layer '{layer}'. Available: {available}")

    alpha = target.opacity if opacity is None else opacity

    if target.kind == "categorical":
        palette = {cid: colour for cid, (_, colour) in target.palette.items()}
        rgba = render.colorize_classes(target.array, palette, opacity=alpha)
    elif target.kind == "mask":
        rgba = render.colorize_mask(target.array, opacity=alpha)
    elif target.kind == "rgb" and target.rgb is not None:
        rgba = render.rgb_composite(*target.rgb)
    else:
        rgba = render.colorize(
            target.array, vmin=target.vmin, vmax=target.vmax,
            colormap=target.colormap, opacity=alpha,
        )

    return Response(
        content=render.to_png(rgba),
        media_type="image/png",
        headers={"Cache-Control": f"public, max-age={settings.cache_ttl_seconds}"},
    )


@router.get("/result/{key}/legend/{layer}", summary="Legend entries for a layer")
def layer_legend(key: str, layer: str) -> dict:
    result: AnalysisResult | None = cache.RESULTS.get(key)
    if result is None:
        raise HTTPException(status_code=404, detail="Result expired or unknown.")
    target = result.layer(layer)
    if target is None:
        raise HTTPException(status_code=404, detail=f"No layer '{layer}'.")

    if target.kind == "categorical":
        return {
            "kind": "categorical",
            "entries": [
                {"label": label, "color": colour}
                for _, (label, colour) in sorted(target.palette.items())
            ],
        }
    return {
        "kind": "continuous",
        "unit": target.unit,
        "entries": render.legend(target.colormap, target.vmin, target.vmax),
    }
