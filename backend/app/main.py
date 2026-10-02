"""AtlasAI Morocco — application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.ai import registry
from app.api.routes import analysis, assistant, catalog, point, timeseries
from app.core.config import settings
from app.eo.catalog import NoDataError

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("atlasai")

DESCRIPTION = """
AI-powered Earth observation for Morocco.

Every response is computed from open satellite imagery at request time —
Sentinel-2, Sentinel-1, Landsat, MODIS, Copernicus DEM and ESA WorldCover, read
directly from cloud-optimized archives. Nothing is precomputed or mocked.

**On "real time":** it isn't, and no Earth observation platform is. Sentinel-2
revisits a given point every ~5 days in the best case, Sentinel-1 every ~6, and
cloud can extend either to weeks. This is *near-real-time* monitoring. Every
response carries the acquisition dates it used and the fraction of the area
actually observed, so you can judge how current an answer is.

**On confidence:** each analysis reports a confidence score driven by clear-sky
coverage, plus explicit caveats about what its method cannot distinguish. Read
them — several modules have failure modes that look like real signal.
"""

app = FastAPI(
    title="AtlasAI Morocco",
    description=DESCRIPTION,
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"
app.include_router(catalog.router, prefix=API_PREFIX)
app.include_router(analysis.router, prefix=API_PREFIX)
app.include_router(point.router, prefix=API_PREFIX)
app.include_router(timeseries.router, prefix=API_PREFIX)
app.include_router(assistant.router, prefix=API_PREFIX)


@app.exception_handler(NoDataError)
async def no_data_handler(_: Request, exc: NoDataError) -> JSONResponse:
    """An empty sky is a 404, not a server error.

    The distinction matters operationally: a 5xx means the platform is broken
    and should page someone; a 404 here means the satellites had nothing to
    offer for that area and window, which is normal and self-correcting.
    """
    return JSONResponse(
        status_code=404,
        content={"error": "no_data", "detail": str(exc), "hint": exc.hint},
    )


@app.get("/", tags=["meta"], summary="Service description and capabilities")
def root() -> dict:
    return {
        "name": "AtlasAI Morocco",
        "version": app.version,
        "description": "AI-powered Earth observation platform for Morocco",
        "docs": "/docs",
        "modules": len(registry.REGISTRY),
        "endpoints": {
            "modules": f"{API_PREFIX}/modules",
            "analysis": f"{API_PREFIX}/analysis/{{module_id}}",
            "point_report": f"{API_PREFIX}/point",
            "timeseries": f"{API_PREFIX}/timeseries",
            "assistant": f"{API_PREFIX}/assistant",
            "places": f"{API_PREFIX}/places",
            "health": f"{API_PREFIX}/health",
        },
        "data_note": (
            "Near-real-time, not live. Optical revisit is ~5 days at best and cloud can "
            "extend that to weeks. Every response lists the dates it actually used."
        ),
    }


@app.on_event("startup")
def _startup() -> None:
    logger.info(
        "AtlasAI ready — %d modules, STAC %s, database %s, LLM assistant %s",
        len(registry.REGISTRY),
        settings.stac_url,
        "on" if settings.has_database else "off",
        "on" if settings.has_llm else "off (rule-based router)",
    )
