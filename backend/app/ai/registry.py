"""Module registry — the single place that knows what the platform can answer.

The API, the assistant's intent router and the frontend layer panel all read
from here, so adding a capability means registering one class rather than
touching three layers.
"""

from __future__ import annotations

from app.ai.base import Module
from app.ai.hazard import FloodModule, WildfireModule
from app.ai.imagery import ImageryModule
from app.ai.infrastructure import RoadsModule
from app.ai.terrain import (
    LandCoverModule,
    SolarPotentialModule,
    TerrainModule,
    WindPotentialModule,
)
from app.ai.urban import ConstructionReviewModule, NightLightsModule, UrbanExpansionModule
from app.ai.vegetation import AgricultureModule, DroughtModule, VegetationModule
from app.ai.water import CoastalModule, WaterModule, WaterQualityModule

_MODULES: tuple[Module, ...] = (
    ImageryModule(),
    VegetationModule(),
    DroughtModule(),
    WaterModule(),
    WaterQualityModule(),
    FloodModule(),
    WildfireModule(),
    UrbanExpansionModule(),
    ConstructionReviewModule(),
    LandCoverModule(),
    AgricultureModule(),
    CoastalModule(),
    TerrainModule(),
    SolarPotentialModule(),
    WindPotentialModule(),
    RoadsModule(),
    NightLightsModule(),
)

REGISTRY: dict[str, Module] = {module.id: module for module in _MODULES}


def get(module_id: str) -> Module:
    if module_id not in REGISTRY:
        raise KeyError(
            f"unknown module '{module_id}'. Available: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[module_id]


def catalog() -> list[dict]:
    """Machine-readable capability list for the API and the assistant."""
    return [
        {
            "id": module.id,
            "available": module.id != "nightlights",
            "unavailable_reason": "No VIIRS nighttime-lights source configured." if module.id == "nightlights" else None,
            "title": module.title,
            "question": module.question,
            "description": module.description,
            "sensors": list(module.sensors),
            "needs_baseline": module.needs_baseline,
            "neural_backend": module.neural_backend,
        }
        for module in _MODULES
    ]


#: Modules cheap and reliable enough to run for a single clicked point.
POINT_REPORT_MODULES: tuple[str, ...] = (
    "landcover",
    "vegetation",
    "drought",
    "water",
    "terrain",
)
