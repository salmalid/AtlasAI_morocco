"""Imagery module — the base satellite view behind every other layer.

Registered as a module rather than a special-cased endpoint so it inherits the
same caching, provenance and rendering path as an analysis result. The map's
"Satellite" toggle is therefore just another layer request.
"""

from __future__ import annotations

import numpy as np

from app.ai import sensing
from app.ai.base import (
    AnalysisRequest,
    AnalysisResult,
    Layer,
    confidence_from_coverage,
    coverage_caveats,
)


class ImageryModule:
    id = "imagery"
    title = "Satellite Imagery"
    question = "What does this place actually look like from orbit?"
    description = (
        "Cloud-free true-colour and false-colour composites from Sentinel-2, with "
        "acquisition dates. False colour puts near-infrared in the red channel, which "
        "makes vegetation glow red and water read black."
    )
    sensors = ("sentinel-2", "landsat")
    needs_baseline = False
    neural_backend = None

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        scene = sensing.load_with_fallback(request, ["blue", "green", "red", "nir", "swir16"])

        red, green, blue = scene.band("red"), scene.band("green"), scene.band("blue")
        nir, swir = scene.band("nir"), scene.band("swir16")

        layers = [
            Layer("truecolor", "True colour", "rgb", red, description="Natural colour, as the eye would see it.",
                  rgb=(red, green, blue), opacity=1.0),
            Layer("falsecolor", "False colour (NIR)", "rgb", nir,
                  description="NIR/red/green — vegetation is red, water is black, bare soil is cyan.",
                  rgb=(nir, red, green), opacity=1.0),
            Layer("agriculture", "Agriculture composite", "rgb", swir,
                  description="SWIR/NIR/blue — separates crop vigour and soil moisture.",
                  rgb=(swir, nir, blue), opacity=1.0),
        ]

        cloud_values = [
            item.properties.get("eo:cloud_cover")
            for item in scene.items
            if item.properties.get("eo:cloud_cover") is not None
        ]

        return AnalysisResult(
            module=self.id,
            title=self.title,
            summary=(
                f"Composite of {len(scene.items)} {scene.collection.title.split()[0]} scene(s) "
                f"between {scene.start.isoformat()} and {scene.end.isoformat()}, "
                f"{scene.coverage():.0%} of the area observed."
            ),
            grid=scene.grid,
            metrics={
                "scene_count": len(scene.items),
                "coverage_pct": round(scene.coverage() * 100, 1),
                "mean_scene_cloud_pct": (
                    round(float(np.mean(cloud_values)), 2) if cloud_values else None
                ),
                "resolution_m": round(scene.grid.resolution_m, 1),
                "sensor": scene.collection.title,
            },
            layers=layers,
            sources=scene.sources(),
            caveats=coverage_caveats(scene.coverage(), request)
            + [
                "This is a median composite across several dates, not a single photograph. "
                "Moving features — vehicles, herds, a river in spate — are averaged away.",
                "Contrast is a 2-98 percentile stretch computed per request, so brightness "
                "is not comparable between two different areas.",
            ],
            confidence=confidence_from_coverage(red),
            observed_dates=scene.dates(),
        )
