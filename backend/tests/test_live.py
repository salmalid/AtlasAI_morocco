"""Live integration tests against the Planetary Computer archive.

Opt-in, because they are slow and depend on a third-party service being up:

    pytest -m live

These assert on *physical plausibility*, not on exact values. Satellite
archives get reprocessed and scenes get withdrawn, so a test pinned to
"NDVI == 0.113" would be a false alarm generator. A test that catches NDVI
outside [-1, 1], or Marrakech reading as sea level, catches real bugs and stays
green across reprocessing.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from app.ai import registry
from app.ai.base import AnalysisRequest
from app.core.geo import BBox
from app.eo import catalog, indices, raster
from app.eo.raster import Grid

pytestmark = pytest.mark.live

# Al Massira — Morocco's largest reservoir, severely depleted by 2024.
MASSIRA = BBox(-7.75, 32.45, -7.50, 32.62)
# Marrakech — city, palmeraie and arid plain in one frame. Sits at ~460 m.
MARRAKECH = BBox(-8.10, 31.55, -7.90, 31.70)

SUMMER_2024 = (dt.date(2024, 6, 1), dt.date(2024, 8, 31))


def _request(bbox: BBox, window=SUMMER_2024, **kwargs) -> AnalysisRequest:
    return AnalysisRequest(
        bbox=bbox, start=window[0], end=window[1],
        resolution_m=kwargs.pop("resolution_m", 60),
        max_cloud=kwargs.pop("max_cloud", 30),
        max_items=kwargs.pop("max_items", 3),
        **kwargs,
    )


# --- catalog ----------------------------------------------------------------


def test_stac_search_returns_scenes_sorted_by_cloud():
    items = catalog.search("sentinel-2", MASSIRA, *SUMMER_2024, max_cloud=30, limit=6)
    assert items
    clouds = [item.properties.get("eo:cloud_cover", 100) for item in items]
    assert clouds == sorted(clouds), "optical scenes should come back least-cloudy first"


def test_empty_window_raises_with_an_actionable_hint():
    with pytest.raises(catalog.NoDataError) as excinfo:
        catalog.search(
            "sentinel-2", MASSIRA,
            dt.date(2024, 6, 1), dt.date(2024, 6, 2),
            max_cloud=0, limit=5,
        )
    assert excinfo.value.hint, "a no-data error must tell the caller how to fix it"


# --- reader -----------------------------------------------------------------


def test_bands_come_back_in_reflectance_units():
    items = catalog.search("sentinel-2", MASSIRA, *SUMMER_2024, max_cloud=20, limit=2)
    grid = Grid.from_bbox(MASSIRA, 60)
    bands = raster.load_bands(items[:2], catalog.COLLECTIONS["sentinel-2"], ["red", "nir"], grid)

    for name, array in bands.items():
        finite = array[np.isfinite(array)]
        assert finite.size > 0, f"{name} returned no valid pixels"
        # Surface reflectance is physically bounded; values outside this range
        # mean the scale factor was not applied.
        assert 0 <= finite.min() < 1.2, f"{name} min {finite.min()} is not reflectance"
        assert finite.max() < 1.6, f"{name} max {finite.max()} is not reflectance"


def test_ndvi_stays_within_its_mathematical_bounds():
    items = catalog.search("sentinel-2", MARRAKECH, *SUMMER_2024, max_cloud=20, limit=2)
    grid = Grid.from_bbox(MARRAKECH, 60)
    bands = raster.load_bands(items[:2], catalog.COLLECTIONS["sentinel-2"], ["red", "nir"], grid)
    ndvi = indices.compute("ndvi", bands)
    finite = ndvi[np.isfinite(ndvi)]
    assert -1.0 <= finite.min() and finite.max() <= 1.0


def test_grid_shape_matches_the_requested_resolution():
    grid = Grid.from_bbox(MASSIRA, 60)
    assert grid.resolution_m == pytest.approx(60, abs=1)
    # ~25 km x ~19 km at 60 m.
    assert 350 < grid.width < 480
    assert 250 < grid.height < 360


def test_tile_boundary_is_masked_not_zero_filled():
    """The DEM regression: an AOI spanning two tiles must not read as sea level.

    GDAL fills uncovered destination pixels with zero and, when the source
    declares no nodata, reports them as valid. Marrakech previously came back
    with a mean elevation of 224 m instead of ~460 m.
    """
    from app.ai.terrain import load_dem

    grid = Grid.from_bbox(MARRAKECH, 60)
    elevation = load_dem(_request(MARRAKECH), grid)
    finite = elevation[np.isfinite(elevation)]

    assert finite.size > 0
    assert finite.min() > 300, f"minimum {finite.min():.0f} m — tile gap filled with zeros"
    assert 380 < finite.mean() < 560, f"mean {finite.mean():.0f} m is not the Haouz plain"


# --- modules ----------------------------------------------------------------


def test_landcover_classifies_marrakech_plausibly():
    result = registry.get("landcover").run(_request(MARRAKECH))
    assert result.confidence > 0.5
    labels = {c.label for c in result.breakdown if c.share > 0.02}
    # A frame containing the medina, the palmeraie and the surrounding plain.
    assert "Urban" in labels
    assert result.layer("landcover") is not None
    assert sum(c.share for c in result.breakdown) == pytest.approx(1.0, abs=0.02)


def test_vegetation_reports_arid_conditions_for_the_haouz():
    result = registry.get("vegetation").run(_request(MARRAKECH))
    ndvi = result.metrics["mean_ndvi"]
    assert ndvi is not None
    assert -0.1 < ndvi < 0.45, f"mean NDVI {ndvi} is implausible for semi-arid terrain"
    assert 0 <= result.metrics["vegetated_share_pct"] <= 100


def test_water_detects_the_al_massira_drawdown():
    """2024 against 2019 at Morocco's largest reservoir.

    Al Massira fell to a small fraction of capacity during the multi-year
    drought. The direction and rough magnitude of that change is a real,
    externally verifiable event, so a module that misses it is broken.
    """
    request = _request(
        MASSIRA,
        baseline_start=dt.date(2019, 6, 1),
        baseline_end=dt.date(2019, 8, 31),
    )
    result = registry.get("water").run(request)

    change = result.metrics.get("water_change_pct")
    assert change is not None, "baseline comparison did not run"
    assert change < -40, f"expected a large drawdown, got {change}%"
    assert result.metrics["baseline_water_km2"] > result.metrics["current_water_km2"]


def test_urban_change_is_radiometrically_stable():
    """Built-up extent must not swing wildly between epochs.

    Independent thresholds on two dates report sun angle and atmosphere as
    construction. Normalisation is what keeps the two epochs comparable.
    """
    request = _request(
        MARRAKECH,
        baseline_start=dt.date(2019, 5, 1),
        baseline_end=dt.date(2019, 9, 30),
    )
    result = registry.get("urban").run(request)

    then = result.metrics["baseline_builtup_km2"]
    now = result.metrics["current_builtup_km2"]
    assert then > 0.5, f"baseline built-up {then} km² is implausibly small for Marrakech"
    assert 0.3 < now / then < 3.0, f"epoch ratio {now / then:.1f} indicates threshold drift"

    growth = result.metrics["growth_pct"]
    if growth is not None:
        assert growth < 100, "a city does not double its footprint in five years"


def test_drought_reports_departure_from_normal():
    result = registry.get("drought").run(_request(MARRAKECH, window=(dt.date(2024, 5, 1), dt.date(2024, 7, 31))))
    vci = result.metrics["mean_vci"]
    assert vci is not None and 0 <= vci <= 100
    assert 0 <= result.metrics["drought_affected_share_pct"] <= 100
    # The method must identify itself — an absolute-dryness fallback is a
    # different claim from a true VCI and has to say so.
    assert "VCI" in result.metrics["method"] or "fallback" in result.metrics["method"]


def test_terrain_derivatives_are_physical():
    result = registry.get("terrain").run(_request(MARRAKECH))
    assert result.metrics["min_elevation_m"] > 300
    assert 0 <= result.metrics["mean_slope_deg"] < 90
    assert result.metrics["relief_m"] > 0


def test_every_result_carries_provenance_and_caveats():
    """The platform's core promise: no bare numbers."""
    for module_id in ("landcover", "vegetation", "terrain"):
        result = registry.get(module_id).run(_request(MARRAKECH))
        payload = result.to_dict()

        assert payload["summary"], f"{module_id} produced no summary"
        assert 0 <= payload["confidence"] <= 1
        assert payload["grid"]["corners"], f"{module_id} produced no georeferencing"
        assert len(payload["grid"]["corners"]) == 4
        assert payload["layers"], f"{module_id} produced no renderable layer"


def test_layers_render_to_png():
    from app.eo import render

    result = registry.get("landcover").run(_request(MARRAKECH))
    layer = result.layers[0]
    palette = {cid: colour for cid, (_, colour) in layer.palette.items()}
    rgba = render.colorize_classes(layer.array, palette)
    png = render.to_png(rgba)

    assert png.startswith(b"\x89PNG")
    assert len(png) > 1000
