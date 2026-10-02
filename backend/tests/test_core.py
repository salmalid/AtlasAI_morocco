"""Unit tests for the pure-computation layer.

These deliberately avoid the network: they test the maths and the routing,
which is where silent wrongness hides. The remote-data path is exercised by
``test_live.py``, which is opt-in.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from app.ai import assistant
from app.ai.base import class_areas, classify, confidence_from_coverage
from app.ai.terrain import distance_to, slope_degrees
from app.ai.urban import _cluster, _normalise_to
from app.core.geo import BBox
from app.eo import indices, render


# --- geometry ---------------------------------------------------------------


def test_bbox_rejects_degenerate():
    with pytest.raises(ValueError):
        BBox(-7.0, 32.0, -7.5, 32.5)  # west east of east
    with pytest.raises(ValueError):
        BBox(-7.5, 32.5, -7.0, 32.0)  # south north of north


def test_bbox_parse_roundtrip():
    box = BBox.parse("-7.75,32.45,-7.50,32.62")
    assert box.as_tuple() == (-7.75, 32.45, -7.50, 32.62)


def test_bbox_area_is_plausible():
    # ~25 km x ~19 km at 32 degrees north.
    box = BBox(-7.75, 32.45, -7.50, 32.62)
    assert 400 < box.area_km2 < 550


def test_fit_resolution_coarsens_to_respect_budget():
    box = BBox(-9.0, 30.0, -5.0, 34.0)  # very large
    coarse = box.fit_resolution(10, max_pixels=1_000_000)
    assert coarse > 10
    height, width = box.shape_at(coarse)
    assert height * width <= 1_100_000  # within rounding of the budget


def test_around_produces_square_ish_box():
    box = BBox.around(-7.6, 32.5, 1000)
    assert 1800 < box.width_m < 2200
    assert 1800 < box.height_m < 2200


# --- spectral indices -------------------------------------------------------


def test_ndvi_matches_hand_computation():
    bands = {"nir": np.array([[0.4]], dtype=np.float32), "red": np.array([[0.1]], dtype=np.float32)}
    assert indices.compute("ndvi", bands)[0, 0] == pytest.approx(0.6)


def test_index_denominator_guard_yields_nan_not_inf():
    bands = {"nir": np.array([[0.0]], dtype=np.float32), "red": np.array([[0.0]], dtype=np.float32)}
    assert np.isnan(indices.compute("ndvi", bands)[0, 0])


def test_savi_suppresses_soil_relative_to_ndvi():
    # Sparse vegetation over bright soil: SAVI should read lower than NDVI,
    # which is the entire reason the module prefers it in arid terrain.
    bands = {"nir": np.array([[0.30]], dtype=np.float32), "red": np.array([[0.22]], dtype=np.float32)}
    assert indices.compute("savi", bands)[0, 0] < indices.compute("ndvi", bands)[0, 0]


def test_water_indices_separate_water_from_vegetation():
    water = {
        "green": np.array([[0.09]], dtype=np.float32),
        "nir": np.array([[0.02]], dtype=np.float32),
        "swir16": np.array([[0.01]], dtype=np.float32),
    }
    vegetation = {
        "green": np.array([[0.06]], dtype=np.float32),
        "nir": np.array([[0.35]], dtype=np.float32),
        "swir16": np.array([[0.20]], dtype=np.float32),
    }
    assert indices.compute("mndwi", water)[0, 0] > 0.3
    assert indices.compute("mndwi", vegetation)[0, 0] < 0


def test_stats_ignores_nan_and_reports_coverage():
    array = np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32)
    stats = indices.stats(array)
    assert stats["count"] == 3
    assert stats["valid_fraction"] == pytest.approx(0.75)
    assert stats["mean"] == pytest.approx(3.0)


def test_stats_on_all_nan_is_json_safe():
    stats = indices.stats(np.full((4, 4), np.nan, dtype=np.float32))
    assert stats["count"] == 0
    assert stats["mean"] is None


def test_required_bands_deduplicates():
    needed = indices.required_bands(["ndvi", "savi", "ndmi"])
    assert needed.count("nir") == 1
    assert set(needed) == {"nir", "red", "swir16"}


# --- classification ---------------------------------------------------------


def test_classify_maps_nan_to_minus_one_not_zero():
    array = np.array([[0.05, 0.15, np.nan, 0.6]], dtype=np.float32)
    codes = classify(array, (0.1, 0.2, 0.5))
    assert codes.tolist() == [[0, 1, -1, 3]]


def test_class_areas_exclude_unobserved_from_shares():
    codes = np.array([[0, 0, 1, -1]], dtype=np.int16)
    palette = {0: ("A", "#000"), 1: ("B", "#fff")}

    class FakeGrid:
        pixel_area_km2 = 1.0

    areas = {c.id: c for c in class_areas(codes, palette, FakeGrid())}
    # Shares are over the 3 observed pixels, not all 4.
    assert areas[0].share == pytest.approx(2 / 3)
    assert areas[1].share == pytest.approx(1 / 3)


def test_confidence_punishes_low_coverage():
    full = confidence_from_coverage(np.ones((10, 10), dtype=np.float32))
    half = np.full((10, 10), np.nan, dtype=np.float32)
    half[:5] = 1.0
    assert full == pytest.approx(1.0)
    assert confidence_from_coverage(half) < 0.4


# --- terrain ----------------------------------------------------------------


def test_slope_of_flat_ground_is_zero():
    assert slope_degrees(np.full((5, 5), 100.0, dtype=np.float32), 30).max() == pytest.approx(0)


def test_slope_of_known_gradient():
    # 30 m rise over each 30 m pixel = 45 degrees.
    elevation = np.tile(np.arange(5, dtype=np.float32) * 30.0, (5, 1))
    assert slope_degrees(elevation, 30)[2, 2] == pytest.approx(45.0, abs=0.5)


def test_distance_to_grows_outward_and_caps():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    distance = distance_to(mask, max_steps=3)
    assert distance[3, 3] == 0
    assert distance[3, 4] == 1
    assert distance[3, 6] == 3
    assert np.isinf(distance[0, 0])  # beyond the cap


# --- change detection -------------------------------------------------------


def test_normalise_removes_a_uniform_brightness_offset():
    reference = np.random.default_rng(0).normal(0.5, 0.1, (60, 60)).astype(np.float32)
    shifted = reference + 0.2  # a pure radiometric offset, not real change
    valid = np.ones_like(reference, dtype=bool)

    corrected, params = _normalise_to(shifted, reference, valid)
    assert params is not None
    assert abs(float(np.median(corrected)) - float(np.median(reference))) < 0.01


def test_normalise_preserves_local_change():
    rng = np.random.default_rng(1)
    reference = rng.normal(0.5, 0.1, (60, 60)).astype(np.float32)
    changed = reference.copy()
    changed[:5, :5] = 0.95  # a small patch that genuinely changed
    corrected, _ = _normalise_to(changed, reference, np.ones_like(reference, dtype=bool))
    # The patch must survive normalisation — erasing it would defeat the point.
    assert corrected[:5, :5].mean() > corrected.mean() + 0.2


def test_cluster_finds_components_and_drops_specks():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True  # 9 px
    mask[7, 7] = True      # 1 px speck
    clusters = _cluster(mask, min_pixels=4)
    assert len(clusters) == 1
    assert clusters[0]["pixels"] == 9


# --- rendering --------------------------------------------------------------


def test_colorize_makes_nan_transparent():
    array = np.array([[0.5, np.nan]], dtype=np.float32)
    rgba = render.colorize(array, vmin=0, vmax=1, colormap="vegetation")
    assert rgba[0, 0, 3] == 255
    assert rgba[0, 1, 3] == 0


def test_png_encodes():
    rgba = render.colorize(np.zeros((8, 8), dtype=np.float32), vmin=0, vmax=1)
    png = render.to_png(rgba)
    assert png.startswith(b"\x89PNG")


def test_legend_spans_the_requested_range():
    entries = render.legend("drought", 0, 100, steps=5)
    assert entries[0]["value"] == 0
    assert entries[-1]["value"] == 100
    assert all(e["color"].startswith("#") for e in entries)


# --- assistant routing ------------------------------------------------------


TODAY = dt.date(2025, 6, 15)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Find reservoirs losing water", "water"),
        ("Show cities with increasing drought", "drought"),
        ("Where did vegetation decrease this year?", "vegetation"),
        ("Show new urban areas", "urban"),
        ("Which provinces have the highest flood risk?", "flood"),
        ("Is the water at Al Massira safe to swim in?", "water_quality"),
        ("How much of Agadir burned?", "wildfire"),
        ("What land cover is around Fes?", "landcover"),
    ],
)
def test_router_picks_the_right_module(question, expected):
    assert assistant.route(question, today=TODAY).module == expected


def test_router_refuses_out_of_scope_questions():
    with pytest.raises(assistant.UnanswerableError):
        assistant.route("What is the capital of France?", today=TODAY)


def test_router_prefers_city_over_its_own_region():
    # "Marrakech" is both a city and an alias of Marrakech-Safi; a question
    # about the city must not silently become a two-area scan.
    plan = assistant.route("Show new urban areas in Marrakech", today=TODAY)
    assert plan.mode == "single"
    assert [p.id for p in plan.places] == ["marrakech"]


def test_router_reads_two_years_as_a_comparison():
    plan = assistant.route("Compare Casablanca in 2020 vs 2024", today=TODAY)
    assert plan.baseline_start == dt.date(2020, 1, 1)
    assert plan.end.year == 2024


def test_change_questions_get_a_seasonal_baseline():
    plan = assistant.route("Where did vegetation decrease in Fes?", today=TODAY)
    assert plan.baseline_start is not None
    # Same season, one year back — not an arbitrary earlier window.
    assert plan.baseline_start.year == plan.start.year - 1


def test_scan_questions_expand_to_many_places():
    plan = assistant.route("Which reservoirs are losing water?", today=TODAY)
    assert plan.mode == "scan"
    assert len(plan.places) > 3
    assert all(p.kind == "reservoir" for p in plan.places)


def test_module_requiring_baseline_always_gets_one():
    plan = assistant.route("Show construction in Tangier", today=TODAY)
    assert plan.baseline_start is not None and plan.baseline_end is not None


# --- place resolution -------------------------------------------------------


@pytest.mark.parametrize(
    "term,expected",
    [
        ("marrakesh", "marrakech"),   # also an alias of Marrakech-Safi
        ("Marrakech", "marrakech"),
        ("casa", "casablanca"),       # also an alias of Casablanca-Settat
        ("fez", "fes"),               # also an alias of Fès-Meknès
        ("safi", "safi"),             # also an alias of Marrakech-Safi
        ("Souss-Massa", "souss-massa"),
        ("agadir", "agadir"),
        ("Béni Mellal", "beni-mellal"),
        ("beni mellal", "beni-mellal"),
    ],
)
def test_specific_place_wins_over_the_region_named_after_it(term, expected):
    from app.services import regions

    resolved = regions.resolve(term)
    assert resolved is not None, f"{term!r} did not resolve"
    assert resolved.id == expected


def test_region_names_still_resolve_to_regions():
    from app.services import regions

    for term in ("Marrakech-Safi", "Casablanca-Settat", "Fès-Meknès", "Drâa-Tafilalet"):
        resolved = regions.resolve(term)
        assert resolved is not None and resolved.kind == "region", term


def test_resolve_tolerates_a_typo():
    from app.services import regions

    assert regions.resolve("marakech") is not None


def test_resolve_rejects_nonsense():
    from app.services import regions

    assert regions.resolve("atlantis") is None
    assert regions.resolve("") is None
