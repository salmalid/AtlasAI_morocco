"""Offline regression coverage for configuration and the public API."""
import datetime as dt

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.ai.base import AnalysisResult, Layer
from app.ai import registry
from app.api.routes.timeseries import _windows, _trend
from app.core.geo import BBox
from app.eo.raster import Grid

client = TestClient(app)


def test_trend_preserves_missing_intervals():
    assert _trend([0.1, 0.103, 0.106]) == "increasing"
    assert _trend([0.1, None, None, 0.103, None, None, 0.106]) == "stable"


@pytest.mark.parametrize("raw", ['http://localhost:3000,http://127.0.0.1:3000', '["http://localhost:3000","http://127.0.0.1:3000"]'])
def test_cors_environment_formats(monkeypatch, raw):
    monkeypatch.setenv("ATLASAI_CORS_ORIGINS", raw)
    assert Settings(_env_file=None).cors_origins == ["http://localhost:3000", "http://127.0.0.1:3000"]


def test_example_dotenv_loads():
    from pathlib import Path
    settings = Settings(_env_file=Path(__file__).resolve().parents[2] / ".env.example")
    assert "http://localhost:3000" in settings.cors_origins


def test_yearly_windows_handle_leap_day():
    windows = _windows(dt.date(2024, 2, 29), dt.date(2025, 3, 1), "year")
    assert windows[0] == (dt.date(2024, 2, 29), dt.date(2025, 2, 27))
    assert windows[1][0] == windows[0][1] + dt.timedelta(days=1)
    assert windows[-1][1] == dt.date(2025, 3, 1)


def test_catalog_discloses_unavailable_module():
    response = client.get("/api/v1/modules")
    assert response.status_code == 200
    modules = {m["id"]: m for m in response.json()["modules"]}
    assert modules["water"]["available"]
    assert not modules["nightlights"]["available"]
    assert modules["nightlights"]["unavailable_reason"]


@pytest.mark.parametrize("path", ["analysis/water", "scenes", "timeseries"])
def test_invalid_windows_are_rejected_without_network(path):
    response = client.get(f"/api/v1/{path}", params={"bbox": "-8,31,-7.9,31.1", "start": "2024-08-01", "end": "2024-01-01"})
    assert response.status_code == 422


def test_analysis_json_and_png_contract(monkeypatch):
    grid = Grid.from_bbox(BBox(-8, 31, -7.999, 31.001), 100)
    def run(request):
        return AnalysisResult(module="vegetation", title="Test", summary="Contract fixture", grid=grid,
                              layers=[Layer("ndvi", "NDVI", "continuous", np.full(grid.shape, 0.5, dtype=np.float32))])
    monkeypatch.setattr(registry.get("vegetation"), "run", run)
    response = client.get("/api/v1/analysis/vegetation", params={"bbox": "-8,31,-7.999,31.001", "start": "2020-01-01", "end": "2020-02-01", "resolution": 100})
    assert response.status_code == 200
    png = client.get(response.json()["layers"][0]["png_url"])
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content.startswith(b"\x89PNG")
