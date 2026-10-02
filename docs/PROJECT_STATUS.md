# Project status

The repository implements an on-demand analytical Earth observation MVP. It
does not implement every research and operational capability in `description.md`.
That file is a vision, not a release acceptance report.

## Implemented

- Sixteen analytical modules plus explicit availability metadata for the
  unconfigured nighttime-lights module.
- Map overlays, legends, date controls, point inspection using the selected
  observation window, and a natural-language analysis router.
- Observation history with selectable spectral index and monthly, quarterly,
  or yearly intervals. Missing observations remain gaps in the chart and trend.
- JSON report downloads containing metrics, source metadata, and caveats.
- Responsive dashboard, catalog connection retry, and request validation.
- Offline API and scientific regression tests; live archive tests are opt-in.
- CI checks for backend tests, frontend lint, and production compilation.
- Standalone frontend Docker image and build-context exclusions.

## Still required for the full vision

- A licensed/configured VIIRS source and an implemented nighttime radiance loader.
- Validated trained weights, training datasets, and evaluation for neural
  segmentation, crop types, road extraction, and yield prediction.
- Wind and weather observations for resource estimates and weather layers.
- Official administrative geometry for authoritative area statistics.
- Scheduled ingestion, persistent analysis jobs, monitoring/alerts, and shared
  caching for a multi-worker national monitoring deployment.

These require data acquisition, scientific validation, or operational design;
they cannot be made complete by renaming the existing analytical approximations.
The dashboard disables the unconfigured nighttime-lights module.

## Local verification

```powershell
# Windows, from the repository root; use an installed Python >= 3.11.
py -3.11 -m venv backend/.venv-check
backend/.venv-check/Scripts/python.exe -m pip install -r backend/requirements.txt
cd backend
.venv-check/Scripts/python.exe -m pytest -q
# Explicit network-dependent checks:
.venv-check/Scripts/python.exe -m pytest -m live -q
cd ../frontend
npm ci
npm run lint
npm run build
```

For local startup, run `backend/.venv-check/Scripts/python.exe -m uvicorn
app.main:app --app-dir backend --port 8000` from the root and `npm run dev` in
`frontend`. Docker requires a running Docker engine. Do not reuse a virtual
environment created against a Python installation that has since been removed.
