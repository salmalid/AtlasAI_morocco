# AtlasAI Morocco

**AI-powered Earth observation platform for Morocco.** Ask where vegetation is
disappearing, which reservoirs are shrinking, or how a city has grown — and get
an answer computed from satellite imagery at the moment you ask, with the
acquisition dates and confidence attached.

---

## What it actually does

Seventeen analysis modules over Morocco, each answering one question:

| Module | Question | Instruments |
|---|---|---|
| `imagery` | What does this place look like from orbit? | Sentinel-2, Landsat |
| `vegetation` | How much healthy vegetation is there, gaining or losing? | Sentinel-2, Landsat |
| `drought` | Which areas are drier than they should be *for this season*? | Sentinel-2, Landsat |
| `water` | Where is surface water, and is it permanent or seasonal? | Sentinel-2, Landsat |
| `water_quality` | How turbid or algae-rich is this water? | Sentinel-2 |
| `flood` | What is under water now, and which ground is exposed? | Sentinel-1 radar, DEM |
| `wildfire` | What burned, and how badly? | Sentinel-2, MODIS |
| `urban` | Where has the city grown, and by how much? | Sentinel-2, Landsat |
| `construction_review` | Which new construction warrants a planning review? | Sentinel-2 |
| `landcover` | What is on the ground, and how much of each? | ESA WorldCover, Sentinel-2, DEM |
| `agriculture` | How are crops doing, and which fields are irrigated? | Sentinel-2, WorldCover |
| `coastal` | Is the shoreline advancing or retreating? | Sentinel-2, Landsat |
| `terrain` | Elevation, slope, aspect, height above drainage | Copernicus DEM |
| `solar` | Where should solar capacity be built? | DEM, WorldCover |
| `wind` | Which terrain suits wind development? | DEM, WorldCover |
| `roads` | What roads are here, and is anything unmapped? | OpenStreetMap, Sentinel-2 |
| `nightlights` | Which areas are lit at night? | *requires a VIIRS source — see below* |

Plus: click-any-point inspection across five modules at once, index time series,
and a natural-language assistant that routes questions to modules and answers
only from what they compute.


---

## Architecture

```
Question / map interaction
        │
        ▼
  Assistant  ──── routes to ────►  Module registry (17 modules)
  (rules, or Claude)                        │
                                            ▼
                                    Scene loader
                                            │
              ┌─────────────────────────────┼──────────────────────────┐
              ▼                             ▼                          ▼
        STAC search              Windowed COG reads            Cloud masking
     (Planetary Computer)      (only the pixels needed)      (SCL / QA_PIXEL)
                                            │
                                            ▼
                                  Median composite → physical units
                                            │
                                            ▼
                        Spectral indices (NDVI, MNDWI, AWEI, dNBR, …)
                                            │
                                            ▼
                        Module logic → metrics, class areas, caveats
                                            │
                            ┌───────────────┴───────────────┐
                            ▼                               ▼
                    JSON (numbers)                  PNG (map overlay)
```

**Backend** Python · FastAPI · rasterio/GDAL · NumPy · pystac-client
**Frontend** Next.js 15 · React 19 · TypeScript · MapLibre GL
**Data** PostgreSQL + PostGIS (optional) · Docker Compose

### Two engineering decisions that carry the platform

**Windowed COG reads with overview selection.** The platform never downloads a
scene. It defines an analysis grid, then pulls only the bytes intersecting it —
and picks the right pyramid level first. Rendering a 30 m grid from 10 m
Sentinel-2 bands without overview selection fetches ~9× more pixels than it can
use. Adding it took one analysis from **12 minutes to 12.9 seconds**
(`app/eo/raster.py`).

**Just-in-time SAS signing.** The obvious setup — signing every asset at STAC
search time — requests a token per asset, trips the anonymous rate limit, and
produces URLs that expire before you use them. Assets are signed immediately
before reading instead: one cached token per storage container
(`app/eo/catalog.py`).

---

## Data sources

All open, all free, none requiring credentials.
Details in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

| Product | Resolution | Revisit | Used for |
|---|---|---|---|
| Sentinel-2 L2A | 10 m | ~5 days | Vegetation, water, urban, fire, agriculture |
| Sentinel-1 RTC | 10 m | ~6 days | Floods, water through cloud and at night |
| Landsat 8/9 L2 | 30 m | ~8 days | Long-term history back to 2013 |
| MODIS | 500 m–1 km | 1–16 days | Active fire, burned area, vegetation |
| Copernicus DEM | 30 m | static | Terrain, floods, solar, wind |
| ESA WorldCover | 10 m | annual | Land-cover baseline |
| OpenStreetMap | vector | live | Roads, buildings, reference geography |


---


## Attribution

Contains modified Copernicus Sentinel data. Landsat and MODIS courtesy of
USGS/NASA. ESA WorldCover © ESA (CC BY 4.0). Copernicus DEM © ESA/Airbus.
Map data © OpenStreetMap contributors (ODbL), basemap © CARTO. Data access via
[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/).
