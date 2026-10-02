# Data Sources

Every collection here is open, free, and reachable without credentials through
the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
STAC API. Collection ids and asset keys were verified against the live API —
they differ per collection in ways that are easy to get wrong (Sentinel-2 names
its red band `B04`, Landsat names it `red`), so treat the tables below as the
reference rather than guessing.

Definitions live in `backend/app/eo/catalog.py`.

---

## Optical

### Sentinel-2 L2A — `sentinel-2-l2a`

The platform's primary instrument. Surface reflectance, atmospherically
corrected, with a scene classification layer for cloud masking.

| Property | Value |
|---|---|
| Resolution | 10 m (visible/NIR), 20 m (red-edge, SWIR), 60 m (atmospheric) |
| Revisit | ~5 days at the equator with both satellites, better at Moroccan latitudes |
| Coverage | 2015→, reliable L2A from 2017 |
| Scaling | uint16, multiply by 1e-4 for reflectance |

Assets used: `B02` blue, `B03` green, `B04` red, `B08` NIR, `B11` SWIR-1,
`B12` SWIR-2, `SCL` scene classification, `visual` true-colour.

**Cloud masking** uses SCL, rejecting classes 0 (no data), 1 (saturated),
3 (cloud shadow), 8/9 (cloud medium/high probability) and 10 (thin cirrus).
Class 11 (snow) is *kept* — in the High Atlas it is real signal, not an
artefact, and masking it would silently delete the snowpack from every
winter analysis.

### Landsat Collection 2 Level-2 — `landsat-c2-l2`

The historical record. Slower and coarser than Sentinel-2 but reaches back to
2013, which is the only way to answer a question about the pre-Sentinel era.

| Property | Value |
|---|---|
| Resolution | 30 m (60 m thermal, resampled) |
| Revisit | ~8 days with Landsat 8 and 9 combined |
| Coverage | 2013→ |
| Scaling | DN × 2.75e-5 − 0.2 |

Assets: `blue`, `green`, `red`, `nir08`, `swir16`, `swir22`, `lwir11` thermal,
`qa_pixel` quality mask.

**Cloud masking** reads the `QA_PIXEL` bitmask — bits 0 (fill), 1 (dilated
cloud), 2 (cirrus), 3 (cloud), 4 (cloud shadow).

The platform falls back from Sentinel-2 to Landsat automatically when a
requested window predates Sentinel-2 or returns nothing usable
(`sensing.load_with_fallback`).

---

## Radar

### Sentinel-1 RTC — `sentinel-1-rtc`

Radiometrically terrain-corrected backscatter. **The reason flood mapping works
at all**: a flood arrives with the storm that caused it, and optical sensors see
only cloud tops. Radar penetrates cloud and works at night.

| Property | Value |
|---|---|
| Resolution | 10 m |
| Revisit | ~6 days |
| Polarisations | VV, VH |
| Units | Linear power (gamma0), converted to dB in `hazard.to_db` |

Calm water is a specular reflector — it bounces radar away from the sensor —
so flooded ground appears dark. Detection requires **both** polarisations below
threshold, because radar shadow darkens VV without darkening VH the same way.

Known false positives: dry smooth surfaces — tarmac, salt flats, sabkha,
runways. Known false negatives: water under dense canopy or inside built-up
areas, where vegetation and buildings scatter the signal back regardless.

---

## Coarse-resolution / thermal

### MODIS active fire — `modis-14A1-061`

Daily thermal anomalies at 1 km. Used as an *independent confirmation* that a
burn scar detected optically was actually combustion, since dNBR alone responds
to harvest and ploughing too. `FireMask` values 7/8/9 are fire at low, nominal
and high confidence.

### MODIS burned area — `modis-64A1-061`

Monthly burned area at 500 m, with `Burn_Date` day-of-year.

### MODIS vegetation indices — `modis-13A1-061`

16-day NDVI/EVI composites at 500 m. Useful for national-scale trends where
Sentinel-2's revisit is unnecessary and its data volume is a liability.

---

## Elevation

### Copernicus DEM GLO-30 — `cop-dem-glo-30`

30 m global elevation, the input to flood susceptibility, solar and wind siting,
and the mountain class in land cover.

**It is a surface model**, not a terrain model: it includes tree canopy and
buildings, so forest and city elevations sit a few metres above true ground.

**Tiles are 1° × 1°.** Any area of interest crossing a tile boundary — routine —
spans multiple items. This was the source of a real bug: GDAL fills the
uncovered part of the grid with **zero** and, because the source declares no
nodata value, reports it as valid data. Half of Marrakech read as sea level and
the mean elevation came out at 224 m instead of 460 m. The reader now supplies
an explicit nodata sentinel so uncovered pixels become NaN
(`raster._nodata_for`).

---

## Land cover

### ESA WorldCover — `esa-worldcover`

Validated 10 m global land cover, ~75% overall accuracy. The platform uses it as
the *base* classification rather than training a worse classifier from the same
Sentinel-2 pixels.

| Code | Class | | Code | Class |
|---|---|---|---|---|
| 10 | Tree cover | | 70 | Snow and ice |
| 20 | Shrubland | | 80 | Permanent water |
| 30 | Grassland | | 90 | Herbaceous wetland |
| 40 | Cropland | | 95 | Mangroves |
| 50 | Built-up | | 100 | Moss and lichen |
| 60 | Bare / sparse vegetation | | | |

The genuine work in `landcover` is reconciling these with the classes this
platform reports: splitting WorldCover's single "bare" class into **desert**
versus **bare soil** using observed greenness, and deriving **mountain** and
**beach** from elevation, slope and distance to water. Those two are terrain
and position rather than surface type, and are applied only to sparsely
vegetated ground so they never overwrite forest or farmland.

### Esri 10 m annual land cover — `io-lulc-annual-v02`

An annual alternative with a different class scheme, registered in the catalog
for year-over-year land-cover change work.

---

## Vector

### OpenStreetMap (Overpass API)

Roads, buildings, amenities and waterways, queried live through
`backend/app/ai/osm.py` with automatic failover between mirrors.

OSM is a **reference, not ground truth**. Coverage in rural Morocco is uneven,
and an absent feature usually means nobody has mapped it yet rather than that it
does not exist. That asymmetry is exactly what makes OSM useful for change
detection and dangerous for enforcement — which is why `roads` reports the
network from OSM and labels its spectral candidates as candidates.

Licensed ODbL; attribution is required for any derived product.

---

## Not included: nighttime lights

**There is no VIIRS nighttime-lights collection in the Planetary Computer STAC
catalog.** The `nightlights` module therefore raises a clear "not configured"
error with wiring instructions instead of returning a fabricated map.

To enable it, register a source in `backend/app/ai/urban.py`:

- **NASA Black Marble (VNP46A2)** — daily, 500 m, via [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/).
  Requires a free Earthdata login. Best for change detection and outage mapping.
- **NOAA/Colorado School of Mines VNL** — annual cloud-free composites, 500 m,
  via [EOG](https://eogdata.mines.edu/products/vnl/). Best for long-term
  electrification trends.

Both need stray-light correction and lunar-cycle handling before radiance is
comparable across dates. That work is why the module is a stub rather than a
half-answer.

---

## Rate limits and access

Anonymous access is rate-limited, and the SAS token endpoint is the part that
bites first. Two mitigations are already in the code:

1. **Just-in-time signing** — assets are signed immediately before reading, not
   at search time, so one cached token per storage container covers an entire
   analysis instead of one token per asset.
2. **Backoff on 429** with a hint pointing at the free subscription key.

Set `ATLASAI_PC_SUBSCRIPTION_KEY` to raise the ceiling substantially. It is
free and requires no payment details — sign in at the [Planetary Computer API
management portal](https://planetarycomputer.developer.azure-api.net/) and
copy the key from your profile.

**A key is never required.** The datasets are anonymously accessible, and SAS
tokens are issued without one; the key only buys you less restrictive rate
limiting.

## Attribution requirements

- Sentinel-1/2: "Contains modified Copernicus Sentinel data [year]"
- Landsat, MODIS: courtesy of the U.S. Geological Survey / NASA
- ESA WorldCover: © ESA WorldCover project, CC BY 4.0
- Copernicus DEM: © ESA / Airbus Defence and Space
- OpenStreetMap: © OpenStreetMap contributors, ODbL
