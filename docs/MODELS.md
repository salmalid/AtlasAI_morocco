# Neural Backends

**No trained weights ship with this repository, and no module pretends
otherwise.** Every analysis module runs a deterministic implementation grounded
in published spectral science. Where a trained network would do better, the
module names the architecture in `Module.neural_backend` and the platform
provides a loading seam — but the analytic path is what actually executes, and
`GET /api/v1/modules` reports which implementation produced any result.

This is a deliberate choice. A model whose weights are random is not a model,
and shipping one behind an impressive architecture name would make the platform
a liar in exactly the situations where it is most trusted.

---

## The analytic baseline is a real baseline

The spectral implementations are not placeholders waiting to be replaced. They
are the operational methods used in the remote-sensing literature, and several
would be hard for a small trained model to beat:

| Module | Method | Source |
|---|---|---|
| `water` | MNDWI + AWEI + NDWI logistic ensemble | Xu 2006; Feyisa 2014; McFeeters 1996 |
| `drought` | Vegetation Condition Index vs multi-year seasonal baseline | Kogan 1990 (FAO/ASIS operational) |
| `vegetation` | SAVI + NDMI composite health score | Huete 1988; Gao 1996 |
| `wildfire` | dNBR with USGS/FIREMON severity classes | Key & Benson 2006 |
| `flood` | Dual-polarisation SAR thresholding + HAND proxy | Standard operational SAR flood mapping |
| `landcover` | ESA WorldCover + terrain/spectral refinement | ESA WorldCover v200 |

The honest gap is narrower than it looks, and sits in specific places:

- **Concrete vs bare rock in arid terrain.** The single biggest weakness. NDBI
  and BSI cannot reliably separate a new building from a quarry or a dry wadi
  at 10 m, and this is where a trained model earns its keep.
- **Crop type.** Not attempted analytically — it needs a temporal model over a
  full growing season.
- **Roads absent from OSM.** A linear-feature extractor is genuinely required;
  the current candidate layer is coarse by construction.
- **Water permanence from few observations.** A temporal model handles sparse,
  irregularly spaced observations better than counting wet fractions.

---

## Adding a backend

```bash
pip install -r backend/requirements-ai.txt
```

Place a **TorchScript** checkpoint in `ATLASAI_MODEL_DIR` (default `models/`)
using the filename in the registry, then verify:

```bash
curl localhost:8000/api/v1/modules | jq '.modules[] | {id, backend}'
```

Registered slots (`backend/app/ai/backends/torch_backend.py`):

| Module | Architecture | Filename | Channel order |
|---|---|---|---|
| `water` | U-Net (ResNet-34) | `water_unet_r34.pt` | green, red, nir, swir16 |
| `landcover` | SegFormer-B2 | `landcover_segformer_b2.pt` | blue, green, red, nir, swir16, swir22 |
| `urban` | Siamese change network | `urban_change_siamese.pt` | blue, green, red, nir, swir16 |

Register a new slot with `backends.register(Checkpoint(...))`.

### Contract

- **Input** — `(C, H, W)` float32, channels in the registered band order,
  surface reflectance 0–1, normalised by the checkpoint's `normalisation`.
- **Output** — `(1, num_classes, H, W)` logits. The loader takes `argmax` and
  aligns ids with the module's palette.
- **Unobserved pixels stay unobserved.** NaN in any input channel produces
  class `-1`, never a predicted class. A model must not invent land cover for a
  pixel hidden by cloud.
- **Inference is tiled with overlap** (default 512 px, 64 px overlap). This is
  not an optimisation: a network applied to a whole scene sees a different
  receptive-field context than it trained with, and non-overlapping tiles leave
  visible seams exactly where features cross a boundary. Only each tile's
  interior is written.

---

## Training data

Nothing here is Morocco-specific out of the box, which is the central problem
with importing a benchmark model into this platform.

| Task | Dataset | Caveat for Morocco |
|---|---|---|
| Water segmentation | [DeepGlobe](http://deepglobe.org/), [Sentinel-2 Water Edges](https://openmldata.ukho.gov.uk/) | Little arid/ephemeral water; sabkha and dry wadis are underrepresented |
| Land cover | [BigEarthNet](https://bigearth.net/), ESA WorldCover as weak labels | European-dominated; Saharan classes are thin |
| Change detection | [LEVIR-CD](https://justchenhao.github.io/LEVIR/), [OSCD](https://rcdaudt.github.io/oscd/) | Temperate cities; arid-terrain confusion is not represented |
| Roads | [SpaceNet](https://spacenet.ai/), [Massachusetts Roads](https://www.cs.toronto.edu/~vmnih/data/) | Unpaved tracks are rare in the labels and common in Morocco |
| Burn severity | [MTBS](https://www.mtbs.gov/) | North American forests, not Mediterranean scrub or cork oak |

**Fine-tune on Moroccan scenes before trusting any of these.** A model trained
on temperate imagery will systematically misread arid terrain — and it will do
so confidently, which is worse than the analytic method's known, stated
limitations.

The same caveat already applies without any model: the `wildfire` module uses
USGS dNBR thresholds calibrated for North American forests, and says so in its
output. Local calibration against field plots would shift those classes.

---

## Evaluating a replacement

Before a backend replaces an analytic path, it should beat it on Moroccan data,
not on a benchmark:

1. Hold out Moroccan scenes across seasons — the Rif in winter and the Drâa in
   summer are different problems.
2. Compare against the analytic implementation on the same pixels, not against
   a published benchmark number.
3. Report per-class recall. Overall accuracy is dominated by the majority class,
   and in Morocco that is bare ground, so a model can score well while being
   useless at exactly the classes anyone cares about.
4. Check the failure cases the analytic method already documents — quarries
   flagged as construction, sabkha flagged as flood, harvest flagged as burn.
   A model that has not fixed those has not improved anything.

If a trained model does not clear the analytic baseline on these, the right
call is to keep the baseline and say so.
