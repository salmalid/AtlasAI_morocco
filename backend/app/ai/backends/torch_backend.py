"""Optional neural backends for the segmentation modules.

Every analysis module in this platform ships a deterministic implementation
grounded in published spectral science, and that is what runs by default. This
file is the seam where a trained network can take over — nothing more.

**There are no weights in this repository, and no module silently uses one.**
A module declares the architecture that would supersede its analytic
implementation via ``Module.neural_backend``; if matching weights are present
in ``ATLASAI_MODEL_DIR`` they are used, and if not the analytic path runs and
the result says so. The platform never presents an untrained model's output as
a prediction.

Adding a backend
----------------
1. ``pip install -r requirements-ai.txt``
2. Put weights in ``models/`` following the naming in ``registry()`` below.
3. Register the checkpoint with :func:`register`, giving the bands it expects
   and the classes it emits.

See ``docs/MODELS.md`` for training data, the class schemes each module
expects, and why the analytic baseline is a genuine baseline rather than a
placeholder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Checkpoint:
    """A trained model and the contract it expects."""

    module_id: str
    architecture: str
    filename: str
    #: Logical band names, in the channel order the network was trained on.
    bands: tuple[str, ...]
    #: Output class ids, aligned with the module's palette.
    classes: tuple[int, ...]
    #: Tile size the network was trained at; inference runs tile-by-tile.
    tile: int = 512
    #: Overlap between tiles, blended to avoid seams at tile boundaries.
    overlap: int = 64
    normalisation: tuple[float, float] = (0.0, 1.0)


#: Checkpoints the platform knows how to load. Extend this rather than
#: hard-coding paths inside modules.
_REGISTRY: dict[str, Checkpoint] = {}


def register(checkpoint: Checkpoint) -> None:
    _REGISTRY[checkpoint.module_id] = checkpoint


def registry() -> dict[str, Checkpoint]:
    return dict(_REGISTRY)


register(
    Checkpoint(
        module_id="water",
        architecture="unet-resnet34",
        filename="water_unet_r34.pt",
        bands=("green", "red", "nir", "swir16"),
        classes=(0, 1, 2),
    )
)
register(
    Checkpoint(
        module_id="landcover",
        architecture="segformer-b2",
        filename="landcover_segformer_b2.pt",
        bands=("blue", "green", "red", "nir", "swir16", "swir22"),
        classes=tuple(range(10)),
    )
)
register(
    Checkpoint(
        module_id="urban",
        architecture="siamese-changeformer",
        filename="urban_change_siamese.pt",
        bands=("blue", "green", "red", "nir", "swir16"),
        classes=(0, 1, 2, 3),
    )
)


def weights_path(module_id: str) -> Path | None:
    """Filesystem path to a module's weights, or ``None`` if not present."""
    checkpoint = _REGISTRY.get(module_id)
    if checkpoint is None:
        return None
    directory = settings.model_dir
    if not directory.is_absolute():
        from app.core.config import REPO_ROOT

        directory = REPO_ROOT / directory
    path = directory / checkpoint.filename
    return path if path.exists() else None


def is_available(module_id: str) -> bool:
    """Whether a neural backend can actually run for this module.

    Requires both the weights and a working torch install. Returning False is
    the normal case, and the caller must fall back rather than degrade.
    """
    if weights_path(module_id) is None:
        return False
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.info(
            "Weights for '%s' are present but torch is not installed. "
            "Install requirements-ai.txt to enable the neural backend.",
            module_id,
        )
        return False
    return True


def _device():
    import torch

    if settings.torch_device != "auto":
        return torch.device(settings.torch_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load(module_id: str):
    """Load and cache a TorchScript module for inference."""
    import torch

    if module_id in _LOADED:
        return _LOADED[module_id]

    path = weights_path(module_id)
    if path is None:
        raise FileNotFoundError(f"No weights registered for '{module_id}'")

    model = torch.jit.load(str(path), map_location=_device())
    model.eval()
    _LOADED[module_id] = model
    logger.info("loaded %s backend from %s on %s", module_id, path.name, _device())
    return model


_LOADED: dict[str, object] = {}


def segment(module_id: str, bands: dict[str, np.ndarray]) -> np.ndarray:
    """Run tiled inference and return a class-id raster.

    Tiling with overlap is not an optimisation — a segmentation network applied
    to a whole scene at once sees a different receptive-field context than it
    was trained with, and stitching non-overlapping tiles leaves visible seams
    exactly where features cross a boundary.

    NaN pixels stay NaN: an unobserved pixel must never be assigned a class.
    """
    import torch

    checkpoint = _REGISTRY[module_id]
    model = _load(module_id)

    missing = [band for band in checkpoint.bands if band not in bands]
    if missing:
        raise ValueError(f"{module_id} backend needs band(s) {', '.join(missing)}")

    stack = np.stack([bands[band] for band in checkpoint.bands])  # (C, H, W)
    observed = np.all(np.isfinite(stack), axis=0)
    mean, scale = checkpoint.normalisation
    normalised = (np.nan_to_num(stack, nan=mean) - mean) / scale

    _, height, width = normalised.shape
    output = np.full((height, width), -1, dtype=np.int16)
    step = checkpoint.tile - checkpoint.overlap

    with torch.inference_mode():
        for top in range(0, max(height - checkpoint.overlap, 1), step):
            for left in range(0, max(width - checkpoint.overlap, 1), step):
                bottom = min(top + checkpoint.tile, height)
                right = min(left + checkpoint.tile, width)
                tile = normalised[:, top:bottom, left:right]

                tensor = torch.from_numpy(tile).float().unsqueeze(0).to(_device())
                logits = model(tensor)
                predicted = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int16)

                # Write only the tile's interior where it overlaps a neighbour,
                # so the seam falls inside a region each tile saw with context.
                inset = checkpoint.overlap // 2
                y0 = top + (inset if top > 0 else 0)
                x0 = left + (inset if left > 0 else 0)
                output[y0:bottom, x0:right] = predicted[y0 - top :, x0 - left :]

    output[~observed] = -1
    return output


def describe(module_id: str) -> dict:
    """Backend status for the API, so callers can see which path produced a result."""
    checkpoint = _REGISTRY.get(module_id)
    if checkpoint is None:
        return {"available": False, "reason": "no neural backend defined for this module"}

    path = weights_path(module_id)
    return {
        "available": is_available(module_id),
        "architecture": checkpoint.architecture,
        "expected_weights": checkpoint.filename,
        "weights_found": path is not None,
        "bands": list(checkpoint.bands),
        "reason": None if is_available(module_id) else (
            "weights not found in ATLASAI_MODEL_DIR"
            if path is None
            else "torch not installed (see requirements-ai.txt)"
        ),
    }


def implementation_note(module_id: str) -> str:
    """One line for a result's caveats, stating which code path actually ran."""
    if is_available(module_id):
        checkpoint = _REGISTRY[module_id]
        return f"Computed by the {checkpoint.architecture} neural backend."
    checkpoint = _REGISTRY.get(module_id)
    if checkpoint is None:
        return "Computed by the module's analytic implementation."
    return (
        f"Computed by the analytic (spectral-index) implementation. A trained "
        f"{checkpoint.architecture} would supersede it; no weights are installed."
    )
