"""Pluggable inference backends for the analysis modules."""

from app.ai.backends.torch_backend import (
    Checkpoint,
    describe,
    implementation_note,
    is_available,
    register,
    registry,
    segment,
    weights_path,
)

__all__ = [
    "Checkpoint",
    "describe",
    "implementation_note",
    "is_available",
    "register",
    "registry",
    "segment",
    "weights_path",
]
