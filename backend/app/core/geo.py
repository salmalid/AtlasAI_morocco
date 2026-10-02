"""Geometry helpers shared by the EO pipeline.

Everything here works in EPSG:4326 (lon/lat degrees) at the API boundary and
converts to metres only where a resolution has to be reasoned about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Bounding box of Morocco including Western Sahara and the Atlantic margin.
#: (west, south, east, north) in EPSG:4326.
MOROCCO_BBOX: tuple[float, float, float, float] = (-17.5, 20.5, -0.8, 36.2)

EARTH_RADIUS_M = 6_378_137.0


class AOITooLargeError(ValueError):
    """Raised when a requested area would exceed the pixel budget."""


@dataclass(frozen=True)
class BBox:
    """An axis-aligned lon/lat bounding box."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if self.west >= self.east or self.south >= self.north:
            raise ValueError(
                f"degenerate bbox: west<east and south<north required, got {self.as_tuple()}"
            )
        if not (-180 <= self.west <= 180 and -180 <= self.east <= 180):
            raise ValueError("longitude out of range")
        if not (-90 <= self.south <= 90 and -90 <= self.north <= 90):
            raise ValueError("latitude out of range")

    @classmethod
    def parse(cls, raw: str) -> "BBox":
        """Parse a ``west,south,east,north`` string."""
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must be 'west,south,east,north'")
        try:
            west, south, east, north = (float(p) for p in parts)
        except ValueError as exc:  # noqa: TRY003
            raise ValueError("bbox components must be numbers") from exc
        return cls(west, south, east, north)

    @classmethod
    def around(cls, lon: float, lat: float, radius_m: float) -> "BBox":
        """Square box of half-width ``radius_m`` centred on a point."""
        dlat = math.degrees(radius_m / EARTH_RADIUS_M)
        # Guard the pole singularity so cos() never collapses to zero.
        dlon = math.degrees(radius_m / (EARTH_RADIUS_M * max(math.cos(math.radians(lat)), 1e-6)))
        return cls(
            max(lon - dlon, -180.0),
            max(lat - dlat, -90.0),
            min(lon + dlon, 180.0),
            min(lat + dlat, 90.0),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def as_geojson(self) -> dict:
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [self.west, self.south],
                    [self.east, self.south],
                    [self.east, self.north],
                    [self.west, self.north],
                    [self.west, self.south],
                ]
            ],
        }

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.west + self.east) / 2, (self.south + self.north) / 2)

    @property
    def width_m(self) -> float:
        lat = math.radians(self.centroid[1])
        return math.radians(self.east - self.west) * EARTH_RADIUS_M * math.cos(lat)

    @property
    def height_m(self) -> float:
        return math.radians(self.north - self.south) * EARTH_RADIUS_M

    @property
    def area_km2(self) -> float:
        return abs(self.width_m * self.height_m) / 1e6

    def shape_at(self, resolution_m: float) -> tuple[int, int]:
        """Raster (height, width) if sampled at ``resolution_m``."""
        return (
            max(int(round(self.height_m / resolution_m)), 1),
            max(int(round(self.width_m / resolution_m)), 1),
        )

    def fit_resolution(self, resolution_m: float, max_pixels: int) -> float:
        """Coarsen ``resolution_m`` just enough to respect the pixel budget.

        Returning a coarser resolution rather than raising keeps large-AOI
        requests useful: a national drought overview does not need 20 m pixels.
        """
        height, width = self.shape_at(resolution_m)
        pixels = height * width
        if pixels <= max_pixels:
            return resolution_m
        return resolution_m * math.sqrt(pixels / max_pixels)

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.east < other.west
            or other.east < self.west
            or self.north < other.south
            or other.north < self.south
        )


def in_morocco(lon: float, lat: float) -> bool:
    """Whether a point falls inside the platform's area of interest."""
    west, south, east, north = MOROCCO_BBOX
    return west <= lon <= east and south <= lat <= north
