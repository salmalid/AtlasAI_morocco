"""Morocco reference geography.

These are **approximate envelopes and centroids**, not official cadastral
boundaries. They exist so the API can resolve a place name to an area of
interest ("drought in Souss-Massa") without shipping a boundary file. For
publication-grade statistics, load official administrative polygons into
PostGIS (see ``infra/postgres/init.sql``) and query those instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

from app.core.geo import BBox


@dataclass(frozen=True)
class Place:
    id: str
    name: str
    kind: str  # region | city | reservoir | basin
    lon: float
    lat: float
    bbox: tuple[float, float, float, float]
    aliases: tuple[str, ...] = field(default=())

    def as_bbox(self) -> BBox:
        return BBox(*self.bbox)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "lon": self.lon,
            "lat": self.lat,
            "bbox": list(self.bbox),
        }


def _city(id_: str, name: str, lon: float, lat: float, *aliases: str, pad: float = 0.12) -> Place:
    return Place(
        id=id_,
        name=name,
        kind="city",
        lon=lon,
        lat=lat,
        bbox=(lon - pad, lat - pad, lon + pad, lat + pad),
        aliases=aliases,
    )


def _reservoir(id_: str, name: str, lon: float, lat: float, *aliases: str, pad: float = 0.15) -> Place:
    return Place(
        id=id_,
        name=name,
        kind="reservoir",
        lon=lon,
        lat=lat,
        bbox=(lon - pad, lat - pad, lon + pad, lat + pad),
        aliases=aliases,
    )


#: The 12 administrative regions, as approximate envelopes.
REGIONS: tuple[Place, ...] = (
    Place("tanger-tetouan-al-hoceima", "Tanger-Tétouan-Al Hoceïma", "region",
          -5.35, 35.30, (-6.35, 34.55, -3.75, 36.00), ("tangier", "tanger", "tetouan")),
    Place("oriental", "L'Oriental", "region",
          -2.60, 33.60, (-4.30, 31.60, -1.00, 35.30), ("oujda", "eastern", "orientale")),
    Place("fes-meknes", "Fès-Meknès", "region",
          -4.80, 33.90, (-6.10, 32.80, -3.20, 34.90), ("fes", "fez", "meknes")),
    Place("rabat-sale-kenitra", "Rabat-Salé-Kénitra", "region",
          -6.20, 34.10, (-7.10, 33.30, -5.20, 34.90), ("rabat", "sale", "kenitra")),
    Place("beni-mellal-khenifra", "Béni Mellal-Khénifra", "region",
          -6.20, 32.60, (-7.30, 31.70, -5.00, 33.60), ("beni mellal", "khenifra")),
    Place("casablanca-settat", "Casablanca-Settat", "region",
          -7.60, 33.00, (-8.90, 32.10, -6.60, 33.90), ("casablanca", "casa", "settat")),
    Place("marrakech-safi", "Marrakech-Safi", "region",
          -8.30, 31.80, (-9.80, 30.80, -7.00, 32.90), ("marrakech", "marrakesh", "safi")),
    Place("draa-tafilalet", "Drâa-Tafilalet", "region",
          -5.20, 31.00, (-7.20, 29.30, -3.20, 32.40), ("ouarzazate", "errachidia", "draa")),
    Place("souss-massa", "Souss-Massa", "region",
          -8.70, 30.30, (-10.30, 29.20, -7.20, 31.30), ("agadir", "souss", "taroudant")),
    Place("guelmim-oued-noun", "Guelmim-Oued Noun", "region",
          -10.20, 28.60, (-12.00, 27.60, -8.60, 29.50), ("guelmim", "tan tan")),
    Place("laayoune-sakia-el-hamra", "Laâyoune-Sakia El Hamra", "region",
          -13.30, 26.50, (-15.00, 24.50, -11.50, 28.10), ("laayoune", "laayoun", "el aaiun")),
    Place("dakhla-oued-ed-dahab", "Dakhla-Oued Ed-Dahab", "region",
          -14.50, 22.60, (-17.10, 20.70, -12.50, 24.50), ("dakhla", "oued ed dahab")),
)

#: Major urban centres — the default AOIs for urban-expansion analysis.
CITIES: tuple[Place, ...] = (
    _city("casablanca", "Casablanca", -7.5898, 33.5731, "casa", "dar el beida", pad=0.22),
    _city("rabat", "Rabat", -6.8416, 34.0209, pad=0.16),
    _city("fes", "Fès", -5.0078, 34.0181, "fez"),
    _city("marrakech", "Marrakech", -7.9811, 31.6295, "marrakesh", pad=0.18),
    _city("tangier", "Tanger", -5.8340, 35.7595, "tangier", "tanja", pad=0.16),
    _city("agadir", "Agadir", -9.5981, 30.4278, pad=0.16),
    _city("meknes", "Meknès", -5.5473, 33.8935),
    _city("oujda", "Oujda", -1.9114, 34.6867),
    _city("kenitra", "Kénitra", -6.5802, 34.2610),
    _city("tetouan", "Tétouan", -5.3684, 35.5785),
    _city("safi", "Safi", -9.2372, 32.2994),
    _city("el-jadida", "El Jadida", -8.5074, 33.2549),
    _city("beni-mellal", "Béni Mellal", -6.3498, 32.3373),
    _city("nador", "Nador", -2.9287, 35.1681),
    _city("khouribga", "Khouribga", -6.9063, 32.8811),
    _city("laayoune", "Laâyoune", -13.2033, 27.1536),
    _city("dakhla", "Dakhla", -15.9580, 23.6848),
    _city("ouarzazate", "Ouarzazate", -6.9370, 30.9335),
    _city("errachidia", "Errachidia", -4.4247, 31.9314),
    _city("essaouira", "Essaouira", -9.7595, 31.5085),
)

#: Principal reservoirs — the default AOIs for water-storage monitoring.
RESERVOIRS: tuple[Place, ...] = (
    _reservoir("al-massira", "Barrage Al Massira", -7.63, 32.53, "massira", pad=0.25),
    _reservoir("bin-el-ouidane", "Bin El Ouidane", -6.45, 32.10),
    _reservoir("al-wahda", "Barrage Al Wahda", -5.62, 34.62, "wahda", pad=0.20),
    _reservoir("mohammed-v", "Barrage Mohammed V", -2.65, 34.68),
    _reservoir("idriss-i", "Barrage Idriss Ier", -4.90, 34.15, "idriss"),
    _reservoir("hassan-addakhil", "Barrage Hassan Addakhil", -4.35, 31.85),
    _reservoir("sidi-mohammed-ben-abdellah", "Barrage Sidi Mohammed Ben Abdellah", -6.65, 33.95, "smba"),
    _reservoir("youssef-ben-tachfine", "Barrage Youssef Ben Tachfine", -9.62, 29.72),
    _reservoir("oued-el-makhazine", "Barrage Oued El Makhazine", -5.65, 35.15),
    _reservoir("al-massira-oum-errabia", "Oum Er-Rbia basin", -7.10, 32.60, "oum er rbia", pad=0.6),
)

ALL_PLACES: tuple[Place, ...] = REGIONS + CITIES + RESERVOIRS

_BY_ID: dict[str, Place] = {p.id: p for p in ALL_PLACES}

#: Normalised search term -> place. Built so that the *most specific* match
#: wins, because several terms legitimately name both a city and the region
#: named after it: "marrakesh" is an alias of Marrakech the city and of
#: Marrakech-Safi the region, and someone asking about Marrakech means the
#: city. Aliases are registered least-specific-first so later groups overwrite,
#: then canonical names are applied last and beat any alias.
_LOOKUP: dict[str, Place] = {}


def _build_lookup() -> None:
    for group in (REGIONS, CITIES, RESERVOIRS):
        for place in group:
            for alias in place.aliases:
                _LOOKUP[_normalise(alias)] = place

    for place in ALL_PLACES:
        _LOOKUP[_normalise(place.name)] = place
        _LOOKUP.setdefault(_normalise(place.id.replace("-", " ")), place)


def get(place_id: str) -> Place | None:
    return _BY_ID.get(place_id)


def resolve(name: str) -> Place | None:
    """Fuzzy-resolve free text to a known place.

    Handles the accent-stripped, lowercase spellings people actually type
    ("marrakesh", "fes", "beni mellal") before falling back to close matching.
    """
    if not name:
        return None
    key = _normalise(name)
    if key in _LOOKUP:
        return _LOOKUP[key]
    matches = get_close_matches(key, list(_LOOKUP), n=1, cutoff=0.82)
    return _LOOKUP[matches[0]] if matches else None


def search(text: str) -> list[Place]:
    """All places whose name or alias contains ``text``."""
    key = _normalise(text)
    if not key:
        return []
    seen: dict[str, Place] = {}
    for term, place in _LOOKUP.items():
        if key in term:
            seen[place.id] = place
    return list(seen.values())


_ACCENTS = str.maketrans("àâäáãçéèêëíìîïñóòôöõúùûüýÿ", "aaaaaceeeeiiiinooooouuuuyy")


def _normalise(text: str) -> str:
    """Fold to the spelling people actually type: lowercase, unaccented, dashless."""
    return " ".join(text.lower().translate(_ACCENTS).replace("-", " ").split())


_build_lookup()
