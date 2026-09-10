"""
phoneme.py — High-Performance Continuous 3D Phoneme Structure.
Optimized with memoized continuous markedness equations, cached IPA labels, and coordinate clamping.
"""

from __future__ import annotations

import functools
import math
from enum import Enum
from typing import Sequence

from articulatory_space import (
    _cached_vowel_lookup,
    _cached_consonant_lookup,
    Airstream,
    SecondaryArticulation,
)

_DEFAULT_VOWEL_LAYERS: tuple[SecondaryArticulation] = (SecondaryArticulation.NONE,)
_DEFAULT_CONSONANT_LAYERS: tuple[Airstream, SecondaryArticulation] = (
    Airstream.PULMONIC,
    SecondaryArticulation.NONE,
)

_AIRSTREAM_PENALTIES: dict[Airstream, float] = {
    Airstream.PULMONIC: 1.00,
    Airstream.IMPLOSIVE: 0.70,
    Airstream.EJECTIVE: 0.65,
    Airstream.CLICK: 0.35,
}

_MANNER_BASE_WEIGHTS: dict[int, float] = {
    0: 3.40,  # Plosive
    1: 3.20,  # Nasal
    2: 2.50,  # Trill
    3: 2.70,  # Tap
    4: 2.90,  # Fricative
    5: 1.40,  # Lateral Fricative
    6: 2.30,  # Approximant
    7: 2.60,  # Lateral Approximant
}


class PhonemeKind(Enum):
    VOWEL = "vowel"
    CONSONANT = "consonant"


# =====================================================================
# Memoized Continuous Markedness Resolvers
# =====================================================================

@functools.lru_cache(maxsize=1024)
def _compute_vowel_weight(hi: int, bk: int, rn: int, layers: tuple[Enum, ...]) -> float:
    """
    Computes continuous markedness weight for vowels:
    - Peripherality & cardinal bonus based on Liljencrants & Lindblom dispersion.
    - Quantal rounding markedness: Open/central /a/ and /ə/ have full 1.00 baseline.
    - Front rounded (y, ø) and back unrounded (ɯ, ɤ) receive natural markedness factors.
    """
    layer_factor = 1.0
    for layer in layers:
        if isinstance(layer, Airstream):
            layer_factor *= _AIRSTREAM_PENALTIES.get(layer, 0.60)
        elif isinstance(layer, SecondaryArticulation) and layer != SecondaryArticulation.NONE:
            layer_factor *= 0.75

    h_norm = hi / 6.0
    b_norm = bk / 2.0

    # Liljencrants & Lindblom dispersion peripherality metric
    dist_center = math.sqrt((h_norm - 0.50) ** 2 + 1.20 * (b_norm - 0.50) ** 2)
    cardinal_bonus = 0.80 + 3.00 * ((dist_center / 0.65) ** 1.30)

    # Quantal Canonical Rounding:
    # Front unrounded (i, e), Back rounded (u, o), and Open/Central unrounded (a, ə, ɨ) = 1.00
    if bk == 0:
        canonical_rounding = 1.00 if rn == 0 else 0.72  # Front rounded (y, ø, œ) is marked
    elif bk == 2:
        canonical_rounding = 1.00 if rn == 1 else 0.75  # Back unrounded (ɯ, ɤ) is marked
    else:
        canonical_rounding = 1.00 if rn == 0 else 0.80  # Central rounded (ʉ, ɵ) is marked

    # Open floor /a/ has universal unmarked priority
    if hi <= 1 and rn == 0:
        canonical_rounding = 1.00

    return round(cardinal_bonus * canonical_rounding * layer_factor, 3)


@functools.lru_cache(maxsize=2048)
def _compute_consonant_weight(pl: int, mn: int, vc: int, ipa: str, layers: tuple[Enum, ...]) -> float:
    """
    Computes continuous markedness weight for consonants:
    - Coronal peak Gaussian distribution centered at alveolar ridge (pl = 3.0).
    - Acoustic stridency factor favoring sibilants (mn=2, 4 at pl in [2.5, 5.5]).
    - Aerodynamic Voicing Constraint on obstruents (Ohala 1983).
    - Spontaneous Voicing Principle on sonorants (Chomsky & Halle 1968: voiceless sonorants marked at 0.35).
    """
    layer_factor = 1.0
    for layer in layers:
        if isinstance(layer, Airstream):
            layer_factor *= _AIRSTREAM_PENALTIES.get(layer, 0.60)
        elif isinstance(layer, SecondaryArticulation) and layer != SecondaryArticulation.NONE:
            layer_factor *= 0.70

    if pl >= 10:
        base_w = 1.40 if ipa in ("ʔ", "h", "ɦ") else 0.45
        return round(base_w * layer_factor, 3)

    manner_w = _MANNER_BASE_WEIGHTS.get(mn, 2.00)
    coronal_peak = 0.55 + 0.45 * math.exp(-((pl - 3.0) ** 2) / 11.52)
    
    # 1. Acoustic Stridency & Markedness
    stridency_factor = 1.00
    is_sibilant_zone = (2.5 <= pl <= 5.5)
    
    if mn == 4:
        stridency_factor = 1.15 if is_sibilant_zone else 0.45
    elif mn == 2:
        stridency_factor = 1.20 if is_sibilant_zone else 0.35  # Sibilant affricates (ts, tʃ, dz, dʒ) favored
    elif mn == 3 and pl == 0:
        stridency_factor = 0.20  # Bilabial trill markedness

    # 2. Aerodynamic Voicing Constraint (Obstruents) & Spontaneous Voicing Principle (Sonorants)
    is_obstruent = (mn in (0, 2, 4, 5))
    if is_obstruent:
        cavity_size = max(0.10, 1.0 - (pl / 10.0))
        voicing_factor = 1.00 - (0.45 * vc * (1.0 - cavity_size * 0.50))
    else:
        # Sonorants (Nasals, Liquids, Approximants): Voiced is universal default (1.00); Voiceless is marked (0.35)
        voicing_factor = 1.00 if vc == 1 else 0.35

    return round(manner_w * coronal_peak * stridency_factor * voicing_factor * layer_factor, 3)


# =====================================================================
# Phoneme Class
# =====================================================================

class Phoneme:
    """A single speech sound with struct-packed attributes, cached IPA, and O(1) memoized continuous weight."""

    __slots__ = ('point', 'kind', 'layers', 'ipa', 'weight')

    def __init__(
        self,
        kind: PhonemeKind,
        point: tuple[float, float, float] | Sequence[float] = (0.0, 0.0, 0.0),
        layers: tuple[Enum, ...] | Sequence[Enum] | None = None,
    ):
        self.kind = kind
        p0, p1, p2 = point
        self.point: tuple[float, float, float] = (float(p0), float(p1), float(p2))

        if layers is None:
            self.layers = _DEFAULT_VOWEL_LAYERS if kind == PhonemeKind.VOWEL else _DEFAULT_CONSONANT_LAYERS
        elif isinstance(layers, tuple):
            self.layers = layers
        else:
            self.layers = tuple(layers)

        if kind == PhonemeKind.VOWEL:
            hi = max(0, min(6, int(round(p0))))
            bk = max(0, min(2, int(round(p1))))
            rn = max(0, min(1, int(round(p2))))
            self.ipa: str = _cached_vowel_lookup(hi, bk, rn, self.layers)
            self.weight: float = _compute_vowel_weight(hi, bk, rn, self.layers)

        else:
            pl = max(0, min(10, int(round(p0))))
            mn = max(0, min(7, int(round(p1))))
            vc = max(0, min(1, int(round(p2))))
            self.ipa: str = _cached_consonant_lookup(pl, mn, vc, self.layers)
            self.weight: float = _compute_consonant_weight(pl, mn, vc, self.ipa, self.layers)

    def drift(
        self,
        delta_point: tuple[float, float, float] | Sequence[float] | None = None,
        point: tuple[float, float, float] | Sequence[float] | None = None,
        layers: tuple[Enum, ...] | Sequence[Enum] | None = None,
    ) -> Phoneme:
        if point is not None:
            p0, p1, p2 = point
        elif delta_point is not None:
            p0 = self.point[0] + delta_point[0]
            p1 = self.point[1] + delta_point[1]
            p2 = self.point[2] + delta_point[2]
        else:
            p0, p1, p2 = self.point

        if self.kind == PhonemeKind.VOWEL:
            clamped_point = (
                max(0.0, min(6.0, float(p0))),
                max(0.0, min(2.0, float(p1))),
                max(0.0, min(1.0, float(p2))),
            )
        else:
            clamped_point = (
                max(0.0, min(10.0, float(p0))),
                max(0.0, min(7.0, float(p1))),
                max(0.0, min(1.0, float(p2))),
            )

        new_layers = self.layers if layers is None else (layers if isinstance(layers, tuple) else tuple(layers))
        return Phoneme(kind=self.kind, point=clamped_point, layers=new_layers)

    def __hash__(self) -> int:
        return hash((self.kind, self.point, self.layers))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phoneme):
            return NotImplemented
        return (
            self.kind is other.kind
            and self.point == other.point
            and self.layers == other.layers
        )

    def __repr__(self) -> str:
        coords = f"{self.point[0]:.1f}, {self.point[1]:.1f}, {self.point[2]:.1f}"
        return f"Phoneme({self.kind.value}, ({coords})) -> /{self.ipa}/ (w={self.weight})"