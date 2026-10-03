"""
phoneme.py — High-Performance Continuous 3D Phoneme Structure with Slotted Biomechanical Signatures.
Precomputes immutable physical, aerodynamic, and auditory cue profiles for O(1) physics evaluation.
Universally compatible with Python 3.8+.
"""

from __future__ import annotations

import functools
import math
from enum import Enum
from typing import Sequence, NamedTuple

from articulatory_space import (
    _cached_vowel_lookup,
    _cached_consonant_lookup,
    Airstream,
    SecondaryArticulation,
)

_DEFAULT_VOWEL_LAYERS: tuple[SecondaryArticulation, ...] = (SecondaryArticulation.NONE,)
_DEFAULT_CONSONANT_LAYERS: tuple[Airstream, SecondaryArticulation] = (
    Airstream.PULMONIC,
    SecondaryArticulation.NONE,
)

_AIRSTREAM_PENALTIES: dict[Airstream, float] = {
    Airstream.PULMONIC: 1.00,
    Airstream.IMPLOSIVE: 0.75,
    Airstream.EJECTIVE: 0.70,
    Airstream.CLICK: 0.45,
}

_MANNER_BASE_WEIGHTS: dict[int, float] = {
    0: 3.40,  # Plosive
    1: 3.20,  # Nasal
    2: 2.50,  # Affricate
    3: 2.70,  # Tap / Trill
    4: 2.90,  # Fricative
    5: 1.40,  # Lateral Fricative
    6: 2.30,  # Approximant
    7: 2.60,  # Lateral Approximant
}


class PhonemeKind(Enum):
    VOWEL = "vowel"
    CONSONANT = "consonant"


class ArticulatoryPhysics(NamedTuple):
    """Intrinsic physical and acoustic invariants governing energy, contrast, and information."""
    oral_aperture: float             # [0.0 to 1.0]: Mouth exposure (thermal/moisture loss)
    nasal_aperture: float            # [0.0 to 1.0]: Velopharyngeal venting (Ohala pressure release)
    pulmonic_volume_velocity: float  # [0.0 to 1.0]: Lung airflow rate (0.0 for clicks/ejectives)
    supraglottal_impedance: float    # [0.0 to 1.0]: Ohala aerodynamic backpressure wall
    ptp_sensitivity: float           # [0.0 to 1.0]: Phonation Threshold Pressure vulnerability
    motor_precision: float           # [0.0 to 1.0]: Neuromuscular coordination complexity
    low_freq_energy: float           # [0.0 to 1.0]: Spectral power < 1.5 kHz (canopy penetration / grave bursts)
    high_freq_energy: float          # [0.0 to 1.0]: Spectral power > 2.5 kHz (wind/distance penetration)
    intrinsic_duration_ms: float     # Temporal duration integration base
    spectral_cog: float              # [0.0 to 1.0]: Spectral Center of Gravity (Burst frequency mass)
    vot_ms: float                    # [-80.0 to +80.0 ms]: Voice Onset Time glottal timing
    envelope_abruptness: float       # [0.0 to 1.0]: Transient shockwave rise time
    information_density: float       # [0.0 to 1.0]: Acoustic cue sharpness (Plosive=0.95, Glottal=0.15)

    @property
    def acoustic_salience(self) -> float:
        return round(0.40 * self.low_freq_energy + 0.60 * self.high_freq_energy, 3)


@functools.lru_cache(maxsize=1024)
def _compute_vowel_physics(hi: int, bk: int, rn: int, layers: tuple[Enum, ...]) -> ArticulatoryPhysics:
    h_norm = hi / 6.0
    aperture = round(max(0.12, 1.0 - h_norm), 3)

    is_nasal = any(l == SecondaryArticulation.NASALIZED for l in layers)
    nasal_aperture = 0.50 if is_nasal else 0.0
    f2_norm = round(0.20 + 0.65 * (1.0 - (bk / 2.0)), 3)

    return ArticulatoryPhysics(
        oral_aperture=aperture,
        nasal_aperture=nasal_aperture,
        pulmonic_volume_velocity=0.35,
        supraglottal_impedance=0.0,
        ptp_sensitivity=0.45,
        motor_precision=round(0.10 + 0.15 * (1.0 - h_norm), 3),
        low_freq_energy=round(0.60 + 0.35 * (1.0 - h_norm), 3),
        high_freq_energy=round(0.20 + 0.50 * (1.0 - (bk / 2.0)), 3),
        intrinsic_duration_ms=round(110.0 + 40.0 * (1.0 - h_norm), 1),
        spectral_cog=f2_norm,
        vot_ms=-60.0,
        envelope_abruptness=0.05,
        information_density=0.85,  # Vowels provide rich formant resonance
    )


@functools.lru_cache(maxsize=2048)
def _compute_consonant_physics(pl: int, mn: int, vc: int, layers: tuple[Enum, ...]) -> ArticulatoryPhysics:
    is_click = any(l == Airstream.CLICK for l in layers)
    is_ejective = any(l == Airstream.EJECTIVE for l in layers)
    is_implosive = any(l == Airstream.IMPLOSIVE for l in layers)
    is_nasal_manner = (mn == 1)
    is_voiced = (vc == 1)

    # 1. Pulmonic Airflow Demand
    if is_click or is_ejective:
        airflow = 0.0
    elif is_implosive:
        airflow = 0.08
    elif mn in (4, 5):
        airflow = 0.85
    elif mn == 0:
        airflow = 0.25
    elif is_nasal_manner:
        airflow = 0.30
    else:
        airflow = 0.40

    # 2. Supraglottal Impedance (Ohala Wall)
    if is_nasal_manner:
        impedance = 0.0
    elif mn in (0, 2) and is_voiced:
        impedance = 0.80
    elif mn in (4, 5) and is_voiced:
        impedance = 0.55
    else:
        impedance = 0.0

    # 3. Motor Precision
    if is_click:
        precision = 0.95
    elif mn == 3:
        precision = 0.80
    elif is_ejective or is_implosive:
        precision = 0.70
    elif 2.0 <= pl <= 5.5 and mn in (2, 4):
        precision = 0.40
    else:
        precision = 0.20

    is_strident = (mn in (2, 4) and 2.0 <= pl <= 5.5) or is_click

    # 4. Spectral Distribution: Restores Labial Low-Frequency Grave Bursts (0.80)
    if pl <= 1.5:
        # Labials produce powerful low-frequency grave transients
        low_freq = 0.80 if mn == 0 else (0.75 if is_nasal_manner else 0.30)
        high_freq = 0.15
        cog = 0.15
    elif 2.0 <= pl <= 3.5:
        # Alveolar sibilants concentrate high-frequency energy (>5 kHz)
        low_freq = 0.70 if is_nasal_manner else (0.40 if is_voiced else 0.10)
        high_freq = 0.92 if is_strident else 0.45
        cog = 0.92 if is_strident else 0.65
    elif 3.8 <= pl <= 5.5:
        # Postalveolars concentrate mid-high energy (2.5-4 kHz)
        low_freq = 0.70 if is_nasal_manner else (0.40 if is_voiced else 0.10)
        high_freq = 0.60 if is_strident else 0.40
        cog = 0.55 if is_strident else 0.65
    elif pl <= 8.5:
        # Velars exhibit compact mid-frequency peaks
        low_freq = 0.65 if is_nasal_manner else (0.35 if is_voiced else 0.25)
        high_freq = 0.45
        cog = 0.50
    else:
        # Glottals carry diffuse, weak friction
        low_freq = 0.15
        high_freq = 0.15
        cog = 0.25

    # 5. Voice Onset Time (VOT)
    if is_nasal_manner or mn in (3, 6, 7):
        vot = -50.0
    elif is_voiced:
        vot = -65.0
    elif is_ejective or is_click:
        vot = 25.0
    elif any(l == SecondaryArticulation.ASPIRATED for l in layers):
        vot = 60.0
    else:
        vot = 10.0

    # 6. Envelope Abruptness
    if mn == 0 or is_click: abruptness = 1.00
    elif mn == 2: abruptness = 0.75
    elif mn in (4, 5): abruptness = 0.20
    else: abruptness = 0.10

    # 7. Acoustic Information Density: Plosives transmit high bits; glottals transmit minimal bits
    if mn == 0 or is_click:
        info_density = 0.95
    elif is_strident:
        info_density = 0.80
    elif mn == 1:
        info_density = 0.75
    elif mn in (4, 5):
        info_density = 0.55
    elif pl >= 9.5:
        info_density = 0.15  # /h/ and /ʔ/ have minimal positional information
    else:
        info_density = 0.40

    duration = 90.0 if (mn == 0 and not is_voiced) else (130.0 if mn == 3 else (100.0 if is_click else 75.0))

    return ArticulatoryPhysics(
        oral_aperture=0.04,
        nasal_aperture=0.70 if is_nasal_manner else 0.0,
        pulmonic_volume_velocity=airflow,
        supraglottal_impedance=impedance,
        ptp_sensitivity=0.60 if is_voiced else 0.0,
        motor_precision=precision,
        low_freq_energy=low_freq,
        high_freq_energy=high_freq,
        intrinsic_duration_ms=duration,
        spectral_cog=cog,
        vot_ms=vot,
        envelope_abruptness=abruptness,
        information_density=info_density,
    )


@functools.lru_cache(maxsize=1024)
def _compute_vowel_weight(hi: int, bk: int, rn: int, layers: tuple[Enum, ...]) -> float:
    layer_factor = 1.0
    for layer in layers:
        if isinstance(layer, Airstream):
            layer_factor *= _AIRSTREAM_PENALTIES.get(layer, 0.60)
        elif isinstance(layer, SecondaryArticulation) and layer != SecondaryArticulation.NONE:
            layer_factor *= 0.75

    h_norm = hi / 6.0
    b_norm = bk / 2.0
    dist_center = math.sqrt((h_norm - 0.50) ** 2 + 1.20 * (b_norm - 0.50) ** 2)
    cardinal_bonus = 0.80 + 3.00 * ((dist_center / 0.65) ** 1.30)

    canonical_rounding = 1.00
    if bk == 0: canonical_rounding = 1.00 if rn == 0 else 0.72
    elif bk == 2: canonical_rounding = 1.00 if rn == 1 else 0.75
    else: canonical_rounding = 1.00 if rn == 0 else 0.80

    if hi <= 1 and rn == 0: canonical_rounding = 1.00
    return round(cardinal_bonus * canonical_rounding * layer_factor, 3)


@functools.lru_cache(maxsize=2048)
def _compute_consonant_weight(pl: int, mn: int, vc: int, ipa: str, layers: tuple[Enum, ...]) -> float:
    layer_factor = 1.0
    for layer in layers:
        if isinstance(layer, Airstream):
            layer_factor *= _AIRSTREAM_PENALTIES.get(layer, 0.60)
        elif isinstance(layer, SecondaryArticulation) and layer != SecondaryArticulation.NONE:
            layer_factor *= 0.70

    if pl >= 10:
        base_w = 0.85 if ipa in ("ʔ", "h", "ɦ") else 0.45
        return round(base_w * layer_factor, 3)

    manner_w = _MANNER_BASE_WEIGHTS.get(mn, 2.00)
    coronal_peak = 0.65 + 0.35 * math.exp(-((pl - 3.0) ** 2) / 14.0)
    
    stridency_factor = 1.00
    is_sibilant_zone = (2.0 <= pl <= 5.5)
    if mn == 4: stridency_factor = 1.10 if is_sibilant_zone else 0.60
    elif mn == 2: stridency_factor = 1.15 if is_sibilant_zone else 0.50

    is_obstruent = (mn in (0, 2, 4, 5))
    if is_obstruent:
        voicing_factor = 1.00 if vc == 0 else 0.95
    else:
        voicing_factor = 1.00 if vc == 1 else 0.35

    return round(manner_w * coronal_peak * stridency_factor * voicing_factor * layer_factor, 3)


class Phoneme:
    """Speech segment encapsulating coordinates, features, and O(1) precomputed physics."""

    __slots__ = ('point', 'kind', 'layers', 'ipa', 'weight', 'physics')

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
            self.physics: ArticulatoryPhysics = _compute_vowel_physics(hi, bk, rn, self.layers)
        else:
            pl = max(0, min(10, int(round(p0))))
            mn = max(0, min(7, int(round(p1))))
            vc = max(0, min(1, int(round(p2))))
            self.ipa: str = _cached_consonant_lookup(pl, mn, vc, self.layers)
            self.weight: float = _compute_consonant_weight(pl, mn, vc, self.ipa, self.layers)
            self.physics: ArticulatoryPhysics = _compute_consonant_physics(pl, mn, vc, self.layers)

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