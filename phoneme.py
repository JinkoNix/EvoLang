"""
phoneme.py — Continuous Multi-Tier Phoneme Structure with Autosegmental Properties.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Sequence, NamedTuple

from articulatory_space import (
    _lookup_vowel,
    _lookup_consonant,
    render_suprasegmentals,
    Airstream,
    SecondaryArticulation,
)

_DEFAULT_VOWEL_LAYERS: tuple[SecondaryArticulation, ...] = (SecondaryArticulation.NONE,)
_DEFAULT_CONSONANT_LAYERS: tuple[Airstream, SecondaryArticulation] = (
    Airstream.PULMONIC,
    SecondaryArticulation.NONE,
)


class PhonemeKind(Enum):
    VOWEL = "vowel"
    CONSONANT = "consonant"


class ArticulatoryPhysics(NamedTuple):
    oral_aperture: float
    nasal_aperture: float
    pulmonic_volume_velocity: float
    supraglottal_impedance: float
    ptp_sensitivity: float
    motor_precision: float
    low_freq_energy: float
    high_freq_energy: float
    intrinsic_duration_ms: float
    spectral_cog: float
    vot_ms: float
    envelope_abruptness: float
    information_density: float

    @property
    def acoustic_salience(self) -> float:
        return round(0.40 * self.low_freq_energy + 0.60 * self.high_freq_energy, 3)


def _compute_vowel_physics(hi: int, bk: int, rn: int, layers: tuple[Enum, ...]) -> ArticulatoryPhysics:
    h_norm = hi / 6.0
    b_norm = bk / 2.0
    is_nasal = any(l == SecondaryArticulation.NASALIZED for l in layers)

    aperture = 0.12 + 0.88 * ((1.0 - h_norm) ** 1.3)
    f2_norm = 0.20 + 0.65 * (1.0 - b_norm)

    return ArticulatoryPhysics(
        oral_aperture=round(aperture, 3),
        nasal_aperture=0.50 if is_nasal else 0.0,
        pulmonic_volume_velocity=0.35,
        supraglottal_impedance=0.0,
        ptp_sensitivity=0.45,
        motor_precision=round(0.10 + 0.15 * (1.0 - h_norm), 3),
        low_freq_energy=round(0.60 + 0.35 * (1.0 - h_norm), 3),
        high_freq_energy=round(0.20 + 0.50 * (1.0 - b_norm), 3),
        intrinsic_duration_ms=round(110.0 + 40.0 * (1.0 - h_norm), 1),
        spectral_cog=round(f2_norm, 3),
        vot_ms=-60.0,
        envelope_abruptness=0.05,
        information_density=0.85,
    )


def _compute_consonant_physics(pl: int, mn: int, vc: int, layers: tuple[Enum, ...]) -> ArticulatoryPhysics:
    is_click = any(l == Airstream.CLICK for l in layers)
    is_ejective = any(l == Airstream.EJECTIVE for l in layers)
    is_implosive = any(l == Airstream.IMPLOSIVE for l in layers)
    is_voiced = (vc == 1)
    is_nasal = (mn == 1)

    if is_click or is_ejective:
        airflow = 0.0
    elif is_implosive:
        airflow = 0.08
    else:
        airflow = 0.85 if mn in (4, 5) else (0.25 if mn == 0 else (0.30 if is_nasal else 0.40))

    if is_nasal:
        impedance = 0.0
    elif mn in (0, 2) and is_voiced:
        impedance = 0.80
    elif mn in (4, 5) and is_voiced:
        impedance = 0.55
    else:
        impedance = 0.0

    pl_f = float(pl)
    cog = (1.0 / (1.0 + math.exp(-1.5 * (pl_f - 2.0)))) if pl_f <= 5.5 else math.exp(-0.6 * (pl_f - 5.5))
    low_freq = 0.80 * math.exp(-((pl_f - 0.5) ** 2) / 4.0) + 0.30
    high_freq = 0.92 * math.exp(-((pl_f - 3.2) ** 2) / 3.0) + 0.15

    if is_nasal or mn in (3, 6, 7):
        vot = -50.0
    elif is_voiced:
        vot = -65.0
    elif is_ejective or is_click:
        vot = 25.0
    elif any(l == SecondaryArticulation.ASPIRATED for l in layers):
        vot = 60.0
    else:
        vot = 10.0

    abruptness = 1.00 if (mn == 0 or is_click) else (0.75 if mn == 2 else (0.20 if mn in (4, 5) else 0.10))
    info_density = 0.95 if (mn == 0 or is_click) else (0.80 if mn in (2, 4) and 2.0 <= pl <= 5.5 else (0.15 if pl >= 9.5 else 0.50))
    duration = 90.0 if (mn == 0 and not is_voiced) else (130.0 if mn == 3 else (100.0 if is_click else 75.0))

    return ArticulatoryPhysics(
        oral_aperture=0.04,
        nasal_aperture=0.70 if is_nasal else 0.0,
        pulmonic_volume_velocity=airflow,
        supraglottal_impedance=impedance,
        ptp_sensitivity=0.60 if is_voiced else 0.0,
        motor_precision=0.95 if is_click else (0.70 if (is_ejective or is_implosive) else 0.35),
        low_freq_energy=round(min(1.0, low_freq), 3),
        high_freq_energy=round(min(1.0, high_freq), 3),
        intrinsic_duration_ms=duration,
        spectral_cog=round(min(1.0, cog), 3),
        vot_ms=vot,
        envelope_abruptness=abruptness,
        information_density=info_density,
    )


class Phoneme:
    __slots__ = (
        'point', 'kind', 'layers', 'duration', 'pitch_offset',
        'pitch_contour', 'base_ipa', 'ipa', 'weight', 'physics', '_key'
    )

    def __init__(
        self,
        kind: PhonemeKind,
        point: tuple[float, float, float] | Sequence[float] = (0.0, 0.0, 0.0),
        layers: tuple[Enum, ...] | Sequence[Enum] | None = None,
        duration: float = 1.0,
        pitch_offset: float = 0.0,
        pitch_contour: float = 0.0,
        _physics: ArticulatoryPhysics | None = None,
    ):
        self.kind = kind
        p0, p1, p2 = point
        self.point = (float(p0), float(p1), float(p2))
        self.duration = max(0.0, min(2.5, float(duration)))
        self.pitch_offset = max(-1.0, min(1.0, float(pitch_offset)))
        self.pitch_contour = max(-1.0, min(1.0, float(pitch_contour)))

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
            self.base_ipa = _lookup_vowel(hi, bk, rn, self.layers)
            self.physics = _physics or _compute_vowel_physics(hi, bk, rn, self.layers)
            self.ipa = render_suprasegmentals(self.base_ipa, self.duration, self.pitch_offset, self.pitch_contour)
        else:
            pl = max(0, min(10, int(round(p0))))
            mn = max(0, min(7, int(round(p1))))
            vc = max(0, min(1, int(round(p2))))
            self.base_ipa = _lookup_consonant(pl, mn, vc, self.layers)
            self.physics = _physics or _compute_consonant_physics(pl, mn, vc, self.layers)
            self.ipa = render_suprasegmentals(self.base_ipa, self.duration, 0.0, 0.0)

        self.weight = 1.0
        # Canonical discrete state key guarantees hash(a) == hash(b) when a == b
        self._key = (
            self.kind,
            round(self.point[0], 2),
            round(self.point[1], 2),
            round(self.point[2], 2),
            self.layers,
            round(self.duration, 2),
            round(self.pitch_offset, 2),
            round(self.pitch_contour, 2),
        )

    @property
    def laryngeal_stiffness(self) -> float:
        if self.kind == PhonemeKind.VOWEL:
            return 0.0
        pl, mn, vc = self.point
        if pl >= 9.5:
            return 0.80 if mn == 0 else -0.70
        if any(l == Airstream.EJECTIVE for l in self.layers):
            return 0.90
        return 1.0 - 2.0 * vc

    @property
    def moraic_mass(self) -> float:
        if self.kind == PhonemeKind.VOWEL:
            return self.duration
        mn = int(round(self.point[1]))
        return self.duration * (0.8 if mn in (3, 6, 7) else 0.5)

    @property
    def articulator_tier(self) -> int:
        """
        Anatomical organ tier:
        0: Labial (Lips)
        1: Coronal (Tongue Tip/Blade)
        2: Dorsal (Tongue Body / Vowels)
        3: Laryngeal / Radical
        """
        if self.kind == PhonemeKind.VOWEL:
            return 2  # All vowels are dorsal
        pl = self.point[0]
        if pl <= 1.5:
            return 0  # Labial
        elif pl <= 5.5:
            return 1  # Coronal
        elif pl <= 8.5:
            return 2  # Dorsal
        return 3      # Laryngeal

    def drift(
        self,
        delta_point: tuple[float, float, float] | Sequence[float] | None = None,
        point: tuple[float, float, float] | Sequence[float] | None = None,
        layers: tuple[Enum, ...] | Sequence[Enum] | None = None,
        duration: float | None = None,
        pitch_offset: float | None = None,
        pitch_contour: float | None = None,
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
            clamped = (max(0.0, min(6.0, float(p0))), max(0.0, min(2.0, float(p1))), max(0.0, min(1.0, float(p2))))
        else:
            clamped = (max(0.0, min(10.0, float(p0))), max(0.0, min(7.0, float(p1))), max(0.0, min(1.0, float(p2))))

        new_layers = self.layers if layers is None else (layers if isinstance(layers, tuple) else tuple(layers))
        new_dur = self.duration if duration is None else duration
        new_po = self.pitch_offset if pitch_offset is None else pitch_offset
        new_pc = self.pitch_contour if pitch_contour is None else pitch_contour

        # Reuse physics if articulatory shape and airflow mode did not change
        cached_phys = self.physics if (clamped == self.point and new_layers == self.layers) else None

        return Phoneme(
            kind=self.kind,
            point=clamped,
            layers=new_layers,
            duration=new_dur,
            pitch_offset=new_po,
            pitch_contour=new_pc,
            _physics=cached_phys,
        )

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phoneme):
            return NotImplemented
        return self._key == other._key

    def __repr__(self) -> str:
        return f"Phoneme({self.kind.value}, {self.point}) -> /{self.ipa}/"