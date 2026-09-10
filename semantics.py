"""
semantics.py — Continuous 7D Cognitive Space, Typed Derivation Trees, and Biological Prototype Anchors.
Optimized with unrolled Euclidean metric math and continuous exponential grammaticalization physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


# =====================================================================
# Typological Enums & Records
# =====================================================================

class ClineStage(Enum):
    FREE_ROOT = "Free Root"
    PERIPHRASIS = "Periphrastic Auxiliary"
    CLITIC = "Clitic"
    BOUND_AFFIX = "Bound Affix"
    FUSED_INTERNAL = "Fused / Suprasegmental"
    ZERO = "Zero / Lost"


class DerivationType(Enum):
    ROOT = "root"
    COMPOUND = "compound"
    AFFIXATION = "affixation"
    LOANWORD = "loanword"
    SEMANTIC_SPLIT = "split"


@dataclass
class LoanProvenance:
    source_language: str
    donor_word_id: int
    donor_form_at_borrowing: str
    donor_vector_at_borrowing: SemanticVector
    generation_borrowed: int = 0


@dataclass
class DerivationRecord:
    root_family_id: int
    derivation_type: DerivationType = DerivationType.ROOT
    local_parent_ids: tuple[int, ...] = ()
    provenance: LoanProvenance | None = None


# =====================================================================
# Continuous Grammaticalization Constants
# =====================================================================

_CLINE_DEPTH: dict[ClineStage, float] = {
    ClineStage.FREE_ROOT: 0.00,
    ClineStage.PERIPHRASIS: 0.20,
    ClineStage.CLITIC: 0.45,
    ClineStage.BOUND_AFFIX: 0.75,
    ClineStage.FUSED_INTERNAL: 0.92,
    ClineStage.ZERO: 1.00,
}

_LAMBDA_CONCRETENESS = 2.40
_LAMBDA_ANIMACY = 2.80
_LAMBDA_POTENCY = 1.80


# =====================================================================
# 7D Semantic Vector
# =====================================================================

class SemanticVector:
    """A continuous coordinate point in 7D cognitive semantic space (0.0 to 1.0)."""

    __slots__ = ('coords',)

    DIMENSION_NAMES = (
        "concreteness",
        "animacy",
        "valence",
        "potency",
        "dynamism",
        "sociality",
        "extension",
    )

    def __init__(
        self,
        concreteness: float,
        animacy: float,
        valence: float,
        potency: float,
        dynamism: float,
        sociality: float,
        extension: float,
    ):
        self.coords: tuple[float, float, float, float, float, float, float] = (
            max(0.0, min(1.0, float(concreteness))),
            max(0.0, min(1.0, float(animacy))),
            max(0.0, min(1.0, float(valence))),
            max(0.0, min(1.0, float(potency))),
            max(0.0, min(1.0, float(dynamism))),
            max(0.0, min(1.0, float(sociality))),
            max(0.0, min(1.0, float(extension))),
        )

    @property
    def concreteness(self) -> float: return self.coords[0]
    @property
    def animacy(self) -> float: return self.coords[1]
    @property
    def valence(self) -> float: return self.coords[2]
    @property
    def potency(self) -> float: return self.coords[3]
    @property
    def dynamism(self) -> float: return self.coords[4]
    @property
    def sociality(self) -> float: return self.coords[5]
    @property
    def extension(self) -> float: return self.coords[6]

    @property
    def grammaticality_index(self) -> float:
        """Continuous measure of grammatical status [0.0 = lexical content, 1.0 = pure functional marker]."""
        c, a = self.coords[0], self.coords[1]
        lexical_mass = (c + 0.60 * a) / 1.60
        return max(0.0, min(1.0, 1.0 - lexical_mass))

    @property
    def is_grammatical(self) -> bool:
        return self.grammaticality_index >= 0.75

    def distance_to(self, other: SemanticVector, weights: Sequence[float] | None = None) -> float:
        c1 = self.coords
        c2 = other.coords
        if weights is None:
            return math.sqrt(
                (c1[0] - c2[0]) ** 2 +
                (c1[1] - c2[1]) ** 2 +
                (c1[2] - c2[2]) ** 2 +
                (c1[3] - c2[3]) ** 2 +
                (c1[4] - c2[4]) ** 2 +
                (c1[5] - c2[5]) ** 2 +
                (c1[6] - c2[6]) ** 2
            )
        w0, w1, w2, w3, w4, w5, w6 = weights
        return math.sqrt(
            w0 * (c1[0] - c2[0]) ** 2 +
            w1 * (c1[1] - c2[1]) ** 2 +
            w2 * (c1[2] - c2[2]) ** 2 +
            w3 * (c1[3] - c2[3]) ** 2 +
            w4 * (c1[4] - c2[4]) ** 2 +
            w5 * (c1[5] - c2[5]) ** 2 +
            w6 * (c1[6] - c2[6]) ** 2
        )

    def weighted_distance_to(self, other: SemanticVector, weights: Sequence[float]) -> float:
        c1 = self.coords
        c2 = other.coords
        w0, w1, w2, w3, w4, w5, w6 = weights
        return math.sqrt(
            w0 * (c1[0] - c2[0]) ** 2 +
            w1 * (c1[1] - c2[1]) ** 2 +
            w2 * (c1[2] - c2[2]) ** 2 +
            w3 * (c1[3] - c2[3]) ** 2 +
            w4 * (c1[4] - c2[4]) ** 2 +
            w5 * (c1[5] - c2[5]) ** 2 +
            w6 * (c1[6] - c2[6]) ** 2
        )

    def drift(self, delta: Sequence[float]) -> SemanticVector:
        c = self.coords
        d0, d1, d2, d3, d4, d5, d6 = delta
        return SemanticVector(
            c[0] + d0, c[1] + d1, c[2] + d2,
            c[3] + d3, c[4] + d4, c[5] + d5, c[6] + d6
        )

    def blend(self, other: SemanticVector, weight: float = 0.5) -> SemanticVector:
        w1 = max(0.0, min(1.0, weight))
        w2 = 1.0 - w1
        c1 = self.coords
        c2 = other.coords
        return SemanticVector(
            w1 * c1[0] + w2 * c2[0],
            w1 * c1[1] + w2 * c2[1],
            w1 * c1[2] + w2 * c2[2],
            w1 * c1[3] + w2 * c2[3],
            w1 * c1[4] + w2 * c2[4],
            w1 * c1[5] + w2 * c2[5],
            w1 * c1[6] + w2 * c2[6],
        )

    def generalize_for_cline(self, stage: ClineStage) -> SemanticVector:
        """Applies continuous exponential desemanticization (bleaching) along the grammaticalization cline."""
        t = _CLINE_DEPTH.get(stage, 0.0)
        if t <= 0.0:
            return self

        c, a, v, p, d, s, e = self.coords
        decay_c = math.exp(-_LAMBDA_CONCRETENESS * t)
        decay_a = math.exp(-_LAMBDA_ANIMACY * t)
        decay_p = math.exp(-_LAMBDA_POTENCY * t)

        return SemanticVector(
            concreteness=c * decay_c,
            animacy=a * decay_a,
            valence=v,
            potency=p * decay_p,
            dynamism=d,
            sociality=s,
            extension=e,
        )

    def __iter__(self):
        return iter(self.coords)

    def __repr__(self) -> str:
        vals = ", ".join(f"{v:.2f}" for v in self.coords)
        return f"SemanticVector({vals})"


# =====================================================================
# Universal Biological & Cognitive Seed Anchors
# =====================================================================

class SemanticSpace:
    """Universal biological & cognitive seed vectors with explicit prototype constants."""

    # 1. Physical & Natural Elements
    WATER = SemanticVector(1.0, 0.0, 0.7, 0.5, 0.5, 0.1, 0.4)
    FIRE  = SemanticVector(0.9, 0.1, 0.4, 0.8, 0.9, 0.2, 0.3)
    SUN   = SemanticVector(0.9, 0.1, 0.8, 1.0, 0.6, 0.1, 0.9)
    MOON  = SemanticVector(0.9, 0.0, 0.6, 0.7, 0.3, 0.1, 0.8)
    EARTH = SemanticVector(1.0, 0.0, 0.6, 0.8, 0.1, 0.1, 0.8)
    STONE = SemanticVector(1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.1)
    TREE  = SemanticVector(1.0, 0.4, 0.7, 0.6, 0.2, 0.1, 0.4)

    # 2. Living Entities & Universal Kinship Primaries
    MOTHER = SemanticVector(0.9, 1.0, 0.85, 0.40, 0.40, 0.95, 0.20)
    FATHER = SemanticVector(0.9, 1.0, 0.75, 0.75, 0.60, 0.85, 0.30)
    PERSON = SemanticVector(0.9, 1.0, 0.60, 0.50, 0.60, 0.70, 0.30)
    BEAST  = SemanticVector(0.9, 0.9, 0.50, 0.70, 0.80, 0.20, 0.40)
    SELF   = SemanticVector(0.8, 1.0, 0.60, 0.50, 0.50, 0.30, 0.10)
    OTHER  = SemanticVector(0.8, 1.0, 0.60, 0.50, 0.50, 0.80, 0.10)

    # 3. Somatic Landmarks
    HEAD  = SemanticVector(0.9, 0.8, 0.6, 0.7, 0.4, 0.3, 0.2)
    BODY  = SemanticVector(1.0, 0.9, 0.6, 0.6, 0.5, 0.4, 0.5)
    EYE   = SemanticVector(0.9, 0.8, 0.6, 0.4, 0.5, 0.3, 0.3)
    HAND  = SemanticVector(1.0, 0.8, 0.6, 0.6, 0.6, 0.4, 0.2)
    HEART = SemanticVector(0.7, 0.9, 0.7, 0.6, 0.4, 0.6, 0.2)
    BLOOD = SemanticVector(0.9, 0.6, 0.4, 0.7, 0.5, 0.3, 0.2)

    # 4. Actions & Sensory Perceptions
    MAKE  = SemanticVector(0.5, 0.8, 0.6, 0.6, 0.8, 0.4, 0.3)
    MOVE  = SemanticVector(0.6, 0.7, 0.5, 0.4, 0.8, 0.2, 0.5)
    SEE   = SemanticVector(0.4, 0.8, 0.6, 0.4, 0.5, 0.3, 0.4)
    HEAR  = SemanticVector(0.4, 0.8, 0.6, 0.4, 0.4, 0.4, 0.4)
    SPEAK = SemanticVector(0.4, 0.9, 0.6, 0.4, 0.7, 1.0, 0.3)
    EAT   = SemanticVector(0.8, 0.8, 0.8, 0.5, 0.6, 0.3, 0.2)
    DIE   = SemanticVector(0.5, 0.8, 0.1, 0.8, 0.4, 0.3, 0.6)

    # 5. Evaluatives & Magnitudes (Mini vs. Huge)
    GOOD  = SemanticVector(0.2, 0.3, 1.0, 0.6, 0.3, 0.7, 0.5)
    BAD   = SemanticVector(0.2, 0.3, 0.0, 0.6, 0.4, 0.3, 0.5)
    HUGE  = SemanticVector(0.4, 0.1, 0.6, 0.95, 0.3, 0.1, 0.85)
    MINI  = SemanticVector(0.4, 0.1, 0.5, 0.10, 0.3, 0.1, 0.10)
    THIS  = SemanticVector(0.3, 0.0, 0.5, 0.3, 0.0, 0.0, 0.1)
    THAT  = SemanticVector(0.3, 0.0, 0.5, 0.3, 0.0, 0.0, 0.6)
    IN    = SemanticVector(0.1, 0.0, 0.5, 0.2, 0.0, 0.0, 0.1)
    ABOVE = SemanticVector(0.2, 0.0, 0.5, 0.6, 0.0, 0.0, 0.5)
    NOT   = SemanticVector(0.0, 0.0, 0.3, 0.7, 0.0, 0.0, 0.1)
    ONE   = SemanticVector(0.1, 0.0, 0.5, 0.2, 0.0, 0.0, 0.1)
    MANY  = SemanticVector(0.1, 0.0, 0.5, 0.6, 0.0, 0.2, 0.6)

    BIOLOGICAL_COGNITIVE_SEEDS = [
        WATER, FIRE, SUN, MOON, EARTH, STONE, TREE,
        MOTHER, FATHER, PERSON, BEAST, SELF, OTHER,
        HEAD, BODY, EYE, HAND, HEART, BLOOD,
        MAKE, MOVE, SEE, HEAR, SPEAK, EAT, DIE,
        GOOD, BAD, HUGE, MINI, THIS, THAT, IN, ABOVE, NOT, ONE, MANY
    ]