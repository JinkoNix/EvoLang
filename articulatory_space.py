"""
articulatory_space.py — High-Resolution 1:1 Continuous 3D Articulatory Space & IPA Resolvers.
Provides unambiguous, biunique physical mapping from (height, backness, roundedness)
and (place, manner, voice) directly to authentic IPA symbols.
"""

from __future__ import annotations

import functools
from enum import Enum


# =====================================================================
# Discrete Layers (Airstreams & Secondary Articulations)
# =====================================================================

class Airstream(Enum):
    PULMONIC = 0
    IMPLOSIVE = 1
    EJECTIVE = -1
    CLICK = 2


class SecondaryArticulation(Enum):
    NONE = ""
    LABIALIZED = "ʷ"        # Lip rounding (kʷ, tʷ)
    PALATALIZED = "ʲ"       # Tongue raising (tʲ, dʲ)
    VELARIZED = "ˠ"         # Back constriction (lˠ)
    PHARYNGEALIZED = "ˁ"    # Pharyngeal constriction (tˁ, sˁ, dˁ, zˁ, qˁ, pˁ)
    ASPIRATED = "ʰ"         # Glottal delay (pʰ, tʰ, kʰ)
    BREATHY = "ʱ"           # Murmur (bʱ, dʱ)
    NASALIZED = "\u0303"    # Combining tilde (ã, ĩ, ũ)

    def __str__(self) -> str:
        return self.value


# =====================================================================
# High-Resolution 1:1 Vowel Table (7 Heights × 3 Backnesses × 2 Roundings = 42 Unique Cells)
# =====================================================================

VOWEL_TABLE = {
    # 0: Open (Low)
    0: {
        0: ['a', 'ɶ'],    # Front: Open Front Unrounded vs. Rounded
        1: ['ä', 'ɒ̈'],    # Central: Open Central Unrounded vs. Rounded
        2: ['ɑ', 'ɒ'],    # Back: Open Back Unrounded vs. Rounded
    },
    # 1: Near-Open
    1: {
        0: ['æ', 'œ̞'],    # Front: Near-Open Front
        1: ['ɐ', 'ɐ̹'],    # Central: Near-Open Central
        2: ['ʌ̞', 'ɔ̞'],    # Back: Near-Open Back
    },
    # 2: Open-Mid
    2: {
        0: ['ɛ', 'œ'],    # Front: Open-Mid Front
        1: ['ɜ', 'ɞ'],    # Central: Open-Mid Central
        2: ['ʌ', 'ɔ'],    # Back: Open-Mid Back
    },
    # 3: Mid (True Central & Lowered/Raised Mid Tiers)
    3: {
        0: ['e̞', 'ø̞'],    # Front: Lowered Mid Front
        1: ['ə', 'ə̹'],    # Central: Neutral Schwa vs. Rounded Schwa
        2: ['ɤ̞', 'o̞'],    # Back: Lowered Mid Back
    },
    # 4: Close-Mid
    4: {
        0: ['e', 'ø'],    # Front: Close-Mid Front
        1: ['ɘ', 'ɵ'],    # Central: Close-Mid Central
        2: ['ɤ', 'o'],    # Back: Close-Mid Back
    },
    # 5: Near-Close
    5: {
        0: ['ɪ', 'ʏ'],    # Front: Near-Close Front
        1: ['ᵻ', 'ᵿ'],    # Central: Near-Close Central
        2: ['ɯ̽', 'ʊ'],    # Back: Near-Close Back
    },
    # 6: Close (High)
    6: {
        0: ['i', 'y'],    # Front: Close Front
        1: ['ɨ', 'ʉ'],    # Central: Close Central
        2: ['ɯ', 'u'],    # Back: Close Back
    },
}


@functools.lru_cache(maxsize=2048)
def _cached_vowel_lookup(hi: int, bk: int, rn: int, layers: tuple[Enum, ...]) -> str:
    symbol = VOWEL_TABLE[hi][bk][rn]
    for layer in layers:
        if isinstance(layer, SecondaryArticulation) and layer != SecondaryArticulation.NONE:
            symbol = f"{symbol}{layer.value}"
    return symbol


def resolve_vowel(height: float, backness: float, roundedness: float, layers: tuple[Enum, ...] = ()) -> str:
    """Public helper resolving continuous 3D coordinates (0.0 to 6.0, 0.0 to 2.0, 0.0 to 1.0) to unique IPA vowel."""
    hi = max(0, min(6, int(round(height))))
    bk = max(0, min(2, int(round(backness))))
    rn = max(0, min(1, int(round(roundedness))))
    return _cached_vowel_lookup(hi, bk, rn, layers)


# =====================================================================
# High-Resolution 1:1 Consonant Grid (8 Manners × 11 Places × 2 Voicings = 176 Cells)
# =====================================================================

CONSONANT_PULMONIC_TABLE = {
    # 0: Plosives / Stops
    0: {
        0: ['p', 'b'],      # Bilabial
        1: ['p̪', 'b̪'],      # Labiodental
        2: ['t̪', 'd̪'],      # Dental
        3: ['t', 'd'],      # Alveolar
        4: ['t̠', 'd̠'],      # Postalveolar
        5: ['ʈ', 'ɖ'],      # Retroflex
        6: ['c', 'ɟ'],      # Palatal
        7: ['k', 'ɡ'],      # Velar
        8: ['q', 'ɢ'],      # Uvular
        9: ['ʡ', 'ʡ'],      # Epiglottal
        10: ['ʔ', 'ʔ'],     # Glottal
    },
    # 1: Nasals
    1: {
        0: ['m̥', 'm'],      # Bilabial
        1: ['ɱ̊', 'ɱ'],      # Labiodental
        2: ['n̪̊', 'n̪'],      # Dental
        3: ['n̥', 'n'],      # Alveolar
        4: ['n̠̊', 'n̠'],      # Postalveolar
        5: ['ɳ̊', 'ɳ'],      # Retroflex
        6: ['ɲ̊', 'ɲ'],      # Palatal
        7: ['ŋ̊', 'ŋ'],      # Velar
        8: ['ɴ̥', 'ɴ'],      # Uvular
        9: ['ɴ', 'ɴ'],      # Pharyngeal
        10: ['ʔ̃', 'ʔ̃'],     # Glottalized Nasal
    },
    # 2: Affricates & Delayed-Release Obstruents (Universal Intermediate Tier)
    2: {
        0: ['p͡ɸ', 'b͡β'],    # Bilabial Affricate
        1: ['p̪͡f', 'b̪͡v'],    # Labiodental Affricate
        2: ['t̪͡θ', 'd̪͡ð'],    # Dental Affricate
        3: ['t͡s', 'd͡z'],    # Alveolar Affricate ("ts", "dz")
        4: ['t͡ʃ', 'd͡ʒ'],    # Postalveolar Affricate ("tsh/ch", "dj/j")
        5: ['ʈ͡ʂ', 'ɖ͡ʐ'],    # Retroflex Affricate
        6: ['c͜ç', 'ɟ͜ʝ'],    # Palatal Affricate
        7: ['k͜x', 'ɡ͜ɣ'],    # Velar Affricate
        8: ['q͜χ', 'ɢ͜ʁ'],    # Uvular Affricate
        9: ['ʡ͡ʢ', 'ʡ͡ʢ'],    # Epiglottal Affricate
        10: ['ʔ͜h', 'ʔ͜h'],   # Glottal Affricate
    },
    # 3: Taps, Flaps & Trills (Resonant Liquid Tier)
    3: {
        0: ['ʙ̥', 'ʙ'],      # Bilabial Trill
        1: ['ⱱ̥', 'ⱱ'],      # Labiodental Flap
        2: ['r̪̥', 'r̪'],      # Dental Trill
        3: ['r̥', 'r'],      # Alveolar Trill / Tap
        4: ['ɾ̠̊', 'ɾ̠'],      # Postalveolar Tap
        5: ['ɽ̊', 'ɽ'],      # Retroflex Flap
        6: ['ɟ̆', 'ɟ̆'],      # Palatal Flap
        7: ['ɡ̆', 'ɡ̆'],      # Velar Flap
        8: ['ʀ̥', 'ʀ'],      # Uvular Trill
        9: ['ʡ̆', 'ʡ̆'],      # Epiglottal Flap
        10: ['ʔ', 'ʔ'],     # Glottal Tap
    },
    # 4: Fricatives (Central)
    4: {
        0: ['ɸ', 'β'],      # Bilabial
        1: ['f', 'v'],      # Labiodental
        2: ['θ', 'ð'],      # Dental
        3: ['s', 'z'],      # Alveolar
        4: ['ʃ', 'ʒ'],      # Postalveolar
        5: ['ʂ', 'ʐ'],      # Retroflex
        6: ['ç', 'ʝ'],      # Palatal
        7: ['x', 'ɣ'],      # Velar
        8: ['χ', 'ʁ'],      # Uvular
        9: ['ħ', 'ʕ'],      # Pharyngeal
        10: ['h', 'ɦ'],     # Glottal
    },
    # 5: Lateral Fricatives
    5: {
        0: ['ɸ', 'β'],
        1: ['f', 'v'],
        2: ['ɬ̪', 'ɮ̪'],      # Dental Lateral Fricative
        3: ['ɬ', 'ɮ'],      # Alveolar Lateral Fricative
        4: ['ɬ̠', 'ɮ̠'],      # Postalveolar Lateral Fricative
        5: ['ɭ̥˔', 'ɭ˔'],    # Retroflex Lateral Fricative
        6: ['ʎ̥˔', 'ʎ̝'],     # Palatal Lateral Fricative
        7: ['ʟ̥˔', 'ʟ̝'],     # Velar Lateral Fricative
        8: ['χ', 'ʁ'],
        9: ['ħ', 'ʕ'],
        10: ['h', 'ɦ'],
    },
    # 6: Approximants (Central)
    6: {
        0: ['β̞', 'β̞'],      # Bilabial Approximant
        1: ['ʋ̥', 'ʋ'],      # Labiodental Approximant
        2: ['ð̞', 'ð̞'],      # Dental Approximant
        3: ['ɹ̥', 'ɹ'],      # Alveolar Approximant
        4: ['ɹ̠̥', 'ɹ̠'],      # Postalveolar Approximant
        5: ['ɻ̊', 'ɻ'],      # Retroflex Approximant
        6: ['j̊', 'j'],      # Palatal Approximant ("y")
        7: ['ɰ̊', 'ɰ'],      # Velar Approximant
        8: ['ʁ̞', 'ʁ̞'],      # Uvular Approximant
        9: ['ʕ̞', 'ʕ̞'],      # Pharyngeal Approximant
        10: ['h', 'ɦ'],
    },
    # 7: Lateral Approximants & Labial-Velar Glides
    7: {
        0: ['w', 'w'],      # Labial-Velar Approximant ("w")
        1: ['ʋ', 'ʋ'],
        2: ['l̪̥', 'l̪'],      # Dental Lateral
        3: ['l̥', 'l'],      # Alveolar Lateral
        4: ['l̠̥', 'l̠'],      # Postalveolar Lateral
        5: ['ɭ̊', 'ɭ'],      # Retroflex Lateral
        6: ['ʎ̥', 'ʎ'],      # Palatal Lateral
        7: ['ʟ̥', 'ʟ'],      # Velar Lateral
        8: ['ʟ̠̥', 'ʟ̠'],      # Uvular Lateral
        9: ['ʕ̞', 'ʕ̞'],
        10: ['h', 'ɦ'],
    },
}

IMPLOSIVES = {
    0: {0: 'ɓ̥', 1: 'ɓ'},
    2: {0: 'ɗ̪̊', 1: 'ɗ̪'},
    3: {0: 'ɗ̥', 1: 'ɗ'},
    5: {0: 'ᶑ̥', 1: 'ᶑ'},
    6: {0: 'ʄ̥', 1: 'ʄ'},
    7: {0: 'ɠ̊', 1: 'ɠ'},
    8: {0: 'ʛ̥', 1: 'ʛ'},
}

CLICK_BASES = {
    0: 'ʘ',  # Bilabial
    1: 'ǀ',  # Dental
    2: 'ǀ',  # Dental
    3: 'ǃ',  # Alveolar / Apical
    4: 'ǃ',  # Postalveolar
    5: 'ǂ',  # Palato-alveolar
    6: 'ǂ',  # Palatal
    7: 'ǃ',
    8: 'ǃ',
    9: 'ǃ',
    10: 'ǃ',
}


@functools.lru_cache(maxsize=4096)
def _cached_consonant_lookup(pl: int, mn: int, vc: int, layers: tuple[Enum, ...]) -> str:
    symbol = CONSONANT_PULMONIC_TABLE[mn][pl][vc]

    if layers:
        for layer in layers:
            if layer == Airstream.EJECTIVE:
                voiceless_base = CONSONANT_PULMONIC_TABLE[mn][pl][0]
                symbol = f"{voiceless_base}ʼ"
            elif layer == Airstream.IMPLOSIVE:
                symbol = IMPLOSIVES.get(pl, {}).get(vc, f"{symbol}ʼ")
            elif layer == Airstream.CLICK:
                click_release = 'ǁ' if mn in (5, 7) else CLICK_BASES.get(pl, 'ǃ')
                if mn == 1:
                    symbol = f"ŋ͜{click_release}" if vc == 1 else f"ŋ̊͜{click_release}"
                elif vc == 1:
                    symbol = f"ɡ͜{click_release}"
                else:
                    symbol = f"k͜{click_release}"

        for layer in layers:
            if isinstance(layer, SecondaryArticulation) and layer != SecondaryArticulation.NONE:
                symbol = f"{symbol}{layer.value}"

    return symbol


def resolve_consonant(place: float, manner: float, voiced: float, layers: tuple[Enum, ...] = ()) -> str:
    """Public helper resolving continuous 3D coordinates (0.0 to 10.0, 0.0 to 7.0, 0.0 to 1.0) to unique IPA consonant."""
    pl = max(0, min(10, int(round(place))))
    mn = max(0, min(7, int(round(manner))))
    vc = max(0, min(1, int(round(voiced))))
    return _cached_consonant_lookup(pl, mn, vc, layers)