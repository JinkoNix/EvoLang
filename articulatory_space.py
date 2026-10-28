"""
articulatory_space.py — Flat Array IPA Mapping & Suprasegmental Projection.
"""

from __future__ import annotations

from enum import Enum

LONG_VOWEL_THRESHOLD = 1.4
TONE_CONTOUR_THRESHOLD = 0.40
TONE_REGISTER_THRESHOLD = 0.40


class Airstream(Enum):
    PULMONIC = 0
    IMPLOSIVE = 1
    EJECTIVE = -1
    CLICK = 2


class SecondaryArticulation(Enum):
    NONE = ""
    LABIALIZED = "ʷ"
    PALATALIZED = "ʲ"
    VELARIZED = "ˠ"
    PHARYNGEALIZED = "ˁ"
    ASPIRATED = "ʰ"
    BREATHY = "ʱ"
    NASALIZED = "\u0303"

    def __str__(self) -> str:
        return self.value


_FLAT_VOWEL_ARRAY = [
    # 0: Open (Central b=1 is standard clean /a/; front b=0 is /æ/)
    'æ', 'ɶ', 'a', 'ɒ̈', 'ɑ', 'ɒ',
    # 1: Near-Open
    'æ', 'œ̞', 'ɐ', 'ɐ̹', 'ʌ̞', 'ɔ̞',
    # 2: Open-Mid
    'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
    # 3: Mid
    'e̞', 'ø̞', 'ə', 'ə̹', 'ɤ̞', 'o̞',
    # 4: Close-Mid
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
    # 5: Near-Close
    'ɪ', 'ʏ', 'ᵻ', 'ᵿ', 'ɯ̽', 'ʊ',
    # 6: Close
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u'
]

_FLAT_CONSONANT_ARRAY = [
    'p', 'b', 'p̪', 'b̪', 't̪', 'd̪', 't', 'd', 't̠', 'd̠', 'ʈ', 'ɖ', 'c', 'ɟ', 'k', 'ɡ', 'q', 'ɢ', 'ʡ', 'ʡ', 'ʔ', 'ʔ',
    'm̥', 'm', 'ɱ̊', 'ɱ', 'n̪̊', 'n̪', 'n̥', 'n', 'n̠̊', 'n̠', 'ɳ̊', 'ɳ', 'ɲ̊', 'ɲ', 'ŋ̊', 'ŋ', 'ɴ̥', 'ɴ', 'ɴ', 'ɴ', 'ʔ̃', 'ʔ̃',
    'p͡ɸ', 'b͡β', 'p̪͡f', 'b̪͡v', 't̪͡θ', 'd̪͡ð', 't͡s', 'd͡z', 't͡ʃ', 'd͡ʒ', 'ʈ͡ʂ', 'ɖ͡ʐ', 'c͜ç', 'ɟ͜ʝ', 'k͜x', 'ɡ͜ɣ', 'q͜χ', 'ɢ͜ʁ', 'ʡ͡ʢ', 'ʡ͡ʢ', 'ʔ͜h', 'ʔ͜h',
    'ʙ̥', 'ʙ', 'ⱱ̥', 'ⱱ', 'r̪̥', 'r̪', 'r̥', 'r', 'ɾ̠̊', 'ɾ̠', 'ɽ̊', 'ɽ', 'ɟ̆', 'ɟ̆', 'ɡ̆', 'ɡ̆', 'ʀ̥', 'ʀ', 'ʡ̆', 'ʡ̆', 'ʔ', 'ʔ',
    'ɸ', 'β', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'ʂ', 'ʐ', 'ç', 'ʝ', 'x', 'ɣ', 'χ', 'ʁ', 'ħ', 'ʕ', 'h', 'ɦ',
    'ɸ', 'β', 'f', 'v', 'ɬ̪', 'ɮ̪', 'ɬ', 'ɮ', 'ɬ̠', 'ɮ̠', 'ɭ̥˔', 'ɭ˔', 'ʎ̥˔', 'ʎ̝', 'ʟ̥˔', 'ʟ̝', 'χ', 'ʁ', 'ħ', 'ʕ', 'h', 'ɦ',
    'β̞', 'β̞', 'ʋ̥', 'ʋ', 'ð̞', 'ð̞', 'ɹ̥', 'ɹ', 'ɹ̠̥', 'ɹ̠', 'ɻ̊', 'ɻ', 'j̊', 'j', 'ɰ̊', 'ɰ', 'ʁ̞', 'ʁ̞', 'ʕ̞', 'ʕ̞', 'h', 'ɦ',
    'w', 'w', 'ʋ', 'ʋ', 'l̪̥', 'l̪', 'l̥', 'l', 'l̠̥', 'l̠', 'ɭ̊', 'ɭ', 'ʎ̥', 'ʎ', 'ʟ̥', 'ʟ', 'ʟ̠̥', 'ʟ̠', 'ʕ̞', 'ʕ̞', 'h', 'ɦ',
]

_IMPLOSIVES_TABLE = {
    0: ('ɓ̥', 'ɓ'), 2: ('ɗ̪̊', 'ɗ̪'), 3: ('ɗ̥', 'ɗ'), 5: ('ᶑ̥', 'ᶑ'), 6: ('ʄ̥', 'ʄ'), 7: ('ɠ̊', 'ɠ'), 8: ('ʛ̥', 'ʛ'),
}

_CLICK_BASES = {
    0: 'ʘ', 1: 'ǀ', 2: 'ǀ', 3: 'ǃ', 4: 'ǃ', 5: 'ǂ', 6: 'ǂ', 7: 'ǃ', 8: 'ǃ', 9: 'ǃ', 10: 'ǃ'
}


def render_suprasegmentals(base_ipa: str, duration: float, pitch_offset: float, pitch_contour: float) -> str:
    length_mark = "ː" if duration >= LONG_VOWEL_THRESHOLD else ""

    tone_mark = ""
    if pitch_contour >= TONE_CONTOUR_THRESHOLD:
        tone_mark = "˧˥" if pitch_offset >= 0.0 else "˩˧"
    elif pitch_contour <= -TONE_CONTOUR_THRESHOLD:
        tone_mark = "˥˩" if pitch_offset >= 0.0 else "˨˩"
    elif pitch_offset >= TONE_REGISTER_THRESHOLD:
        tone_mark = "˥"
    elif pitch_offset <= -TONE_REGISTER_THRESHOLD:
        tone_mark = "˩"

    return f"{base_ipa}{length_mark}{tone_mark}"


def _lookup_vowel(hi: int, bk: int, rn: int, layers: tuple[Enum, ...]) -> str:
    idx = (hi * 6) + (bk * 2) + rn
    symbol = _FLAT_VOWEL_ARRAY[idx]
    for layer in layers:
        if isinstance(layer, SecondaryArticulation) and layer != SecondaryArticulation.NONE:
            symbol = f"{symbol}{layer.value}"
    return symbol


def _lookup_consonant(pl: int, mn: int, vc: int, layers: tuple[Enum, ...]) -> str:
    symbol = _FLAT_CONSONANT_ARRAY[(mn * 22) + (pl * 2) + vc]

    if layers:
        for layer in layers:
            if layer == Airstream.EJECTIVE:
                base_vl = _FLAT_CONSONANT_ARRAY[(mn * 22) + (pl * 2)]
                symbol = f"{base_vl}ʼ"
            elif layer == Airstream.IMPLOSIVE:
                pair = _IMPLOSIVES_TABLE.get(pl)
                symbol = pair[vc] if pair else f"{symbol}ʼ"
            elif layer == Airstream.CLICK:
                click_rel = 'ǁ' if mn in (5, 7) else _CLICK_BASES.get(pl, 'ǃ')
                symbol = f"ŋ͜{click_rel}" if mn == 1 else (f"ɡ͜{click_rel}" if vc == 1 else f"k͜{click_rel}")

        for layer in layers:
            if isinstance(layer, SecondaryArticulation) and layer != SecondaryArticulation.NONE:
                symbol = f"{symbol}{layer.value}"

    return symbol


def resolve_vowel(height: float, backness: float, roundedness: float, layers: tuple[Enum, ...] = ()) -> str:
    hi = max(0, min(6, int(round(height))))
    bk = max(0, min(2, int(round(backness))))
    rn = max(0, min(1, int(round(roundedness))))
    return _lookup_vowel(hi, bk, rn, layers)


def resolve_consonant(place: float, manner: float, voiced: float, layers: tuple[Enum, ...] = ()) -> str:
    pl = max(0, min(10, int(round(place))))
    mn = max(0, min(7, int(round(manner))))
    vc = max(0, min(1, int(round(voiced))))
    return _lookup_consonant(pl, mn, vc, layers)