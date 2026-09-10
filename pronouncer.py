"""
pronouncer.py — High-Performance Universal IPA Speech Synthesizer for EvoLang.
Features multi-dialect phonetic transpilation (Latin/Spanish/English voices),
full secondary articulation mapping (ʷ, ʲ, ˁ, ʰ, ʱ), click synthesis, and WAV export.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import unicodedata

# =====================================================================
# Comprehensive IPA-to-eSpeak Phonetic Translation Mapping
# =====================================================================

_PHONETIC_MAP = {
    # 1. Multi-Character Sequences & Affricates
    't͡s': 'ts', 'tʃ': 'tS', 'd͡z': 'dz', 'dʒ': 'dZ',
    'k͜x': 'kx', 'ɡ͜ɣ': 'g', 't͡ɬ': 'tl',

    # 2. Clicks (Transpiled to Crisp Acoustic Equivalents)
    'k͜ǃ': 'k!', 'ɡ͜ǃ': 'g!', 'ŋ͜ǃ': 'n!',
    'k͜ʘ': 'p!', 'ɡ͜ʘ': 'b!', 'ŋ͜ʘ': 'm!',
    'k͜ǂ': 'c!', 'ɡ͜ǂ': 'ɟ!', 'ŋ͜ǂ': 'ɲ!',
    'k͜ǁ': 'kl!', 'ɡ͜ǁ': 'gl!', 'ŋ͜ǁ': 'nl!',
    'k͜ǀ': 't!', 'ɡ͜ǀ': 'd!', 'ŋ͜ǀ': 'n!',

    # 3. Implosives (Voiced & Voiceless)
    'ɠ̊': 'k', 'ɠ': 'g', 'ɗ̥': 't', 'ɗ': 'd', 'ɗ̪̊': 't', 'ɗ̪': 'd',
    'ɓ̥': 'p', 'ɓ': 'b', 'ᶑ': 'd', 'ʄ': 'j', 'ʛ': 'g',

    # 4. Secondary Articulation Superscripts (Crucial for Clean Speech!)
    'ʷ': 'w',   # Labialization: kʷ -> kw, tʷ -> tw
    'ʲ': 'j',   # Palatalization: tʲ -> tj, dʲ -> dj
    'ˁ': 'Q',   # Pharyngealization: tˁ -> tQ (guttural coloring)
    'ˠ': 'G',   # Velarization: lˠ -> lG
    'ʰ': '_h',  # Aspiration: pʰ -> p_h
    'ʱ': '_h',  # Breathy voice: bʱ -> b_h
    'ʼ': '',    # Ejective release mark (e.g. kʼ -> k)

    # 5. Vowels (Universal Cardinal Triangle & Centrals)
    'i': 'i', 'y': 'y', 'ɨ': 'I', 'ʉ': 'u', 'ɯ': 'u', 'u': 'u',
    'ɪ': 'I', 'ʏ': 'Y', 'ʊ': 'U', 'ᵻ': 'I', 'ᵿ': 'U',
    'e': 'e', 'ø': 'e', 'ɘ': '@', 'ɵ': 'o', 'ɤ': 'o', 'o': 'o',
    'ə': '@', 'ɛ': 'E', 'œ': 'E', 'ɜ': '3', 'ʌ': 'V', 'ɔ': 'O',
    'æ': 'a', 'ɐ': 'a', 'a': 'a', 'ä': 'a', 'ɑ': 'A', 'ɒ': 'O',

    # 6. Consonants
    'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'ʈ': 't', 'ɖ': 'd',
    'c': 'c', 'ɟ': 'j', 'k': 'k', 'ɡ': 'g', 'g': 'g', 'q': 'k',
    'ɢ': 'g', 'ʔ': '?', 'ʡ': '?', 'ɸ': 'f', 'β': 'b', 'f': 'f',
    'v': 'v', 'θ': 'T', 'ð': 'D', 's': 's', 'z': 'z', 'ʃ': 'S',
    'ʒ': 'Z', 'ʂ': 's', 'ʐ': 'z', 'ç': 'C', 'ʝ': 'j', 'x': 'x',
    'ɣ': 'g', 'χ': 'x', 'ʁ': 'r', 'ħ': 'x', 'ʕ': '?', 'h': 'h',
    'ɦ': 'h', 'ɬ': 's', 'ɮ': 'z', 'm': 'm', 'ɱ': 'm', 'n': 'n',
    'ɳ': 'n', 'ɲ': 'n', 'ŋ': 'N', 'ɴ': 'N', 'r': 'r', 'ɾ': 'r',
    'ɽ': 'r', 'ʀ': 'r', 'ⱱ': 'v', 'l': 'l', 'ɭ': 'l', 'ʎ': 'l',
    'ʟ': 'l', 'w': 'w', 'j': 'j', 'ʋ': 'v', 'ɹ': 'r', 'ɰ': 'w',

    # 7. Suprasegmentals & Stress
    'ˈ': "'", 'ː': ':', 'ˑ': ':',
}

# Master regex compiled with longest tokens first
_MASTER_PATTERN = re.compile("|".join(re.escape(k) for k in sorted(_PHONETIC_MAP.keys(), key=len, reverse=True)))


class IPANormalizer:
    """Translates EvoLang Unicode IPA into clean phoneme tokens for eSpeak."""

    @classmethod
    def to_espeak_phonemes(cls, ipa_string: str) -> str:
        # 1. Decompose Unicode to NFD
        text = unicodedata.normalize('NFD', str(ipa_string))

        # 2. Strip brackets and punctuation WITHOUT deleting constituent content
        text = re.sub(r'[/\[\]\(\)\.\-\+\u0361\u035c\u2040˥˦˧˨˩«»]', '', text)

        # 3. Handle Nasal Vowels (Ṽ -> Vn: ã -> an, õ -> on, ẽ -> en)
        text = re.sub(r'([a-zA-Z\u0080-\u02AF])\u0303', r'\1n', text)

        # 4. Master Single-Pass Phonetic Translation
        text = _MASTER_PATTERN.sub(lambda m: _PHONETIC_MAP[m.group(0)], text)

        # 5. Clean residual unmapped diacritics
        text = re.sub(r'[\u0300-\u036F]', '', text)
        text = re.sub(r"([ptkbdgqcɟ])ʼ", r"\1", text)
        text = text.replace('ʼ', '').replace('’', '')

        return re.sub(r'\s+', ' ', text).strip()


class Pronouncer:
    """Universal IPA speech synthesizer supporting Latin/Spanish/English vocal engines."""

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def find_espeak_binary() -> str:
        path = shutil.which("espeak-ng") or shutil.which("espeak")
        if path:
            return path

        candidates = [
            r"C:\Program Files\eSpeak NG\espeak-ng.exe",
            r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
            r"C:\Program Files\eSpeak\command_line\espeak.exe",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p

        raise FileNotFoundError("Could not locate 'espeak-ng.exe'. Ensure eSpeak is installed.")

    @classmethod
    def speak(
        cls,
        target: str | object,
        lang=None,
        speed: int = 120,
        pitch: int = 50,
        voice: str = "la",  # Default to 'la' (Latin) for pure cardinal vowels, with fallback
    ) -> str:
        """
        Speaks any IPA string, Word, NounPhrase, or Clause in real-time.
        Uses pure cardinal Latin/Spanish phoneme engine for natural conlang acoustics.
        """
        if hasattr(target, "render"):
            if lang is None:
                raise ValueError("Must provide `lang` when speaking a Clause/NounPhrase.")
            raw_ipa, _ = target.render(lang)
        elif hasattr(target, "form"):
            raw_ipa = target.form
        elif isinstance(target, tuple) and len(target) == 2:
            raw_ipa = target[0]
        else:
            raw_ipa = str(target)

        phoneme_str = IPANormalizer.to_espeak_phonemes(raw_ipa)
        if not phoneme_str:
            return ""

        try:
            espeak_bin = cls.find_espeak_binary()
        except FileNotFoundError:
            return phoneme_str

        # Primary command with pure vowel voice (Latin 'la' or Spanish 'es')
        cmd = [
            espeak_bin,
            "-v", voice,
            "-s", str(speed),
            "-p", str(pitch),
        ]

        payload = f"[[{phoneme_str}]]"
        try:
            subprocess.run(cmd, input=payload, text=True, encoding="utf-8", check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to standard English voice if Latin voice is unavailable
            cmd[2] = "en-us"
            cmd[-1] = phoneme_str
            try:
                subprocess.run(cmd, check=False)
            except Exception:
                pass

        return phoneme_str

    @classmethod
    def save_wav(
        cls,
        target: str | object,
        output_filepath: str,
        lang=None,
        speed: int = 120,
        pitch: int = 50,
        voice: str = "la",
    ) -> str:
        """Exports spoken speech directly to a WAV audio file on disk."""
        if hasattr(target, "render"):
            raw_ipa, _ = target.render(lang)
        elif hasattr(target, "form"):
            raw_ipa = target.form
        else:
            raw_ipa = str(target)

        phoneme_str = IPANormalizer.to_espeak_phonemes(raw_ipa)
        if not phoneme_str:
            return ""

        espeak_bin = cls.find_espeak_binary()
        cmd = [
            espeak_bin,
            "-v", voice,
            "-s", str(speed),
            "-p", str(pitch),
            "-w", output_filepath,
        ]

        payload = f"[[{phoneme_str}]]"
        subprocess.run(cmd, input=payload, text=True, encoding="utf-8", check=True)
        return output_filepath