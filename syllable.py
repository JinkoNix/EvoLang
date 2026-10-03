"""
syllable.py — High-Performance Syllabification and Natural Emergent Metrical Prominence.
Features:
- Weight-to-Stress Principle: Stress emerges naturally from physical acoustic mass (no hardcoded presets).
- Continuous Emergent Tonogenesis (F0 register from onset voicing; contour from coda decay).
- Minimal Word Constraint to prevent structural collapse.
"""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import Sequence

from phoneme import Phoneme, PhonemeKind
from word import Word
from energy import ArticulatoryEnergyModel, calculate_syllable_mass, _get_sonority


class StressPattern(Enum):
    NATURAL_WEIGHT = "natural_weight"  # Emerges from acoustic mass
    INITIAL = "initial"
    PENULTIMATE = "penultimate"
    FINAL = "final"
    LATIN = "latin"


class Tone(Enum):
    NONE = ""
    HIGH_LEVEL = "˥"
    MID_LEVEL = "˧"
    LOW_LEVEL = "˩"
    HIGH_RISING = "˧˥"
    LOW_RISING = "˩˧"
    HIGH_FALLING = "˥˩"
    LOW_FALLING = "˨˩"
    DIPPING = "˨˩˦"


class Syllable:
    __slots__ = ('onset', 'nucleus', 'coda', 'stressed', 'tone')

    def __init__(
        self,
        onset: Sequence[Phoneme] = (),
        nucleus: Sequence[Phoneme] | Phoneme | None = None,
        coda: Sequence[Phoneme] = (),
        stressed: bool = False,
        tone: Tone = Tone.NONE,
    ):
        self.onset: list[Phoneme] = list(onset) if not isinstance(onset, list) else onset
        if nucleus is None:
            self.nucleus: list[Phoneme] = []
        elif isinstance(nucleus, list):
            self.nucleus = nucleus
        elif isinstance(nucleus, tuple):
            self.nucleus = list(nucleus)
        else:
            self.nucleus = [nucleus]
        self.coda: list[Phoneme] = list(coda) if not isinstance(coda, list) else coda
        self.stressed: bool = stressed
        self.tone: Tone = tone

    @property
    def is_heavy(self) -> bool:
        if not self.nucleus:
            return False
        return len(self.nucleus) > 1 or len(self.coda) > 0

    def to_phonemes(self) -> list[Phoneme]:
        return self.onset + self.nucleus + self.coda

    @property
    def form(self) -> str:
        prefix = "ˈ" if (self.stressed and self.tone == Tone.NONE) else ""
        tone_suffix = self.tone.value if self.tone != Tone.NONE else ""
        
        if (
            len(self.nucleus) >= 2 
            and abs(self.nucleus[0].point[0] - self.nucleus[1].point[0]) < 0.60 
            and abs(self.nucleus[0].point[1] - self.nucleus[1].point[1]) < 0.60
        ):
            nuc_str = f"{self.nucleus[0].ipa}ː"
        else:
            nuc_str = "".join(p.ipa for p in self.nucleus)

        onset_str = "".join(p.ipa for p in self.onset)
        coda_str = "".join(p.ipa for p in self.coda)
        return prefix + onset_str + nuc_str + coda_str + tone_suffix

    def __repr__(self) -> str:
        return f"Syllable({self.form!r}, heavy={self.is_heavy}, stressed={self.stressed})"


class SyllableEngine:
    """Handles syllabification, Natural Weight-to-Stress prominence, and Diachronic Tonogenesis."""

    @staticmethod
    def resolve_vowel_sequence(phonemes: list[Phoneme]) -> list[Phoneme]:
        if len(phonemes) < 2:
            return phonemes

        resolved: list[Phoneme] = []
        i = 0
        n = len(phonemes)

        while i < n:
            p = phonemes[i]
            if p.kind == PhonemeKind.VOWEL and i + 1 < n and phonemes[i + 1].kind == PhonemeKind.VOWEL:
                next_p = phonemes[i + 1]
                dh = abs(p.point[0] - next_p.point[0])
                db = abs(p.point[1] - next_p.point[1])

                if dh < 0.60 and db < 0.60:
                    resolved.extend([p, p])
                    i += 2
                    while i < n and phonemes[i].kind == PhonemeKind.VOWEL and abs(p.point[0] - phonemes[i].point[0]) < 0.60 and abs(p.point[1] - phonemes[i].point[1]) < 0.60:
                        i += 1
                    continue

                resolved.append(p)
                i += 1
                continue

            resolved.append(p)
            i += 1

        return resolved

    @classmethod
    def _is_legal_onset(cls, cons_seq: list[Phoneme], syllable_complexity: float = 0.50) -> bool:
        if len(cons_seq) <= 1:
            return True

        if len(cons_seq) == 2:
            son1 = _get_sonority(cons_seq[0])
            son2 = _get_sonority(cons_seq[1])
            mn1 = int(round(cons_seq[0].point[1]))
            mn2 = int(round(cons_seq[1].point[1]))

            is_identical = (abs(cons_seq[0].point[0] - cons_seq[1].point[0]) < 0.6 and mn1 == mn2)
            is_sibilant_lead = (mn1 in (4, 5) and mn2 in (0, 2) and syllable_complexity >= 0.35)
            min_son_distance = 3.60 - (2.20 * syllable_complexity)
            delta_son = son2 - son1

            if is_identical or (delta_son < min_son_distance and not is_sibilant_lead):
                return False
            return True

        return len(cons_seq) <= 3 and syllable_complexity >= 0.65

    @classmethod
    def syllabify(cls, word: Word, syllable_complexity: float = 0.50) -> list[Syllable]:
        if not word.phonemes:
            return []

        raw_phonemes = cls.resolve_vowel_sequence(list(word.phonemes))
        n = len(raw_phonemes)

        nuclei_ranges: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if raw_phonemes[i].kind == PhonemeKind.VOWEL:
                start = i
                if i + 1 < n and raw_phonemes[i + 1].kind == PhonemeKind.VOWEL:
                    p1, p2 = raw_phonemes[i].point, raw_phonemes[i+1].point
                    is_diph = (abs(p1[0] - p2[0]) < 0.60 and abs(p1[1] - p2[1]) < 0.60) or (p2[0] > p1[0] and p2[0] >= 3.8)
                    end = i + 2 if is_diph else i + 1
                    i = end
                else:
                    end = i + 1
                    i += 1
                nuclei_ranges.append((start, end))
            else:
                i += 1

        if not nuclei_ranges:
            return [Syllable(onset=raw_phonemes)]

        syllables: list[Syllable] = []
        first_start, first_end = nuclei_ranges[0]
        syllables.append(Syllable(onset=raw_phonemes[:first_start], nucleus=raw_phonemes[first_start:first_end]))

        for k in range(1, len(nuclei_ranges)):
            prev_end = nuclei_ranges[k - 1][1]
            curr_start, curr_end = nuclei_ranges[k]
            intervocalic = raw_phonemes[prev_end:curr_start]

            if len(intervocalic) == 0:
                coda, onset = [], []
            elif len(intervocalic) == 1:
                coda, onset = [], intervocalic
            else:
                split_idx = 0
                while split_idx < len(intervocalic):
                    if cls._is_legal_onset(intervocalic[split_idx:], syllable_complexity=syllable_complexity):
                        break
                    split_idx += 1
                split_idx = min(split_idx, len(intervocalic) - 1)
                coda = intervocalic[:split_idx]
                onset = intervocalic[split_idx:]

            syllables[-1].coda = coda
            syllables.append(Syllable(onset=onset, nucleus=raw_phonemes[curr_start:curr_end]))

        trailing = raw_phonemes[nuclei_ranges[-1][1]:]
        if trailing:
            syllables[-1].coda = list(trailing)

        return syllables

    @classmethod
    def assign_stress(cls, syllables: list[Syllable], pattern: StressPattern = StressPattern.NATURAL_WEIGHT) -> None:
        """
        Assigns primary stress:
        By default, stress emerges naturally on the syllable with highest physical acoustic mass.
        """
        if not syllables: return
        for s in syllables: s.stressed = False
        num_syl = len(syllables)
        if num_syl == 1:
            syllables[0].stressed = True
            return

        if pattern == StressPattern.NATURAL_WEIGHT:
            # Physical Weight-to-Stress: Syllable with maximal acoustic mass wins prominence
            masses = [calculate_syllable_mass(s.to_phonemes()) for s in syllables]
            max_m = max(masses)
            # If equal mass, fall back to natural boundary demarcation (penultimate)
            if masses.count(max_m) == len(masses):
                syllables[num_syl - 2].stressed = True
            else:
                winning_idx = masses.index(max_m)
                syllables[winning_idx].stressed = True

        elif pattern == StressPattern.INITIAL:
            syllables[0].stressed = True
        elif pattern == StressPattern.FINAL:
            syllables[-1].stressed = True
        elif pattern == StressPattern.PENULTIMATE:
            syllables[-2].stressed = True
        elif pattern == StressPattern.LATIN:
            penult_idx = num_syl - 2
            if syllables[penult_idx].is_heavy or num_syl == 2:
                syllables[penult_idx].stressed = True
            else:
                syllables[max(0, penult_idx - 1)].stressed = True

    @classmethod
    def reduce_and_prosodify(
        cls,
        word: Word,
        pattern: StressPattern = StressPattern.NATURAL_WEIGHT,
        reduction_strength: float = 0.50,
        apocope_rate: float = 0.50,
        syncope_rate: float = 0.50,
        reduction_mode: str = "inventory_snap",
        environment=None,
    ) -> Word:
        s_complex = getattr(environment, "syllable_complexity", 0.50) if environment else 0.50
        syllables = cls.syllabify(word, syllable_complexity=s_complex)
        if not syllables:
            return word.copy()

        cls.assign_stress(syllables, pattern)
        num_syl = len(syllables)
        can_reduce_structure = (num_syl >= 2)

        pop = getattr(environment, "population", 0.50) if environment else 0.50
        alt = getattr(environment, "altitude", 0.0) if environment else 0.0
        environmental_tempo = 0.20 + (pop * 0.30) + (alt * 0.15)

        processed_syllables: list[Syllable] = []

        for idx, syl in enumerate(syllables):
            if not syl.nucleus:
                processed_syllables.append(syl)
                continue

            is_final = (idx == num_syl - 1)
            is_medial = (0 < idx < num_syl - 1)
            nucleus = list(syl.nucleus)

            # Centralization & Syncope
            if not syl.stressed and reduction_mode != "none" and len(nucleus) == 1:
                v = nucleus[0]
                dist_to_schwa = math.sqrt(((v.point[0] - 3.0) / 3.0) ** 2 + ((v.point[1] - 1.0) / 1.0) ** 2)

                # Stage 1: Centralization
                rho = min(0.40, reduction_strength * environmental_tempo)
                h_new = v.point[0] + rho * (3.0 - v.point[0])
                b_new = v.point[1] + rho * (1.0 - v.point[1])
                nucleus = [Phoneme(PhonemeKind.VOWEL, (h_new, b_new, v.point[2]), layers=v.layers)]

                # Stage 2: Deletion conditioned on prior reduction
                drop_prob = reduction_strength * environmental_tempo * 0.35
                if can_reduce_structure and dist_to_schwa < 0.38:
                    should_drop = (is_final and random.random() < drop_prob * apocope_rate) or \
                                  (is_medial and random.random() < drop_prob * syncope_rate)
                    if should_drop:
                        if processed_syllables:
                            processed_syllables[-1].coda.extend(syl.onset + syl.coda)
                        elif idx + 1 < num_syl:
                            syllables[idx + 1].onset = syl.onset + syl.coda + syllables[idx + 1].onset
                        continue

            # Diachronic Tonogenesis
            if reduction_mode != "none":
                has_glottal_coda = any(c.ipa == "ʔ" or (c.point[0] >= 9.5 and int(round(c.point[1])) == 0) for c in syl.coda)
                has_laryngeal_coda = any(int(round(c.point[1])) in (4, 5) or c.ipa in ("s", "h", "x") for c in syl.coda)

                if syl.onset and syl.onset[0].kind == PhonemeKind.CONSONANT:
                    register_high = (int(round(syl.onset[0].point[2])) == 0)
                else:
                    register_high = True

                if has_glottal_coda:
                    assigned_tone = Tone.HIGH_RISING if register_high else Tone.LOW_RISING
                    retained_coda = [c for c in syl.coda if not (c.ipa == "ʔ" or c.point[0] >= 9.5)]
                elif has_laryngeal_coda and random.random() < 0.35:
                    assigned_tone = Tone.HIGH_FALLING if register_high else Tone.LOW_FALLING
                    retained_coda = [c for c in syl.coda if c.ipa not in ("h", "x")]
                else:
                    assigned_tone = getattr(syl, "tone", Tone.NONE)
                    retained_coda = list(syl.coda)
            else:
                assigned_tone = getattr(syl, "tone", Tone.NONE)
                retained_coda = list(syl.coda)

            processed_syllables.append(
                Syllable(onset=syl.onset, nucleus=nucleus, coda=retained_coda, stressed=syl.stressed, tone=assigned_tone)
            )

        reconstructed = []
        for s in processed_syllables: reconstructed.extend(s.to_phonemes())
        cleaned = ArticulatoryEnergyModel.clean_clusters_and_degeminate(reconstructed, syllable_complexity=s_complex)

        if len(cleaned) != len(reconstructed):
            temp_word = Word(vector=word.vector, phonemes=cleaned, is_ephemeral=True)
            processed_syllables = cls.syllabify(temp_word, syllable_complexity=s_complex)
            cls.assign_stress(processed_syllables, pattern)

        out_morphemes = [m.copy() for m in word.morphemes]
        n_clean = len(cleaned)
        if len(out_morphemes) == 1:
            out_morphemes[0].phonemes = list(cleaned)
        elif len(out_morphemes) > 1 and n_clean > 0:
            total_old = sum(len(m.phonemes) for m in out_morphemes) or 1
            cursor = 0
            for m_idx, m in enumerate(out_morphemes):
                ratio = len(m.phonemes) / total_old
                m_len = max(1, int(round(n_clean * ratio))) if m_idx < len(out_morphemes) - 1 else (n_clean - cursor)
                m.phonemes = list(cleaned[cursor:cursor + m_len])
                cursor += m_len

        return Word(
            word_id=word.id,
            vector=word.vector,
            phonemes=cleaned,
            syllables=processed_syllables,
            morphemes=out_morphemes,
            derivation=word.derivation,
            usage_frequency=word.usage_frequency,
            category=word.category,
            generation_born=word.generation_born,
            senses=word.senses,
        )

    @classmethod
    def apply_prosody(cls, word: Word, pattern: StressPattern = StressPattern.NATURAL_WEIGHT, **kwargs) -> Word:
        return cls.reduce_and_prosodify(word=word, pattern=pattern, reduction_strength=0.0, reduction_mode="none")