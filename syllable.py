"""
syllable.py — Syllabification and Prosodic Reduction.
"""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import Sequence

from articulatory_space import (
    LONG_VOWEL_THRESHOLD,
    TONE_CONTOUR_THRESHOLD,
    TONE_REGISTER_THRESHOLD,
)
from phoneme import Phoneme, PhonemeKind
from word import Word, Morpheme
from energy import ArticulatoryEnergyModel, calculate_syllable_mass, _get_sonority


class StressPattern(Enum):
    NATURAL_WEIGHT = "natural_weight"
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
    __slots__ = ('onset', 'nucleus', 'coda', 'stressed')

    def __init__(
        self,
        onset: Sequence[Phoneme] = (),
        nucleus: Sequence[Phoneme] | Phoneme | None = None,
        coda: Sequence[Phoneme] = (),
        stressed: bool = False,
        tone: Tone = Tone.NONE,
    ):
        self.onset = list(onset)
        if nucleus is None:
            self.nucleus = []
        elif isinstance(nucleus, (list, tuple)):
            self.nucleus = list(nucleus)
        else:
            self.nucleus = [nucleus]
        self.coda = list(coda)
        self.stressed = stressed

    @property
    def tone(self) -> Tone:
        if not self.nucleus:
            return Tone.NONE
        v = self.nucleus[0]
        if v.pitch_contour >= TONE_CONTOUR_THRESHOLD:
            return Tone.HIGH_RISING if v.pitch_offset >= 0.0 else Tone.LOW_RISING
        elif v.pitch_contour <= -TONE_CONTOUR_THRESHOLD:
            return Tone.HIGH_FALLING if v.pitch_offset >= 0.0 else Tone.LOW_FALLING
        elif v.pitch_offset >= TONE_REGISTER_THRESHOLD:
            return Tone.HIGH_LEVEL
        elif v.pitch_offset <= -TONE_REGISTER_THRESHOLD:
            return Tone.LOW_LEVEL
        return Tone.NONE

    @property
    def is_heavy(self) -> bool:
        if not self.nucleus:
            return False
        return self.nucleus[0].duration >= LONG_VOWEL_THRESHOLD or len(self.nucleus) > 1 or len(self.coda) > 0

    def to_phonemes(self) -> list[Phoneme]:
        return self.onset + self.nucleus + self.coda

    @property
    def form(self) -> str:
        prefix = "ˈ" if self.stressed else ""
        return prefix + "".join(p.ipa for p in self.onset) + "".join(p.ipa for p in self.nucleus) + "".join(p.ipa for p in self.coda)

    def __repr__(self) -> str:
        return f"Syllable({self.form!r}, heavy={self.is_heavy}, stressed={self.stressed})"


class SyllableEngine:
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
                    resolved.append(p.drift(duration=min(2.2, p.duration + next_p.duration)))
                    i += 2
                    while i < n and phonemes[i].kind == PhonemeKind.VOWEL and abs(p.point[0] - phonemes[i].point[0]) < 0.60:
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

        if len(cons_seq) == 3 and syllable_complexity >= 0.65:
            mn1 = int(round(cons_seq[0].point[1]))
            mn2 = int(round(cons_seq[1].point[1]))
            son3 = _get_sonority(cons_seq[2])
            return (mn1 in (4, 5)) and (mn2 in (0, 2)) and (son3 >= 4.0)

        return False

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
        if not syllables:
            return
        for s in syllables:
            s.stressed = False

        num_syl = len(syllables)
        if num_syl == 1:
            syllables[0].stressed = True
            return

        if pattern == StressPattern.NATURAL_WEIGHT:
            masses = [calculate_syllable_mass(s.to_phonemes()) for s in syllables]
            max_m = max(masses)
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
        can_reduce = (num_syl >= 2)

        pop = getattr(environment, "population", 0.50) if environment else 0.50
        alt = getattr(environment, "altitude", 0.0) if environment else 0.0
        environmental_tempo = 0.20 + (pop * 0.30) + (alt * 0.15)

        processed_syllables: list[Syllable] = []
        pending_onset: list[Phoneme] = []

        for idx, syl in enumerate(syllables):
            current_onset = pending_onset + syl.onset
            pending_onset = []

            if not syl.nucleus:
                processed_syllables.append(Syllable(onset=current_onset, nucleus=[], coda=syl.coda, stressed=syl.stressed))
                continue

            is_final = (idx == num_syl - 1)
            is_medial = (0 < idx < num_syl - 1)
            is_initial = (idx == 0)
            nucleus = list(syl.nucleus)

            # 1. Centralization of unstressed vowels
            dist_to_schwa = 999.0
            if not syl.stressed and reduction_mode != "none" and len(nucleus) == 1:
                v = nucleus[0]
                dist_to_schwa = math.sqrt(((v.point[0] - 3.0) / 3.0) ** 2 + ((v.point[1] - 1.0) / 1.0) ** 2)

                rho = min(0.40, reduction_strength * environmental_tempo)
                h_new = v.point[0] + rho * (3.0 - v.point[0])
                b_new = v.point[1] + rho * (1.0 - v.point[1])
                new_dur = max(0.4, v.duration * (1.0 - rho * 0.5))

                nucleus = [v.drift(point=(h_new, b_new, v.point[2]), duration=new_dur)]

            # 2. Conditioned deletion / syncope
            drop_prob = reduction_strength * environmental_tempo * 0.35
            if can_reduce and dist_to_schwa < 0.38:
                should_drop = (
                    (is_final and random.random() < drop_prob * apocope_rate)
                    or (is_medial and random.random() < drop_prob * syncope_rate)
                    or (is_initial and random.random() < drop_prob * syncope_rate * 0.60)
                )
                if should_drop:
                    if is_initial and idx + 1 < num_syl:
                        next_onset = syllables[idx + 1].onset
                        merged_onset = current_onset + next_onset
                        # Gated syncope: verify legality without doubling next_onset
                        if not cls._is_legal_onset(merged_onset, syllable_complexity=s_complex):
                            if random.random() < 0.50:
                                # Friction blocks syncope: keep as reduced vowel
                                processed_syllables.append(Syllable(onset=current_onset, nucleus=nucleus, coda=syl.coda, stressed=False))
                                continue
                            else:
                                # Cluster reduction: drop the weak onset entirely
                                pending_onset = []
                                continue
                        else:
                            # Forward current_onset only; next iteration will add next_onset once
                            pending_onset = current_onset
                            continue

                    if processed_syllables:
                        processed_syllables[-1].coda.extend(current_onset + syl.coda)
                    else:
                        pending_onset = current_onset + syl.coda
                    continue

            processed_syllables.append(
                Syllable(onset=current_onset, nucleus=nucleus, coda=syl.coda, stressed=syl.stressed)
            )

        if pending_onset and processed_syllables:
            processed_syllables[0].onset = pending_onset + processed_syllables[0].onset

        reconstructed: list[Phoneme] = []
        for s in processed_syllables:
            reconstructed.extend(s.to_phonemes())

        cleaned = ArticulatoryEnergyModel.clean_clusters_and_degeminate(reconstructed, syllable_complexity=s_complex)

        if len(cleaned) != len(reconstructed):
            temp_word = Word(vector=word.vector, phonemes=cleaned, is_ephemeral=True)
            processed_syllables = cls.syllabify(temp_word, syllable_complexity=s_complex)
            cls.assign_stress(processed_syllables, pattern)

        out_morphemes = [m.copy() for m in word.morphemes]
        if len(out_morphemes) <= 1:
            out_morphemes = [Morpheme(cleaned, word.derivation.root_family_id)]
        else:
            total_old = sum(len(m.phonemes) for m in out_morphemes) or 1
            cursor = 0
            n_clean = len(cleaned)
            for m_idx, m in enumerate(out_morphemes):
                if m_idx == len(out_morphemes) - 1:
                    m.phonemes = list(cleaned[cursor:])
                else:
                    m_len = max(0, min(n_clean - cursor, int(round(n_clean * (len(m.phonemes) / total_old)))))
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