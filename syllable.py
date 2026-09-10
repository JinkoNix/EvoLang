"""
syllable.py — High-Performance Syllabification, Weight-Sensitive Latin Stress, and Continuous Prosody.
Features Sonority-Aware Maximal Onset Principle (MOP), Non-Amputating Syncope/Apocope,
Lindblom target-undershoot vowel reduction, and Haudricourt tonogenesis.
"""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import Sequence

from phoneme import Phoneme, PhonemeKind
from word import Word
from energy import ArticulatoryEnergyModel, _get_sonority


NEUTRAL_TRACT_CENTER = (3.0, 1.0, 0.0)  # Neutral barycenter /ə/


class StressPattern(Enum):
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
    """A single syllable constituent: Onset, Nucleus, Coda, Stress, and Tone."""

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
        """Evaluates syllable weight (bimoraic): long nucleus/diphthong or closed coda."""
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
    """Handles syllabification, Latin stress assignment, continuous vowel undershoot, and prosody."""

    @staticmethod
    def resolve_vowel_sequence(phonemes: list[Phoneme]) -> list[Phoneme]:
        """
        Normalizes vocalic sequences before syllabification:
        - Fuses identical adjacent vowel triples into a bimoraic long vowel (V1 V1 V1 -> V1 V1).
        - Preserves natural diphthongs and hiatus sequences.
        """
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

                # Identical Vowels -> Geminate Long Vowel (Max 2 moras)
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

    @staticmethod
    def is_diphthong_or_geminate(v1: Phoneme, v2: Phoneme) -> bool:
        """Determines if two adjacent vowels form a tautosyllabic diphthong or geminate long vowel."""
        p1 = v1.point
        p2 = v2.point
        h1, h2 = p1[0], p2[0]
        dh = abs(h1 - h2)
        db = abs(p1[1] - p2[1])

        # 1. Identical / Near-identical vowels -> Geminate Monophthong (Vː)
        if dh < 0.60 and db < 0.60:
            return True

        # 2. Closing Diphthongs (e.g. ai, au, ei, ou: rising in height to close target)
        if h2 > h1 and h2 >= 3.8:
            return True

        # 3. High Vowel Trajectory Glides (e.g. iu, ui)
        if h1 >= 4.0 and h2 >= 4.0 and db >= 0.8:
            return True

        # 4. Opening on-glide diphthongs (e.g. ia, ua, ie, uo)
        if h1 >= 4.5 and h2 <= 3.5:
            return True

        return False

    @classmethod
    def _is_legal_onset(cls, cons_seq: list[Phoneme], syllable_complexity: float = 0.50) -> bool:
        """
        Continuous Minimal Sonority Distance Onset Licensing:
        ΔSon_min = 3.6 - 2.2 * S_comp
        """
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

        if len(cons_seq) == 3:
            # Sibilant + Stop + Liquid (str-, skr-, spl-) licensed in high complexity
            if syllable_complexity < 0.65:
                return False
            is_sibilant = (int(round(cons_seq[0].point[1])) in (4, 5))
            is_stop = (int(round(cons_seq[1].point[1])) in (0, 2))
            is_liquid = (_get_sonority(cons_seq[2]) >= 4.0)
            return is_sibilant and is_stop and is_liquid

        return False

    @classmethod
    def syllabify(cls, word: Word, syllable_complexity: float = 0.50) -> list[Syllable]:
        """
        Syllabifies a word using the Universal Maximal Onset Principle (MOP)
        coupled to continuous environmental syllable complexity.
        """
        if not word.phonemes:
            return []

        raw_phonemes = cls.resolve_vowel_sequence(list(word.phonemes))
        n = len(raw_phonemes)

        # 1. Locate Nucleus Ranges
        nuclei_ranges: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if raw_phonemes[i].kind == PhonemeKind.VOWEL:
                start = i
                if i + 1 < n and raw_phonemes[i + 1].kind == PhonemeKind.VOWEL:
                    if cls.is_diphthong_or_geminate(raw_phonemes[i], raw_phonemes[i + 1]):
                        end = i + 2
                        i += 2
                    else:
                        end = i + 1
                        i += 1
                else:
                    end = i + 1
                    i += 1
                nuclei_ranges.append((start, end))
            else:
                i += 1

        if not nuclei_ranges:
            return [Syllable(onset=raw_phonemes)]

        syllables: list[Syllable] = []

        # 2. First Syllable Onset & Nucleus
        first_start, first_end = nuclei_ranges[0]
        syllables.append(
            Syllable(
                onset=raw_phonemes[:first_start],
                nucleus=raw_phonemes[first_start:first_end],
            )
        )

        # 3. Maximal Onset Intervocalic Allocation (Uses environmental S_comp!)
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
                    candidate_onset = intervocalic[split_idx:]
                    if cls._is_legal_onset(candidate_onset, syllable_complexity=syllable_complexity):
                        break
                    split_idx += 1

                split_idx = min(split_idx, len(intervocalic) - 1)
                coda = intervocalic[:split_idx]
                onset = intervocalic[split_idx:]

            syllables[-1].coda = coda
            syllables.append(
                Syllable(
                    onset=onset,
                    nucleus=raw_phonemes[curr_start:curr_end],
                )
            )

        # 4. Word-Final Coda
        last_end = nuclei_ranges[-1][1]
        trailing = raw_phonemes[last_end:]
        if trailing:
            syllables[-1].coda = list(trailing)

        return syllables

    @classmethod
    def assign_stress(
        cls,
        syllables: list[Syllable],
        pattern: StressPattern = StressPattern.PENULTIMATE,
    ) -> None:
        """Assigns primary stress based on metrical pattern and syllable weight."""
        if not syllables:
            return
        for s in syllables:
            s.stressed = False

        num_syl = len(syllables)
        if num_syl == 1:
            syllables[0].stressed = True
            return

        if pattern == StressPattern.INITIAL:
            syllables[0].stressed = True
        elif pattern == StressPattern.FINAL:
            syllables[-1].stressed = True
        elif pattern == StressPattern.PENULTIMATE:
            syllables[-2].stressed = True
        elif pattern == StressPattern.LATIN:
            penult_idx = num_syl - 2
            # Latin Stress: If penult is heavy or word is disyllabic -> stress penult; else antepenult
            if syllables[penult_idx].is_heavy or num_syl == 2:
                syllables[penult_idx].stressed = True
            else:
                syllables[max(0, penult_idx - 1)].stressed = True

    @classmethod
    def reduce_and_prosodify(
        cls,
        word: Word,
        pattern: StressPattern = StressPattern.PENULTIMATE,
        reduction_strength: float = 0.50,
        apocope_rate: float = 0.50,
        syncope_rate: float = 0.50,
        reduction_mode: str = "inventory_snap",
        environment=None,
        tone_tier: int = 0,
    ) -> Word:
        # Passes the actual environmental syllable complexity into syllabification!
        s_complex = getattr(environment, "syllable_complexity", 0.50) if environment else 0.50
        syllables = cls.syllabify(word, syllable_complexity=s_complex)
        if not syllables:
            return word.copy()

        cls.assign_stress(syllables, pattern)
        num_syl = len(syllables)

        pop = getattr(environment, "population", 0.50) if environment else 0.50
        alt = getattr(environment, "altitude", 0.0) if environment else 0.0
        environmental_tempo = 0.35 + (pop * 0.45) + (alt * 0.20)

        processed_syllables: list[Syllable] = []

        for idx, syl in enumerate(syllables):
            if not syl.nucleus:
                processed_syllables.append(syl)
                continue

            is_final = (idx == num_syl - 1)
            is_medial = (0 < idx < num_syl - 1)

            nucleus = list(syl.nucleus)
            is_geminate = (
                len(nucleus) >= 2 
                and abs(nucleus[0].point[0] - nucleus[1].point[0]) < 0.60 
                and abs(nucleus[0].point[1] - nucleus[1].point[1]) < 0.60
            )

            # Metrical & Tonal Degemination in Unstressed Positions
            if is_geminate and (not syl.stressed or is_final or is_medial or tone_tier > 0):
                nucleus = [nucleus[0]]

            # Continuous Vowel Reduction / Syncope / Apocope
            if tone_tier == 0 and not syl.stressed and reduction_mode != "none":
                if len(nucleus) == 1:
                    v = nucleus[0]
                    centrality = 1.0 - abs(v.point[0] - 3.0) / 3.0
                    drop_prob = reduction_strength * environmental_tempo * (0.50 + centrality * 0.50)

                    should_drop = False
                    if is_final and (random.random() < (drop_prob * apocope_rate)):
                        should_drop = True
                    elif is_medial and (random.random() < (drop_prob * syncope_rate)):
                        should_drop = True

                    if should_drop:
                        nucleus = []
                        if processed_syllables:
                            processed_syllables[-1].coda.extend(syl.onset + syl.coda)
                        continue

                    elif reduction_mode == "centralize":
                        rho = min(0.60, reduction_strength * environmental_tempo * 0.50)
                        h_t, b_t, r_t = v.point
                        
                        target_center_h = 5.0 if h_t >= 4.5 else (1.0 if h_t <= 1.5 else 3.0)
                        target_center_b = 1.0
                        
                        h_new = h_t + rho * (target_center_h - h_t)
                        b_new = b_t + rho * (target_center_b - b_t)
                        r_new = r_t
                        nucleus = [Phoneme(PhonemeKind.VOWEL, (h_new, b_new, r_new), layers=v.layers)]

            if not nucleus:
                continue

            # Dynamic Haudricourt Tonogenesis
            if tone_tier > 0:
                v_height = nucleus[0].point[0] if nucleus else 3.0
                is_high_vowel = (v_height >= 4.0)

                has_glottal = any(
                    (c.ipa == "ʔ" or (c.point[0] >= 9.5 and int(round(c.point[1])) == 0))
                    for c in syl.coda
                )
                has_fricative_laryngeal = any(
                    (int(round(c.point[1])) in (4, 5) or c.ipa in ("s", "h", "x", "ħ", "χ", "ɦ"))
                    for c in syl.coda
                )
                has_stop_coda = any(int(round(c.point[1])) in (0, 2) and c.point[0] < 9.5 for c in syl.coda)
                has_nasal_coda = any(int(round(c.point[1])) == 1 for c in syl.coda)
                is_diphthong = (len(nucleus) >= 2 and abs(nucleus[0].point[0] - nucleus[1].point[0]) >= 0.50)

                retained_coda = [
                    c for c in syl.coda 
                    if c.ipa != "ʔ" and not (c.point[0] >= 9.5 and int(round(c.point[1])) == 0) and int(round(c.point[1])) not in (4, 5)
                ]

                # Register Split
                if not syl.onset:
                    register = "HIGH"
                else:
                    first_c = syl.onset[0]
                    if first_c.kind == PhonemeKind.CONSONANT:
                        _, mn_f, vc_f = first_c.point
                        vc = int(round(vc_f))
                        mn = int(round(mn_f))
                        if vc == 0:
                            register = "HIGH"
                        elif mn in (0, 2, 4, 5):
                            register = "LOW"
                        else:
                            register = "HIGH" if (is_high_vowel or syl.stressed) else "LOW"
                    else:
                        register = "HIGH"

                # Pitch Contour Formation
                if has_glottal:
                    raw_tone = Tone.HIGH_RISING if register == "HIGH" else Tone.LOW_RISING
                elif has_fricative_laryngeal:
                    raw_tone = Tone.HIGH_FALLING if register == "HIGH" else Tone.LOW_FALLING
                elif has_stop_coda:
                    raw_tone = Tone.HIGH_LEVEL if register == "HIGH" else Tone.LOW_LEVEL
                elif is_diphthong or has_nasal_coda or syl.stressed:
                    if register == "HIGH":
                        raw_tone = Tone.HIGH_FALLING if is_high_vowel else Tone.HIGH_RISING
                    else:
                        raw_tone = Tone.DIPPING if is_high_vowel else Tone.LOW_FALLING
                else:
                    raw_tone = Tone.HIGH_LEVEL if (register == "HIGH" and is_high_vowel) else (Tone.MID_LEVEL if register == "HIGH" else Tone.LOW_LEVEL)

                # Tier Snapping
                if tone_tier <= 2:
                    assigned_tone = Tone.HIGH_LEVEL if raw_tone in (Tone.HIGH_LEVEL, Tone.HIGH_RISING, Tone.HIGH_FALLING, Tone.MID_LEVEL) else Tone.LOW_LEVEL
                elif tone_tier <= 4:
                    if raw_tone in (Tone.HIGH_LEVEL, Tone.MID_LEVEL): assigned_tone = Tone.HIGH_LEVEL
                    elif raw_tone in (Tone.HIGH_RISING, Tone.LOW_RISING): assigned_tone = Tone.HIGH_RISING
                    elif raw_tone in (Tone.HIGH_FALLING, Tone.LOW_FALLING): assigned_tone = Tone.HIGH_FALLING
                    else: assigned_tone = Tone.DIPPING
                elif tone_tier <= 6:
                    assigned_tone = Tone.LOW_RISING if raw_tone == Tone.DIPPING else raw_tone
                else:
                    assigned_tone = raw_tone

                processed_syllables.append(
                    Syllable(onset=syl.onset, nucleus=nucleus, coda=retained_coda, stressed=syl.stressed, tone=assigned_tone)
                )
            else:
                processed_syllables.append(
                    Syllable(onset=syl.onset, nucleus=nucleus, coda=syl.coda, stressed=syl.stressed, tone=Tone.NONE)
                )

        reconstructed_phonemes: list[Phoneme] = []
        for s in processed_syllables:
            reconstructed_phonemes.extend(s.to_phonemes())

        # Clean boundary clusters with the actual environmental S_comp
        s_complex = getattr(environment, "syllable_complexity", 0.50) if environment else 0.50
        cleaned_phonemes = ArticulatoryEnergyModel.clean_clusters_and_degeminate(
            reconstructed_phonemes, 
            syllable_complexity=s_complex
        )

        # CRITICAL FIX: If phonemes were pruned (e.g. ll- -> l-, zz# -> z#),
        # re-syllabify from the cleaned phonemes so syllables and phonemes are synchronized!
        if len(cleaned_phonemes) != len(reconstructed_phonemes):
            temp_word = Word(vector=word.vector, phonemes=cleaned_phonemes, is_ephemeral=True)
            processed_syllables = cls.syllabify(temp_word, syllable_complexity=s_complex)
            cls.assign_stress(processed_syllables, pattern)

        out_morphemes = [m.copy() for m in word.morphemes]
        if len(out_morphemes) == 1:
            out_morphemes[0].phonemes = list(cleaned_phonemes)

        return Word(
            word_id=word.id,
            vector=word.vector,
            phonemes=cleaned_phonemes,
            syllables=processed_syllables,
            morphemes=out_morphemes,
            derivation=word.derivation,
            usage_frequency=word.usage_frequency,
            category=word.category,
            generation_born=word.generation_born,
            senses=word.senses,
        )

    @classmethod
    def apply_prosody(
        cls,
        word: Word,
        pattern: StressPattern = StressPattern.PENULTIMATE,
        tone_tier: int = 0,
    ) -> Word:
        """Applies stress and tone prosody without generational reduction."""
        return cls.reduce_and_prosodify(
            word=word,
            pattern=pattern,
            reduction_strength=0.0,
            reduction_mode="none",
            tone_tier=tone_tier,
        )