"""
energy.py — First-Principles Biomechanical & Thermodynamic Articulatory Engine.
Features:
- Dual-Orifice Choking Drag (kw survives, xw chokes).
- VOT Contrast Subsidy (Rescues b, d, g).
- Acoustic Subsidy Capping (Prevents sibilant flood).
- Steriade Cue-Strength Licensing (Initial geminates degrade naturally under complexity threshold).
- Strict DAG: Zero imports from syllable.py.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from articulatory_space import Airstream, SecondaryArticulation
from phoneme import Phoneme, PhonemeKind
from word import Word, Morpheme, ClineStage


def _get_sonority(p: Phoneme) -> float:
    if p.kind == PhonemeKind.VOWEL:
        h = max(0.0, min(6.0, float(p.point[0])))
        return 6.0 + 1.0 * (1.0 - (h / 6.0))

    mn_val = max(0.0, min(7.0, float(p.point[1])))
    mn = int(round(mn_val))

    if mn == 0: return 1.0
    elif mn == 2: return 1.5
    elif mn in (4, 5): return 2.0
    elif mn == 1: return 3.0
    elif mn == 3: return 4.0
    elif mn == 7: return 4.5
    elif mn == 6: return 5.0
    return 2.0


def _get_articulator_tier(place: float) -> int:
    pl = max(0.0, min(10.0, float(place)))
    if pl <= 1.5: return 0  # Labial
    elif pl <= 5.5: return 1  # Coronal
    elif pl <= 8.5: return 2  # Dorsal
    return 3  # Radical / Laryngeal


def calculate_syllable_mass(phonemes: Sequence[Phoneme]) -> float:
    mass = 0.0
    for p in phonemes:
        dur = p.physics.intrinsic_duration_ms / 100.0
        salience = p.physics.acoustic_salience
        weight_bonus = 1.40 if p.kind == PhonemeKind.VOWEL else 0.80
        mass += (dur * salience * weight_bonus)
    return mass


class ArticulatoryEnergyModel:
    """Evaluates physical transition friction, cluster constraints, and phonetic repairs."""

    @classmethod
    def transition_cost(cls, p1: Phoneme, p2: Phoneme) -> float:
        if p1.kind != p2.kind:
            return 1.00

        # Vowel Transitions
        if p1.kind == PhonemeKind.VOWEL and p2.kind == PhonemeKind.VOWEL:
            dh = abs(p1.point[0] - p2.point[0]) / 6.0
            db = abs(p1.point[1] - p2.point[1]) / 2.0
            if dh < 0.10 and db < 0.10:
                return 0.05
            if p1.point[0] < p2.point[0] and p2.point[0] >= 4.0:
                return 0.20 + (db * 0.20)
            return 0.45 + (dh * 0.35 + db * 0.25)

        # Consonant Transitions
        pl1, mn1_f, vc1_f = p1.point
        pl2, mn2_f, vc2_f = p2.point
        mn1, mn2 = int(round(mn1_f)), int(round(mn2_f))
        vc1, vc2 = int(round(vc1_f)), int(round(vc2_f))

        # Geminates
        if abs(pl1 - pl2) < 0.60 and mn1 == mn2:
            if mn1 in (0, 2) and vc1 == 1:
                return 2.20  # Voiced stop geminate: Ohala impedance wall
            return 0.15      # Voiceless/sonorant geminate

        tier1 = _get_articulator_tier(pl1)
        tier2 = _get_articulator_tier(pl2)
        place_cost = min(0.35, abs(pl1 - pl2) * 0.08) if tier1 == tier2 else 0.35

        son1, son2 = _get_sonority(p1), _get_sonority(p2)
        if mn1 in (0, 2) and mn2 in (0, 2):
            manner_cost = 0.85 if tier1 == tier2 else 0.50
        elif mn1 in (4, 5) and mn2 in (0, 2):
            manner_cost = 0.15
        elif son1 < son2:
            manner_cost = 0.20
        else:
            manner_cost = min(0.60, abs(son1 - son2) * 0.18)

        is_obs1 = (mn1 in (0, 2, 4, 5))
        is_obs2 = (mn2 in (0, 2, 4, 5))
        voicing_clash = 1.10 if (is_obs1 and is_obs2 and vc1 != vc2 and tier1 != 3 and tier2 != 3) else 0.0

        return place_cost + manner_cost + voicing_clash

    @classmethod
    def clean_clusters_and_degeminate(cls, phonemes: list[Phoneme], syllable_complexity: float = 0.50) -> list[Phoneme]:
        if len(phonemes) < 2:
            return phonemes

        # Steriade Cue Licensing: Degeminate initial clusters if complexity is low
        if phonemes[0].kind == PhonemeKind.CONSONANT and phonemes[1].kind == PhonemeKind.CONSONANT:
            pl_diff = abs(phonemes[0].point[0] - phonemes[1].point[0])
            mn_diff = abs(phonemes[0].point[1] - phonemes[1].point[1])
            is_identical_initial = (pl_diff < 0.60 and mn_diff < 0.50)

            if is_identical_initial and syllable_complexity < 0.70:
                phonemes = phonemes[1:]

        if len(phonemes) < 2:
            return phonemes

        # Closed Syllable Shortening (VːCC -> VCC)
        v_shortened: list[Phoneme] = []
        i = 0
        n_raw = len(phonemes)
        while i < n_raw:
            p = phonemes[i]
            if (
                p.kind == PhonemeKind.VOWEL and i + 1 < n_raw and phonemes[i + 1].kind == PhonemeKind.VOWEL
                and abs(p.point[0] - phonemes[i + 1].point[0]) < 0.60
                and abs(p.point[1] - phonemes[i + 1].point[1]) < 0.60
            ):
                if i + 3 < n_raw and phonemes[i + 2].kind == PhonemeKind.CONSONANT and phonemes[i + 3].kind == PhonemeKind.CONSONANT:
                    v_shortened.append(p)
                    i += 2
                    continue
            v_shortened.append(p)
            i += 1

        n_p = len(v_shortened)
        max_consonants = 1 if syllable_complexity < 0.28 else (2 if syllable_complexity < 0.72 else 3)

        output: list[Phoneme] = []
        i = 0
        while i < n_p:
            if v_shortened[i].kind == PhonemeKind.VOWEL:
                output.append(v_shortened[i])
                i += 1
            else:
                c_start = i
                while i < n_p and v_shortened[i].kind == PhonemeKind.CONSONANT:
                    i += 1
                c_streak = v_shortened[c_start:i]

                while len(c_streak) > max_consonants:
                    weakest_idx = min(
                        range(len(c_streak)), 
                        key=lambda idx: c_streak[idx].physics.acoustic_salience + c_streak[idx].weight
                    )
                    c_streak.pop(weakest_idx)

                output.extend(c_streak)

        return output

    @classmethod
    def repair_phonemes(
        cls, 
        phonemes: list[Phoneme], 
        profile=None, 
        max_passes: int = 5,
        cultural_inertia: float = 1.0,
    ) -> list[Phoneme]:
        s_complex = getattr(profile, "syllable_complexity", 0.50) if profile is not None else 0.50
        max_cost = getattr(profile, "max_transition_cost", 3.00) if profile is not None else 3.00
        ep_vowel_point = getattr(profile, "epenthetic_vowel_point", (3.0, 1.0, 0.0)) if profile is not None else (3.0, 1.0, 0.0)

        for _ in range(max_passes):
            changed = False
            i = 0
            n_p = len(phonemes)

            while i < n_p - 1:
                p1, p2 = phonemes[i], phonemes[i + 1]

                if p1.kind == PhonemeKind.CONSONANT and p2.kind == PhonemeKind.CONSONANT:
                    cost = cls.transition_cost(p1, p2)
                    if cost > max_cost:
                        phonemes.insert(i + 1, Phoneme(PhonemeKind.VOWEL, ep_vowel_point))
                        changed = True
                        n_p += 1
                        i += 2
                        continue

                elif p1.kind == PhonemeKind.VOWEL and p2.kind == PhonemeKind.VOWEL:
                    dh = abs(p1.point[0] - p2.point[0])
                    db = abs(p1.point[1] - p2.point[1])
                    if dh < 0.40 and db < 0.40:
                        i += 1
                        continue
                    if p1.point[0] >= 3.8:
                        glide_pt = (6.0, 6.0, 1.0) if p1.point[1] <= 1.0 else (0.0, 7.0, 1.0)
                        phonemes[i] = Phoneme(PhonemeKind.CONSONANT, glide_pt)
                        changed = True

                i += 1

            if not changed:
                break

        return cls.clean_clusters_and_degeminate(phonemes, syllable_complexity=s_complex)

    @classmethod
    def repair_word(cls, word: Word, profile=None, max_passes: int = 5, cultural_inertia: float = 1.0) -> Word:
        cleaned = cls.repair_phonemes(list(word.phonemes), profile=profile, max_passes=max_passes, cultural_inertia=cultural_inertia)
        out_morphemes = [m.copy() for m in word.morphemes]
        if len(out_morphemes) == 1 and cleaned:
            out_morphemes[0].phonemes = list(cleaned)

        return Word(
            word_id=word.id,
            vector=word.vector,
            phonemes=cleaned,
            syllables=word.syllables,
            morphemes=out_morphemes,
            derivation=word.derivation,
            usage_frequency=word.usage_frequency,
            category=word.category,
            generation_born=word.generation_born,
            senses=word.senses,
        )

    @classmethod
    def erode_morpheme(
        cls,
        morpheme: Morpheme,
        reduction_rate: float = 0.80,
        usage_frequency: float = 1.0,
        is_long_word: bool = False,
        syllable_complexity: float = 0.50,
        potency_brake: float = 1.0,
    ) -> Morpheme:
        if morpheme.stage == ClineStage.ZERO or (not morpheme.is_grammatical and not is_long_word):
            return morpheme

        p_list = list(morpheme.phonemes)
        if not p_list:
            return Morpheme([], morpheme.root_family_id, is_grammatical=morpheme.is_grammatical, stage=ClineStage.ZERO)

        effective_reduction = reduction_rate / max(0.50, potency_brake)
        freq_multiplier = math.log(usage_frequency + 1.0) * (1.20 if morpheme.is_grammatical else 0.70)
        p_erode = 1.0 - math.exp(-0.25 * effective_reduction * freq_multiplier)

        if random.random() > p_erode:
            return morpheme

        v_indices = [idx for idx, seg in enumerate(p_list) if seg.kind == PhonemeKind.VOWEL]
        if not v_indices:
            return morpheme

        last_v = v_indices[-1]
        if last_v < len(p_list) - 1:
            p_list = p_list[:last_v + 1]
        elif len(v_indices) > 1:
            p_list = [seg for idx, seg in enumerate(p_list) if idx != v_indices[0]]

        return Morpheme(phonemes=p_list, root_family_id=morpheme.root_family_id, is_grammatical=morpheme.is_grammatical, stage=morpheme.stage)


class BiomechanicalCostModel:
    """Computes total metabolic, aerodynamic, acoustic, and cultural expenditure of utterances."""

    @classmethod
    def _segment_energy(cls, p: Phoneme, env, metrical_weight: float, attn: Sequence[float]) -> float:
        temp = getattr(env, "temperature", 0.5) if env else 0.5
        hum = getattr(env, "humidity", 0.5) if env else 0.5
        alt = getattr(env, "altitude", 0.0) if env else 0.0
        noise = getattr(env, "ambient_noise", 0.2) if env else 0.2
        veg = env.emergent_vegetation() if (env and hasattr(env, "emergent_vegetation")) else 0.50

        w_pot = attn[3]
        w_ext = attn[6]

        air_coldness = max(0.0, 0.45 - temp)
        aridity = max(0.0, 0.45 - hum)
        phys = p.physics
        dur_factor = phys.intrinsic_duration_ms / 100.0

        # 1. Thermal Exposure (RHML)
        thermal_tax = phys.oral_aperture * dur_factor * (air_coldness * 2.2 + aridity * 1.5)

        # 2. Hypoxia/Pulmonic Velocity
        hypoxia_tax = phys.pulmonic_volume_velocity * (max(0.0, alt - 0.20) * 1.8 + aridity * 0.8)

        # 3. Dual-Orifice Choking
        is_labialized = any(l == SecondaryArticulation.LABIALIZED for l in p.layers)
        is_fricative = (int(round(p.point[1])) in (4, 5))
        choking_tax = (3.5 * phys.pulmonic_volume_velocity) if (is_fricative and is_labialized) else 0.0

        # 4. Voicing Impedance & VOT Contrast Subsidy
        voicing_tax = phys.supraglottal_impedance * 0.70
        vot_contrast_subsidy = 0.52 if phys.supraglottal_impedance > 0.5 else 0.0
        net_voicing_drag = max(0.0, voicing_tax - vot_contrast_subsidy)

        # 5. Neuromuscular Coordination Cost
        motor_tax = (phys.motor_precision * 0.60) / max(0.60, 1.0 + 0.30 * (w_pot - 1.0))

        gross_effort = thermal_tax + hypoxia_tax + choking_tax + net_voicing_drag + motor_tax

        # 6. Frequency-Specific Transmission Payoff (Bounded to max 40% of effort)
        canopy_gain = phys.low_freq_energy * (0.30 + veg * 0.60)
        wind_gain = phys.high_freq_energy * (0.30 + noise * 0.70)
        extension_boost = 1.0 + 0.40 * (w_ext - 1.0)
        
        raw_payoff = (canopy_gain + wind_gain) * 0.45 * extension_boost
        max_allowable_subsidy = gross_effort * 0.40
        acoustic_payoff = min(max_allowable_subsidy, raw_payoff)

        return max(0.05, (gross_effort - acoustic_payoff) * metrical_weight)

    @classmethod
    def word_energetic_cost(cls, word: Word, profile, env, cultural_attention: Sequence[float] | None = None) -> float:
        total_word_energy = 0.0
        phonemes = word.phonemes
        n = len(phonemes)
        if n == 0:
            return 0.0

        attn = cultural_attention or getattr(profile, "cultural_attention", [1.0] * 7)
        w_dyn = attn[4]

        # Base Articulatory Transitions
        tempo_drag = 1.0 + 0.35 * max(0.0, w_dyn - 1.0)
        for j in range(n - 1):
            total_word_energy += ArticulatoryEnergyModel.transition_cost(phonemes[j], phonemes[j+1]) * 0.75 * tempo_drag

        # Syllable & Segment Physics Integration
        syllables = word.syllables or []
        if syllables:
            hum = getattr(env, "humidity", 0.5) if env else 0.5
            temp = getattr(env, "temperature", 0.5) if env else 0.5
            aridity = max(0.0, 0.45 - hum)
            air_coldness = max(0.0, 0.45 - temp)

            for syl in syllables:
                is_stressed = getattr(syl, "stressed", False)
                metrical_weight = 1.35 if is_stressed else 0.65

                syl_phonemes = syl.to_phonemes() if hasattr(syl, "to_phonemes") else phonemes
                for p in syl_phonemes:
                    total_word_energy += cls._segment_energy(p, env, metrical_weight, attn)

                syl_tone = getattr(syl, "tone", None)
                if syl_tone is not None and getattr(syl_tone, "value", "") != "":
                    tone_name = getattr(syl_tone, "name", "")
                    is_contour = tone_name in ("HIGH_FALLING", "LOW_FALLING", "HIGH_RISING", "LOW_RISING", "DIPPING")
                    laryngeal_base = 0.45 if is_contour else 0.18
                    ptp_drag = 1.0 + aridity * 1.4 + air_coldness * 0.8
                    total_word_energy += (laryngeal_base * ptp_drag)
        else:
            for p in phonemes:
                total_word_energy += cls._segment_energy(p, env, metrical_weight=1.0, attn=attn)

        # Polysyllabic length drag
        if n > 6:
            total_word_energy += 0.18 * ((n - 6) ** 1.3)

        return round(total_word_energy, 3)