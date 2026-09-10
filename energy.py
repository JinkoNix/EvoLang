"""
energy.py — High-Performance Articulatory Physics, Continuous Transition Friction, and Morpheme Erosion.
Optimized with continuous sonority hierarchy math, anatomical organ-distance transition costs,
in-situ phonotactic cluster pruning, and constituent-aware syllable skeleton erosion.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from articulatory_space import Airstream, SecondaryArticulation
from phoneme import Phoneme, PhonemeKind
from word import Word, Morpheme, ClineStage


# =====================================================================
# 1. Continuous Sonority & Anatomical Organ Classification
# =====================================================================

def _get_sonority(p: Phoneme) -> float:
    """
    Returns continuous acoustic sonority on a 1.0 (Plosive) to 7.0 (Open Vowel) scale.
    Grounded in Clements (1990) and Parker (2002).
    """
    if p.kind == PhonemeKind.VOWEL:
        h = max(0.0, min(6.0, float(p.point[0])))
        # Open /a/ has higher acoustic intensity (7.0) than close /i, u/ (6.0)
        return 6.0 + 1.0 * (1.0 - (h / 6.0))

    mn_val = max(0.0, min(7.0, float(p.point[1])))
    mn = int(round(mn_val))

    # Continuous Manner Mapping
    if mn == 0:
        base_son = 1.0  # Plosives / Stops (Total oral occlusion)
    elif mn == 2:
        base_son = 1.5  # Affricates & Delayed-Release (Occlusion + strident burst)
    elif mn in (4, 5):
        base_son = 2.0  # Fricatives (Central & Lateral continuous friction)
    elif mn == 1:
        base_son = 3.0  # Nasals (Oral stop with open velopharyngeal murmur)
    elif mn == 3:
        base_son = 4.0  # Taps, Flaps & Trills (Intermittent fluid contact)
    elif mn == 7:
        base_son = 4.5  # Lateral Approximants (Unobstructed lateral resonance)
    elif mn == 6:
        base_son = 5.0  # Central Approximants / Glides (Semi-vocalic resonance)
    else:
        base_son = 2.0

    return base_son


def _get_articulator_tier(place: float) -> int:
    """
    Maps Place coordinate into the 4 active anatomical motor tiers (Sagey & Halle 1986):
    0: Labial (Lips)
    1: Coronal (Tongue Tip/Blade)
    2: Dorsal (Tongue Body)
    3: Laryngeal/Radical (Pharynx & Glottis)
    """
    pl = max(0.0, min(10.0, float(place)))
    if pl <= 1.5:
        return 0  # Labial
    elif pl <= 5.5:
        return 1  # Coronal
    elif pl <= 8.5:
        return 2  # Dorsal
    return 3      # Laryngeal / Radical


# =====================================================================
# 2. Articulatory Energy Model
# =====================================================================

class ArticulatoryEnergyModel:
    """Evaluates transition friction, repairs clusters, and executes constituent-aware erosion."""

    @staticmethod
    def get_articulator_tier(place: float) -> int:
        return _get_articulator_tier(place)

    @classmethod
    def transition_cost(cls, p1: Phoneme, p2: Phoneme) -> float:
        """
        Computes continuous articulatory transition friction between adjacent phonemes.
        Normalized to [0.10, 2.50].
        """
        # Baseline CV / VC transition cost
        if p1.kind != p2.kind:
            return 1.00

        # -----------------------------------------------------------------
        # A. Vowel-to-Vowel Transition Cost (Diphthong vs. Hiatus)
        # -----------------------------------------------------------------
        if p1.kind == PhonemeKind.VOWEL and p2.kind == PhonemeKind.VOWEL:
            h1, h2 = p1.point[0], p2.point[0]
            b1, b2 = p1.point[1], p2.point[1]
            dh = abs(h1 - h2) / 6.0
            db = abs(b1 - b2) / 2.0

            # Closing Diphthongs (e.g. ai, au, ei, ou: h1 < h2 and h2 >= 4.0)
            if h1 < h2 and h2 >= 4.0:
                return 0.20 + (db * 0.20)

            # Opening Hiatus (e.g. ia, ua, io: h1 >= h2)
            return 0.45 + (dh * 0.35 + db * 0.25)

        # -----------------------------------------------------------------
        # B. Consonant-to-Consonant Transition Cost
        # -----------------------------------------------------------------
        pl1, mn1_f, vc1_f = p1.point
        pl2, mn2_f, vc2_f = p2.point
        mn1, mn2 = int(round(mn1_f)), int(round(mn2_f))
        vc1, vc2 = int(round(vc1_f)), int(round(vc2_f))

        tier1 = _get_articulator_tier(pl1)
        tier2 = _get_articulator_tier(pl2)

        # 1. Place / Anatomical Organ Cost
        if tier1 == tier2:
            # Intra-organ deformation bounded by muscle tissue elasticity
            place_cost = min(0.35, abs(pl1 - pl2) * 0.08)
        else:
            # Independent motor organ transition (e.g. Labial -> Coronal)
            if tier1 == 3 or tier2 == 3:
                place_cost = 0.20  # Laryngeal coordination
            else:
                place_cost = 0.35  # Organ switching (pt, ks, tw)

        # 2. Manner & Aerodynamic Pressure Cost
        son1 = _get_sonority(p1)
        son2 = _get_sonority(p2)

        if mn1 in (0, 2) and mn2 in (0, 2):
            # Double stops (tt, pt, kt): Intraoral pressure buildup
            manner_cost = 0.85 if tier1 == tier2 else 0.50
        elif mn1 in (4, 5) and mn2 in (0, 2):
            manner_cost = 0.15  # Natural sibilant-stop cluster (st, sp, sk)
        elif mn1 == 1 and mn2 in (0, 2) and abs(pl1 - pl2) < 1.0:
            manner_cost = 0.15  # Homorganic nasal-stop (mp, nt, ŋk)
        elif son1 < son2:
            manner_cost = 0.20  # Obstruent + Sonorant rising sonority (pl, tr, kw)
        else:
            manner_cost = min(0.60, abs(son1 - son2) * 0.18)

        # 3. Voicing Clash Friction
        is_obs1 = (mn1 in (0, 2, 4, 5))
        is_obs2 = (mn2 in (0, 2, 4, 5))
        is_glottal = (tier1 == 3 or tier2 == 3)

        if is_obs1 and is_obs2 and vc1 != vc2 and not is_glottal:
            voicing_clash_cost = 1.10  # Penalizes mismatch without breaking the repair loop
        else:
            voicing_clash_cost = 0.00

        return place_cost + manner_cost + voicing_clash_cost

    @classmethod
    def clean_clusters_and_degeminate(cls, phonemes: list[Phoneme], syllable_complexity: float = 0.50) -> list[Phoneme]:
        """
        Continuous In-Situ Cluster Capacity Shearing & Phonotactic Licensing:
        - Closed Syllable Shortening (VːCC -> VCC).
        - Affricate-Obstruent Shearing in Low/Moderate S_comp (d+ɮ -> ɮ, p+t͡s -> t͡s, t͡s+k -> t͡s).
        - Homorganic Degemination (dd͡z -> d͡z, t͡st -> t͡s).
        - Word-Boundary Degemination (ll- -> l-, zz# -> z#).
        - Minimal Sonority Distance on Onsets (dm- -> m-, bm- -> m-, tp- -> p-).
        """
        if len(phonemes) < 2:
            return phonemes

        # 1. Closed Syllable Shortening (VːCC -> VCC)
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
                has_cluster_ahead = (
                    i + 3 < n_raw and phonemes[i + 2].kind == PhonemeKind.CONSONANT and phonemes[i + 3].kind == PhonemeKind.CONSONANT
                )
                if has_cluster_ahead:
                    v_shortened.append(p)
                    i += 2
                    continue
            v_shortened.append(p)
            i += 1

        n_p = len(v_shortened)
        max_consonants = 1 if syllable_complexity < 0.28 else (2 if syllable_complexity < 0.72 else 3)

        # 2. Steriade's Cue Licensing & Boundary Degemination
        licensed: list[Phoneme] = []
        for i, p in enumerate(v_shortened):
            if p.kind == PhonemeKind.CONSONANT:
                is_preconsonantal = (i + 1 < n_p and v_shortened[i + 1].kind == PhonemeKind.CONSONANT)
                is_word_final = (i == n_p - 1)

                if is_word_final and len(licensed) > 0 and licensed[-1].kind == PhonemeKind.CONSONANT:
                    if (
                        abs(p.point[0] - licensed[-1].point[0]) < 0.80 and
                        abs(p.point[1] - licensed[-1].point[1]) < 0.80
                    ):
                        continue

                if is_preconsonantal and i + 2 < n_p and v_shortened[i + 2].kind == PhonemeKind.CONSONANT:
                    if (
                        abs(p.point[0] - v_shortened[i + 1].point[0]) < 0.60 and
                        abs(p.point[1] - v_shortened[i + 1].point[1]) < 0.60
                    ):
                        continue

                if is_preconsonantal or is_word_final:
                    if p.layers[0] == Airstream.IMPLOSIVE:
                        p = p.drift(point=(p.point[0], p.point[1], 0.0), layers=(Airstream.PULMONIC, SecondaryArticulation.NONE))
                    elif is_preconsonantal and p.layers[0] == Airstream.EJECTIVE:
                        p = p.drift(layers=(Airstream.PULMONIC, SecondaryArticulation.NONE))
                    elif any(l in (SecondaryArticulation.ASPIRATED, SecondaryArticulation.BREATHY, SecondaryArticulation.LABIALIZED) for l in p.layers):
                        p = p.drift(layers=(p.layers[0], SecondaryArticulation.NONE))

            licensed.append(p)

        # -----------------------------------------------------------------
        # 3. Universal OCP Affricate Gestural Merger & Homorganic Degemination
        # -----------------------------------------------------------------
        affricate_cleaned: list[Phoneme] = []
        i = 0
        n_lic = len(licensed)

        while i < n_lic:
            p = licensed[i]
            if p.kind == PhonemeKind.CONSONANT and i + 1 < n_lic and licensed[i + 1].kind == PhonemeKind.CONSONANT:
                p_next = licensed[i + 1]
                pl_diff = abs(p.point[0] - p_next.point[0])
                mn1 = int(round(p.point[1]))
                mn2 = int(round(p_next.point[1]))

                is_affricate_1 = (mn1 == 2)
                is_affricate_2 = (mn2 == 2)
                is_obs_1 = (mn1 in (0, 4, 5))
                is_obs_2 = (mn2 in (0, 4, 5))

                # A. OCP Gesture Merger on Adjacent Affricates (d͡z + d͡z -> d͡z, t͡s + t͡s -> t͡s)
                # Two consecutive delayed releases cannot fire twice at the same place
                if is_affricate_1 and is_affricate_2 and pl_diff < 1.2:
                    affricate_cleaned.append(p)  # Fuses into single affricate!
                    i += 2
                    continue

                # B. Obstruent Clustering Prohibition in Low/Moderate S_comp
                if (is_affricate_1 and is_obs_2) or (is_obs_1 and is_affricate_2):
                    if syllable_complexity <= 0.68:
                        survivor = p if is_affricate_1 else p_next
                        affricate_cleaned.append(survivor)
                        i += 2
                        continue

                # C. Homorganic Stop + Affricate Degemination (dd͡z -> d͡z, t͡st -> t͡s)
                if pl_diff < 1.4:
                    if mn1 in (0, 4) and mn2 == 2:
                        affricate_cleaned.append(p_next)
                        i += 2
                        continue
                    elif mn1 == 2 and mn2 in (0, 4):
                        affricate_cleaned.append(p)
                        i += 2
                        continue

            affricate_cleaned.append(p)
            i += 1

        # -----------------------------------------------------------------
        # 4. In-Situ Consonant Cluster Capacity Shearing
        # -----------------------------------------------------------------
        output: list[Phoneme] = []
        i = 0
        n_aff = len(affricate_cleaned)

        while i < n_aff:
            if affricate_cleaned[i].kind == PhonemeKind.VOWEL:
                output.append(affricate_cleaned[i])
                i += 1
            else:
                c_start = i
                while i < n_aff and affricate_cleaned[i].kind == PhonemeKind.CONSONANT:
                    i += 1
                c_streak = affricate_cleaned[c_start:i]

                while len(c_streak) > max_consonants:
                    def get_salience(seg: Phoneme) -> float:
                        mn_code = int(round(seg.point[1]))
                        pl_code = seg.point[0]
                        # Only central coronal sibilants (mn=2, 4 at pl in [2.5, 5.5]) get stridency bonus
                        # Lateral fricatives (mn=5) do NOT receive the sibilant bonus (cheek flesh damping!)
                        is_sibilant = (mn_code in (2, 4) and 2.2 <= pl_code <= 5.5)
                        strident_bonus = 2.0 if is_sibilant else (1.0 if mn_code == 0 else 0.0)
                        return strident_bonus + seg.weight

                    weakest_idx = min(range(len(c_streak)), key=lambda idx: get_salience(c_streak[idx]))
                    c_streak.pop(weakest_idx)

                output.extend(c_streak)

        # -----------------------------------------------------------------
        # 5. Continuous Minimal Sonority Distance (MSD) & Bidirectional SSP
        # -----------------------------------------------------------------
        if len(output) >= 2:
            # A. Continuous Minimal Sonority Distance on Onsets:
            # ΔSon_min = 3.6 - 2.2 * (S_comp * R_rough)
            # Low S_comp / Low Roughness (Hawaiian, Japonic, Italian) -> ΔSon_min ~ 3.3 (Rejects bz-, ps-, dm-!)
            # High S_comp / High Roughness (German, Russian) -> ΔSon_min ~ 2.2 (Licenses steeper onsets)
            if output[0].kind == PhonemeKind.CONSONANT and output[1].kind == PhonemeKind.CONSONANT:
                son1 = _get_sonority(output[0])
                son2 = _get_sonority(output[1])
                mn1 = int(round(output[0].point[1]))
                mn2 = int(round(output[1].point[1]))

                is_identical = (abs(output[0].point[0] - output[1].point[0]) < 0.6 and mn1 == mn2)
                is_sibilant_lead = (mn1 in (4, 5) and mn2 in (0, 2) and syllable_complexity >= 0.35)
                
                # Continuous Minimal Sonority Distance equation
                min_son_distance = 3.60 - (2.20 * syllable_complexity)
                delta_son = son2 - son1

                # Rejects:
                # 1. Initial geminates (ll-, rr-, ss-)
                # 2. Insufficient sonority rise (bz-, ps-, dm-, bm-, tp-)
                # 3. Double stops (tp-, kt-)
                if is_identical or (delta_son < min_son_distance and not is_sibilant_lead):
                    output = output[1:]  # Prunes illegal initial cluster element

            if len(output) >= 2 and output[-1].kind == PhonemeKind.CONSONANT and output[-2].kind == PhonemeKind.CONSONANT:
                son_penult = _get_sonority(output[-2])
                son_final = _get_sonority(output[-1])
                mn_penult = int(round(output[-2].point[1]))
                mn_final = int(round(output[-1].point[1]))

                is_identical_coda = (abs(output[-1].point[0] - output[-2].point[0]) < 0.6 and mn_final == mn_penult)
                is_liquid_liquid = (son_penult >= 4.0 and son_final >= 4.0)
                is_stop_nasal_coda = (mn_penult in (0, 2) and mn_final == 1)

                if is_identical_coda or is_liquid_liquid or is_stop_nasal_coda or (son_final > son_penult and mn_final not in (4, 5)):
                    output = output[:-1]

        return output


    @classmethod
    def resolve_hiatus(
        cls,
        p1: Phoneme,
        p2: Phoneme,
        profile=None,
        cv_ratio: float = 1.20,
        cultural_inertia: float = 1.0,
    ) -> list[Phoneme]:
        """
        Continuous Boltzmann Hiatus Resolution Engine:
        Dynamically selects between Gliding, Contraction, Elision, and Epenthesis
        using continuous vocalic kinematics, C:V macro-state balance, and cultural inertia.
        """
        h1, b1, r1 = p1.point
        h2, b2, r2 = p2.point
        dh = abs(h1 - h2)
        db = abs(b1 - b2)

        # 1. Identical/Near-Identical Vowels -> Fused Geminate Long Vowel (V1 V1 -> V1ː)
        if dh < 0.50 and db < 0.50:
            return [p1, p1]

        s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
        inertia = max(0.20, float(cultural_inertia))
        cv = max(0.50, float(cv_ratio))

        # 2. Continuous Energy Costs for the 4 Strategies
        # Strategy A: Gliding (V1 -> Glide: i+a -> ja, u+a -> wa)
        if h1 >= 3.8:
            e_glide = 0.60 * (6.0 - h1) + 0.40 * abs(b1 - 1.0) + 1.20 * max(0.0, cv - 1.30) + 0.50 * s_comp
        else:
            e_glide = 6.00  # Low vowels (/a/) cannot naturally glide without backing

        # Strategy B: Crasis / Contraction (V1 + V2 -> V_fused: ai -> e, au -> o, ae -> eː)
        is_closing = (h2 >= h1)
        e_contract = 0.80 * (dh / 6.0) + (0.00 if is_closing else 0.85) + 0.40 * s_comp

        # Strategy C: Elision (V1 drops: V1 + V2 -> V2, e.g. sˁä + e -> sˁe)
        e_elide = 1.00 * inertia + 1.60 * max(0.0, 1.25 - cv)

        # Strategy D: Epenthesis (V1 + C + V2: inserts j, w, or ʔ based on S_comp)
        e_epenthesis = 2.20 * max(0.0, 1.40 - cv) + 1.20 * (1.00 - s_comp)

        # 3. Normalized Boltzmann Distribution Sampling (T = 0.65)
        strategies = ["glide", "contract", "elide", "epenthesis"]
        energies = [e_glide, e_contract, e_elide, e_epenthesis]
        
        # Softmax probabilities
        exp_weights = [math.exp(-e / 0.65) for e in energies]
        total_weight = sum(exp_weights) or 1.0
        probs = [w / total_weight for w in exp_weights]

        chosen_strat = random.choices(strategies, weights=probs, k=1)[0]

        # 4. Execute Winning Strategy
        if chosen_strat == "glide":
            # High-front -> /j/ (Place 6, Manner 6); High-back -> /w/ (Place 0, Manner 7)
            glide_pt = (6.0, 6.0, 1.0) if b1 <= 1.0 else (0.0, 7.0, 1.0)
            return [Phoneme(PhonemeKind.CONSONANT, glide_pt), p2]

        elif chosen_strat == "contract":
            # Quantal Formant Blending: ai -> e (h=4, b=0), au -> o (h=4, b=2)
            if h1 <= 1.8 and h2 >= 3.8:
                h_fused = 4.0
                b_fused = 2.0 if b2 >= 1.2 else 0.0
                r_fused = 1.0 if b2 >= 1.2 else 0.0
            else:
                h_fused = (h1 + h2) / 2.0
                b_fused = (b1 + b2) / 2.0
                r_fused = 1.0 if (r1 > 0.5 or r2 > 0.5) and b_fused >= 1.3 else 0.0
            return [Phoneme(PhonemeKind.VOWEL, (h_fused, b_fused, r_fused))]

        elif chosen_strat == "elide":
            return [p2]

        else:  # epenthesis
            # Vocalic profiles insert smooth glides (j, w); Consonant-heavy profiles insert ʔ
            if s_comp < 0.60 or cv < 1.40:
                ep_pt = (6.0, 6.0, 1.0) if b1 <= 1.0 else (0.0, 7.0, 1.0)  # j or w
            else:
                ep_pt = (10.0, 0.0, 0.0)  # ʔ
            return [p1, Phoneme(PhonemeKind.CONSONANT, ep_pt), p2]

    
    @classmethod
    def repair_phonemes(
        cls, 
        phonemes: list[Phoneme], 
        profile=None, 
        max_passes: int = 5,
        cultural_inertia: float = 1.0,
    ) -> list[Phoneme]:
        """
        Executes Natural 3-Step Phonological Repair Sequence:
        Step 1A: Local Consonant Feature Assimilation.
        Step 1B: Continuous Boltzmann Vocalic Hiatus Resolution (modulated by Cultural Inertia).
        Step 2: Re-evaluate transition cost.
        Step 3: Epenthesis (Last resort for unresolved friction or strict open CV).
        """
        s_complex = getattr(profile, "syllable_complexity", 0.50) if profile is not None else 0.50
        max_cost = getattr(profile, "max_transition_cost", 3.00) if profile is not None else 3.00
        ep_vowel_point = getattr(profile, "epenthetic_vowel_point", (3.0, 1.0, 0.0)) if profile is not None else (3.0, 1.0, 0.0)
        double_stop_strat = getattr(profile, "double_stop_strategy", "spirantization") if profile is not None else "spirantization"
        bias_regressive = (getattr(profile, "voicing_assimilation_bias", "regressive") == "regressive") if profile is not None else True

        strict_open = (s_complex < 0.25)

        for _ in range(max_passes):
            changed = False
            i = 0
            n_p = len(phonemes)

            while i < n_p - 1:
                p1, p2 = phonemes[i], phonemes[i + 1]

                # -------------------------------------------------------------
                # STEP 1A: CONSONANT FEATURE ASSIMILATION
                # -------------------------------------------------------------
                if p1.kind == PhonemeKind.CONSONANT and p2.kind == PhonemeKind.CONSONANT:
                    pl1, mn1_f, vc1_f = p1.point
                    pl2, mn2_f, vc2_f = p2.point
                    mn1, mn2 = int(round(mn1_f)), int(round(mn2_f))
                    vc1, vc2 = int(round(vc1_f)), int(round(vc2_f))

                    # A. Obstruent Voicing Harmony (pd -> bd, zt -> st)
                    if mn1 in (0, 2, 4, 5) and mn2 in (0, 2, 4, 5) and vc1 != vc2:
                        target_vc = float(vc2) if bias_regressive else float(vc1)
                        phonemes[i] = p1.drift(point=(pl1, mn1_f, target_vc))
                        changed = True

                    # B. Continuous Homorganic Nasal Assimilation
                    elif mn1 == 1 and mn2 in (0, 2) and abs(pl1 - pl2) > 0.40:
                        phonemes[i] = p1.drift(point=(pl2, mn1_f, vc1_f))
                        changed = True

                    # C. Double Stop Resolution
                    elif mn1 in (0, 2) and mn2 in (0, 2) and abs(pl1 - pl2) > 0.50:
                        if double_stop_strat == "gemination":
                            phonemes[i] = p1.drift(point=(pl2, mn1_f, vc1_f))
                        else:
                            if pl1 <= 1.5:
                                target_p = (1.0, 4.0, 0.0) if vc1 == 0 else (0.0, 7.0, 1.0)
                            elif pl1 <= 5.5:
                                target_p = (3.0, 4.0, float(vc1))
                            else:
                                target_p = (7.0, 4.0, 0.0) if vc1 == 0 else (6.0, 6.0, 1.0)
                            phonemes[i] = p1.drift(point=target_p)
                        changed = True

                    # -------------------------------------------------------------
                    # STEPS 2 & 3: RE-EVALUATE AND EPENTHESIS
                    # -------------------------------------------------------------
                    cost = cls.transition_cost(phonemes[i], phonemes[i + 1])
                    if strict_open or (cost > max_cost):
                        ep_vowel = Phoneme(PhonemeKind.VOWEL, ep_vowel_point)
                        phonemes.insert(i + 1, ep_vowel)
                        changed = True
                        n_p += 1
                        i += 2
                        continue

                # -------------------------------------------------------------
                # STEP 1B: VOCALIC HIATUS RESOLUTION (Coupled to Cultural Inertia)
                # -------------------------------------------------------------
                elif p1.kind == PhonemeKind.VOWEL and p2.kind == PhonemeKind.VOWEL:
                    dh = abs(p1.point[0] - p2.point[0])
                    db = abs(p1.point[1] - p2.point[1])

                    if dh < 0.40 and db < 0.40:
                        i += 1
                        continue

                    num_c = sum(1 for p in phonemes if p.kind == PhonemeKind.CONSONANT)
                    num_v = sum(1 for p in phonemes if p.kind == PhonemeKind.VOWEL)
                    cv_local = num_c / max(1.0, float(num_v))

                    resolved_hiatus = cls.resolve_hiatus(
                        p1, p2, 
                        profile=profile, 
                        cv_ratio=cv_local,
                        cultural_inertia=cultural_inertia
                    )

                    phonemes[i:i + 2] = resolved_hiatus
                    changed = True
                    n_p = len(phonemes)
                    i += max(1, len(resolved_hiatus) - 1)
                    continue

                i += 1

            if not changed:
                break

        if strict_open and phonemes and phonemes[-1].kind == PhonemeKind.CONSONANT:
            phonemes.append(Phoneme(PhonemeKind.VOWEL, ep_vowel_point))

        return cls.clean_clusters_and_degeminate(phonemes, syllable_complexity=s_complex)

    @classmethod
    def repair_word(
        cls, 
        word: Word, 
        profile=None, 
        max_passes: int = 5,
        cultural_inertia: float = 1.0,
    ) -> Word:
        """Repairs word phonemes and resynchronizes internal morpheme slice boundaries."""
        cleaned = cls.repair_phonemes(
            list(word.phonemes), 
            profile=profile, 
            max_passes=max_passes,
            cultural_inertia=cultural_inertia
        )
        
        resynced_morphemes = cls.clean_morpheme_structure(
            word.morphemes,
            root_family_id=word.derivation.root_family_id,
            total_phonemes=len(cleaned)
        )
        if len(resynced_morphemes) == 1 and cleaned:
            resynced_morphemes[0].phonemes = list(cleaned)

        return Word(
            word_id=word.id,
            vector=word.vector,
            phonemes=cleaned,
            syllables=word.syllables,
            morphemes=resynced_morphemes,
            derivation=word.derivation,
            usage_frequency=word.usage_frequency,
            category=word.category,
            generation_born=word.generation_born,
            senses=word.senses,
        )

    @classmethod
    def clean_morpheme_structure(cls, morphemes: list[Morpheme], root_family_id: int, total_phonemes: int | None = None) -> list[Morpheme]:
        """Preserves multi-morphemic structure while cleaning dead or zeroed morphemes."""
        living = [m for m in morphemes if len(m.phonemes) > 0 and m.stage != ClineStage.ZERO]
        if not living:
            return [Morpheme([], root_family_id=root_family_id, stage=ClineStage.ZERO)]

        if len(living) == 1:
            return living

        if total_phonemes is not None and total_phonemes <= 3:
            for m in living[1:]:
                m.stage = ClineStage.FUSED_INTERNAL

        return living

    @classmethod
    def erode_morpheme(
        cls,
        morpheme: Morpheme,
        reduction_rate: float = 0.80,
        usage_frequency: float = 1.0,
        is_long_word: bool = False,
        syllable_complexity: float = 0.50,
    ) -> Morpheme:
        """
        Continuous Constituent-Aware Morpheme Erosion:
        - Non-linear length-scaling: oversized words accelerate tail & coda shedding.
        - Preserves root nucleus integrity without arbitrary character slicing.
        """
        if morpheme.stage == ClineStage.ZERO:
            return morpheme

        if not morpheme.is_grammatical and not is_long_word:
            return morpheme

        p_list = list(morpheme.phonemes)
        if not p_list:
            return Morpheme([], morpheme.root_family_id, is_grammatical=morpheme.is_grammatical, stage=ClineStage.ZERO)

        vocalic_openness_drive = max(0.20, 1.0 - syllable_complexity)
        freq_multiplier = math.log(usage_frequency + 1.0) * (1.20 if morpheme.is_grammatical else 0.70)
        
        # Length-accelerated erosion probability
        m_len = len(p_list)
        length_scaling = 1.0 + 0.12 * (max(0, m_len - 3) ** 1.3)
        p_erode = 1.0 - math.exp(-0.25 * reduction_rate * freq_multiplier * length_scaling * (1.0 + vocalic_openness_drive * 0.50))

        if random.random() > p_erode:
            return morpheme

        # Constituent Decomposition: Partition into Onsets, Vowels, Codas
        v_indices = [idx for idx, seg in enumerate(p_list) if seg.kind == PhonemeKind.VOWEL]

        if not v_indices:
            p_list.append(Phoneme(PhonemeKind.VOWEL, (0.0, 1.0, 0.0)))
            v_indices = [len(p_list) - 1]

        first_v_idx = v_indices[0]
        last_v_idx = v_indices[-1]

        # Step 1: Coda Shedding (Drop post-vocalic consonants first: CVC -> CV)
        if last_v_idx < len(p_list) - 1:
            p_list = p_list[:last_v_idx + 1]

        # Step 2: Complex Onset Simplification (Drop pre-vocalic cluster consonants: CCV -> CV)
        elif first_v_idx > 1:
            p_list = [p_list[0]] + p_list[first_v_idx:]

        # Step 3: Nucleus Monophthongization (VV -> V)
        elif len(v_indices) > 1:
            p_list = [seg for idx, seg in enumerate(p_list) if idx not in v_indices[1:]]

        n_p = len(p_list)
        if n_p == 0:
            stage = ClineStage.ZERO
        elif n_p == 1:
            stage = ClineStage.FUSED_INTERNAL
        elif n_p <= 2:
            stage = ClineStage.BOUND_AFFIX if morpheme.is_grammatical else morpheme.stage
        else:
            stage = ClineStage.CLITIC if morpheme.is_grammatical else morpheme.stage

        return Morpheme(
            phonemes=p_list,
            root_family_id=morpheme.root_family_id,
            is_grammatical=morpheme.is_grammatical,
            stage=stage,
        )