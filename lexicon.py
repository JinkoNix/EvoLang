"""
lexicon.py — Procedural Seed Generation, Dynamic Derivational Morphology, and Logistic Saturation.
Driven by Damped 7D Cross-Modal Projection Tensors, Sonority-Governed Cluster Generation,
Bimoraic Metrical Foot Compounding, and Synchronized Morpheme Boundary Phonotactics.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from articulatory_space import Airstream, SecondaryArticulation
from phoneme import Phoneme, PhonemeKind
from word import Word, Morpheme, ClineStage, LexicalCategory
from semantics import SemanticVector, SemanticSpace, DerivationRecord, DerivationType
from energy import ArticulatoryEnergyModel, _get_sonority
from syllable import SyllableEngine, StressPattern, Syllable


# =====================================================================
# Continuous 7D -> 3D Cross-Modal Projection Tensors
# =====================================================================

def _project_vowel_target(
    vec: SemanticVector, 
    bias: Sequence[float] | None = None
) -> tuple[float, float, float]:
    """
    Projects 7D cognitive dimensions onto continuous 3D target vowel coordinates [h*, b*, r*]
    using damped hyperbolic tangents modulated by language-specific aesthetic bias.
    """
    c, a, v, p, d, s, e = vec.coords
    b0 = bias[0] if bias and len(bias) > 0 else 0.0
    b1 = bias[1] if bias and len(bias) > 1 else 0.0
    b2 = bias[2] if bias and len(bias) > 2 else 0.0

    h_shift = math.tanh(-0.90 * (e - 0.50) - 0.70 * (p - 0.50) + 0.65 * (d - 0.50) + 0.45 * (v - 0.50) + b0)
    target_h = max(0.0, min(6.0, 3.0 + 2.50 * h_shift))

    b_shift = math.tanh(0.85 * (e - 0.50) + 0.65 * (p - 0.50) - 0.80 * (v - 0.50) - 0.45 * (a - 0.50) + b1)
    target_b = max(0.0, min(2.0, 1.0 + 0.85 * b_shift))

    r_shift = math.tanh(0.70 * (e - 0.50) + 0.55 * (p - 0.50) - 0.55 * (d - 0.50) + b2)
    target_r = max(0.0, min(1.0, 0.50 + 0.45 * r_shift))

    return (target_h, target_b, target_r)


def _project_consonant_target(
    vec: SemanticVector, 
    bias: Sequence[float] | None = None
) -> tuple[float, float, float]:
    """
    Projects 7D cognitive dimensions onto continuous 3D target consonant coordinates [pl*, mn*, vc*]
    modulated by language-specific aesthetic bias.
    """
    c, a, v, p, d, s, e = vec.coords
    b3 = bias[3] if bias and len(bias) > 3 else 0.0
    b4 = bias[4] if bias and len(bias) > 4 else 0.0
    b5 = bias[5] if bias and len(bias) > 5 else 0.0

    pl_shift = -3.00 * (s - 0.50) * a + 3.50 * (p - 0.50) + 2.50 * (1.0 - c) + (b3 * 2.5)
    target_pl = max(0.0, min(10.0, 3.0 + pl_shift))

    mn_bias = int(round(b4 * 2.0))
    if a >= 0.65 and s >= 0.65:
        target_mn = 1.0
    elif d >= 0.60 and p >= 0.60:
        target_mn = 0.0 if (v <= 0.50) else 2.0
    elif e >= 0.65:
        target_mn = 7.0 if (a <= 0.50) else 6.0
    elif p >= 0.55:
        target_mn = 0.0
    else:
        target_mn = 4.0

    target_mn = max(0.0, min(7.0, target_mn + mn_bias))

    vc_shift = 0.80 * (v - 0.50) + 0.60 * (s - 0.50) - 0.50 * (p - 0.50) * d + b5
    target_vc = max(0.0, min(1.0, 0.50 + 0.50 * vc_shift))

    return (target_pl, target_mn, target_vc)


def _extract_metrical_head(word: Word, mode: str = "head") -> list[Phoneme]:
    syls = word.syllables if word.syllables else SyllableEngine.syllabify(word)
    if not syls or len(syls) <= 2:
        return list(word.phonemes)

    stressed_idx = next((i for i, s in enumerate(syls) if s.stressed), 0)

    if mode == "tail":
        extracted_syls = syls[-2:] if len(syls) >= 2 else syls[-1:]
    elif mode == "head":
        extracted_syls = syls[:2]
    else:
        if stressed_idx + 1 < len(syls):
            extracted_syls = syls[stressed_idx:stressed_idx + 2]
        elif stressed_idx > 0:
            extracted_syls = syls[stressed_idx - 1:stressed_idx + 1]
        else:
            extracted_syls = syls[:2]

    extracted: list[Phoneme] = []
    for s in extracted_syls:
        extracted.extend(s.to_phonemes())

    return extracted if extracted else list(word.phonemes)


# =====================================================================
# Lexicon Generator Class
# =====================================================================

class LexiconGenerator:
    """Procedurally synthesizes initial roots and dynamically expands vocabulary with carrying capacity."""

    @classmethod
    def weighted_choice_phoneme(
        cls,
        pool: Sequence[Phoneme],
        target_vector: SemanticVector | None = None,
        aesthetic_bias: Sequence[float] | None = None,
    ) -> Phoneme:
        """Samples a phoneme via continuous 3D Euclidean cross-modal Boltzmann resonance with bias."""
        if not pool:
            return Phoneme(PhonemeKind.VOWEL, (0.0, 1.0, 0.0))

        if target_vector is None:
            weights = [p.weight for p in pool]
            return random.choices(pool, weights=weights, k=1)[0].drift()

        is_vowel = (pool[0].kind == PhonemeKind.VOWEL)
        if is_vowel:
            th, tb, tr = _project_vowel_target(target_vector, bias=aesthetic_bias)
            resonance_weights = []
            for p in pool:
                h, b, r = p.point
                dist_sq = ((h - th) / 6.0) ** 2 + ((b - tb) / 2.0) ** 2 + ((r - tr) / 1.0) ** 2
                resonance = math.exp(-dist_sq / 0.35)
                resonance_weights.append(p.weight * resonance)
        else:
            tpl, tmn, tvc = _project_consonant_target(target_vector, bias=aesthetic_bias)
            resonance_weights = []
            for p in pool:
                pl, mn, vc = p.point
                dist_sq = ((pl - tpl) / 10.0) ** 2 + ((mn - tmn) / 7.0) ** 2 + ((vc - tvc) / 1.0) ** 2
                resonance = math.exp(-dist_sq / 0.35)
                resonance_weights.append(p.weight * resonance)

        return random.choices(pool, weights=resonance_weights, k=1)[0].drift()

    @classmethod
    def generate_legal_cluster(
        cls,
        c_pool: Sequence[Phoneme],
        is_onset: bool = True,
        syllable_complexity: float = 0.50,
    ) -> list[Phoneme]:
        if not c_pool:
            return []
        if len(c_pool) < 2:
            return [c_pool[0].drift(), c_pool[0].drift()]

        min_son_distance = 3.60 - (2.20 * syllable_complexity)

        for _ in range(35):
            c1 = random.choice(c_pool).drift()
            c2 = random.choice(c_pool).drift()

            son1 = _get_sonority(c1)
            son2 = _get_sonority(c2)

            mn1 = int(round(c1.point[1]))
            mn2 = int(round(c2.point[1]))
            is_sibilant_lead = (mn1 in (4, 5) and mn2 in (0, 2) and syllable_complexity >= 0.35)
            is_double_stop = (mn1 in (0, 2) and mn2 in (0, 2))

            if is_onset:
                delta_son = son2 - son1
                if (delta_son >= min_son_distance or is_sibilant_lead) and not is_double_stop:
                    return [c1, c2]
            else:
                if (son1 - son2 >= 0.5 or mn2 in (4, 5)):
                    return [c1, c2]

        stops = [c for c in c_pool if int(round(c.point[1])) in (0, 2)]
        liquids = [c for c in c_pool if _get_sonority(c) >= 4.0]
        sibilants = [c for c in c_pool if int(round(c.point[1])) in (4, 5)]

        if is_onset:
            if stops and liquids:
                return [random.choice(stops).drift(), random.choice(liquids).drift()]
            elif sibilants and stops and syllable_complexity >= 0.35:
                return [random.choice(sibilants).drift(), random.choice(stops).drift()]
        else:
            if liquids and stops:
                return [random.choice(liquids).drift(), random.choice(stops).drift()]
            elif stops and sibilants:
                return [random.choice(stops).drift(), random.choice(sibilants).drift()]

        return [c_pool[0].drift(), c_pool[-1].drift()]

    @classmethod
    def sample_phonotactic_template(cls, profile=None) -> str:
        s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
        vocalic_bias = max(0.0, 1.0 - s_comp)

        raw_on_none = 0.08 + 0.32 * vocalic_bias
        raw_on_cc = 0.65 / (1.0 + math.exp(-7.0 * (s_comp - 0.45)))
        raw_on_c = 1.00
        total_on = raw_on_none + raw_on_c + raw_on_cc
        p_on_none = raw_on_none / total_on
        p_on_c = raw_on_c / total_on

        r_on = random.random()
        if r_on < p_on_none:
            onset = "NONE"
        elif r_on < (p_on_none + p_on_c):
            onset = "C"
        else:
            onset = "CC"

        raw_nuc_vv = 0.10 + 0.20 * vocalic_bias
        raw_nuc_v1v2 = 0.15 + 0.25 * vocalic_bias
        raw_nuc_v = 1.00
        total_nuc = raw_nuc_v + raw_nuc_vv + raw_nuc_v1v2

        r_nuc = random.random()
        if r_nuc < (raw_nuc_v / total_nuc):
            nuc = "V"
        elif r_nuc < ((raw_nuc_v + raw_nuc_vv) / total_nuc):
            nuc = "VV"
        else:
            nuc = "V1V2"

        raw_coda_none = 1.00 / (1.0 + math.exp(6.0 * (s_comp - 0.35)))
        raw_coda_cc = 0.60 / (1.0 + math.exp(-8.0 * (s_comp - 0.60)))
        raw_coda_c = 0.85 if s_comp >= 0.28 else 0.15
        total_coda = raw_coda_none + raw_coda_c + raw_coda_cc

        r_coda = random.random()
        if r_coda < (raw_coda_none / total_coda):
            coda = "NONE"
        elif r_coda < ((raw_coda_none + raw_coda_c) / total_coda):
            coda = "C"
        else:
            coda = "CC"

        if onset == "NONE" and coda == "NONE":
            onset = "C"

        p_disyllabic = 0.20 + 0.35 * vocalic_bias
        if random.random() < p_disyllabic:
            ext_on = "C" if (coda != "NONE" or random.random() < 0.70) else "NONE"
            ext_nuc = "V" if nuc in ("VV", "V1V2") else ("VV" if random.random() < 0.15 else "V")
            ext_coda = "NONE" if (s_comp < 0.40 or random.random() < 0.60) else "C"
            return f"{onset}_{nuc}_{coda}.{ext_on}_{ext_nuc}_{ext_coda}"

        return f"{onset}_{nuc}_{coda}"

    @classmethod
    def generate_phonetic_root(
        cls,
        consonants: Sequence[Phoneme],
        vowels: Sequence[Phoneme],
        seed_vector: SemanticVector | None = None,
        template: str = "C_V_C",
        existing_forms: set[str] | None = None,
        syllable_complexity: float = 0.50,
        aesthetic_bias: Sequence[float] | None = None,
    ) -> list[Phoneme]:
        existing = existing_forms or set()
        tpl = template or "C_V_C"

        for _ in range(250):
            root_phonemes: list[Phoneme] = []
            syl_patterns = tpl.split(".")

            for s_pat in syl_patterns:
                tokens = s_pat.split("_") if "_" in s_pat else ["C", "V", "C"]
                onset_tok = tokens[0] if len(tokens) > 0 else "C"
                nuc_tok   = tokens[1] if len(tokens) > 1 else "V"
                coda_tok  = tokens[2] if len(tokens) > 2 else "NONE"

                if onset_tok == "C" and consonants:
                    root_phonemes.append(cls.weighted_choice_phoneme(consonants, seed_vector, aesthetic_bias=aesthetic_bias))
                elif onset_tok == "CC" and consonants:
                    root_phonemes.extend(cls.generate_legal_cluster(consonants, is_onset=True, syllable_complexity=syllable_complexity))

                if nuc_tok == "V" and vowels:
                    root_phonemes.append(cls.weighted_choice_phoneme(vowels, seed_vector, aesthetic_bias=aesthetic_bias))
                elif nuc_tok == "VV" and vowels:
                    chosen_v = cls.weighted_choice_phoneme(vowels, seed_vector, aesthetic_bias=aesthetic_bias)
                    root_phonemes.extend([chosen_v, chosen_v])
                elif nuc_tok == "V1V2" and vowels:
                    v1 = cls.weighted_choice_phoneme(vowels, seed_vector, aesthetic_bias=aesthetic_bias)
                    high_targets = [v for v in vowels if v.point[0] >= 3.8]
                    v2 = cls.weighted_choice_phoneme(high_targets, aesthetic_bias=aesthetic_bias) if high_targets else cls.weighted_choice_phoneme(vowels, aesthetic_bias=aesthetic_bias)
                    root_phonemes.extend([v1, v2])

                if coda_tok == "C" and consonants:
                    root_phonemes.append(cls.weighted_choice_phoneme(consonants, seed_vector, aesthetic_bias=aesthetic_bias))
                elif coda_tok == "CC" and consonants:
                    root_phonemes.extend(cls.generate_legal_cluster(consonants, is_onset=False, syllable_complexity=syllable_complexity))

            form_candidate = "".join(p.ipa for p in root_phonemes)
            if form_candidate not in existing and root_phonemes:
                return root_phonemes

        return root_phonemes

    @classmethod
    def populate_proto_lexicon(
        cls,
        consonants: Sequence[Phoneme],
        vowels: Sequence[Phoneme],
        env=None,
        profile=None,
        population: float = 0.50,
        **kwargs,
    ) -> list[Word]:
        words: list[Word] = []
        used_forms: set[str] = set()
        stress_pat = getattr(profile, "stress_pattern", StressPattern.PENULTIMATE) if profile else StressPattern.PENULTIMATE
        tone_tier = getattr(profile, "tone_tier", 0) if profile else 0
        s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
        aesthetic_bias = getattr(profile, "aesthetic_bias", None)

        for seed_vec in SemanticSpace.BIOLOGICAL_COGNITIVE_SEEDS:
            template = cls.sample_phonotactic_template(profile)
            root_phonemes = cls.generate_phonetic_root(
                consonants=consonants,
                vowels=vowels,
                seed_vector=seed_vec,
                template=template,
                existing_forms=used_forms,
                syllable_complexity=s_comp,
                aesthetic_bias=aesthetic_bias,
            )

            category = LexicalCategory.FUNCTIONAL_CLOSED if seed_vec.is_grammatical else LexicalCategory.CONTENT_OPEN
            w = Word(
                vector=seed_vec,
                phonemes=root_phonemes,
                derivation=DerivationRecord(root_family_id=len(words) + 1),
                usage_frequency=1.5,
                category=category,
                generation_born=0,
            )
            w = SyllableEngine.apply_prosody(w, pattern=stress_pat, tone_tier=tone_tier)
            words.append(w)
            used_forms.add(w.plain_form)

        entities = [w for w in words if w.vector.concreteness >= 0.65]
        modulators = [w for w in words if w.vector.concreteness < 0.65]

        pop = population if population is not None else (getattr(env, "population", None) or 0.50)
        num_compounds = int(round(15 + (pop * 25)))

        for _ in range(num_compounds):
            if not entities or not modulators:
                break
            w_head = random.choice(entities)
            w_mod = random.choice(modulators)

            w_head.record_usage(0.5)
            w_mod.record_usage(0.5)

            blended_vec = w_head.vector.blend(w_mod.vector, weight=0.60)
            
            p_h = _extract_metrical_head(w_head)
            p_m = _extract_metrical_head(w_mod)

            p_h_clean = ArticulatoryEnergyModel.repair_phonemes(p_h, profile)
            p_m_clean = ArticulatoryEnergyModel.repair_phonemes(p_m, profile)

            m1 = Morpheme(p_h_clean, root_family_id=w_head.derivation.root_family_id, stage=ClineStage.FREE_ROOT)
            m2 = Morpheme(p_m_clean, root_family_id=w_mod.derivation.root_family_id, stage=ClineStage.FREE_ROOT)

            raw_phonemes = m1.phonemes + m2.phonemes
            cleaned_phonemes = ArticulatoryEnergyModel.repair_phonemes(raw_phonemes, profile)

            l1 = len(m1.phonemes)
            l_raw = len(raw_phonemes) or 1
            l_clean = len(cleaned_phonemes)
            split_idx = max(1, min(l_clean - 1, int(round(l_clean * (l1 / l_raw)))))

            m1.phonemes = list(cleaned_phonemes[:split_idx])
            m2.phonemes = list(cleaned_phonemes[split_idx:])

            compound_word = Word(
                vector=blended_vec,
                phonemes=cleaned_phonemes,
                morphemes=[m1, m2],
                derivation=DerivationRecord(
                    root_family_id=w_head.derivation.root_family_id,
                    derivation_type=DerivationType.COMPOUND,
                    local_parent_ids=(w_head.id, w_mod.id),
                ),
                usage_frequency=1.2,
                category=LexicalCategory.CONTENT_OPEN,
                generation_born=0,
            )
            compound_word = SyllableEngine.apply_prosody(compound_word, pattern=stress_pat, tone_tier=tone_tier)
            words.append(compound_word)
            used_forms.add(compound_word.plain_form)

        return words

    @classmethod
    def expand_vocabulary(
        cls,
        lang,
        growth_rate: float = 0.03,
        d_sem_min: float = 0.06,
        current_generation: int = 0,
        profile=None,
    ) -> list[Word]:
        living_words = list(lang.words.values())
        if not living_words:
            return []

        prof = profile or getattr(lang, "profile", None)
        pop = getattr(lang, "population", None) or 0.50
        
        attn = lang.cultural_attention
        cultural_inertia = (attn[2] ** 1.4 * attn[5]) / (1.0 + attn[4] * 0.8)
        
        # 1. Asymptotic Biological Carrying Capacity
        k_carrying_capacity = round(3200.0 + 1800.0 * math.tanh(min(10.0, float(pop)) * 1.2) * (1.0 - min(0.35, float(cultural_inertia) * 0.20)))
        n_current = float(len(living_words))
        
        if n_current >= k_carrying_capacity:
            return []
        
        # 2. Logistic Saturation Damping & Semantic Deficit Boost
        logistic_damping = max(0.0, 1.0 - (n_current / k_carrying_capacity)) ** 1.5
        semantic_deficit_boost = max(1.0, 600.0 / max(45.0, n_current))

        # 3. Effective Generation Attempts
        effective_attempts = int(round(n_current * growth_rate * semantic_deficit_boost * logistic_damping))
        effective_attempts = max(2, min(40, effective_attempts))

        weights = lang.cultural_attention
        new_born: list[Word] = []
        occupied_forms: set[str] = {w.plain_form for w in living_words}

        c_pool = lang.consonants if lang.consonants else list(lang.phonemes.values())
        v_pool = lang.vowels if lang.vowels else [Phoneme(PhonemeKind.VOWEL, (0.0, 1.0, 0.0))]
        d_deriv_min = d_sem_min * 0.35

        family_members: dict[int, list[Word]] = {}
        for w in living_words:
            fam_id = w.derivation.root_family_id
            if fam_id not in family_members:
                family_members[fam_id] = []
            family_members[fam_id].append(w)
        all_family_ids = list(family_members.keys())

        max_existing_family = max((w.derivation.root_family_id for w in living_words), default=0)

        s_comp = getattr(prof, "syllable_complexity", 0.50) if prof else 0.50
        s_idx  = getattr(prof, "synthesis_index", 0.50) if prof else 0.50
        aesthetic_bias = getattr(prof, "aesthetic_bias", None)

        # =====================================================================
        # NUMERICALLY STABLE EXPONENTS (Prevents math.exp OverflowError!)
        # =====================================================================
        c_inertia = max(0.10, min(10.0, float(cultural_inertia)))
        c_pop = float(pop)
        if c_pop > 50.0:
            c_pop = c_pop / 100000.0  # Normalize raw citizen count down to index scale
        c_pop = max(0.05, min(8.0, c_pop))

        e_affix = 2.00 * s_idx
        e_compound = 2.00 * (1.00 - s_idx)
        e_mitosis = 1.00 * c_inertia + 0.80 * (1.00 - s_idx)
        e_denovo = 1.00 * c_pop * (1.00 - min(0.80, c_inertia))

        # Softmax max-subtraction: all exponents are <= 0, so math.exp never overflows!
        max_e = max(e_affix, e_compound, e_mitosis, e_denovo)
        z_affix = math.exp(max(-50.0, e_affix - max_e))
        z_compound = math.exp(max(-50.0, e_compound - max_e))
        z_mitosis = math.exp(max(-50.0, e_mitosis - max_e))
        z_denovo = math.exp(max(-50.0, e_denovo - max_e))

        total_z = z_affix + z_compound + z_mitosis + z_denovo
        p_denovo = z_denovo / total_z
        p_mitosis = p_denovo + (z_mitosis / total_z)
        p_affix = p_mitosis + (z_affix / total_z)

        stress_pat = getattr(prof, "stress_pattern", StressPattern.NATURAL_WEIGHT) if prof else StressPattern.NATURAL_WEIGHT

        def get_compounding_fitness(w: Word) -> float:
            w_len = len(w.phonemes)
            length_penalty = 1.0 + 0.18 * (max(0, w_len - 3) ** 2.2)
            return math.sqrt(w.usage_frequency + 0.10) / length_penalty

        # =====================================================================
        # BATCH CANDIDATE GENERATION & VECTORIZED GPU VALIDATION
        # =====================================================================
        from gpu_accelerator import GPUSemanticEngine

        candidate_plans = []

        for _ in range(effective_attempts):
            roll = random.random()

            # Pathway 1: De Novo Root
            if roll < p_denovo:
                coords = [max(0.05, min(0.95, random.betavariate(1.5, 1.5))) for _ in range(7)]
                candidate_plans.append({
                    "pathway": "denovo",
                    "vec": SemanticVector(*coords),
                    "d_min": d_sem_min,
                    "parent_a": None,
                    "parent_b": None,
                })

            # Pathway 2: Semantic Mitosis
            elif roll < p_mitosis:
                chosen_fam_id = random.choice(all_family_ids)
                candidates_in_fam = [w for w in family_members[chosen_fam_id] if w.category != LexicalCategory.FUNCTIONAL_CLOSED]
                if not candidates_in_fam:
                    continue
                parent_a = random.choices(candidates_in_fam, weights=[get_compounding_fitness(w) for w in candidates_in_fam], k=1)[0]
                sigma = 0.16 / (1.0 + math.sqrt(parent_a.usage_frequency))
                split_dim = max(range(7), key=lambda i: weights[i] * random.random())
                offset = [0.0] * 7
                offset[split_dim] = random.gauss(0.0, sigma)
                cand_vec = parent_a.vector.drift(offset)
                candidate_plans.append({
                    "pathway": "mitosis",
                    "vec": cand_vec,
                    "d_min": d_deriv_min,
                    "parent_a": parent_a,
                    "parent_b": None,
                })

            # Pathway 3: Derivational Affixation
            elif roll < p_affix:
                chosen_fam_id = random.choice(all_family_ids)
                base_cands = [w for w in family_members[chosen_fam_id] if w.category != LexicalCategory.FUNCTIONAL_CLOSED]
                if not base_cands:
                    continue
                base_stem = random.choices(base_cands, weights=[get_compounding_fitness(w) for w in base_cands], k=1)[0]
                affix_fam = random.choice(all_family_ids)
                affix_cands = [w for w in family_members[affix_fam] if w.category != LexicalCategory.FUNCTIONAL_CLOSED]
                if not affix_cands:
                    continue
                affix_host = min(affix_cands, key=lambda w: (len(w.phonemes), -w.usage_frequency))
                deriv_vec = base_stem.vector.blend(affix_host.vector, weight=0.70)
                candidate_plans.append({
                    "pathway": "affix",
                    "vec": deriv_vec,
                    "d_min": d_deriv_min,
                    "parent_a": base_stem,
                    "parent_b": affix_host,
                })

            # Pathway 4: Interstitial Compounding
            else:
                fam_a = random.choice(all_family_ids)
                fam_b = random.choice(all_family_ids)
                if fam_a == fam_b:
                    continue
                cands_a = [w for w in family_members[fam_a] if w.category != LexicalCategory.FUNCTIONAL_CLOSED]
                cands_b = [w for w in family_members[fam_b] if w.category != LexicalCategory.FUNCTIONAL_CLOSED]
                if not cands_a or not cands_b:
                    continue
                parent_a = random.choices(cands_a, weights=[get_compounding_fitness(w) for w in cands_a], k=1)[0]
                parent_b = random.choices(cands_b, weights=[get_compounding_fitness(w) for w in cands_b], k=1)[0]
                blended_vec = parent_a.vector.blend(parent_b.vector, weight=random.uniform(0.40, 0.60))
                candidate_plans.append({
                    "pathway": "compound",
                    "vec": blended_vec,
                    "d_min": d_deriv_min,
                    "parent_a": parent_a,
                    "parent_b": parent_b,
                })

        if not candidate_plans:
            return []

        existing_coords = [w.vector.coords for w in living_words]
        batch_coords = [plan["vec"].coords for plan in candidate_plans]
        effective_d_min = min(plan["d_min"] for plan in candidate_plans)

        accepted_mask = GPUSemanticEngine.filter_candidate_density_gpu(
            candidates_coords=batch_coords,
            existing_coords=existing_coords,
            weights=weights,
            d_min=effective_d_min,
        )

        for plan, is_valid in zip(candidate_plans, accepted_mask):
            if not is_valid:
                continue

            pw = plan["pathway"]
            cand_vec = plan["vec"]

            if pw == "denovo" and c_pool:
                template = cls.sample_phonotactic_template(prof)
                new_phonemes = cls.generate_phonetic_root(
                    consonants=c_pool, vowels=v_pool, seed_vector=cand_vec,
                    template=template, syllable_complexity=s_comp, aesthetic_bias=aesthetic_bias,
                )
                cleaned = ArticulatoryEnergyModel.repair_phonemes(new_phonemes, prof)
                max_existing_family += 1
                new_word = Word(
                    vector=cand_vec, phonemes=cleaned,
                    derivation=DerivationRecord(root_family_id=max_existing_family, derivation_type=DerivationType.ROOT, local_parent_ids=()),
                    usage_frequency=0.85, category=LexicalCategory.CONTENT_OPEN, generation_born=current_generation,
                )
                new_word = SyllableEngine.apply_prosody(new_word, pattern=stress_pat)
                if new_word.plain_form not in occupied_forms:
                    lang.add_word(new_word)
                    living_words.append(new_word)
                    new_born.append(new_word)
                    occupied_forms.add(new_word.plain_form)

            elif pw == "mitosis" and c_pool:
                p_a = plan["parent_a"]
                p_a.record_usage(0.30)
                p_stem = list(p_a.phonemes)
                v_indices = [i for i, p in enumerate(p_stem) if p.kind == PhonemeKind.VOWEL]
                c_indices = [i for i, p in enumerate(p_stem) if p.kind == PhonemeKind.CONSONANT]

                if len(p_stem) > 3:
                    if v_indices and random.random() < 0.60:
                        v_idx = random.choice(v_indices)
                        p_stem[v_idx] = cls.weighted_choice_phoneme(v_pool, cand_vec, aesthetic_bias=aesthetic_bias)
                    elif c_indices:
                        c_idx = random.choice(c_indices)
                        p_stem[c_idx] = cls.weighted_choice_phoneme(c_pool, cand_vec, aesthetic_bias=aesthetic_bias)
                    raw_p = p_stem
                else:
                    deriv_p = cls.weighted_choice_phoneme(c_pool, cand_vec, aesthetic_bias=aesthetic_bias)
                    raw_p = [deriv_p] + p_stem if getattr(prof, "prefix_ratio", 0.50) > 0.50 else p_stem + [deriv_p]

                cleaned = ArticulatoryEnergyModel.repair_phonemes(raw_p, prof)
                new_word = Word(
                    vector=cand_vec, phonemes=cleaned,
                    derivation=DerivationRecord(root_family_id=p_a.derivation.root_family_id, derivation_type=DerivationType.SEMANTIC_SPLIT, local_parent_ids=(p_a.id,)),
                    usage_frequency=0.90, category=LexicalCategory.CONTENT_OPEN, generation_born=current_generation,
                )
                new_word = SyllableEngine.apply_prosody(new_word, pattern=stress_pat)
                if new_word.plain_form not in occupied_forms:
                    lang.add_word(new_word)
                    living_words.append(new_word)
                    new_born.append(new_word)
                    family_members[p_a.derivation.root_family_id].append(new_word)
                    occupied_forms.add(new_word.plain_form)

            elif pw == "affix":
                b_stem = plan["parent_a"]
                a_host = plan["parent_b"]
                is_prefix = (random.random() < getattr(prof, "prefix_ratio", 0.50))
                p_stem_foot = _extract_metrical_head(b_stem, mode="head")
                p_affix_foot = _extract_metrical_head(a_host, mode="tail" if is_prefix else "head")
                p_s_clean = ArticulatoryEnergyModel.repair_phonemes(p_stem_foot, prof)
                p_a_clean = ArticulatoryEnergyModel.repair_phonemes(p_affix_foot, prof)
                m_stem = Morpheme(p_s_clean, root_family_id=b_stem.derivation.root_family_id)
                m_affix = Morpheme(p_a_clean, root_family_id=a_host.derivation.root_family_id, stage=ClineStage.BOUND_AFFIX)
                morphemes = [m_affix, m_stem] if is_prefix else [m_stem, m_affix]
                raw_phonemes = (m_affix.phonemes + m_stem.phonemes) if is_prefix else (m_stem.phonemes + m_affix.phonemes)
                cleaned = ArticulatoryEnergyModel.repair_phonemes(raw_phonemes, prof)

                l0 = len(morphemes[0].phonemes)
                l_raw = len(raw_phonemes) or 1
                l_clean = len(cleaned)
                split_idx = max(1, min(l_clean - 1, int(round(l_clean * (l0 / l_raw)))))
                morphemes[0].phonemes = list(cleaned[:split_idx])
                morphemes[1].phonemes = list(cleaned[split_idx:])

                new_word = Word(
                    vector=cand_vec, phonemes=cleaned, morphemes=morphemes,
                    derivation=DerivationRecord(root_family_id=b_stem.derivation.root_family_id, derivation_type=DerivationType.AFFIXATION, local_parent_ids=(b_stem.id, a_host.id)),
                    usage_frequency=1.0, category=LexicalCategory.CONTENT_OPEN, generation_born=current_generation,
                )
                new_word = SyllableEngine.apply_prosody(new_word, pattern=stress_pat)
                if new_word.plain_form not in occupied_forms:
                    lang.add_word(new_word)
                    living_words.append(new_word)
                    new_born.append(new_word)
                    family_members[b_stem.derivation.root_family_id].append(new_word)
                    occupied_forms.add(new_word.plain_form)

            elif pw == "compound":
                p_a = plan["parent_a"]
                p_b = plan["parent_b"]
                p_a.record_usage(0.30)
                p_b.record_usage(0.30)
                is_prefix = (random.random() < getattr(prof, "prefix_ratio", 0.50))
                p_clip_a = 1.0 - math.exp(-0.25 * max(0, len(p_a.phonemes) - 4))
                p_clip_b = 1.0 - math.exp(-0.25 * max(0, len(p_b.phonemes) - 4))
                p_a_foot = _extract_metrical_head(p_a, mode="head" if random.random() < 0.60 else "tail") if random.random() < p_clip_a else list(p_a.phonemes)
                p_b_foot = _extract_metrical_head(p_b, mode="tail" if random.random() < 0.60 else "head") if random.random() < p_clip_b else list(p_b.phonemes)
                p_a_clean = ArticulatoryEnergyModel.repair_phonemes(p_a_foot, prof)
                p_b_clean = ArticulatoryEnergyModel.repair_phonemes(p_b_foot, prof)
                m1 = Morpheme(p_a_clean, root_family_id=p_a.derivation.root_family_id)
                m2 = Morpheme(p_b_clean, root_family_id=p_b.derivation.root_family_id)
                morphemes = [m2, m1] if is_prefix else [m1, m2]
                raw_phonemes = (m2.phonemes + m1.phonemes) if is_prefix else (m1.phonemes + m2.phonemes)
                cleaned = ArticulatoryEnergyModel.repair_phonemes(raw_phonemes, prof)

                l0 = len(morphemes[0].phonemes)
                l_raw = len(raw_phonemes) or 1
                l_clean = len(cleaned)
                split_idx = max(1, min(l_clean - 1, int(round(l_clean * (l0 / l_raw)))))
                morphemes[0].phonemes = list(cleaned[:split_idx])
                morphemes[1].phonemes = list(cleaned[split_idx:])

                new_word = Word(
                    vector=cand_vec, phonemes=cleaned, morphemes=morphemes,
                    derivation=DerivationRecord(root_family_id=p_a.derivation.root_family_id, derivation_type=DerivationType.COMPOUND, local_parent_ids=(p_a.id, p_b.id)),
                    usage_frequency=1.1, category=LexicalCategory.CONTENT_OPEN, generation_born=current_generation,
                )
                new_word = SyllableEngine.apply_prosody(new_word, pattern=stress_pat)
                if new_word.plain_form not in occupied_forms:
                    lang.add_word(new_word)
                    living_words.append(new_word)
                    new_born.append(new_word)
                    family_members[p_a.derivation.root_family_id].append(new_word)
                    occupied_forms.add(new_word.plain_form)

        return new_born