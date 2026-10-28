"""
lexicon.py — Continuous Thermodynamic Neologisms, Compression, and Information Dynamics.
"""

from __future__ import annotations

import math
import random
from typing import Sequence
import numpy as np

from articulatory_space import Airstream, SecondaryArticulation
from phoneme import Phoneme, PhonemeKind
from word import Word, Morpheme, ClineStage, LexicalCategory
from semantics import SemanticVector, SemanticSpace, DerivationRecord, DerivationType
from energy import ArticulatoryEnergyModel, _get_sonority
from syllable import SyllableEngine, StressPattern
from gpu_accelerator import GPUSemanticEngine


def _project_vowel_target(vec: SemanticVector, bias: Sequence[float] | None = None) -> tuple[float, float, float]:
    c, a, v, p, d, s, e = vec.coords
    b0 = bias[0] if bias and len(bias) > 0 else 0.0
    b1 = bias[1] if bias and len(bias) > 1 else 0.0
    b2 = bias[2] if bias and len(bias) > 2 else 0.0

    size_factor = (p * 0.60 + e * 0.40) - (v * 0.30)
    target_h = max(0.0, min(6.0, 3.0 - 2.80 * (size_factor - 0.50) + b0))
    target_b = max(0.0, min(2.0, 1.0 + 1.20 * (size_factor - 0.50) + b1))
    target_r = max(0.0, min(1.0, 0.50 + 0.80 * (size_factor - 0.50) + b2))
    return (target_h, target_b, target_r)


def _project_consonant_target(vec: SemanticVector, bias: Sequence[float] | None = None) -> tuple[float, float, float]:
    c, a, v, p, d, s, e = vec.coords
    b3 = bias[3] if bias and len(bias) > 3 else 0.0
    b4 = bias[4] if bias and len(bias) > 4 else 0.0
    b5 = bias[5] if bias and len(bias) > 5 else 0.0

    target_pl = max(0.0, min(10.0, 3.0 + 4.0 * (p - s) + b3))

    # Continuous manner projection based on dynamism and potency
    target_mn = 7.0 * (1.0 - math.tanh(d * 1.5)) * (v * 0.8) + (0.0 if p >= 0.5 else 4.0) * math.tanh(d * 1.2)
    target_mn = max(0.0, min(7.0, target_mn + b4))
    target_vc = max(0.0, min(1.0, 0.50 + 0.50 * (v - p) + b5))
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


class LexiconGenerator:
    @classmethod
    def weighted_choice_phoneme(
        cls,
        pool: Sequence[Phoneme],
        target_vector: SemanticVector | None = None,
        aesthetic_bias: Sequence[float] | None = None,
    ) -> Phoneme:
        if not pool:
            return Phoneme(PhonemeKind.VOWEL, (0.0, 1.0, 0.0))

        if target_vector is None:
            weights = [p.weight for p in pool]
            return random.choices(pool, weights=weights, k=1)[0].drift()

        is_vowel = (pool[0].kind == PhonemeKind.VOWEL)
        if is_vowel:
            th, tb, tr = _project_vowel_target(target_vector, bias=aesthetic_bias)
            weights = []
            for p in pool:
                h, b, r = p.point
                dist_sq = ((h - th) / 6.0) ** 2 + ((b - tb) / 2.0) ** 2 + ((r - tr) / 1.0) ** 2
                weights.append(p.weight * math.exp(-dist_sq / 0.40))
        else:
            tpl, tmn, tvc = _project_consonant_target(target_vector, bias=aesthetic_bias)
            weights = []
            for p in pool:
                pl, mn, vc = p.point
                dist_sq = ((pl - tpl) / 10.0) ** 2 + ((mn - tmn) / 7.0) ** 2 + ((vc - tvc) / 1.0) ** 2
                weights.append(p.weight * math.exp(-dist_sq / 0.40))

        total_w = sum(weights) or 1.0
        normalized = [w / total_w for w in weights]
        return random.choices(pool, weights=normalized, k=1)[0].drift()

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
            return [c_pool[0].drift()]

        min_son_distance = 3.60 - (2.20 * syllable_complexity)

        for _ in range(40):
            c1 = random.choice(c_pool)
            c2 = random.choice(c_pool)

            mn1 = int(round(c1.point[1]))
            mn2 = int(round(c2.point[1]))

            is_homorganic = abs(c1.point[0] - c2.point[0]) < 0.8
            is_double_stop = (mn1 in (0, 2) and mn2 in (0, 2))
            if is_homorganic or is_double_stop:
                continue

            son1 = _get_sonority(c1)
            son2 = _get_sonority(c2)

            if is_onset:
                is_sibilant_lead = (mn1 in (4, 5) and mn2 in (0, 2) and syllable_complexity >= 0.35)
                delta_son = son2 - son1
                if delta_son >= min_son_distance or is_sibilant_lead:
                    return [c1.drift(), c2.drift()]
            else:
                is_sibilant_coda = (mn2 in (4, 5))
                delta_son = son1 - son2
                if delta_son >= 0.5 or is_sibilant_coda:
                    return [c1.drift(), c2.drift()]

        return [random.choice(c_pool).drift()]

    @classmethod
    def sample_phonotactic_template(cls, profile=None) -> list[tuple[int, str, int]]:
        s_comp = getattr(profile, "syllable_complexity", 0.50) if profile else 0.50
        vocalic_bias = max(0.0, 1.0 - s_comp)

        # Continuous logistic activation for onset and coda complexity
        p_cc_onset = 1.0 / (1.0 + math.exp(-8.0 * (s_comp - 0.60)))
        p_no_onset = 0.05 + 0.25 * vocalic_bias

        r_on = random.random()
        on_len = 0 if r_on < p_no_onset else (2 if r_on > (1.0 - p_cc_onset) else 1)

        r_nuc = random.random()
        nuc_type = "VV" if r_nuc < 0.15 else ("V1V2" if r_nuc < 0.30 else "V")

        p_coda = 1.0 / (1.0 + math.exp(-6.0 * (s_comp - 0.40)))
        p_cc_coda = 1.0 / (1.0 + math.exp(-9.0 * (s_comp - 0.70)))
        coda_len = 2 if random.random() < p_cc_coda else (1 if random.random() < p_coda else 0)

        syl1 = (on_len, nuc_type, coda_len)
        if random.random() < (0.25 + 0.35 * vocalic_bias):
            syl2 = (1 if coda_len > 0 or random.random() < 0.70 else 0, "V", 0 if random.random() < 0.65 else 1)
            return [syl1, syl2]

        return [syl1]

    @classmethod
    def generate_phonetic_root(
        cls,
        consonants: Sequence[Phoneme],
        vowels: Sequence[Phoneme],
        seed_vector: SemanticVector | None = None,
        template: list[tuple[int, str, int]] | None = None,
        existing_forms: set[str] | None = None,
        syllable_complexity: float = 0.50,
        aesthetic_bias: Sequence[float] | None = None,
    ) -> list[Phoneme]:
        existing = existing_forms or set()
        tpl = template or [(1, "V", 1)]

        for _ in range(120):
            root_phonemes: list[Phoneme] = []

            for on_len, nuc_type, coda_len in tpl:
                if on_len == 1 and consonants:
                    root_phonemes.append(cls.weighted_choice_phoneme(consonants, seed_vector, aesthetic_bias))
                elif on_len >= 2 and consonants:
                    root_phonemes.extend(cls.generate_legal_cluster(consonants, is_onset=True, syllable_complexity=syllable_complexity))

                if nuc_type == "V" and vowels:
                    root_phonemes.append(cls.weighted_choice_phoneme(vowels, seed_vector, aesthetic_bias))
                elif nuc_type == "VV" and vowels:
                    v = cls.weighted_choice_phoneme(vowels, seed_vector, aesthetic_bias)
                    root_phonemes.extend([v, v])
                elif nuc_type == "V1V2" and vowels:
                    v1 = cls.weighted_choice_phoneme(vowels, seed_vector, aesthetic_bias)
                    high_v = [v for v in vowels if v.point[0] >= 3.8] or vowels
                    v2 = cls.weighted_choice_phoneme(high_v, aesthetic_bias=aesthetic_bias)
                    root_phonemes.extend([v1, v2])

                if coda_len == 1 and consonants:
                    root_phonemes.append(cls.weighted_choice_phoneme(consonants, seed_vector, aesthetic_bias))
                elif coda_len >= 2 and consonants:
                    root_phonemes.extend(cls.generate_legal_cluster(consonants, is_onset=False, syllable_complexity=syllable_complexity))

            candidate_form = "".join(p.ipa for p in root_phonemes)
            if candidate_form and candidate_form not in existing:
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
            w = SyllableEngine.apply_prosody(w, pattern=stress_pat)
            words.append(w)
            used_forms.add(w.plain_form)

        entities = [w for w in words if w.vector.concreteness >= 0.65]
        modulators = [w for w in words if w.vector.concreteness < 0.65]

        num_compounds = int(round(12 + (population * 20)))
        for _ in range(num_compounds):
            if not entities or not modulators:
                break
            w_head = random.choice(entities)
            w_mod = random.choice(modulators)

            blended_vec = w_head.vector.blend(w_mod.vector, weight=0.60)
            p_h = ArticulatoryEnergyModel.repair_phonemes(_extract_metrical_head(w_head), profile)
            p_m = ArticulatoryEnergyModel.repair_phonemes(_extract_metrical_head(w_mod), profile)

            m1 = Morpheme(p_h, root_family_id=w_head.derivation.root_family_id)
            m2 = Morpheme(p_m, root_family_id=w_mod.derivation.root_family_id)

            raw = m1.phonemes + m2.phonemes
            cleaned = ArticulatoryEnergyModel.repair_phonemes(raw, profile)

            split = max(1, min(len(cleaned) - 1, len(m1.phonemes)))
            m1.phonemes = cleaned[:split]
            m2.phonemes = cleaned[split:]

            comp_word = Word(
                vector=blended_vec,
                phonemes=cleaned,
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
            comp_word = SyllableEngine.apply_prosody(comp_word, pattern=stress_pat)
            words.append(comp_word)
            used_forms.add(comp_word.plain_form)

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
        pop = getattr(lang, "population", 0.50)
        s_idx = getattr(prof, "synthesis_index", 0.50) if prof else 0.50
        attn = lang.cultural_attention
        weights = attn

        k_capacity = round(2500.0 * (1.0 + math.tanh(pop * 1.5)))
        n_current = len(living_words)
        if n_current >= k_capacity:
            return []

        damping = max(0.0, 1.0 - (n_current / k_capacity))
        attempts = max(2, min(30, int(round(n_current * growth_rate * damping))))

        c_pool = lang.consonants or list(lang.phonemes.values())
        v_pool = lang.vowels or [Phoneme(PhonemeKind.VOWEL, (0.0, 1.0, 0.0))]
        s_comp = getattr(prof, "syllable_complexity", 0.50) if prof else 0.50
        stress_pat = getattr(prof, "stress_pattern", StressPattern.NATURAL_WEIGHT) if prof else StressPattern.NATURAL_WEIGHT

        # Continuous Boltzmann Neologism Selector
        # High synthesis index -> favors Affixation
        # Low synthesis index (analytic) -> favors Compounding
        # Saturated semantic space / high density -> drives De Novo Root splitting
        semantic_density = min(1.0, n_current / 800.0)
        mean_len = sum(len(w.phonemes) for w in living_words) / max(1, n_current)
        length_penalty = math.tanh(max(0.0, mean_len - 5.0))

        e_root = 1.2 * (1.0 - semantic_density) - 0.4 * length_penalty
        e_comp = 1.4 * (1.0 - s_idx) - 0.8 * length_penalty
        e_affx = 1.5 * s_idx + 0.6 * length_penalty

        potentials = np.array([e_root, e_comp, e_affx], dtype=np.float32)
        exp_pots = np.exp(potentials - np.max(potentials))
        neologism_probs = exp_pots / np.sum(exp_pots)

        max_family_id = max((w.derivation.root_family_id for w in living_words), default=0)
        occupied_forms = {w.plain_form for w in living_words}
        new_born: list[Word] = []

        candidate_plans = []
        for _ in range(attempts):
            strategy = random.choices(["root", "compound", "affix"], weights=neologism_probs, k=1)[0]
            if strategy == "root" or len(living_words) < 10:
                coords = [float(min(0.98, max(0.02, random.gauss(0.5, 0.22)))) for _ in range(7)]
                candidate_plans.append({"type": "root", "vec": SemanticVector(*coords), "parents": ()})
            elif strategy == "compound":
                w1 = random.choice(living_words)
                w2 = random.choice(living_words)
                if w1.id != w2.id:
                    blended = w1.vector.blend(w2.vector, weight=0.55)
                    candidate_plans.append({"type": "compound", "vec": blended, "parents": (w1, w2)})
            else:
                base = random.choice(living_words)
                affix_host = min(living_words, key=lambda w: (len(w.phonemes), -w.usage_frequency))
                if base.id != affix_host.id:
                    deriv_vec = base.vector.blend(affix_host.vector, weight=0.70)
                    candidate_plans.append({"type": "affix", "vec": deriv_vec, "parents": (base, affix_host)})

        if not candidate_plans:
            return []

        existing_coords = [w.vector.coords for w in living_words]
        cand_coords = [p["vec"].coords for p in candidate_plans]
        valid_mask = GPUSemanticEngine.filter_candidate_density_gpu(
            candidates_coords=cand_coords,
            existing_coords=existing_coords,
            weights=weights,
            d_min=d_sem_min,
        )

        for plan, is_valid in zip(candidate_plans, valid_mask):
            if not is_valid:
                continue

            ptype = plan["type"]
            cand_vec = plan["vec"]

            if ptype == "root":
                template = cls.sample_phonotactic_template(prof)
                new_phonemes = cls.generate_phonetic_root(
                    consonants=c_pool,
                    vowels=v_pool,
                    seed_vector=cand_vec,
                    template=template,
                    syllable_complexity=s_comp,
                    aesthetic_bias=getattr(prof, "aesthetic_bias", None),
                )
                cleaned = ArticulatoryEnergyModel.repair_phonemes(new_phonemes, prof)
                if not cleaned:
                    continue
                max_family_id += 1
                word = Word(
                    vector=cand_vec,
                    phonemes=cleaned,
                    derivation=DerivationRecord(root_family_id=max_family_id, derivation_type=DerivationType.ROOT),
                    usage_frequency=0.90,
                    category=LexicalCategory.CONTENT_OPEN,
                    generation_born=current_generation,
                )

            elif ptype == "compound":
                w1, w2 = plan["parents"]
                h1 = ArticulatoryEnergyModel.repair_phonemes(_extract_metrical_head(w1, mode="head"), prof)
                h2 = ArticulatoryEnergyModel.repair_phonemes(_extract_metrical_head(w2, mode="tail"), prof)
                m1 = Morpheme(h1, root_family_id=w1.derivation.root_family_id)
                m2 = Morpheme(h2, root_family_id=w2.derivation.root_family_id)
                raw = m1.phonemes + m2.phonemes
                cleaned = ArticulatoryEnergyModel.repair_phonemes(raw, prof)
                split = max(1, min(len(cleaned) - 1, len(m1.phonemes)))
                m1.phonemes = cleaned[:split]
                m2.phonemes = cleaned[split:]
                word = Word(
                    vector=cand_vec,
                    phonemes=cleaned,
                    morphemes=[m1, m2],
                    derivation=DerivationRecord(root_family_id=w1.derivation.root_family_id, derivation_type=DerivationType.COMPOUND, local_parent_ids=(w1.id, w2.id)),
                    usage_frequency=1.10,
                    category=LexicalCategory.CONTENT_OPEN,
                    generation_born=current_generation,
                )

            else:  # Affixation
                base, host = plan["parents"]
                p_base = ArticulatoryEnergyModel.repair_phonemes(_extract_metrical_head(base, mode="head"), prof)
                p_aff = ArticulatoryEnergyModel.repair_phonemes(host.phonemes[:2], prof)
                m_base = Morpheme(p_base, root_family_id=base.derivation.root_family_id)
                m_aff = Morpheme(p_aff, root_family_id=host.derivation.root_family_id, is_grammatical=True, stage=ClineStage.BOUND_AFFIX)
                raw = p_base + p_aff
                cleaned = ArticulatoryEnergyModel.repair_phonemes(raw, prof)
                split = max(1, min(len(cleaned) - 1, len(m_base.phonemes)))
                m_base.phonemes = cleaned[:split]
                m_aff.phonemes = cleaned[split:]
                word = Word(
                    vector=cand_vec,
                    phonemes=cleaned,
                    morphemes=[m_base, m_aff],
                    derivation=DerivationRecord(root_family_id=base.derivation.root_family_id, derivation_type=DerivationType.AFFIXATION, local_parent_ids=(base.id, host.id)),
                    usage_frequency=1.00,
                    category=LexicalCategory.CONTENT_OPEN,
                    generation_born=current_generation,
                )

            word = SyllableEngine.apply_prosody(word, pattern=stress_pat)
            if word.plain_form not in occupied_forms:
                lang.add_word(word)
                living_words.append(word)
                new_born.append(word)
                occupied_forms.add(word.plain_form)

        return new_born