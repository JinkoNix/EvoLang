"""
language.py — Living Speech Community with Cultural Genome Physics and Carnot Communicative Efficiency.
Features:
- Incubation Threshold for High-Contrast Innovated Features.
- Vowel Trapezoid Formant Scaling.
- Clements Feature Economy without Bootstrap Deadlocks.
- Purged Procedural Sound Rewrite Rules.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Sequence, NamedTuple

from articulatory_space import Airstream, SecondaryArticulation
from energy import ArticulatoryEnergyModel, BiomechanicalCostModel, _get_articulator_tier
from lexicon import LexiconGenerator
from phoneme import Phoneme, PhonemeKind
from semantics import SemanticVector, DerivationRecord, DerivationType, LoanProvenance
from syllable import SyllableEngine, StressPattern
from word import Word, Morpheme, ClineStage, LexicalCategory


class LanguageProfile(NamedTuple):
    name: str
    max_transition_cost: float
    voicing_assimilation_bias: str
    double_stop_strategy: str
    epenthetic_vowel_point: tuple[float, float, float]
    stress_pattern: StressPattern
    vowel_reduction_mode: str
    apocope_rate: float = 0.50
    syncope_rate: float = 0.50
    head_directionality: float = 0.50
    d_min_vowel: float = 0.38
    d_min_consonant: float = 0.28
    synthesis_index: float = 0.50
    syllable_complexity: float = 0.50
    acoustic_roughness: float = 0.50
    alignment: str = "nominative_accusative"
    aesthetic_bias: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def allow_clusters(self) -> bool:
        return self.syllable_complexity >= 0.28

    @property
    def prefix_ratio(self) -> float:
        return max(0.05, min(0.95, self.head_directionality))

    @property
    def dominant_word_order(self) -> str:
        if self.head_directionality < 0.35: return "SOV"
        elif self.head_directionality > 0.65: return "VSO"
        return "SVO"


def _resync_word_structure(word: Word, new_phonemes: list[Phoneme], profile: LanguageProfile) -> None:
    word.phonemes = new_phonemes
    prosodified = SyllableEngine.apply_prosody(word, pattern=profile.stress_pattern)
    word.syllables = prosodified.syllables

    n_p = len(word.phonemes)
    if len(word.morphemes) <= 1:
        if not word.morphemes:
            word.morphemes = [Morpheme(list(word.phonemes), root_family_id=word.derivation.root_family_id)]
        else:
            word.morphemes[0].phonemes = list(word.phonemes)
    else:
        n_m = len(word.morphemes)
        split_idx = max(1, n_p // n_m)
        word.morphemes[0].phonemes = list(word.phonemes[:split_idx])
        word.morphemes[1].phonemes = list(word.phonemes[split_idx:])


class Language:
    """A speech community evolving under continuous physical acoustics and Clements feature economy."""

    def __init__(
        self,
        name: str,
        phonemes: Sequence[Phoneme] | None = None,
        profile: LanguageProfile | None = None,
        environment=None,
        population: float = 0.50,
    ):
        self.name = name
        self.environment = environment
        self.population = float(population)
        self.systemic_instability: float = 0.0

        alt = getattr(self.environment, "altitude", 0.0) if self.environment else 0.0
        hum = getattr(self.environment, "humidity", 0.5) if self.environment else 0.5
        temp = getattr(self.environment, "temperature", 0.5) if self.environment else 0.5
        noise = getattr(self.environment, "ambient_noise", 0.2) if self.environment else 0.2
        pop = self.population

        veg = self.environment.emergent_vegetation(population=pop) if hasattr(self.environment, "emergent_vegetation") else 0.50
        aridity = 1.0 - hum

        agri_dist = ((hum - 0.55) / 0.22) ** 2 + ((veg - 0.45) / 0.22) ** 2 + ((temp - 0.55) / 0.22) ** 2
        arable_agri = math.exp(-agri_dist) * max(0.0, 1.0 - alt * 1.5)
        pastoral_dist = ((veg - 0.35) / 0.18) ** 2 + ((hum - 0.40) / 0.20) ** 2
        pastoral_herding = math.exp(-pastoral_dist) * (1.0 - pop * 0.70)
        steppe_mobility = (1.0 - veg) * (1.0 - pop * 0.80)
        imperial_potency = pop * (1.0 - veg * 0.50)

        pop_eff = math.tanh(max(0.0, pop) / 1.8)

        raw_env_targets = [
            0.45 + 0.70 * alt + 0.85 * arable_agri + 0.45 * pop_eff * (1.0 - veg),      # Concreteness
            0.45 + 0.80 * (1.0 - pop_eff) * veg + 0.95 * pastoral_herding,              # Animacy
            0.45 + 0.90 * aridity + 0.45 * pop_eff * (1.0 - arable_agri * 0.4),         # Valence
            0.45 + 0.75 * alt + 0.95 * imperial_potency + 0.30 * noise,                 # Potency
            0.45 + 0.75 * hum * (0.50 + noise) + 0.95 * steppe_mobility,                # Dynamism
            0.45 + 0.85 * pop_eff + 0.20 * temp,                                        # Sociality
            0.45 + 0.90 * (1.0 - veg) + 0.65 * imperial_potency,                        # Extension
        ]

        total_raw = sum(raw_env_targets) or 7.0
        self.cultural_attention = [round(7.0 * (w / total_raw), 3) for w in raw_env_targets]

        self.profile = profile or (environment.generate_profile(name=f"{name} Profile", cultural_attention=self.cultural_attention) if environment else LanguageProfile(
            name=f"{name} Profile",
            max_transition_cost=3.00,
            voicing_assimilation_bias="regressive",
            double_stop_strategy="spirantization",
            epenthetic_vowel_point=(3.0, 1.0, 0.0),
            stress_pattern=StressPattern.NATURAL_WEIGHT,
            vowel_reduction_mode="inventory_snap",
            apocope_rate=0.50,
            syncope_rate=0.50,
            head_directionality=0.50,
            d_min_vowel=0.38,
            d_min_consonant=0.28,
            synthesis_index=0.50,
            syllable_complexity=0.50,
            acoustic_roughness=0.50,
            alignment="nominative_accusative",
            aesthetic_bias=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ))

        self.phonemes: dict[str, Phoneme] = {}
        if phonemes:
            for p in phonemes: self.phonemes[p.ipa] = p

        self.words: dict[int, Word] = {}
        self.contacts: dict[Language, float] = {}
        self.cached_mean_cost: float = 1.20
        self._cached_grammar_paradigm = None

    def populate_default_phonemes(
        self,
        vowel_points: Sequence[tuple[float, float, float] | Phoneme] | None = None,
        consonant_points: Sequence[tuple[float, float, float] | Phoneme] | None = None,
    ) -> None:
        default_vowels = vowel_points or [
            (6.0, 0.0, 0.0), (6.0, 2.0, 1.0), (4.0, 0.0, 0.0), (4.0, 2.0, 1.0), (0.0, 1.0, 0.0)
        ]
        default_consonants = consonant_points or [
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (3.0, 0.0, 0.0), (3.0, 0.0, 1.0),
            (7.0, 0.0, 0.0), (7.0, 0.0, 1.0), (1.0, 4.0, 0.0), (1.0, 4.0, 1.0),
            (3.0, 4.0, 0.0), (3.0, 4.0, 1.0), (4.0, 4.0, 0.0), (7.0, 4.0, 0.0),
            (10.0, 4.0, 0.0), (0.0, 1.0, 1.0), (3.0, 1.0, 1.0), (3.0, 7.0, 1.0), (0.0, 7.0, 1.0)
        ]
        for pt in default_vowels:
            if isinstance(pt, Phoneme): self.add_phoneme(pt)
            else: self.add_phoneme(Phoneme(PhonemeKind.VOWEL, pt))

        for pt in default_consonants:
            if isinstance(pt, Phoneme): self.add_phoneme(pt)
            else: self.add_phoneme(Phoneme(PhonemeKind.CONSONANT, pt))

    @property
    def cultural_inertia(self) -> float:
        w = self.cultural_attention
        v_clamped = max(0.01, float(w[2]))
        d_denom = max(0.20, 1.0 + float(w[4]) * 0.8)
        return float((v_clamped ** 1.4 * max(0.01, float(w[5]))) / d_denom)

    @property
    def consonants(self) -> list[Phoneme]:
        return [p for p in self.phonemes.values() if p.kind == PhonemeKind.CONSONANT]

    @property
    def vowels(self) -> list[Phoneme]:
        return [p for p in self.phonemes.values() if p.kind == PhonemeKind.VOWEL]

    @property
    def word_order(self) -> str:
        return self.profile.dominant_word_order

    def set_contact(self, other: Language, intensity: float) -> None:
        clamped = max(0.0, min(1.0, float(intensity)))
        self.contacts[other] = clamped
        other.contacts[self] = clamped

    def add_phoneme(self, p: Phoneme) -> Phoneme:
        self.phonemes[p.ipa] = p
        return p

    def add_word(self, word: Word) -> Word:
        self.words[word.id] = word
        for p in word.phonemes: self.phonemes[p.ipa] = p
        return word

    def nearest_word_to_vector(self, target_vec: SemanticVector) -> tuple[Word, float]:
        closest = min(self.words.values(), key=lambda w: w.vector.weighted_distance_to(target_vec, self.cultural_attention))
        dist = closest.vector.weighted_distance_to(target_vec, self.cultural_attention)
        return closest, dist

    def get_active_feature_series(self) -> set[str]:
        series = set()
        for p in self.phonemes.values():
            if any(l == Airstream.EJECTIVE for l in p.layers): series.add("ejective")
            if any(l == Airstream.CLICK for l in p.layers): series.add("click")
            if any(l == Airstream.IMPLOSIVE for l in p.layers): series.add("implosive")
            if any(l == SecondaryArticulation.NASALIZED for l in p.layers): series.add("nasalized")
            if any(l == SecondaryArticulation.LABIALIZED for l in p.layers): series.add("labialized")
            if any(l == SecondaryArticulation.PHARYNGEALIZED for l in p.layers): series.add("pharyngealized")
            if any(l == SecondaryArticulation.ASPIRATED for l in p.layers): series.add("aspirated")
            if int(round(p.point[1])) in (4, 5): series.add("fricative")
        return series

    def identify_emergent_auxiliaries(self, frequency_threshold: int | None = None) -> list[int]:
        if frequency_threshold is None:
            multi_m_count = sum(1 for w in self.words.values() if len(w.morphemes) >= 2)
            threshold = max(3, int(round(multi_m_count * 0.05)))
        else:
            threshold = frequency_threshold

        family_counts = Counter()
        for word in self.words.values():
            if len(word.morphemes) >= 2:
                stem_family = word.derivation.root_family_id
                for m in word.morphemes:
                    if m.root_family_id != stem_family:
                        family_counts[m.root_family_id] += 1
                        
        return [fid for fid, count in family_counts.most_common(12) if count >= threshold]

    def borrow_word(
        self,
        foreign_word: Word,
        source_language_name: str,
        cultural_delta: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        current_generation: int = 0,
    ) -> Word:
        adapted_phonemes: list[Phoneme] = []
        for p in foreign_word.phonemes:
            pool = self.vowels if p.kind == PhonemeKind.VOWEL else self.consonants
            if pool:
                closest_local = min(
                    pool,
                    key=lambda lp: math.sqrt(sum((p.point[i] - lp.point[i]) ** 2 for i in range(3)))
                )
                adapted_phonemes.append(closest_local.drift())
            else:
                adapted_phonemes.append(p.drift())

        revalued_vector = foreign_word.vector.drift(cultural_delta)
        max_existing_family = max((w.derivation.root_family_id for w in self.words.values()), default=0)
        new_local_family_id = max_existing_family + 1

        provenance = LoanProvenance(
            source_language=source_language_name,
            donor_word_id=foreign_word.id,
            donor_form_at_borrowing=foreign_word.form,
            donor_vector_at_borrowing=SemanticVector(*foreign_word.vector.coords),
            generation_borrowed=current_generation,
        )

        borrowed_word = Word(
            vector=revalued_vector,
            phonemes=adapted_phonemes,
            derivation=DerivationRecord(
                root_family_id=new_local_family_id,
                derivation_type=DerivationType.LOANWORD,
                local_parent_ids=(),
                provenance=provenance,
            ),
            usage_frequency=1.5,
            category=foreign_word.category,
            generation_born=current_generation,
        )

        borrowed_word = ArticulatoryEnergyModel.repair_word(borrowed_word, self.profile)
        self.add_word(borrowed_word)
        return borrowed_word

    def calque_word(
        self,
        foreign_word: Word,
        source_language_name: str,
        current_generation: int = 0,
    ) -> Word:
        living_content = [w for w in self.words.values() if w.category == LexicalCategory.CONTENT_OPEN]
        if len(living_content) < 2:
            return self.borrow_word(foreign_word, source_language_name, current_generation=current_generation)

        head_root = min(living_content, key=lambda w: w.vector.distance_to(foreign_word.vector))
        mod_pool = [w for w in living_content if w.id != head_root.id]
        mod_root = min(mod_pool, key=lambda w: w.vector.distance_to(foreign_word.vector)) if mod_pool else living_content[0]

        blended_vec = head_root.vector.blend(foreign_word.vector, weight=0.60)
        s_head = SyllableEngine.syllabify(head_root)
        s_mod = SyllableEngine.syllabify(mod_root)

        p1_clean = ArticulatoryEnergyModel.repair_phonemes(s_head[0].to_phonemes() if s_head else list(head_root.phonemes[:2]), self.profile)
        p2_clean = ArticulatoryEnergyModel.repair_phonemes(s_mod[0].to_phonemes() if s_mod else list(mod_root.phonemes[:2]), self.profile)
        
        is_prefix = self.profile.prefix_ratio > 0.50
        m1 = Morpheme(p1_clean, root_family_id=head_root.derivation.root_family_id, stage=ClineStage.FREE_ROOT)
        m2 = Morpheme(p2_clean, root_family_id=mod_root.derivation.root_family_id, stage=ClineStage.FREE_ROOT)
        
        morphemes = [m2, m1] if is_prefix else [m1, m2]
        raw_phonemes = (m2.phonemes + m1.phonemes) if is_prefix else (m1.phonemes + m2.phonemes)
        cleaned_phonemes = ArticulatoryEnergyModel.repair_phonemes(raw_phonemes, self.profile)

        l0 = len(morphemes[0].phonemes)
        l_raw = len(raw_phonemes) or 1
        l_clean = len(cleaned_phonemes)
        split_idx = max(1, min(l_clean - 1, int(round(l_clean * (l0 / l_raw)))))
        morphemes[0].phonemes = list(cleaned_phonemes[:split_idx])
        morphemes[1].phonemes = list(cleaned_phonemes[split_idx:])

        provenance = LoanProvenance(
            source_language=source_language_name,
            donor_word_id=foreign_word.id,
            donor_form_at_borrowing=foreign_word.form,
            donor_vector_at_borrowing=SemanticVector(*foreign_word.vector.coords),
            generation_borrowed=current_generation,
        )

        calqued_word = Word(
            vector=blended_vec,
            phonemes=cleaned_phonemes,
            morphemes=morphemes,
            derivation=DerivationRecord(
                root_family_id=head_root.derivation.root_family_id,
                derivation_type=DerivationType.COMPOUND,
                local_parent_ids=(head_root.id, mod_root.id),
                provenance=provenance,
            ),
            usage_frequency=1.3,
            category=foreign_word.category,
            generation_born=current_generation,
        )
        calqued_word = SyllableEngine.apply_prosody(calqued_word, pattern=self.profile.stress_pattern)
        self.add_word(calqued_word)
        return calqued_word

    @staticmethod
    def acoustic_vowel_distance(p1: Phoneme, p2: Phoneme) -> float:
        """
        Psychoacoustic Formant Distance across the Vowel Trapezoid:
        Scales F2 backness resolution by jaw depression (0.25 + 0.75 * h_norm).
        At open floor (h -> 0), backness distinction contracts by 75%, unifying /a, ä, ɑ/.
        """
        h1, h2 = p1.point[0] / 6.0, p2.point[0] / 6.0
        b1, b2 = p1.point[1] / 2.0, p2.point[1] / 2.0
        avg_h = (h1 + h2) * 0.50

        d_f1 = abs(h1 - h2) * 1.40
        trapezoid_factor = 0.25 + 0.75 * avg_h
        d_f2 = abs(b1 - b2) * 1.25 * trapezoid_factor

        d_rnd = abs(p1.point[2] - p2.point[2]) * 0.35
        d_nas = 0.30 if (p1.layers != p2.layers and (SecondaryArticulation.NASALIZED in p1.layers or SecondaryArticulation.NASALIZED in p2.layers)) else 0.0

        return math.sqrt(d_f1**2 + d_f2**2 + d_rnd**2 + d_nas**2)

    @staticmethod
    def acoustic_consonant_distance(p1: Phoneme, p2: Phoneme) -> float:
        """
        Multi-Dimensional Categorical Consonant Contrast:
        Measures distance across VOT timing, Spectral CoG (burst place), and Envelope Abruptness.
        """
        phys1, phys2 = p1.physics, p2.physics

        # 1. Voice Onset Time
        d_vot = (abs(phys1.vot_ms - phys2.vot_ms) / 160.0) * 0.90

        # 2. Spectral Center of Gravity & Motor Tier
        tier1 = _get_articulator_tier(p1.point[0])
        tier2 = _get_articulator_tier(p2.point[0])
        tier_boost = 0.40 if tier1 != tier2 else 0.0
        intra_place = (abs(p1.point[0] - p2.point[0]) / 4.0) * 0.35 if tier1 == tier2 else 0.0
        d_cog = (abs(phys1.spectral_cog - phys2.spectral_cog) * 0.95) + tier_boost + intra_place

        # 3. Envelope Abruptness
        d_env = abs(phys1.envelope_abruptness - phys2.envelope_abruptness) * 0.80

        # 4. Layer differences
        d_layer = 0.80 if p1.layers != p2.layers else 0.0

        return math.sqrt(d_vot**2 + d_cog**2 + d_env**2 + d_layer**2)

    def calculate_lexical_usage(self) -> None:
        """
        Computes Carnot Communicative Efficiency with O(1) cached energetic costs.
        """
        if not self.words: return
        living_words = list(self.words.values())
        attn = self.cultural_attention
        attn_sum = max(0.01, sum(attn))
        pop = self.population
        w_pot = attn[3]
        w_val = attn[2]

        k_cap = 2500.0 * (1.0 + 15.0 * (pop ** 1.6) * (1.0 - min(1.0, self.cultural_inertia) * 0.40))
        k_effort = 0.35 / (1.0 + 0.60 * self.cultural_inertia)
        energy_ceiling = 1.80 * (1.0 + 0.50 * (w_pot - 1.0))

        raw_loads: list[tuple[float, Word]] = []
        total_energy_burden = 0.0

        for word in living_words:
            u_base = 3.50 if word.is_proto_root else 1.00
            relevance = sum(word.vector.coords[i] * attn[i] for i in range(7)) / attn_sum
            relevance = max(0.001, float(relevance))  # Prevents complex numbers!
            
            is_functional = (word.category == LexicalCategory.FUNCTIONAL_CLOSED)
            sacred_bonus = 1.0 + 1.20 * max(0.0, float(word.vector.valence) * float(w_val))

            if getattr(word, '_cached_cost', None) is None:
                word._cached_cost = BiomechanicalCostModel.word_energetic_cost(
                    word, self.profile, self.environment, cultural_attention=attn
                )
            word_cost = float(word._cached_cost)
            total_energy_burden += word_cost

            effort_drag = 1.0 + float(k_effort) * max(0.0, word_cost - energy_ceiling)
            score = (u_base * (relevance ** 1.4) * (6.0 if is_functional else 1.0) * sacred_bonus) / max(0.1, effort_drag)
            
            # Guarantee pure float
            raw_loads.append((float(score.real if isinstance(score, complex) else score), word))

        raw_loads.sort(key=lambda x: x[0], reverse=True)
        mean_word_cost = total_energy_burden / max(1, len(living_words))
        self.cached_mean_cost = mean_word_cost

        for rank, (_, word) in enumerate(raw_loads, start=1):
            zipf_epoch_load = 45.0 / (rank ** 0.72)
            word.usage_frequency = (0.75 * word.usage_frequency) + (0.25 * zipf_epoch_load)
            if word.usage_frequency < 0.28 and not word.is_proto_root and word.category != LexicalCategory.FUNCTIONAL_CLOSED:
                word.epochs_idle += 1
            else:
                word.epochs_idle = 0

        cost_pressure = max(0.02, mean_word_cost * 0.05)
        turnover_pressure = (len(self.words) / max(1.0, k_cap)) * 0.04
        self.systemic_instability += (cost_pressure + turnover_pressure)

    def apply_punctuated_sound_shifts(self, words_dict: dict[int, Word], drift_rate: float) -> None:
        CRITICAL_INSTABILITY_THRESHOLD = 1.50
        active_series = self.get_active_feature_series()
        alt = getattr(self.environment, "altitude", 0.0) if self.environment else 0.0
        hum = getattr(self.environment, "humidity", 0.5) if self.environment else 0.5
        temp = getattr(self.environment, "temperature", 0.5) if self.environment else 0.5
        noise = getattr(self.environment, "ambient_noise", 0.2) if self.environment else 0.2
        aridity = max(0.0, 1.0 - hum)
        w_val = self.cultural_attention[2]

        # 1. Catastrophic Phase Shift
        if self.systemic_instability >= CRITICAL_INSTABILITY_THRESHOLD:
            self.systemic_instability = 0.0

            sound_rules = [
                # Spirantization: Relieves Ohala intraoral impedance on stops
                (
                    lambda p, next_p: p.kind == PhonemeKind.CONSONANT and p.physics.supraglottal_impedance >= 0.70,
                    lambda p: p.drift(point=(p.point[0], 4.0, p.point[2]))
                ),
                # Coda Devoicing: Relieves PTP dehydration strain in coda
                (
                    lambda p, next_p: p.kind == PhonemeKind.CONSONANT and p.point[2] >= 0.5 and (next_p is None or next_p.kind == PhonemeKind.CONSONANT),
                    lambda p: p.drift(point=(p.point[0], p.point[1], 0.0))
                ),
                # Conditioned Velar Palatalization: k -> t͡ʃ strictly before high front vowels (i, e)
                (
                    lambda p, next_p: (
                        p.kind == PhonemeKind.CONSONANT and 6.0 <= p.point[0] <= 8.0 and p.point[1] <= 0.8
                        and next_p is not None and next_p.kind == PhonemeKind.VOWEL and next_p.point[0] >= 3.8 and next_p.point[1] <= 0.8
                    ),
                    lambda p: p.drift(point=(4.0, 2.0, p.point[2]))
                ),
            ]

            predicate, transform = random.choice(sound_rules)

            for word in words_dict.values():
                # Valence Archaic Anchor: High-valence sacred roots resist sound shifts
                if (word.is_proto_root or (word.vector.valence * w_val >= 1.4)) and word.usage_frequency > 2.5:
                    continue
                p_list = list(word.phonemes)
                modified = False
                n_p = len(p_list)

                for i in range(n_p):
                    p = p_list[i]
                    next_p = p_list[i + 1] if i + 1 < n_p else None
                    if predicate(p, next_p):
                        p_list[i] = transform(p)
                        modified = True

                if modified:
                    _resync_word_structure(word, p_list, self.profile)

        # 2. Continuous Micro-Drift with Coulomb Dispersion Field & De Novo Innovation
        active_vowels = [p for p in self.phonemes.values() if p.kind == PhonemeKind.VOWEL]
        effective_drift = min(0.04, drift_rate * 0.12)

        p_ejective_seed = 0.08 if "ejective" in active_series else (0.015 * max(0.0, alt - 0.25))
        p_implosive_seed = 0.08 if "implosive" in active_series else (0.012 * max(0.0, hum - 0.55))
        p_click_seed = 0.08 if "click" in active_series else (0.010 * max(0.0, noise - 0.35) * aridity)

        for word in words_dict.values():
            if word.is_proto_root or random.random() > (effective_drift / (1.0 + math.log(word.usage_frequency + 1.0))):
                continue

            p_list = list(word.phonemes)
            changed = False
            n_p = len(p_list)

            for i in range(n_p):
                p = p_list[i]
                next_p = p_list[i + 1] if i + 1 < n_p else None

                if p.kind == PhonemeKind.CONSONANT:
                    is_plosive = (p.point[1] <= 0.8)
                    is_velar = (6.0 <= p.point[0] <= 8.5)
                    is_coronal = (2.0 <= p.point[0] <= 5.5)

                    if is_plosive and p.point[2] == 0.0 and random.random() < p_ejective_seed:
                        p_list[i] = p.drift(layers=(Airstream.EJECTIVE, SecondaryArticulation.NONE))
                        changed = True

                    elif is_plosive and p.point[2] == 1.0 and random.random() < p_implosive_seed:
                        p_list[i] = p.drift(layers=(Airstream.IMPLOSIVE, SecondaryArticulation.NONE))
                        changed = True

                    elif (p.point[1] in (0, 2)) and random.random() < p_click_seed:
                        p_list[i] = p.drift(layers=(Airstream.CLICK, SecondaryArticulation.NONE))
                        changed = True

                    elif (
                        is_velar and is_plosive and next_p and next_p.kind == PhonemeKind.VOWEL
                        and next_p.point[1] >= 1.2 and next_p.point[2] >= 0.5
                        and random.random() < 0.18
                    ):
                        p_list[i] = p.drift(layers=(p.layers[0], SecondaryArticulation.LABIALIZED))
                        changed = True

                    elif (
                        is_coronal and aridity > 0.40 and next_p and next_p.kind == PhonemeKind.VOWEL
                        and next_p.point[0] <= 1.5 and next_p.point[1] >= 1.0
                        and random.random() < 0.15
                    ):
                        p_list[i] = p.drift(layers=(p.layers[0], SecondaryArticulation.PHARYNGEALIZED))
                        changed = True

                    else:
                        pl, mn, vc = p.point
                        p_list[i] = p.drift(point=(max(0.0, min(10.0, pl + random.gauss(0, 0.02))),
                                                   max(0.0, min(7.0, mn + random.gauss(0, 0.01))), vc))
                        changed = True

                elif p.kind == PhonemeKind.VOWEL:
                    h, b, r = p.point
                    f_h, f_b = 0.0, 0.0

                    # Active Coulomb Repulsion (Holds /a/ open against thermal pull)
                    for other_v in active_vowels:
                        if other_v.ipa == p.ipa: continue
                        dh = (h - other_v.point[0]) / 6.0
                        db = (b - other_v.point[1]) / 2.0
                        dist_sq = dh**2 + db**2 + 0.01
                        dist = math.sqrt(dist_sq)
                        repel_force = 0.035 / dist_sq
                        f_h += repel_force * (dh / dist)
                        f_b += repel_force * (db / dist)

                    coldness = max(0.0, 0.45 - temp)
                    f_h += coldness * 0.08

                    new_h = max(0.0, min(6.0, h + f_h + random.gauss(0, 0.02)))
                    new_b = max(0.0, min(2.0, b + f_b + random.gauss(0, 0.01)))
                    p_list[i] = p.drift(point=(new_h, new_b, r))
                    changed = True

            if changed:
                _resync_word_structure(word, p_list, self.profile)

    def evolve(
        self,
        reduction_strength: float = 0.80,
        generational_drift: bool = True,
        drift_rate: float = 0.35,
        semantic_drift_rate: float = 0.20,
        enable_neologisms: bool = True,
        current_generation: int = 0,
    ) -> None:
        # Only invalidate grammar paradigm periodically or when catastrophic instability occurs
        if current_generation % 10 == 0 or self.systemic_instability >= 1.50:
            self._cached_grammar_paradigm = None

        attn = self.cultural_attention
        w_pot = attn[3]
        w_con = attn[0]
        w_val = attn[2]

        if enable_neologisms:
            LexiconGenerator.expand_vocabulary(
                lang=self,
                growth_rate=0.04,
                d_sem_min=0.06,
                current_generation=current_generation,
                profile=self.profile,
            )

        self.calculate_lexical_usage()
        new_words: dict[int, Word] = {}
        effective_reduction = reduction_strength / (1.0 + self.cultural_inertia * 0.40)

        for word in list(self.words.values()):
            if word.is_obsolete(max_idle_epochs=25, cultural_inertia=self.cultural_inertia):
                continue

            current_word = word
            current_vector = current_word.vector
            is_mutated = False

            updated_morphemes: list[Morpheme] = []
            total_word_len = len(current_word.phonemes)
            s_comp = getattr(self.profile, "syllable_complexity", 0.50)
            is_long_word = (total_word_len >= 6 and random.random() < 0.25)

            for morph in current_word.morphemes:
                if morph.is_grammatical or is_long_word:
                    old_len = len(morph.phonemes)
                    eroded_m = ArticulatoryEnergyModel.erode_morpheme(
                        morph, reduction_rate=effective_reduction,
                        usage_frequency=current_word.usage_frequency,
                        is_long_word=is_long_word, syllable_complexity=s_comp,
                        potency_brake=w_pot,
                    )
                    if len(eroded_m.phonemes) != old_len: is_mutated = True
                    updated_morphemes.append(eroded_m)
                else:
                    updated_morphemes.append(morph)

            concreteness_decay_brake = max(0.40, 1.0 - 0.40 * (w_con - 1.0))
            if random.random() < (semantic_drift_rate * concreteness_decay_brake / (1.0 + math.log(current_word.usage_frequency + 1.0))):
                s_delta = [random.uniform(-0.030, 0.030) * (self.cultural_attention[d] / 1.0) for d in range(7)]
                current_vector = current_vector.drift(s_delta)
                current_word.vector = current_vector

            if not is_mutated:
                new_words[current_word.id] = current_word
                continue

            fused_phonemes = []
            for m in updated_morphemes: fused_phonemes.extend(m.phonemes)
            repaired = ArticulatoryEnergyModel.repair_phonemes(fused_phonemes or list(current_word.phonemes), self.profile)

            temp_word = Word(
                word_id=current_word.id, vector=current_vector, phonemes=repaired or [current_word.phonemes[0]],
                morphemes=updated_morphemes, derivation=current_word.derivation,
                usage_frequency=current_word.usage_frequency, category=current_word.category,
                generation_born=current_generation, senses=current_word.senses
            )

            final_word = SyllableEngine.reduce_and_prosodify(
                temp_word, pattern=self.profile.stress_pattern,
                reduction_strength=effective_reduction, apocope_rate=self.profile.apocope_rate,
                syncope_rate=self.profile.syncope_rate, reduction_mode=self.profile.vowel_reduction_mode,
                environment=self.environment
            )
            new_words[current_word.id] = final_word

        if generational_drift:
            self.apply_punctuated_sound_shifts(new_words, drift_rate)

        # Contrast Preservation & Taboo Suppletion
        form_clusters: dict[str, list[Word]] = defaultdict(list)
        for w in new_words.values(): form_clusters[w.plain_form].append(w)
        words_to_remove: set[int] = set()
        occupied_forms: set[str] = set(form_clusters.keys())

        for form, cluster in form_clusters.items():
            if len(cluster) >= 2:
                cluster.sort(key=lambda item: item.usage_frequency, reverse=True)
                dominant = cluster[0]
                for subordinate in cluster[1:]:
                    sem_dist = dominant.vector.weighted_distance_to(subordinate.vector, self.cultural_attention)
                    is_sacred_collision = (dominant.vector.valence * w_val >= 1.3 or subordinate.vector.valence * w_val >= 1.3)
                    if is_sacred_collision or sem_dist < 0.25:
                        dominant.record_usage(subordinate.usage_frequency * 0.60)
                        words_to_remove.add(subordinate.id)
                    elif subordinate.phonemes:
                        p_cand = list(subordinate.phonemes)
                        for i_p in range(len(p_cand) - 1, -1, -1):
                            seg = p_cand[i_p]
                            if seg.kind == PhonemeKind.CONSONANT:
                                p_cand[i_p] = seg.drift(point=(seg.point[0], seg.point[1], 0.0 if seg.point[2] >= 0.5 else 1.0))
                                break
                            elif seg.kind == PhonemeKind.VOWEL:
                                new_h = min(6.0, seg.point[0] + 1.0) if seg.point[0] <= 3.0 else max(0.0, seg.point[0] - 1.0)
                                p_cand[i_p] = seg.drift(point=(new_h, seg.point[1], seg.point[2]))
                                break

                        cleaned_cand = ArticulatoryEnergyModel.repair_phonemes(p_cand, self.profile)
                        temp_w = Word(vector=subordinate.vector, phonemes=cleaned_cand, is_ephemeral=True)
                        pros_w = SyllableEngine.apply_prosody(temp_w, pattern=self.profile.stress_pattern)
                        if pros_w.plain_form not in occupied_forms:
                            _resync_word_structure(subordinate, list(pros_w.phonemes), self.profile)
                            occupied_forms.add(pros_w.plain_form)

        for w_id in words_to_remove:
            if w_id in new_words: del new_words[w_id]

        self.words = new_words
        self.rebuild_inventory()

    def rebuild_inventory(self) -> None:
        """
        Usage-Based Emergent Inventory Clustering:
        - Vowel Trapezoid Formant Clustering.
        - Auditory Consonant Contrast Matrix.
        - Incubation Threshold (0.5% for high-contrast features, preventing infant mortality).
        """
        if not self.words:
            self.phonemes = {}
            return

        c_counts = Counter(p.ipa for w in self.words.values() for p in w.phonemes if p.kind == PhonemeKind.CONSONANT)
        v_counts = Counter(p.ipa for w in self.words.values() for p in w.phonemes if p.kind == PhonemeKind.VOWEL)
        total_c = sum(c_counts.values()) or 1.0
        total_v = sum(v_counts.values()) or 1.0

        unique_phonemes = {p.ipa: p for w in self.words.values() for p in w.phonemes}
        core_inventory: dict[str, Phoneme] = {}

        # 1. Vowel Trapezoid Formant Clustering
        active_vowels = [p for p in unique_phonemes.values() if p.kind == PhonemeKind.VOWEL]
        if active_vowels:
            sorted_v = sorted(active_vowels, key=lambda v: v_counts.get(v.ipa, 0), reverse=True)
            v_clusters: list[list[Phoneme]] = []
            effective_v_min = max(0.30, self.profile.d_min_vowel * 1.35)

            for v in sorted_v:
                matched = False
                for cl in v_clusters:
                    rep = cl[0]
                    if self.acoustic_vowel_distance(v, rep) < effective_v_min:
                        cl.append(v)
                        matched = True
                        break
                if not matched: v_clusters.append([v])

            for cl in v_clusters:
                if (sum(v_counts.get(v.ipa, 0) for v in cl) / total_v) >= 0.025:
                    dom_v = max(cl, key=lambda v: v_counts.get(v.ipa, 0) * v.weight)
                    core_inventory[dom_v.ipa] = dom_v

        # 2. Auditory Consonant Clustering with Incubation Protection
        active_consonants = [p for p in unique_phonemes.values() if p.kind == PhonemeKind.CONSONANT]
        if active_consonants:
            sorted_c = sorted(active_consonants, key=lambda c: c_counts.get(c.ipa, 0), reverse=True)
            c_clusters: list[list[Phoneme]] = []
            for c in sorted_c:
                matched = False
                for cl in c_clusters:
                    rep = cl[0]
                    if self.acoustic_consonant_distance(c, rep) < self.profile.d_min_consonant:
                        cl.append(c)
                        matched = True
                        break
                if not matched: c_clusters.append([c])

            for cl in c_clusters:
                cluster_share = sum(c_counts.get(c.ipa, 0) for c in cl) / total_c
                rep_c = cl[0]
                
                # High-contrast features (ejectives, clicks, secondary layers) receive incubation protection (0.5% cutoff)
                is_exotic = any(l != Airstream.PULMONIC for l in rep_c.layers) or any(l != SecondaryArticulation.NONE for l in rep_c.layers)
                cutoff = 0.005 if is_exotic else 0.025

                if cluster_share >= cutoff:
                    dom_c = max(cl, key=lambda seg: c_counts.get(seg.ipa, 0) * seg.weight)
                    core_inventory[dom_c.ipa] = dom_c

        self.phonemes = core_inventory or unique_phonemes

    def generate_lexicon(self) -> int:
        generated_words = LexiconGenerator.populate_proto_lexicon(
            consonants=self.consonants,
            vowels=self.vowels,
            env=self.environment,
            profile=self.profile,
            population=self.population,
        )
        for w in generated_words:
            self.words[w.id] = w
        self.rebuild_inventory()
        return len(generated_words)

    def fork(self, daughter_name: str, environment=None, initial_head_directionality: float | None = None) -> Language:
        env = environment or self.environment
        h = initial_head_directionality if initial_head_directionality is not None else self.profile.head_directionality
        profile = env.generate_profile(
            name=f"{daughter_name} Profile",
            initial_head_directionality=h,
            cultural_attention=self.cultural_attention,
            parent_bias=self.profile.aesthetic_bias,
        ) if env else None

        daughter = Language(name=daughter_name, profile=profile, environment=env, population=self.population)
        daughter.phonemes = {k: p.drift() for k, p in self.phonemes.items()}
        for word in self.words.values(): daughter.add_word(word.copy())
        return daughter

    def get_phoneme_distribution_stats(self) -> dict:
        if not self.words:
            return {}

        c_token_counts: Counter = Counter()
        v_token_counts: Counter = Counter()
        total_c_tokens = 0.0
        total_v_tokens = 0.0
        raw_c_segments = 0
        raw_v_segments = 0

        for word in self.words.values():
            u = word.usage_frequency
            for p in word.phonemes:
                if p.kind == PhonemeKind.CONSONANT:
                    c_token_counts[p.ipa] += u
                    total_c_tokens += u
                    raw_c_segments += 1
                else:
                    v_token_counts[p.ipa] += u
                    total_v_tokens += u
                    raw_v_segments += 1

        total_tokens = total_c_tokens + total_v_tokens
        total_segments = raw_c_segments + raw_v_segments

        c_percentages = {ipa: round((count / max(1.0, total_c_tokens)) * 100, 2) for ipa, count in c_token_counts.most_common()}
        v_percentages = {ipa: round((count / max(1.0, total_v_tokens)) * 100, 2) for ipa, count in v_token_counts.most_common()}

        entropy = 0.0
        if total_tokens > 0:
            for count in list(c_token_counts.values()) + list(v_token_counts.values()):
                prob = count / total_tokens
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            num_sound_types = len(c_token_counts) + len(v_token_counts)
            max_entropy = math.log2(num_sound_types) if num_sound_types > 1 else 1.0
            evenness = round(min(1.0, entropy / max_entropy), 3)
        else:
            evenness = 0.0

        return {
            "total_living_words": len(self.words),
            "raw_total_segments": total_segments,
            "weighted_token_volume": int(total_tokens),
            "cv_ratio": round(total_c_tokens / max(1.0, total_v_tokens), 2),
            "entropy_evenness": evenness,
            "consonants": c_percentages,
            "vowels": v_percentages,
            "top_5_consonants": list(c_percentages.items())[:5],
            "top_3_vowels": list(v_percentages.items())[:3],
        }

    def __repr__(self) -> str:
        return f"Language({self.name!r}, {len(self.vowels)} vowels, {len(self.consonants)} consonants, {len(self.words)} words)"