"""
language.py — Diachronic Speech Community Engine with Continuous Physical Transmission and Topological Mode-Seeking.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Sequence, NamedTuple
import numpy as np

from articulatory_space import Airstream, SecondaryArticulation, resolve_vowel, resolve_consonant
from energy import ArticulatoryEnergyModel, BiomechanicalCostModel
from lexicon import LexiconGenerator
from phoneme import Phoneme, PhonemeKind
from semantics import SemanticVector, DerivationRecord, DerivationType, LoanProvenance
from syllable import SyllableEngine, StressPattern
from word import Word, Morpheme, ClineStage, LexicalCategory
from gpu_accelerator import GPUSemanticEngine


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
    d_min_vowel: float = 0.28
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
        if self.head_directionality < 0.35:
            return "SOV"
        elif self.head_directionality > 0.65:
            return "VSO"
        return "SVO"


def _resync_word_structure(word: Word, new_phonemes: list[Phoneme], profile: LanguageProfile) -> None:
    word.phonemes = new_phonemes
    prosodified = SyllableEngine.apply_prosody(word, pattern=profile.stress_pattern)
    word.syllables = prosodified.syllables

    if len(word.morphemes) <= 1:
        if not word.morphemes:
            word.morphemes = [Morpheme(list(word.phonemes), root_family_id=word.derivation.root_family_id)]
        else:
            word.morphemes[0].phonemes = list(word.phonemes)
    else:
        total_old = sum(len(m.phonemes) for m in word.morphemes) or 1
        cursor = 0
        n_clean = len(word.phonemes)
        for m_idx, m in enumerate(word.morphemes):
            if m_idx == len(word.morphemes) - 1:
                m.phonemes = list(word.phonemes[cursor:])
            else:
                m_len = max(0, min(n_clean - cursor, int(round(n_clean * (len(m.phonemes) / total_old)))))
                m.phonemes = list(word.phonemes[cursor:cursor + m_len])
                cursor += m_len


class Language:
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

        self.standard_canon: dict[int, Word] = {}
        self.sociolinguistic_tension: float = 0.0
        self.last_reform_epoch: int = 0

        alt = getattr(self.environment, "altitude", 0.0) if self.environment else 0.0
        hum = getattr(self.environment, "humidity", 0.5) if self.environment else 0.5
        temp = getattr(self.environment, "temperature", 0.5) if self.environment else 0.5
        noise = getattr(self.environment, "ambient_noise", 0.2) if self.environment else 0.2
        pop = self.population
        veg = self.environment.emergent_vegetation(population=pop) if hasattr(self.environment, "emergent_vegetation") else 0.50

        raw_targets = [
            0.50 + 0.60 * alt + 0.40 * (1.0 - veg),
            0.50 + 0.80 * veg * (1.0 - pop * 0.5),
            0.50 + 0.70 * (1.0 - hum),
            0.50 + 0.70 * alt + 0.60 * pop,
            0.50 + 0.70 * hum + 0.30 * noise,
            0.50 + 0.80 * pop + 0.20 * temp,
            0.50 + 0.80 * (1.0 - veg) + 0.50 * pop,
        ]
        total_w = sum(raw_targets) or 7.0
        self.cultural_attention = [round(7.0 * (v / total_w), 3) for v in raw_targets]

        self.profile = profile or (
            environment.generate_profile(name=f"{name} Profile", cultural_attention=self.cultural_attention)
            if environment
            else LanguageProfile(
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
                d_min_vowel=0.28,
                d_min_consonant=0.28,
                synthesis_index=0.50,
                syllable_complexity=0.50,
                acoustic_roughness=0.50,
                alignment="nominative_accusative",
                aesthetic_bias=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        )

        self.phonemes: dict[str, Phoneme] = {}
        if phonemes:
            for p in phonemes:
                self.phonemes[p.ipa] = p

        self.words: dict[int, Word] = {}
        self.contacts: dict[Language, float] = {}
        self.cached_mean_cost: float = 1.20
        self._cached_grammar_paradigm = None

    def populate_default_phonemes(
        self,
        vowel_points: Sequence[tuple[float, float, float] | Phoneme] | None = None,
        consonant_points: Sequence[tuple[float, float, float] | Phoneme] | None = None,
    ) -> None:
        default_consonants = consonant_points or [
            (0.0, 0.0, 0.0),  # p
            (3.0, 0.0, 0.0),  # t
            (7.0, 0.0, 0.0),  # k
            (0.0, 1.0, 1.0),  # m
            (3.0, 1.0, 1.0),  # n
            (3.0, 4.0, 0.0),  # s
            (3.0, 7.0, 1.0),  # l
            (0.0, 7.0, 1.0),  # w
        ]
        default_vowels = vowel_points or [
            (6.0, 0.0, 0.0),  # i
            (6.0, 2.0, 1.0),  # u
            (0.0, 1.0, 0.0),  # a
        ]
        for pt in default_vowels:
            p = pt if isinstance(pt, Phoneme) else Phoneme(PhonemeKind.VOWEL, pt)
            self.add_phoneme(p)

        for pt in default_consonants:
            p = pt if isinstance(pt, Phoneme) else Phoneme(PhonemeKind.CONSONANT, pt)
            self.add_phoneme(p)

    @property
    def synthesis_index(self) -> float:
        if not self.words:
            return 0.50
        total_words = len(self.words)
        total_morphemes = sum(max(1, len(w.morphemes)) for w in self.words.values())
        m_ratio = total_morphemes / total_words
        return round(float(np.clip(0.18 + (m_ratio - 1.0) * 0.55, 0.15, 0.90)), 3)

    def codify_standard(self, epoch: int = 0) -> None:
        self.standard_canon = {w_id: w.copy() for w_id, w in self.words.items()}
        self.sociolinguistic_tension = 0.0
        self.last_reform_epoch = epoch

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
        self.codify_standard(epoch=0)
        self.rebuild_inventory()
        return len(generated_words)

    def nearest_word_to_vector(self, target_vec: SemanticVector) -> tuple[Word, float]:
        closest = min(
            self.words.values(),
            key=lambda w: w.vector.weighted_distance_to(target_vec, self.cultural_attention)
        )
        dist = closest.vector.weighted_distance_to(target_vec, self.cultural_attention)
        return closest, dist

    @property
    def cultural_inertia(self) -> float:
        w = self.cultural_attention
        return float((max(0.01, w[2]) ** 1.4 * max(0.01, w[5])) / max(0.20, 1.0 + w[4] * 0.8))

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
        for p in word.phonemes:
            self.phonemes[p.ipa] = p
        return word

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
                wells_mu = np.array([lp.point for lp in pool], dtype=np.float32)
                wells_weights = np.ones(len(pool), dtype=np.float32)
                token_coord = np.array([p.point], dtype=np.float32)

                best_idx, _ = GPUSemanticEngine.decode_listener_wells_batch(token_coord, wells_mu, wells_weights)
                adapted_phonemes.append(pool[best_idx[0]].drift(duration=p.duration, pitch_offset=p.pitch_offset, pitch_contour=p.pitch_contour))
            else:
                adapted_phonemes.append(p.drift())

        revalued_vector = foreign_word.vector.drift(cultural_delta)
        max_fam = max((w.derivation.root_family_id for w in self.words.values()), default=0)

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
                root_family_id=max_fam + 1,
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

    def calculate_lexical_usage(self) -> None:
        if not self.words:
            return

        living_words = list(self.words.values())
        attn = np.array(self.cultural_attention, dtype=np.float32)
        attn_sum = max(0.01, float(np.sum(attn)))

        total_burden = 0.0
        word_scores = []
        divergence_accum = 0.0

        form_counts = Counter(w.plain_form for w in living_words)

        for word in living_words:
            relevance = float(np.dot(word.vector.data, attn) / attn_sum)
            relevance = max(0.001, relevance)

            if getattr(word, '_cached_cost', None) is None:
                word._cached_cost = BiomechanicalCostModel.word_energetic_cost(
                    word, self.profile, self.environment, cultural_attention=self.cultural_attention
                )
            cost = float(word._cached_cost)
            total_burden += cost

            clash_penalty = 1.0 / (1.0 + 0.35 * max(0, form_counts[word.plain_form] - 1))

            base = 2.00 if word.is_proto_root else 1.00
            score = (base * (relevance ** 1.3) * clash_penalty) / max(0.1, 1.0 + cost * 0.20)
            word_scores.append((score, word))

            if word.id in self.standard_canon:
                std_w = self.standard_canon[word.id]
                divergence_accum += (word.form != std_w.form) * 1.0

        word_scores.sort(key=lambda x: x[0], reverse=True)
        self.cached_mean_cost = total_burden / max(1, len(living_words))

        for rank, (_, word) in enumerate(word_scores, start=1):
            load = 45.0 / (rank ** 0.72)
            word.usage_frequency = (0.75 * word.usage_frequency) + (0.25 * load)
            if word.usage_frequency < 0.28 and not word.is_proto_root and word.category != LexicalCategory.FUNCTIONAL_CLOSED:
                word.epochs_idle += 1
            else:
                word.epochs_idle = 0

        self.sociolinguistic_tension = round(divergence_accum / max(1.0, float(len(self.standard_canon))), 3)
        excess_cost = max(0.0, self.cached_mean_cost - 1.50)
        self.systemic_instability = max(0.0, (self.systemic_instability * 0.92) + (excess_cost * 0.12))

    def apply_coarticulatory_and_climate_shifts(self, words_dict: dict[int, Word], drift_rate: float, infrastructure: float = 1.0) -> None:
        alt = getattr(self.environment, "altitude", 0.0) if self.environment else 0.0
        hum = getattr(self.environment, "humidity", 0.5) if self.environment else 0.5
        temp = getattr(self.environment, "temperature", 0.5) if self.environment else 0.5

        rate = min(0.08, drift_rate * 0.20)
        kappa_standard = min(0.35, 0.05 * math.log(1.0 + max(0.0, infrastructure)))
        mucosal_lubrication = 1.0 / (1.0 + math.exp(-6.0 * (hum - 0.45))) * (0.8 + 0.4 * temp)

        for word in words_dict.values():
            freq_damping = 1.0 + math.log(word.usage_frequency + 1.0)
            step_size = rate / freq_damping

            p_list = list(word.phonemes)
            n_p = len(p_list)
            if n_p == 0:
                continue

            changed = False
            delta_buffer = list(p_list)

            # 1. Continuous Articulatory Field Assimilation
            for i in range(n_p):
                p = p_list[i]

                # Continuous Umlaut
                if p.kind == PhonemeKind.VOWEL:
                    for k in range(i + 1, n_p):
                        if p_list[k].kind == PhonemeKind.VOWEL:
                            next_v = p_list[k]
                            d_back = next_v.point[1] - p.point[1]
                            atten = 1.0 / float(k - i)
                            assim_pull = 0.35 * step_size * d_back * atten
                            if abs(assim_pull) > 0.005:
                                new_b = max(0.0, min(2.0, p.point[1] + assim_pull))
                                delta_buffer[i] = delta_buffer[i].drift(point=(p.point[0], new_b, p.point[2]))
                                changed = True
                            break

                # Continuous Dorsal Palatalization
                if p.kind == PhonemeKind.CONSONANT and p.articulator_tier == 2 and i + 1 < n_p:
                    next_seg = p_list[i + 1]
                    if next_seg.kind == PhonemeKind.VOWEL:
                        d_palatal = math.sqrt(((next_seg.point[0] - 6.0) / 6.0) ** 2 + (next_seg.point[1] / 2.0) ** 2)
                        palatal_pull = math.exp(-2.5 * d_palatal)
                        if palatal_pull > 0.05:
                            shift_pl = (4.0 - p.point[0]) * (0.45 * step_size * palatal_pull)
                            if abs(shift_pl) > 0.01:
                                new_pl = max(3.5, min(8.5, p.point[0] + shift_pl))
                                delta_buffer[i] = delta_buffer[i].drift(point=(new_pl, p.point[1], p.point[2]))
                                changed = True

                # Continuous Intervocalic Lenition
                if p.kind == PhonemeKind.CONSONANT and int(round(p.point[1])) == 0:
                    if i > 0 and i + 1 < n_p and p_list[i - 1].kind == PhonemeKind.VOWEL and p_list[i + 1].kind == PhonemeKind.VOWEL:
                        p_lenite = 0.25 * step_size * mucosal_lubrication
                        if random.random() < p_lenite:
                            delta_buffer[i] = delta_buffer[i].drift(point=(p.point[0], 4.0, 1.0))
                            changed = True

                # Tautosyllabic Coda Tonogenesis
                if p.kind == PhonemeKind.VOWEL and i + 1 < n_p and p_list[i + 1].kind == PhonemeKind.CONSONANT:
                    is_coda = (i + 2 >= n_p or p_list[i + 2].kind == PhonemeKind.CONSONANT)
                    if is_coda:
                        stiff = p_list[i + 1].laryngeal_stiffness
                        if abs(stiff) > 0.15:
                            new_pc = max(-1.0, min(1.0, p.pitch_contour + 0.35 * step_size * stiff))
                            delta_buffer[i] = delta_buffer[i].drift(pitch_contour=new_pc)
                            changed = True

                # Onset Voice Register Shift
                if p.kind == PhonemeKind.VOWEL and i > 0 and p_list[i - 1].kind == PhonemeKind.CONSONANT:
                    stiff = p_list[i - 1].laryngeal_stiffness
                    if abs(stiff) > 0.15:
                        new_po = max(-1.0, min(1.0, p.pitch_offset + 0.30 * step_size * stiff))
                        delta_buffer[i] = delta_buffer[i].drift(pitch_offset=new_po)
                        changed = True

            # 2. Continuous Atmospheric & Biomechanical Thermodynamics
            active_vowel_coords = [p_list[k].point for k in range(n_p) if p_list[k].kind == PhonemeKind.VOWEL]
            drying_power = max(0.0, 1.0 - hum) * max(0.2, 1.0 - temp * 0.4) + max(0.0, 0.35 - temp) * 1.4

            # Barometric density calibrated: zero hypoxia drive below ~1,500m (alt ~ 0.35)
            rho_air = math.exp(-0.90 * max(0.0, alt))
            altitude_hypoxia_drive = max(0.0, (1.0 - rho_air) - 0.20)
            p_ejective_prob = step_size * 0.15 * altitude_hypoxia_drive * max(0.0, drying_power - 0.25)

            # Continuous deglottalization in dense, moist lowlands
            lowland_moisture_gain = max(0.0, rho_air - 0.70) * (0.50 + 0.50 * hum)
            p_deglottalize = step_size * 0.40 * lowland_moisture_gain

            for i in range(n_p):
                p = delta_buffer[i]
                if p.kind == PhonemeKind.VOWEL:
                    h, b, r = p.point
                    dh_climate = drying_power * 0.25 * (5.0 - h) - max(0.0, 1.0 - drying_power) * 0.15 * h

                    f_h_repel, f_b_repel = 0.0, 0.0
                    for other_h, other_b, _ in active_vowel_coords:
                        dist_sq = ((h - other_h) / 6.0) ** 2 + ((b - other_b) / 2.0) ** 2 + 0.02
                        if dist_sq > 0.001:
                            repel = 0.025 / dist_sq
                            f_h_repel += repel * ((h - other_h) / 6.0)
                            f_b_repel += repel * ((b - other_b) / 2.0)

                    new_h = max(0.0, min(6.0, h + step_size * (dh_climate + f_h_repel * 6.0)))
                    new_b = max(0.0, min(2.0, b + step_size * (f_b_repel * 2.0)))
                    new_po = p.pitch_offset * 0.88
                    new_pc = p.pitch_contour * 0.88
                    new_dur = max(0.5, p.duration - 0.04 * step_size * (p.duration - 1.0))

                    if abs(new_h - h) > 0.02 or abs(new_b - b) > 0.02 or abs(new_po - p.pitch_offset) > 0.02:
                        delta_buffer[i] = delta_buffer[i].drift(
                            point=(new_h, new_b, r),
                            duration=new_dur,
                            pitch_offset=new_po,
                            pitch_contour=new_pc,
                        )
                        changed = True
                else:
                    pl, mn, vc = p.point
                    is_ejective = any(l == Airstream.EJECTIVE for l in p.layers)

                    if not is_ejective and mn == 0 and vc == 0 and p_ejective_prob > 0.0:
                        if random.random() < p_ejective_prob:
                            delta_buffer[i] = delta_buffer[i].drift(layers=(Airstream.EJECTIVE, SecondaryArticulation.NONE))
                            changed = True
                    elif is_ejective:
                        # Fricatives and sonorants lose ejective closure; lowlands relax ejectives back to pulmonic
                        is_non_stop = (mn != 0)
                        if is_non_stop or random.random() < p_deglottalize:
                            clean_layers = tuple(l for l in p.layers if l != Airstream.EJECTIVE)
                            if not clean_layers:
                                clean_layers = (Airstream.PULMONIC, SecondaryArticulation.NONE)
                            delta_buffer[i] = delta_buffer[i].drift(layers=clean_layers)
                            changed = True

            # 3. Prescriptive Pull Toward Standard Canon
            if word.id in self.standard_canon and kappa_standard > 0.0:
                std_w = self.standard_canon[word.id]
                if len(std_w.phonemes) == n_p:
                    for i, sp in enumerate(std_w.phonemes):
                        cur_p = delta_buffer[i]
                        if cur_p.kind == sp.kind:
                            dp0 = sp.point[0] - cur_p.point[0]
                            dp1 = sp.point[1] - cur_p.point[1]
                            new_p0 = cur_p.point[0] + kappa_standard * step_size * dp0
                            new_p1 = cur_p.point[1] + kappa_standard * step_size * dp1
                            delta_buffer[i] = delta_buffer[i].drift(point=(new_p0, new_p1, cur_p.point[2]))
                            changed = True

            if changed:
                _resync_word_structure(word, delta_buffer, self.profile)
                self.systemic_instability = max(0.0, self.systemic_instability - 0.08)

    def evolve(
        self,
        reduction_strength: float = 0.80,
        generational_drift: bool = True,
        drift_rate: float = 0.35,
        semantic_drift_rate: float = 0.20,
        enable_neologisms: bool = True,
        current_generation: int = 0,
        infrastructure: float = 1.0,
    ) -> None:
        if current_generation % 10 == 0 or self.systemic_instability >= 1.50:
            self._cached_grammar_paradigm = None

        if enable_neologisms:
            LexiconGenerator.expand_vocabulary(
                lang=self,
                growth_rate=0.04,
                d_sem_min=0.06,
                current_generation=current_generation,
                profile=self.profile,
            )

        self.calculate_lexical_usage()

        if self.sociolinguistic_tension >= 0.45 and infrastructure >= 1.20:
            self.codify_standard(epoch=current_generation)

        new_words: dict[int, Word] = {}
        effective_reduction = reduction_strength / (1.0 + self.cultural_inertia * 0.40)

        for word in list(self.words.values()):
            if word.is_obsolete(max_idle_epochs=25, cultural_inertia=self.cultural_inertia):
                continue

            current_word = word
            current_vector = current_word.vector

            updated_morphemes = []
            for morph in current_word.morphemes:
                if morph.is_grammatical or len(current_word.phonemes) >= 6:
                    eroded = ArticulatoryEnergyModel.erode_morpheme(
                        morph,
                        reduction_rate=effective_reduction,
                        usage_frequency=current_word.usage_frequency,
                        is_long_word=(len(current_word.phonemes) >= 6),
                        syllable_complexity=self.profile.syllable_complexity,
                    )
                    if len(eroded.phonemes) < len(morph.phonemes) and eroded.phonemes:
                        dropped_mass = sum(p.moraic_mass for p in morph.phonemes[len(eroded.phonemes):])
                        for idx_rev in range(len(eroded.phonemes) - 1, -1, -1):
                            target_v = eroded.phonemes[idx_rev]
                            if target_v.kind == PhonemeKind.VOWEL:
                                new_dur = min(2.2, target_v.duration + dropped_mass * 0.5)
                                eroded.phonemes[idx_rev] = target_v.drift(duration=new_dur)
                                break
                    updated_morphemes.append(eroded)
                else:
                    updated_morphemes.append(morph)

            freq_damping = 1.0 + math.log(current_word.usage_frequency + 1.0)
            if random.random() < (semantic_drift_rate / freq_damping):
                delta = [random.uniform(-0.025, 0.025) * (self.cultural_attention[d] / 1.0) for d in range(7)]
                current_vector = current_vector.drift(delta)

            fused_p = []
            for m in updated_morphemes:
                fused_p.extend(m.phonemes)
            repaired = ArticulatoryEnergyModel.repair_phonemes(fused_p or list(current_word.phonemes), self.profile)

            temp_word = Word(
                word_id=current_word.id,
                vector=current_vector,
                phonemes=repaired or [current_word.phonemes[0]],
                morphemes=updated_morphemes,
                derivation=current_word.derivation,
                usage_frequency=current_word.usage_frequency,
                category=current_word.category,
                generation_born=current_word.generation_born,
                senses=current_word.senses,
            )

            final_word = SyllableEngine.reduce_and_prosodify(
                temp_word,
                pattern=self.profile.stress_pattern,
                reduction_strength=effective_reduction,
                apocope_rate=self.profile.apocope_rate,
                syncope_rate=self.profile.syncope_rate,
                reduction_mode=self.profile.vowel_reduction_mode,
                environment=self.environment,
            )
            new_words[current_word.id] = final_word

        if generational_drift:
            self.apply_coarticulatory_and_climate_shifts(new_words, drift_rate, infrastructure=infrastructure)

        self.words = new_words
        self.rebuild_inventory()

    def rebuild_inventory(self) -> None:
        if not self.words:
            self.phonemes = {}
            return

        c_token_counts: Counter[str] = Counter()
        v_tokens: list[tuple[float, float, float, float]] = []
        c_tokens: list[tuple[float, float, float, float, tuple]] = []

        total_v_weight = 0.0
        total_c_weight = 0.0

        for w in self.words.values():
            u = w.usage_frequency
            for p in w.phonemes:
                seg_energy = u * p.duration * p.physics.acoustic_salience
                if p.kind == PhonemeKind.VOWEL:
                    v_tokens.append((p.point[0] / 6.0, p.point[1] / 2.0, p.point[2], seg_energy))
                    total_v_weight += seg_energy
                else:
                    c_tokens.append((p.point[0] / 10.0, p.point[1] / 7.0, p.point[2], seg_energy, p.layers))
                    total_c_weight += seg_energy
                    c_token_counts[p.ipa] += u

        core_inventory: dict[str, Phoneme] = {}

        # 1. 3D Vowel Peaks with Sharpness Gating
        if v_tokens and total_v_weight > 0.0:
            v_grid = np.zeros((10, 6, 2), dtype=np.float32)
            for h_n, b_n, r, wgt in v_tokens:
                r_i = int(np.clip(h_n * 9.0, 0, 9))
                c_i = int(np.clip(b_n * 5.0, 0, 5))
                rnd_i = 1 if r >= 0.5 else 0
                v_grid[r_i, c_i, rnd_i] += wgt

            cutoff_v = total_v_weight * 0.035
            for r in range(10):
                for c in range(6):
                    for rnd in range(2):
                        val = v_grid[r, c, rnd]
                        if val < cutoff_v:
                            continue
                        r0, r1 = max(0, r - 1), min(10, r + 2)
                        c0, c1 = max(0, c - 1), min(6, c + 2)
                        patch = v_grid[r0:r1, c0:c1, rnd]

                        if val == np.max(patch):
                            surround_mean = (np.sum(patch) - val) / max(1, patch.size - 1)
                            sharpness = (val - surround_mean) / val
                            if sharpness >= 0.22:
                                h_peak = (r / 9.0) * 6.0
                                b_peak = (c / 5.0) * 2.0
                                rn_peak = float(rnd)
                                sym = resolve_vowel(h_peak, b_peak, rn_peak)
                                core_inventory[sym] = Phoneme(PhonemeKind.VOWEL, (h_peak, b_peak, rn_peak))

        # 2. 3D Consonant Peaks with Topological Prominence
        if c_tokens and total_c_weight > 0.0:
            c_grid = np.zeros((11, 8, 2), dtype=np.float32)
            for pl_n, mn_n, vc, wgt, _ in c_tokens:
                p_i = int(np.clip(pl_n * 10.0, 0, 10))
                m_i = int(np.clip(mn_n * 7.0, 0, 7))
                vc_i = 1 if vc >= 0.5 else 0
                c_grid[p_i, m_i, vc_i] += wgt

            cutoff_c = total_c_weight * 0.025
            for p_i in range(11):
                for m_i in range(8):
                    for vc_i in range(2):
                        val = c_grid[p_i, m_i, vc_i]
                        if val < cutoff_c:
                            continue
                        p0, p1 = max(0, p_i - 1), min(11, p_i + 2)
                        m0, m1 = max(0, m_i - 1), min(8, m_i + 2)
                        patch = c_grid[p0:p1, m0:m1, vc_i]

                        if val == np.max(patch):
                            surround_mean = (np.sum(patch) - val) / max(1, patch.size - 1)
                            sharpness = (val - surround_mean) / val
                            if sharpness >= 0.20:
                                pl_val = float(p_i)
                                mn_val = float(m_i)
                                vc_val = float(vc_i)
                                sym = resolve_consonant(pl_val, mn_val, vc_val)
                                core_inventory[sym] = Phoneme(PhonemeKind.CONSONANT, (pl_val, mn_val, vc_val))

            # Only preserve active non-pulmonic or secondary-articulated segments with volume >= 2.0%
            for w in self.words.values():
                for p in w.phonemes:
                    if p.kind == PhonemeKind.CONSONANT:
                        has_special_layer = any(
                            (isinstance(l, Airstream) and l != Airstream.PULMONIC) or
                            (isinstance(l, SecondaryArticulation) and l != SecondaryArticulation.NONE)
                            for l in p.layers
                        )
                        if has_special_layer:
                            if (c_token_counts[p.ipa] / max(1.0, total_c_weight)) >= 0.020:
                                core_inventory[p.ipa] = p

        self.phonemes = core_inventory or {p.ipa: p for w in self.words.values() for p in w.phonemes}

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
        for word in self.words.values():
            daughter.add_word(word.copy())
        daughter.codify_standard(epoch=0)
        return daughter

    def get_phoneme_distribution_stats(self) -> dict:
        if not self.words:
            return {}

        c_counts = Counter()
        v_counts = Counter()
        total_c = 0.0
        total_v = 0.0

        for w in self.words.values():
            u = w.usage_frequency
            for p in w.phonemes:
                if p.kind == PhonemeKind.CONSONANT:
                    c_counts[p.ipa] += u
                    total_c += u
                else:
                    v_counts[p.ipa] += u
                    total_v += u

        c_pct = {ipa: round((cnt / max(1.0, total_c)) * 100, 2) for ipa, cnt in c_counts.most_common()}
        v_pct = {ipa: round((cnt / max(1.0, total_v)) * 100, 2) for ipa, cnt in v_counts.most_common()}

        total_tokens = total_c + total_v
        entropy = 0.0
        if total_tokens > 0:
            for cnt in list(c_counts.values()) + list(v_counts.values()):
                p_val = cnt / total_tokens
                if p_val > 0:
                    entropy -= p_val * math.log2(p_val)
            n_types = len(c_counts) + len(v_counts)
            max_e = math.log2(n_types) if n_types > 1 else 1.0
            evenness = round(min(1.0, entropy / max_e), 3)
        else:
            evenness = 0.0

        return {
            "total_living_words": len(self.words),
            "weighted_token_volume": int(total_tokens),
            "cv_ratio": round(total_c / max(1.0, total_v), 2),
            "entropy_evenness": evenness,
            "consonants": c_pct,
            "vowels": v_pct,
            "top_5_consonants": list(c_pct.items())[:5],
            "top_3_vowels": list(v_pct.items())[:3],
        }

    def __repr__(self) -> str:
        return f"Language({self.name!r}, {len(self.vowels)} vowels, {len(self.consonants)} consonants, {len(self.words)} words)"