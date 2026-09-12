"""
language.py — Pure Linguistic Container with Two-Layer Acoustic Physics and Cultural Genome.
Features Continuous 3D Articulatory Potential Fields, Demographic Sociolinguistic Leveling,
Frequency-First Quantal Inventory Clustering, and Proportional Morpheme Boundary Resynchronization.
"""

from __future__ import annotations

from typing import Sequence, NamedTuple
import random
import math
from collections import Counter, defaultdict

from articulatory_space import Airstream, SecondaryArticulation
from phoneme import Phoneme, PhonemeKind
from word import Word, Morpheme, ClineStage, LexicalCategory
from syllable import SyllableEngine, StressPattern
from energy import ArticulatoryEnergyModel
from semantics import SemanticVector, DerivationRecord, DerivationType, LoanProvenance
from lexicon import LexiconGenerator


# =====================================================================
# Language Profile (The Linguistic State)
# =====================================================================

class LanguageProfile(NamedTuple):
    name: str
    max_transition_cost: float
    voicing_assimilation_bias: str
    allow_clusters: bool
    double_stop_strategy: str
    epenthetic_vowel_point: tuple[float, float, float]
    stress_pattern: StressPattern
    vowel_reduction_mode: str
    apocope_rate: float = 0.50
    syncope_rate: float = 0.50
    tone_tier: int = 0
    head_directionality: float = 0.50
    d_min_vowel: float = 0.38
    d_min_consonant: float = 0.28
    synthesis_index: float = 0.50
    syllable_complexity: float = 0.50
    acoustic_roughness: float = 0.50

    @property
    def is_tonal(self) -> bool:
        return self.tone_tier > 0

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

    @property
    def word_order(self) -> str:
        return self.dominant_word_order


# =====================================================================
# Language Class
# =====================================================================

class Language:
    """A living speech community governed by physical acoustics and a 7D cultural attention genome."""

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

        alt = getattr(self.environment, "altitude", 0.0) if self.environment else 0.0
        hum = getattr(self.environment, "humidity", 0.5) if self.environment else 0.5
        temp = getattr(self.environment, "temperature", 0.5) if self.environment else 0.5
        noise = getattr(self.environment, "ambient_noise", 0.2) if self.environment else 0.2
        pop = self.population

        # Emergent vegetation
        veg = self.environment.emergent_vegetation(population=pop) if hasattr(self.environment, "emergent_vegetation") else 0.50
        aridity = 1.0 - hum

        # 1. Human Subsistence & Ecological Coupling Functions
        # A. Arable Agriculture: Peaks in temperate, well-watered, arable soils
        agri_dist = ((hum - 0.55) / 0.22) ** 2 + ((veg - 0.45) / 0.22) ** 2 + ((temp - 0.55) / 0.22) ** 2
        arable_agri = math.exp(-agri_dist) * max(0.0, 1.0 - alt * 1.5)

        # B. Pastoral Grazing: Peaks in open grasslands/semi-arid steppes with low population
        pastoral_dist = ((veg - 0.35) / 0.18) ** 2 + ((hum - 0.40) / 0.20) ** 2
        pastoral_herding = math.exp(-pastoral_dist) * (1.0 - pop * 0.70)

        # C. Steppe Nomadic Mobility: Open terrain with low population density
        steppe_mobility = (1.0 - veg) * (1.0 - pop * 0.80)

        # D. Imperial Military Expansion: Organized population across open/conquerable plains
        imperial_potency = pop * (1.0 - veg * 0.50)

        raw_env_targets = [
            0.45 + 0.70 * alt + 0.85 * arable_agri + 0.45 * pop * (1.0 - veg),      # Concreteness (Soil, tools, stone, engineering)
            0.45 + 0.80 * (1.0 - pop) * veg + 0.95 * pastoral_herding,              # Animacy (Faunal dependency, herds, kinship)
            0.45 + 0.90 * aridity + 0.45 * pop * (1.0 - arable_agri * 0.4),         # Valence (Desert asceticism & institutional taboo)
            0.45 + 0.75 * alt + 0.95 * imperial_potency + 0.30 * noise,              # Potency (Mountain fortitude & military imperium)
            0.45 + 0.75 * hum * (0.50 + noise) + 0.95 * steppe_mobility,            # Dynamism (Maritime trade ports & nomadic speed)
            0.45 + 1.05 * pop + 0.20 * temp,                                        # Sociality (Balanced civic social contracts)
            0.45 + 0.90 * (1.0 - veg) + 0.65 * imperial_potency,                    # Extension (Geographic horizons & imperial reach)
        ]

        # Finite Cognitive Budget Normalization (Sum = 7.0, Mean = 1.0)
        total_raw = sum(raw_env_targets) or 7.0
        self.cultural_attention = [round(7.0 * (w / total_raw), 3) for w in raw_env_targets]

        self.profile = profile or (environment.generate_profile(name=f"{name} Profile", cultural_attention=self.cultural_attention) if environment else LanguageProfile(
            name=f"{name} Profile",
            max_transition_cost=3.00,
            voicing_assimilation_bias="regressive",
            allow_clusters=True,
            double_stop_strategy="spirantization",
            epenthetic_vowel_point=(3.0, 1.0, 0.0),
            stress_pattern=StressPattern.PENULTIMATE,
            vowel_reduction_mode="inventory_snap",
            apocope_rate=0.50,
            syncope_rate=0.50,
            tone_tier=0,
            head_directionality=0.50,
            d_min_vowel=0.38,
            d_min_consonant=0.28,
            synthesis_index=0.50,
            syllable_complexity=0.50,
        ))

        self.phonemes: dict[str, Phoneme] = {}
        if phonemes:
            for p in phonemes:
                self.phonemes[p.ipa] = p

        self.words: dict[int, Word] = {}
        self.contacts: dict[Language, float] = {}
        self._cached_grammar_paradigm = None


    def populate_default_phonemes(
        self,
        vowel_points: Sequence[tuple[float, float, float] | Phoneme] | None = None,
        consonant_points: Sequence[tuple[float, float, float] | Phoneme] | None = None,
    ) -> None:
        """
        Initializes the proto-phonemic inventory.
        Can be customized by passing coordinate tuples or Phoneme objects.
        Defaults to the universal 5-vowel cardinal grid and balanced baseline consonants.
        """
        # Default 5-Vowel Cardinal Grid: /i, u, e, o, a/
        default_vowels = vowel_points or [
            (6.0, 0.0, 0.0),  # /i/
            (6.0, 2.0, 1.0),  # /u/
            (4.0, 0.0, 0.0),  # /e/
            (4.0, 2.0, 1.0),  # /o/
            (0.0, 1.0, 0.0),  # /a/
        ]

        # Default Balanced Consonants: p, b, t, d, k, g, f, v, s, z, ʃ, x, h, m, n, l, w
        default_consonants = consonant_points or [
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),  # p, b
            (3.0, 0.0, 0.0), (3.0, 0.0, 1.0),  # t, d
            (7.0, 0.0, 0.0), (7.0, 0.0, 1.0),  # k, g
            (1.0, 4.0, 0.0), (1.0, 4.0, 1.0),  # f, v
            (3.0, 4.0, 0.0), (3.0, 4.0, 1.0),  # s, z
            (4.0, 4.0, 0.0),                   # ʃ
            (7.0, 4.0, 0.0),                   # x
            (10.0, 4.0, 0.0),                  # h
            (0.0, 1.0, 1.0), (3.0, 1.0, 1.0),  # m, n
            (3.0, 7.0, 1.0), (0.0, 7.0, 1.0),  # l, w
        ]

        for pt in default_vowels:
            if isinstance(pt, Phoneme):
                self.add_phoneme(pt)
            else:
                self.add_phoneme(Phoneme(PhonemeKind.VOWEL, pt))

        for pt in default_consonants:
            if isinstance(pt, Phoneme):
                self.add_phoneme(pt)
            else:
                self.add_phoneme(Phoneme(PhonemeKind.CONSONANT, pt))
    
    
    @property
    def cultural_inertia(self) -> float:
        """Layer 2 Derived Cultural Inertia (Purism): V^1.4 * S / (1 + D)."""
        w = self.cultural_attention
        return (w[2] ** 1.4 * w[5]) / (1.0 + w[4] * 0.8)

    @property
    def consonants(self) -> list[Phoneme]:
        return [p for p in self.phonemes.values() if p.kind == PhonemeKind.CONSONANT]

    @property
    def vowels(self) -> list[Phoneme]:
        return [p for p in self.phonemes.values() if p.kind == PhonemeKind.VOWEL]

    @property
    def word_order(self) -> str:
        return self.profile.dominant_word_order

    def set_contact(self, other_language: Language, intensity: float) -> None:
        clamped = max(0.0, min(1.0, float(intensity)))
        self.contacts[other_language] = clamped
        other_language.contacts[self] = clamped

    def add_phoneme(self, phoneme: Phoneme) -> Phoneme:
        self.phonemes[phoneme.ipa] = phoneme
        return phoneme

    def add_word(self, word: Word) -> Word:
        self.words[word.id] = word
        for p in word.phonemes:
            self.phonemes[p.ipa] = p
        return word

    def nearest_word_to_vector(self, target_vec: SemanticVector) -> tuple[Word, float]:
        closest = min(self.words.values(), key=lambda w: w.vector.weighted_distance_to(target_vec, self.cultural_attention))
        dist = closest.vector.weighted_distance_to(target_vec, self.cultural_attention)
        return closest, dist

    def calculate_lexical_usage(self) -> None:
        """Calculates power-law Zipfian communicative load across orders of magnitude."""
        if not self.words:
            return

        living_words = list(self.words.values())
        family_tree_counts = Counter(w.derivation.root_family_id for w in living_words)
        
        attn = self.cultural_attention
        attn_sum = max(0.01, sum(attn))
        pop = self.population
        grammatical_concreteness_cutoff = 0.12 + (1.0 - pop) * 0.12
        k_cap = 2500.0 * (1.0 + 15.0 * (pop ** 1.6) * (1.0 - min(1.0, self.cultural_inertia) * 0.40))

        # Continuous Articulatory Impedance Drag (Zipf 1935 / Martinet 1955)
        # Purist cultures (high inertia) tolerate difficult words; dynamic cultures penalize friction heavily
        k_effort = 0.35 / (1.0 + 0.60 * self.cultural_inertia)

        raw_loads: list[tuple[float, Word]] = []
        for word in living_words:
            u_base = 3.50 if word.is_proto_root else 1.00
            relevance = sum(word.vector.coords[i] * attn[i] for i in range(7)) / attn_sum
            is_functional = (word.category == LexicalCategory.FUNCTIONAL_CLOSED or word.vector.concreteness < grammatical_concreteness_cutoff)
            
            children_count = family_tree_counts.get(word.derivation.root_family_id, 1)
            hub_bonus = 1.0 + 0.40 * math.tanh(children_count / 6.0)

            # Calculate articulatory difficulty of the word
            p_len = len(word.phonemes)
            if p_len >= 2:
                transition_sum = sum(
                    ArticulatoryEnergyModel.transition_cost(word.phonemes[j], word.phonemes[j+1])
                    for j in range(p_len - 1)
                )
                markedness_sum = sum(p.weight for p in word.phonemes)
                art_difficulty = (transition_sum + markedness_sum) / float(p_len)
            else:
                art_difficulty = 1.00

            # Communicative load: high articulatory difficulty creates frequency drag unless eroded!
            effort_drag = 1.0 + k_effort * max(0.0, art_difficulty - 1.20)
            score = (u_base * (relevance ** 1.4) * (6.0 if is_functional else 1.0) * hub_bonus) / effort_drag
            raw_loads.append((score, word))

        raw_loads.sort(key=lambda x: x[0], reverse=True)

        for rank, (_, word) in enumerate(raw_loads, start=1):
            zipf_epoch_load = 45.0 / (rank ** 0.72)
            word.usage_frequency = (0.75 * word.usage_frequency) + (0.25 * zipf_epoch_load)
            
            if (rank > k_cap or word.usage_frequency < 0.28) and not word.is_proto_root and word.category != LexicalCategory.FUNCTIONAL_CLOSED:
                word.epochs_idle += 1
            else:
                word.epochs_idle = 0

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

        # Proportional Morpheme Resynchronization
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
        calqued_word = SyllableEngine.apply_prosody(calqued_word, pattern=self.profile.stress_pattern, tone_tier=self.profile.tone_tier)
        self.add_word(calqued_word)
        return calqued_word

    @staticmethod
    def acoustic_vowel_distance(p1: Phoneme, p2: Phoneme) -> float:
        """
        Psychoacoustic Formant Distance (Zwicker Bark Scale / Lindblom Dispersion).
        - Height (F1) weighted with primary perceptual salience (separates /e, o/ from /i, u/).
        - Backness (F2) spread scales with trapezoid height.
        """
        h1, h2 = p1.point[0] / 6.0, p2.point[0] / 6.0
        b1, b2 = p1.point[1] / 2.0, p2.point[1] / 2.0
        f2_1 = 0.5 + (b1 - 0.5) * (0.20 + 0.80 * h1)
        f2_2 = 0.5 + (b2 - 0.5) * (0.20 + 0.80 * h2)

        d_f1 = abs(h1 - h2) * 1.40
        d_f2 = abs(f2_1 - f2_2) * 1.25
        d_rnd = abs(p1.point[2] - p2.point[2]) * 0.35
        d_nas = 0.30 if (p1.layers != p2.layers and (SecondaryArticulation.NASALIZED in p1.layers or SecondaryArticulation.NASALIZED in p2.layers)) else 0.0

        return math.sqrt(d_f1**2 + d_f2**2 + d_rnd**2 + d_nas**2)

    @staticmethod
    def acoustic_consonant_distance(p1: Phoneme, p2: Phoneme) -> float:
        """
        Computes continuous acoustic and articulatory distance between consonants.
        Distinctly categorizes Affricates (mn=2) from Liquids/Trills (mn=3).
        """
        pl1, mn1, vc1 = p1.point
        pl2, mn2, vc2 = p2.point

        d_place = abs(pl1 - pl2) / 10.0 * 1.10

        def get_manner_class(mn: float) -> int:
            m = int(round(mn))
            if m == 0: return 0       # Stops
            if m == 1: return 1       # Nasals
            if m == 2: return 2       # Affricates
            if m == 3: return 3       # Taps / Trills
            if m in (4, 5): return 4  # Central & Lateral Fricatives
            return 5                  # Approximants (6, 7)

        mc1, mc2 = get_manner_class(mn1), get_manner_class(mn2)
        d_manner = abs(mn1 - mn2) * 0.08 if mc1 == mc2 else (0.38 + abs(mc1 - mc2) * 0.06)

        is_obs1 = (int(round(mn1)) in (0, 2, 4, 5))
        is_obs2 = (int(round(mn2)) in (0, 2, 4, 5))
        d_voice = abs(vc1 - vc2) * (0.45 if is_obs1 and is_obs2 else 0.08)

        d_layer = 0.0
        if p1.layers != p2.layers:
            a1 = next((layer for layer in p1.layers if isinstance(layer, Airstream)), Airstream.PULMONIC)
            a2 = next((layer for layer in p2.layers if isinstance(layer, Airstream)), Airstream.PULMONIC)
            if a1 != a2: d_layer += 0.55
            s1 = next((layer for layer in p1.layers if isinstance(layer, SecondaryArticulation)), SecondaryArticulation.NONE)
            s2 = next((layer for layer in p2.layers if isinstance(layer, SecondaryArticulation)), SecondaryArticulation.NONE)
            if s1 != s2: d_layer += 0.35

        return math.sqrt(d_place**2 + d_manner**2 + d_voice**2 + d_layer**2)

    def apply_regular_sound_laws(self, words_dict: dict[int, Word], drift_rate: float) -> None:
        """
        Unified Closed-Loop Continuous Articulatory & Acoustic Field Dynamics:
        - Dual Anatomical Energy Basins: Coronal (pl=3.0) and Velar/Dorsal (pl=7.0).
        - Open-Floor Acoustic Rounding Decoupling (unifies open vowels into /a/ /ä/, eliminating /ɒ/ leakage).
        - Complete Closed-Loop Manner Transitions (Lateral Fricatives mn=5 lenite to /l/ or /z/).
        - Symmetrical Boundary Potential Field & Universal Aerodynamic Voicing Equilibrium.
        """
        pop = self.population
        alt = getattr(self.environment, "altitude", 0.0) if self.environment else 0.0
        hum = getattr(self.environment, "humidity", 0.5) if self.environment else 0.5
        temp = getattr(self.environment, "temperature", 0.5) if self.environment else 0.5
        noise = getattr(self.environment, "ambient_noise", 0.2) if self.environment else 0.2
        veg = self.environment.emergent_vegetation(population=pop, cultural_attention=self.cultural_attention) if hasattr(self.environment, "emergent_vegetation") else 0.50

        # Gas Law Air Density: Strictly real & positive across all temperatures (-100 to +1000)
        rho_air = math.exp(-2.40 * max(0.0, alt)) * math.exp(-0.25 * (temp - 0.50))
        p_ejective_field = max(0.0, 0.42 - rho_air) ** 2.2 * 0.35
        p_ejective_relax = (1.0 / (1.0 + math.exp(-7.0 * (rho_air - 0.42)))) * 0.40

        p_implosive_field = max(0.0, hum - 0.55) * 0.30
        aridity_gradient = max(0.0, 0.50 - hum) * (1.0 + max(0.0, temp) * 0.50)
        canopy_damping = max(0.0, veg * 0.65 + hum * 0.35 - 0.40)

        savannah_dist_sq = (
            ((temp - 0.70) / 0.22) ** 2 +
            (hum / 0.25) ** 2 +
            (veg / 0.25) ** 2 +
            (noise / 0.25) ** 2 +
            (pop / 0.35) ** 2 +
            (alt / 0.35) ** 2
        )
        click_propensity = math.exp(-savannah_dist_sq / 2.0)
        click_relaxation = max(0.0, 0.30 - click_propensity) * 1.50

        d_v_min = self.profile.d_min_vowel

        for word in words_dict.values():
            u = word.usage_frequency
            c_v = word.vector
            v_sacred = c_v.valence * self.cultural_attention[2]
            s_social = c_v.sociality * self.cultural_attention[5]
            
            word_inertia = math.log(u + 1.0) * 0.45 + (v_sacred + s_social) * 0.25
            word_drift_prob = drift_rate / (1.0 + word_inertia)

            if random.random() > word_drift_prob:
                continue

            p_list: list[Phoneme | None] = list(word.phonemes)
            changed = False
            n = len(p_list)

            for i in range(n):
                p = p_list[i]
                if p is None:
                    continue

                prev_p = next((p_list[k] for k in range(i - 1, -1, -1) if p_list[k] is not None), None)
                next_p = next((p_list[k] for k in range(i + 1, n) if p_list[k] is not None), None)

                is_initial = (prev_p is None)
                is_intervocalic = (prev_p and prev_p.kind == PhonemeKind.VOWEL and next_p and next_p.kind == PhonemeKind.VOWEL)
                is_final = (next_p is None or next_p.kind == PhonemeKind.CONSONANT)

                # =============================================================
                # SECTION A: CONTINUOUS 3D CONSONANT CLOSED-LOOP FIELD
                # =============================================================
                if p.kind == PhonemeKind.CONSONANT:
                    pl, mn, vc = p.point
                    f_pl, f_mn, f_vc = 0.0, 0.0, 0.0
                    new_layers = p.layers

                    # 1. Dual Anatomical Energy Basins:
                    # A. Coronal Basin (Restoring Attractor at pl=3.0)
                    if 1.8 < pl < 5.5:
                        f_pl += -0.12 * (pl - 3.0)

                    # B. Velar/Dorsal Basin (Restoring Attractor at pl=7.0: Protects /k, ɡ/ from extinction!)
                    elif 6.0 <= pl <= 8.0:
                        f_pl += -0.15 * (pl - 7.0)

                    # 2. Epiglottic Sphincter Energy Barrier (Place 9)
                    if 8.5 <= pl <= 9.6 and aridity_gradient < 0.10:
                        f_pl += 0.50 * (pl - 8.5)

                    # 3. Articulator-Bounded Co-articulation Field
                    adj_vowels = [adj for adj in (prev_p, next_p) if adj and adj.kind == PhonemeKind.VOWEL]
                    if adj_vowels:
                        avg_backness = sum(v.point[1] for v in adj_vowels) / len(adj_vowels)
                        avg_height = sum(v.point[0] for v in adj_vowels) / len(adj_vowels)

                        # Tongue-Root Retraction on Dorsals & Glottals (Requires true aridity to retract velars to uvulars)
                        if (avg_backness >= 1.0 or avg_height <= 1.5) and pl >= 6.8 and aridity_gradient > 0.05:
                            f_pl += 0.35 * (avg_backness - 0.50) * aridity_gradient
                            if pl >= 8.8 and aridity_gradient > 0.10 and random.random() < 0.28:
                                f_mn += 4.0 if mn <= 1.0 else 0.0

                        # Palatal Fronting on Dorsals before high-front vowels (i, e)
                        if avg_height >= 4.0 and avg_backness <= 0.6 and 6.0 <= pl <= 8.0:
                            f_pl -= 0.35 * (1.0 - avg_backness)
                            if mn <= 1.0 and random.random() < 0.18:
                                f_mn += 2.0  # k -> t͡ʃ

                    # 4. Environmental Potentials & Calibrated Demographic Leveling
                    if aridity_gradient > 0.08 and pl >= 7.5:
                        f_pl += 0.35 * aridity_gradient

                    if canopy_damping > 0.15 and 3.5 <= mn <= 5.5 and 2.2 <= pl <= 5.5:
                        f_mn -= 0.30

                    # Calibrated Demographic Sociolinguistic Leveling
                    f_pl += -0.18 * (pop ** 1.4) * math.exp(-((pl - 8.8) ** 2) / 1.5) * (pl - 7.0)

                    # 5. Soft Local Dispersion
                    f_pl += random.gauss(0.0, 0.08)
                    f_mn += random.gauss(0.0, 0.05)

                    # 6. Continuous Aerodynamic Lenition, Fortition & Roughness-Coupled Cycles
                    s_comp = getattr(self.profile, "syllable_complexity", 0.50)
                    r_rough = getattr(self.profile, "acoustic_roughness", 0.50)

                    if is_intervocalic:
                        f_vc += 0.16

                        # A. Plosive Lenition (Coupled to Acoustic Roughness & C:V ratio)
                        if mn <= 0.8:
                            if pl >= 8.8:
                                if random.random() < 0.35:
                                    p_list[i] = None
                                    changed = True
                                    continue
                                else:
                                    f_mn += 4.0  # Breathy voice (ɦ)
                                    f_vc += 0.5
                            else:
                                # High Roughness (Arabic, German) licenses affricates (t -> t͡s)
                                # Low Roughness / Vocalic Dominance (Hawaiian, Japonic) routes into sonorants (l, ɾ, w)
                                p_affricate = 0.38 * (s_comp ** 1.2) * (r_rough ** 1.2)
                                r_len = random.random()

                                if r_len < p_affricate:
                                    f_mn += 2.0  # Stop -> Affricate
                                elif r_len < (p_affricate + 0.40) or canopy_damping > 0.10:
                                    f_mn += 7.0 if random.random() < 0.50 else 3.0  # Sonorant (Liquid/Glide)
                                else:
                                    f_mn += 4.0  # Fricative

                        # B. Sibilant Aerodynamics (Ohala Voicing Wall & Rhotacism)
                        elif 3.5 <= mn <= 4.5 and 2.2 <= pl <= 5.5:
                            if vc >= 0.5:
                                if random.random() < 0.15:
                                    f_mn -= 1.0  # Intervocalic Rhotacism: z -> ɾ
                                elif random.random() < 0.20:
                                    f_vc -= 0.40  # Devoicing: z -> s
                            elif (hum > 0.60 or canopy_damping > 0.15) and random.random() < 0.12:
                                p_list[i] = Phoneme(PhonemeKind.CONSONANT, (10.0, 4.0, 0.0))  # s -> h
                                changed = True
                                continue

                        # C. Affricate De-affrication (Accelerated in soft languages)
                        elif 1.5 <= mn <= 2.5:
                            p_deaffricate = 0.25 * (1.20 - r_rough)
                            if random.random() < p_deaffricate:
                                f_mn += 2.0  # t͡s -> s

                        # D. Liquid / Rhotic Outflow (Lambdacism: r -> l or j)
                        elif 2.8 <= mn <= 3.5 and random.random() < 0.20:
                            f_mn += 4.0 if random.random() < 0.60 else 3.0

                        # E. Lateral Fricative Outflow (Opens to /l/ or delateralizes to /z/)
                        elif 4.6 <= mn <= 5.5:
                            if random.random() < 0.35:
                                f_mn += 2.0  # Manner 5 -> Manner 7: Opens to Lateral Approximant /l/!
                            else:
                                f_mn -= 1.0  # Manner 5 -> Manner 4: Delateralizes to Sibilant /z/ or /s/!

                        # F. Non-sibilant Voiced Fricatives -> Glides/Laterals
                        elif 3.5 <= mn <= 4.5 and vc >= 0.45 and not (2.2 <= pl <= 5.5):
                            f_mn += 2.5 if random.random() < 0.50 else 3.0

                    if is_final:
                        if 2.8 <= mn <= 3.5 and (hum > 0.50 or random.random() < 0.20):
                            p_list[i] = None
                            changed = True
                            continue
                        elif 3.5 <= mn <= 4.5 and 2.2 <= pl <= 5.5 and random.random() < 0.18:
                            p_list[i] = None
                            changed = True
                            continue

                    # Onset Aerodynamics
                    if is_initial:
                        if mn in (0, 2, 4) and vc >= 0.5 and random.random() < 0.22:
                            f_vc -= 0.50  # Initial devoicing

                        if mn >= 5.5 and random.random() < 0.22:
                            if pl <= 1.5:
                                p_list[i] = Phoneme(PhonemeKind.CONSONANT, (1.0, 4.0, 1.0))
                                changed = True
                                continue
                            elif 5.2 <= pl <= 6.8:
                                p_list[i] = Phoneme(PhonemeKind.CONSONANT, (3.0, 4.0, 1.0))
                                changed = True
                                continue
                        elif 2.8 <= mn <= 3.5 and random.random() < 0.18:
                            p_list[i] = Phoneme(PhonemeKind.CONSONANT, (3.0, 0.0, 1.0))
                            changed = True
                            continue
                        elif pl >= 9.2 and random.random() < 0.25:
                            target_fort = (3.0, 0.0, 0.0) if random.random() < 0.60 else (7.0, 0.0, 0.0)
                            p_list[i] = Phoneme(PhonemeKind.CONSONANT, target_fort)
                            changed = True
                            continue

                    if pl >= 9.2 and aridity_gradient < 0.08 and random.random() < 0.35:
                        p_list[i] = None
                        changed = True
                        continue

                    # 7. Non-Pulmonic Layer Vector Dynamics
                    if new_layers[0] == Airstream.EJECTIVE:
                        if random.random() < p_ejective_relax:
                            new_layers = (Airstream.PULMONIC, SecondaryArticulation.NONE)
                    elif p_ejective_field > 0.01 and mn <= 0.8 and vc <= 0.35 and pl < 9.2:
                        if random.random() < p_ejective_field:
                            new_layers = (Airstream.EJECTIVE, SecondaryArticulation.NONE)
                            f_vc = -1.0

                    if new_layers[0] == Airstream.CLICK:
                        if random.random() < click_relaxation:
                            new_layers = (Airstream.PULMONIC, SecondaryArticulation.NONE)
                    elif click_propensity > 0.20 and new_layers[0] != Airstream.CLICK:
                        if is_initial and pl <= 4.8 and (mn <= 0.8 or 3.2 <= mn <= 4.8) and random.random() < (click_propensity * 0.18):
                            new_layers = (Airstream.CLICK, SecondaryArticulation.NONE)

                    if new_layers[0] == Airstream.IMPLOSIVE:
                        p_implosive_relax = max(0.12, (1.0 - hum) * 0.45)
                        if random.random() < p_implosive_relax:
                            new_layers = (Airstream.PULMONIC, SecondaryArticulation.NONE)
                    elif p_implosive_field > 0.01 and mn <= 0.8 and vc >= 0.65 and (pl <= 1.5 or 2.2 <= pl <= 3.8 or 6.2 <= pl <= 7.8):
                        if random.random() < p_implosive_field:
                            new_layers = (Airstream.IMPLOSIVE, SecondaryArticulation.NONE)

                    if (
                        aridity_gradient > 0.08 
                        and 1.8 <= pl <= 4.8 
                        and (mn <= 0.8 or 3.2 <= mn <= 4.8)
                        and SecondaryArticulation.PHARYNGEALIZED not in new_layers
                    ):
                        has_pharyngeal = any(
                            seg is not None and seg.kind == PhonemeKind.CONSONANT and 
                            (SecondaryArticulation.PHARYNGEALIZED in seg.layers or seg.point[0] >= 8.5)
                            for seg in p_list
                        )
                        if not has_pharyngeal and adj_vowels and any(v.point[1] >= 0.8 or v.point[0] <= 1.8 for v in adj_vowels):
                            if random.random() < 0.28:
                                new_layers = (new_layers[0], SecondaryArticulation.PHARYNGEALIZED)

                    if SecondaryArticulation.PHARYNGEALIZED in new_layers and adj_vowels:
                        is_pure_front = all(v.point[0] >= 3.8 and v.point[1] <= 0.5 for v in adj_vowels)
                        if (is_pure_front or aridity_gradient < 0.06) and random.random() < 0.18:
                            new_layers = (new_layers[0], SecondaryArticulation.NONE)

                    raw_pl = max(0.0, min(10.0, pl + f_pl * 0.40))
                    raw_mn = max(0.0, min(7.0, mn + f_mn * 0.40))
                    raw_vc = max(0.0, min(1.0, vc + f_vc * 0.35))

                    p_list[i] = p.drift(point=(raw_pl, raw_mn, raw_vc), layers=new_layers)
                    changed = True
                    continue

                # =============================================================
                # SECTION B: CONTINUOUS 3D VOWEL POTENTIAL FIELD
                # =============================================================
                elif p.kind == PhonemeKind.VOWEL:
                    h, b, r = p.point
                    f_h, f_b, f_r = 0.0, 0.0, 0.0

                    if (
                        h <= 1.8 and next_p and next_p.kind == PhonemeKind.VOWEL and next_p.point[0] >= 4.2
                        and random.random() < 0.28
                    ):
                        is_back_glide = (next_p.point[1] >= 1.2)
                        p_list[i] = p.drift(point=(4.0, 2.0 if is_back_glide else 0.0, 1.0 if is_back_glide else 0.0))
                        for k in range(i + 1, n):
                            if p_list[k] is next_p:
                                p_list[k] = None
                                break
                        changed = True
                        continue

                    if (
                        h >= 4.2 and next_p and next_p.kind == PhonemeKind.VOWEL and next_p.point[0] <= 4.0
                        and random.random() < 0.32
                    ):
                        is_front_i = (b <= 0.8)
                        p_list[i] = Phoneme(PhonemeKind.CONSONANT, (6.0, 6.0, 1.0) if is_front_i else (0.0, 7.0, 1.0))
                        changed = True
                        continue

                    if b >= 1.2 and next_p and next_p.kind in (PhonemeKind.VOWEL, PhonemeKind.CONSONANT):
                        has_front_trigger = (
                            (next_p.kind == PhonemeKind.VOWEL and next_p.point[0] >= 4.0 and next_p.point[1] <= 0.5) or
                            (next_p.kind == PhonemeKind.CONSONANT and next_p.point[0] == 6.0)
                        )
                        if has_front_trigger and random.random() < 0.28:
                            f_b -= 1.4
                            f_r += 0.5

                    s_comp = getattr(self.profile, "syllable_complexity", 0.50)
                    if s_comp >= 0.70 and is_final:
                        if h >= 5.5 and random.random() < 0.25:
                            f_h -= 1.0
                        elif 3.5 <= h <= 4.5 and random.random() < 0.25:
                            f_h -= 1.8

                    is_nasal = (SecondaryArticulation.NASALIZED in p.layers)
                    if not is_nasal:
                        if (
                            hum > 0.60 and next_p and next_p.kind == PhonemeKind.CONSONANT 
                            and 0.5 <= next_p.point[1] <= 1.5 and is_final and random.random() < (hum * 0.28)
                        ):
                            p_list[i] = p.drift(layers=(SecondaryArticulation.NASALIZED,))
                            for k in range(i + 1, n):
                                if p_list[k] is next_p:
                                    p_list[k] = None
                                    break
                            changed = True
                            continue
                    else:
                        p_denasal = max(0.08, (1.0 - hum) * 0.45)
                        if random.random() < p_denasal:
                            p_list[i] = p.drift(layers=(SecondaryArticulation.NONE,))
                            changed = True
                            continue

                    # Resting Mandibular Equilibrium
                    h_rest = 3.0 - 0.80 * (temp - 0.50)
                    f_h += -0.15 * (h - h_rest)

                    active_v_points = [v.point for v in self.phonemes.values() if v.kind == PhonemeKind.VOWEL and v.ipa != p.ipa]
                    for v_h, v_b, _ in active_v_points:
                        dh = (h - v_h) / 6.0
                        db = (b - v_b) / 2.0
                        dist_sq = dh**2 + db**2 + 0.02
                        dist = math.sqrt(dist_sq)

                        repel_mag = (0.040 / (dist_sq ** 0.85)) * (max(0.20, d_v_min) / 0.38)
                        f_h += repel_mag * (dh / dist)
                        f_b += repel_mag * (db / dist)

                    # Symmetrical Boundary Potential
                    f_h += 0.08 * (1.0 / ((h + 0.35) ** 2) - 1.0 / ((6.35 - h) ** 2))
                    
                    # Natural Open-Floor Prominence: Wide jaw opening centers /a/ as primary acoustic mass
                    if h <= 1.4:
                        f_b += -0.30 * (b - 1.0)

                    # Stabilizes Close-Mid (/e, o/ at h=4.0) and Open-Mid (/ɛ, ɔ/ at h=2.0)
                    if d_v_min < 0.42:
                        tier_spacing = 2.0
                        nearest_tier = round(h / tier_spacing) * tier_spacing
                        quantal_strength = max(0.0, 1.0 - (d_v_min / 0.42))
                        
                        quantal_attractor = -0.42 * (h - nearest_tier) * quantal_strength
                        f_h += quantal_attractor

                        # Open-Mid Tier Symmetry:
                        # Polarizes front open-mid to /ɛ/ (b -> 0.0) and back open-mid to /ɔ/ (b -> 2.0),
                        # preventing open-mid vowels from stagnating in the ambiguous central /ɜ/!
                        if 1.5 <= h <= 2.5:
                            if b < 1.0:
                                f_b += -0.22 * b * quantal_strength         # Pulls to front /ɛ/
                            else:
                                f_b += 0.22 * (2.0 - b) * quantal_strength  # Pulls to back /ɔ/

                    total_c = sum(1 for seg in p_list if seg and seg.kind == PhonemeKind.CONSONANT)
                    total_v = sum(1 for seg in p_list if seg and seg.kind == PhonemeKind.VOWEL)
                    cv_ratio = total_c / max(1.0, float(total_v))

                    if cv_ratio > 2.4 and d_v_min >= 0.40:
                        f_b += -0.35 * (b - 1.0)

                    raw_h = h + f_h * 0.80 + random.gauss(0.0, 0.08)
                    clamped_h = max(0.0, min(6.0, raw_h))

                    h_ratio = clamped_h / 6.0
                    min_b = max(0.0, 0.85 * (1.0 - h_ratio))
                    max_b = 2.0

                    raw_b = b + f_b * 0.60 + random.gauss(0.0, 0.05)
                    clamped_b = max(min_b, min(max_b, raw_b))

                    # Quantal Lip-Rounding Coupling:
                    # Open Floor (h <= 1.4) decouples from lip rounding (prevents /ä/ from fragmenting into /ɒ/!)
                    if clamped_h <= 1.4:
                        raw_r = 0.0  # Open floor unifies cleanly to unrounded /a/ (/ä/)!
                    elif clamped_b >= 1.35:
                        raw_r = r + f_r + 0.35  # Mid/Close back vowels round (/o, u, ɔ/)
                    elif clamped_b <= 0.6 and f_r > 0.2:
                        raw_r = r + f_r        # Retains front-rounded umlaut (/y, ø/)
                    else:
                        raw_r = r + f_r - 0.35  # Front/Central unrounded default (/i, e, ɨ/)

                    clamped_r = max(0.0, min(1.0, raw_r))

                    p_list[i] = p.drift(point=(clamped_h, clamped_b, clamped_r))
                    changed = True
                    continue

            if changed:
                valid_phonemes = [p_item for p_item in p_list if p_item is not None]

                if word.category == LexicalCategory.CONTENT_OPEN and len(valid_phonemes) < 2 and word.phonemes:
                    valid_phonemes = list(word.phonemes[:2]) if len(word.phonemes) >= 2 else [word.phonemes[0], word.phonemes[0]]
                elif not valid_phonemes and word.phonemes:
                    valid_phonemes = [word.phonemes[0]]

                word.phonemes = valid_phonemes
                
                if len(word.morphemes) <= 1:
                    if not word.morphemes:
                        word.morphemes = [Morpheme(list(valid_phonemes), root_family_id=word.derivation.root_family_id)]
                    else:
                        word.morphemes[0].phonemes = list(valid_phonemes)
                else:
                    total_p = len(valid_phonemes)
                    n_m = len(word.morphemes)
                    split_idx = max(1, total_p // n_m)
                    word.morphemes[0].phonemes = list(valid_phonemes[:split_idx])
                    word.morphemes[1].phonemes = list(valid_phonemes[split_idx:])

                prosodified = SyllableEngine.apply_prosody(
                    word,
                    pattern=self.profile.stress_pattern,
                    tone_tier=self.profile.tone_tier,
                )
                word.phonemes = list(prosodified.phonemes)
                word.syllables = prosodified.syllables
                word.morphemes = prosodified.morphemes

                for p_item in word.phonemes:
                    if p_item.ipa not in self.phonemes:
                        self.add_phoneme(p_item)

    def evolve(
        self,
        reduction_strength: float = 0.80,
        generational_drift: bool = True,
        drift_rate: float = 0.35,
        semantic_drift_rate: float = 0.20,
        enable_neologisms: bool = True,
        current_generation: int = 0,
    ) -> None:
        self._cached_grammar_paradigm = None

        if self.profile.tone_tier > 0 and current_generation % 50 == 0:
            pop = self.population
            if pop > 0.75 and self.profile.tone_tier > 2 and (random.random() > self.cultural_inertia * 0.60):
                self.profile = self.profile._replace(
                    tone_tier=max(2, self.profile.tone_tier - 1)
                )

        # -------------------------------------------------------------
        # Continuous Ornstein-Uhlenbeck Cultural Evolution
        # Continuous Environmental Gravity + Inertia-Resisted Stochastic Drift + Finite Attention Budget
        # -------------------------------------------------------------
        pop = self.population
        alt = getattr(self.environment, "altitude", 0.0) if self.environment else 0.0
        hum = getattr(self.environment, "humidity", 0.5) if self.environment else 0.5
        temp = getattr(self.environment, "temperature", 0.5) if self.environment else 0.5
        noise = getattr(self.environment, "ambient_noise", 0.2) if self.environment else 0.2
        veg = self.environment.emergent_vegetation(population=pop, cultural_attention=self.cultural_attention) if hasattr(self.environment, "emergent_vegetation") else 0.50
        aridity = 1.0 - hum

        # Subsistence & Ecological Couplers
        agri_dist = ((hum - 0.55) / 0.22) ** 2 + ((veg - 0.45) / 0.22) ** 2 + ((temp - 0.55) / 0.22) ** 2
        arable_agri = math.exp(-agri_dist) * max(0.0, 1.0 - alt * 1.5)

        pastoral_dist = ((veg - 0.35) / 0.18) ** 2 + ((hum - 0.40) / 0.20) ** 2
        pastoral_herding = math.exp(-pastoral_dist) * (1.0 - pop * 0.70)

        steppe_mobility = (1.0 - veg) * (1.0 - pop * 0.80)
        imperial_potency = pop * (1.0 - veg * 0.50)

        env_attractors = [
            0.45 + 0.70 * alt + 0.85 * arable_agri + 0.45 * pop * (1.0 - veg),      # Concreteness Target
            0.45 + 0.80 * (1.0 - pop) * veg + 0.95 * pastoral_herding,              # Animacy Target
            0.45 + 0.90 * aridity + 0.45 * pop * (1.0 - arable_agri * 0.4),         # Valence Target
            0.45 + 0.75 * alt + 0.95 * imperial_potency + 0.30 * noise,              # Potency Target
            0.45 + 0.75 * hum * (0.50 + noise) + 0.95 * steppe_mobility,            # Dynamism Target
            0.45 + 1.05 * pop + 0.20 * temp,                                        # Sociality Target
            0.45 + 0.90 * (1.0 - veg) + 0.65 * imperial_potency,                    # Extension Target
        ]

        total_env_attractor = sum(env_attractors) or 7.0
        normalized_attractors = [7.0 * (w / total_env_attractor) for w in env_attractors]

        inertia = self.cultural_inertia
        k_env_gravity = 0.035 / (1.0 + 0.60 * inertia)
        drift_damping = 1.0 + 0.80 * inertia

        updated_attention = []
        for i in range(7):
            curr_w = self.cultural_attention[i]
            target_w = normalized_attractors[i]
            
            # Continuous Environmental Gravitational Pull
            env_pull = k_env_gravity * (target_w - curr_w)
            
            # Internal Stochastic Cultural Drift
            internal_drift = random.gauss(0.0, 0.025) / drift_damping
            
            updated_attention.append(max(0.30, curr_w + env_pull + internal_drift))

        # Enforce Finite Cognitive Attention Budget (Sum = 7.0)
        total_budget = sum(updated_attention) or 7.0
        self.cultural_attention = [
            round(max(0.30, min(2.60, 7.0 * (w / total_budget))), 3)
            for w in updated_attention
        ]

        avail_vowels = self.vowels or [Phoneme(PhonemeKind.VOWEL, (0.0, 1.0, 0.0))]
        avail_consonants = self.consonants or list(self.phonemes.values())

        if enable_neologisms:
            LexiconGenerator.expand_vocabulary(
                lang=self,
                growth_rate=0.04,
                d_sem_min=0.06,
                current_generation=current_generation,
                profile=self.profile,
            )

        self.calculate_lexical_usage()

        active_operator_ids: set[int] = set()
        if current_generation % 10 == 0 or current_generation == 1:
            from grammar import GrammarEngine
            paradigm = GrammarEngine.discover_grammar_system(self)
            active_operator_ids = {
                op.id for op in (
                    paradigm.temporal_operators[:3] + 
                    paradigm.spatial_operators[:3] + 
                    paradigm.modal_operators[:2] + 
                    list(paradigm.pronouns.values()) +
                    list(paradigm.articles.values())
                ) if op
            }

        emergent_auxiliaries = set(self.identify_emergent_auxiliaries())
        new_words: dict[int, Word] = {}
        effective_reduction = reduction_strength / (1.0 + self.cultural_inertia * 0.40)

        for word in list(self.words.values()):
            if word.is_obsolete(max_idle_epochs=4, cultural_inertia=self.cultural_inertia):
                continue

            current_word = word
            stem_family = current_word.derivation.root_family_id
            current_vector = current_word.vector
            is_mutated = False

            is_functional_operator = (
                current_word.id in active_operator_ids
                or current_word.category == LexicalCategory.FUNCTIONAL_CLOSED
                or stem_family in emergent_auxiliaries
            )

            # 1. Morpheme & Word-Length Erosion
            updated_morphemes: list[Morpheme] = []
            total_word_len = len(current_word.phonemes)
            s_comp = getattr(self.profile, "syllable_complexity", 0.50)
            
            p_length_erode = 1.0 - math.exp(
                -0.18 * max(0, total_word_len - 5) * math.log(current_word.usage_frequency + 1.0)
                / (1.0 + self.cultural_inertia * 0.50)
            )

            is_long_word = (total_word_len >= 6 and random.random() < p_length_erode)

            for morph in current_word.morphemes:
                if is_functional_operator:
                    morph.is_grammatical = True
                    if morph.stage == ClineStage.FREE_ROOT:
                        morph.stage = ClineStage.CLITIC
                        current_vector = current_vector.generalize_for_cline(ClineStage.CLITIC)
                        is_mutated = True

                if morph.is_grammatical or is_long_word:
                    old_len = len(morph.phonemes)
                    eroded_m = ArticulatoryEnergyModel.erode_morpheme(
                        morph, 
                        reduction_rate=effective_reduction,
                        usage_frequency=current_word.usage_frequency,
                        is_long_word=is_long_word,
                        syllable_complexity=s_comp,
                    )
                    if len(eroded_m.phonemes) != old_len:
                        is_mutated = True
                    updated_morphemes.append(eroded_m)
                else:
                    updated_morphemes.append(morph)

            cleaned_morphemes = updated_morphemes

            if is_functional_operator:
                current_word.category = LexicalCategory.FUNCTIONAL_CLOSED

            # 2. Continuous Semantic Drift
            word_inertia = math.log(current_word.usage_frequency + 1.0) * 0.45 + (current_vector.valence * self.cultural_attention[2] + current_vector.sociality * self.cultural_attention[5]) * 0.25
            effective_sem_rate = (semantic_drift_rate * current_word.semantic_drift_multiplier) / (1.0 + word_inertia)

            if random.random() < effective_sem_rate:
                s_delta = [random.uniform(-0.030, 0.030) * (self.cultural_attention[d] / 1.0) for d in range(7)]
                current_vector = current_vector.drift(s_delta)
                current_word.vector = current_vector

            # 3. Continuous Metaphor Projection
            if current_word.usage_frequency >= 2.20 and random.random() < 0.05:
                c, a, v, p, d, s, e = current_vector.coords
                eta = random.uniform(0.25, 0.35)
                continuous_delta = [
                    -eta * c,
                    -eta * a,
                    eta * (v - 0.50) * self.cultural_attention[2],
                    eta * (p - 0.50) * self.cultural_attention[3],
                    eta * (d - 0.50) * self.cultural_attention[4],
                    eta * (s - 0.50) * self.cultural_attention[5],
                    eta * (e - 0.50) * self.cultural_attention[6],
                ]
                extended_sense = current_vector.drift(continuous_delta)
                current_word.add_sense(extended_sense)

            if not is_mutated:
                new_words[current_word.id] = current_word
                continue

            fused_phonemes: list[Phoneme] = []
            for m in cleaned_morphemes:
                fused_phonemes.extend(m.phonemes)

            phonemes = fused_phonemes if fused_phonemes else list(current_word.phonemes)
            repaired = ArticulatoryEnergyModel.repair_phonemes(phonemes, self.profile)
            if not repaired and current_word.phonemes:
                repaired = [current_word.phonemes[0]]

            temp_word = Word(
                word_id=current_word.id,
                vector=current_vector,
                phonemes=repaired,
                morphemes=cleaned_morphemes,
                derivation=current_word.derivation,
                usage_frequency=current_word.usage_frequency,
                category=current_word.category,
                generation_born=current_generation,
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
                tone_tier=self.profile.tone_tier,
            )

            new_words[current_word.id] = final_word

        # Step 1: Regular Sound Laws
        if generational_drift:
            self.apply_regular_sound_laws(new_words, drift_rate)

        # Step 2: Principle of Contrast (Homophone Disambiguation)
        form_clusters: dict[str, list[Word]] = defaultdict(list)
        for w in new_words.values():
            form_clusters[w.plain_form].append(w)

        words_to_remove: set[int] = set()
        occupied_forms: set[str] = set(form_clusters.keys())

        for form, cluster in form_clusters.items():
            if len(cluster) >= 2:
                cluster.sort(key=lambda item: item.usage_frequency, reverse=True)
                dominant = cluster[0]
                
                for idx_sub, subordinate in enumerate(cluster[1:], start=1):
                    sem_dist = dominant.vector.weighted_distance_to(subordinate.vector, self.cultural_attention)
                    
                    if sem_dist < 0.25:
                        dominant.record_usage(subordinate.usage_frequency * 0.60)
                        if sem_dist > 0.08:
                            dominant.add_sense(subordinate.vector)
                        words_to_remove.add(subordinate.id)

                    elif subordinate.phonemes:
                        p_orig = list(subordinate.phonemes)

                        for attempt in range(1, 8):
                            p_cand = list(p_orig)
                            if attempt % 2 == 1 and avail_consonants:
                                for i_p, p_item in enumerate(p_cand):
                                    if p_item.kind == PhonemeKind.CONSONANT:
                                        p_cand[i_p] = avail_consonants[(i_p + attempt * 3 + idx_sub) % len(avail_consonants)].drift()
                                        break
                            elif avail_vowels:
                                for i_p, p_item in enumerate(p_cand):
                                    if p_item.kind == PhonemeKind.VOWEL:
                                        p_cand[i_p] = avail_vowels[(i_p + attempt * 2 + idx_sub) % len(avail_vowels)].drift()
                                        break

                            cleaned_cand = ArticulatoryEnergyModel.repair_phonemes(p_cand, self.profile)
                            temp_w = Word(vector=subordinate.vector, phonemes=cleaned_cand, is_ephemeral=True)
                            pros_w = SyllableEngine.apply_prosody(temp_w, pattern=self.profile.stress_pattern, tone_tier=self.profile.tone_tier)
                            cand_form = pros_w.plain_form

                            if cand_form not in occupied_forms:
                                subordinate.phonemes = list(pros_w.phonemes)
                                subordinate.syllables = pros_w.syllables
                                
                                # Resynchronize morphemes with differentiated candidate
                                if len(subordinate.morphemes) <= 1:
                                    if not subordinate.morphemes:
                                        subordinate.morphemes = [Morpheme(list(pros_w.phonemes), root_family_id=subordinate.derivation.root_family_id)]
                                    else:
                                        subordinate.morphemes[0].phonemes = list(pros_w.phonemes)
                                else:
                                    split_idx = max(1, len(pros_w.phonemes) // len(subordinate.morphemes))
                                    subordinate.morphemes[0].phonemes = list(pros_w.phonemes[:split_idx])
                                    subordinate.morphemes[1].phonemes = list(pros_w.phonemes[split_idx:])

                                occupied_forms.add(cand_form)
                                break

        for w_id in words_to_remove:
            if w_id in new_words:
                del new_words[w_id]

        self.words = new_words

        # Step 3: Rebuilt on every turn
        self.rebuild_inventory()

    def rebuild_inventory(self) -> None:
        """
        Usage-Based Spatial Inventory Extraction (Stevens 1989 / de Boer 2000 / Trudgill 2011).
        - Clusters acoustic vowel space continuously using environmental d_min_vowel.
        - Selects cluster representatives strictly by TOKEN FREQUENCY in the living language.
        - Scales extraction threshold continuously by Population (Linguistic Niche Hypothesis).
        """
        if not self.words:
            self.phonemes = {}
            return

        c_counts = Counter(p.ipa for w in self.words.values() for p in w.phonemes if p.kind == PhonemeKind.CONSONANT)
        v_counts = Counter(p.ipa for w in self.words.values() for p in w.phonemes if p.kind == PhonemeKind.VOWEL)

        total_c = sum(c_counts.values()) or 1.0
        total_v = sum(v_counts.values()) or 1.0

        unique_phonemes: dict[str, Phoneme] = {p.ipa: p for w in self.words.values() for p in w.phonemes}
        core_inventory: dict[str, Phoneme] = {}

        pop = self.population
        d_v_min = self.profile.d_min_vowel
        d_c_min = self.profile.d_min_consonant

        # Demographic Inventory Extraction Thresholds (Lupyan & Dale 2010)
        consonant_threshold = d_c_min * 0.025 * (1.0 + 1.50 * (pop ** 1.3))
        vowel_threshold = 0.025 * (1.0 + 1.20 * (pop ** 1.3))

        # =====================================================================
        # 1. CONTINUOUS VOWEL SPATIAL CLUSTERING (Frequency-First Selection)
        # =====================================================================
        active_vowels = [p for p in unique_phonemes.values() if p.kind == PhonemeKind.VOWEL]

        if active_vowels:
            sorted_v = sorted(active_vowels, key=lambda v: v_counts.get(v.ipa, 0), reverse=True)
            vowel_clusters: list[list[Phoneme]] = []

            for v in sorted_v:
                matched_cluster = False
                for cluster in vowel_clusters:
                    rep_v = cluster[0]
                    effective_d_min = d_v_min * (1.20 if (v.point[0] <= 1.4 and rep_v.point[0] <= 1.4) else 1.0)
                    if self.acoustic_vowel_distance(v, rep_v) < effective_d_min:
                        cluster.append(v)
                        matched_cluster = True
                        break
                if not matched_cluster:
                    vowel_clusters.append([v])

            for cluster in vowel_clusters:
                total_cluster_tokens = sum(v_counts.get(v.ipa, 0) for v in cluster)
                cluster_share = total_cluster_tokens / total_v

                if cluster_share >= vowel_threshold:
                    dominant_v = max(cluster, key=lambda v: (v_counts.get(v.ipa, 0), v.weight))
                    core_inventory[dominant_v.ipa] = dominant_v

        # =====================================================================
        # 2. CONTINUOUS CONSONANT INTRA-MANNER CLUSTERING
        # =====================================================================
        active_consonants = [p for p in unique_phonemes.values() if p.kind == PhonemeKind.CONSONANT]

        if active_consonants:
            sorted_c = sorted(active_consonants, key=lambda c: c_counts.get(c.ipa, 0), reverse=True)
            consonant_clusters: list[list[Phoneme]] = []

            for c in sorted_c:
                matched_c_cluster = False
                for c_cluster in consonant_clusters:
                    rep_c = c_cluster[0]
                    if int(round(c.point[1])) == int(round(rep_c.point[1])):
                        if self.acoustic_consonant_distance(c, rep_c) < (d_c_min * 0.90):
                            c_cluster.append(c)
                            matched_c_cluster = True
                            break
                if not matched_c_cluster:
                    consonant_clusters.append([c])

            for c_cluster in consonant_clusters:
                total_c_tokens = sum(c_counts.get(c.ipa, 0) for c in c_cluster)
                share = total_c_tokens / total_c
                if share >= consonant_threshold:
                    dominant_c = max(c_cluster, key=lambda seg: c_counts.get(seg.ipa, 0) * seg.weight)
                    core_inventory[dominant_c.ipa] = dominant_c

        if not core_inventory:
            core_inventory = unique_phonemes

        self.phonemes = core_inventory

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
        ) if env else None

        daughter = Language(name=daughter_name, profile=profile, environment=env, population=self.population)
        daughter.phonemes = {k: p.drift() for k, p in self.phonemes.items()}
        for word in self.words.values():
            daughter.add_word(word.copy())
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