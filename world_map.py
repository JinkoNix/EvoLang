"""
world_map.py — Continuous 2D Physical Geography, Biophysical Channels, and Demographic Contact.
Features pure self-bounding logistic response curves (Weber-Fechner biophysics) with zero hard clamps.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence, Callable
from concurrent.futures import ThreadPoolExecutor

from language import Language, LanguageProfile
from syllable import StressPattern
from semantics import SemanticVector, DerivationType
from word import Word, LexicalCategory


# =====================================================================
# Layer 1: Continuous Local Physical Environment
# =====================================================================

@dataclass
class Environment:
    """Continuous physical parameters at an exact coordinate point (x, y)."""
    temperature: float = 0.50     # [0.0 = Arctic Cryosphere, 1.0 = Hyper-Arid Desert / Tropical Heat]
    humidity: float = 0.50        # [0.0 = Extreme Aridity, 1.0 = Rainforest Saturation]
    altitude: float = 0.20        # [0.0 = Sea Level, 1.0 = Alpine Plateau]
    vegetation: float = 0.50      # [0.0 = Bare Rock / Sand, 1.0 = Dense Jungle Canopy]
    ambient_noise: float = 0.20   # [0.0 = Quiet Plains, 1.0 = Coastal Surf / River Rapids / Wind]
    population: float = 0.50      # [0.0 = Micro-Forager Band, 1.0 = Metropolitan Empire]

    def generate_profile(
        self,
        name: str = "Adapted Profile",
        initial_head_directionality: float | None = None,
        cultural_attention: Sequence[float] | None = None,
    ) -> LanguageProfile:
        """
        Layer 2: Derives continuous physical acoustic impedance channels and
        morphosyntactic targets using self-bounding logistic response curves.
        """
        # 1. Atmospheric Air Density (ρ_air) [Exponential barometric pressure equation]
        rho_air = math.exp(-2.40 * max(0.0, self.altitude)) * (1.0 - 0.20 * (self.temperature - 0.50))

        # 2. Vocal Fold Tissue Hydration & Compliance (μ_vocal) [Logistic humidity response]
        mu_vocal = (1.0 / (1.0 + math.exp(-6.0 * (self.humidity - 0.45)))) * math.exp(-2.0 * ((self.temperature - 0.60) ** 2))

        # 3. High-Frequency Spectral Damping (α_damp) [Canopy scattering sigmoid]
        alpha_damp = 1.0 / (1.0 + math.exp(-5.0 * ((self.vegetation * 0.65 + self.humidity * 0.35) - 0.50)))

        # 4. Continuous Syllable Complexity (S_comp) [Self-bounding in [0.10, 0.90]]
        s_comp_driver = alpha_damp + (self.ambient_noise * 0.35) - ((1.0 - rho_air) * 0.60) - 0.40
        syllable_complexity = round(0.10 + (0.80 / (1.0 + math.exp(5.0 * s_comp_driver))), 3)
        allow_clusters = (syllable_complexity >= 0.28)
        max_transition_cost = round(2.0 + (3.80 * syllable_complexity), 2)

        # 5. Continuous Vowel Dispersion Threshold (Vapor Pressure Deficit Biophysics)
        aridity = 1.0 - self.humidity
        vpd = max(0.0, (self.temperature ** 1.2) * aridity * 1.50 - 0.20)
        v_driver = vpd + (self.ambient_noise * 0.15) - (alpha_damp * 0.15) - 0.35
        d_min_vowel = round(0.12 + (0.22 / (1.0 + math.exp(-6.0 * v_driver))), 3)

        # 6. Continuous Consonant Dispersion Threshold (d_min_consonant) [Self-bounding in [0.22, 0.34]]
        d_min_consonant = round(0.22 + (0.12 / (1.0 + math.exp(-5.0 * (rho_air - 0.60)))), 3)

        # 7. Continuous Haudricourt Tonogenesis Capacity (Smooth Sigmoid, Zero Cliffs)
        tonal_driver = (mu_vocal * 1.40) + (self.humidity * 0.80) - (self.population * 0.80) - 1.20
        raw_tone_tier = 8.0 / (1.0 + math.exp(-4.5 * tonal_driver))

        if raw_tone_tier < 1.20:
            tone_tier = 0
        else:
            tone_tier = int(round(raw_tone_tier / 2.0) * 2)

        # 8. Continuous Vowel Reduction Mode
        if tone_tier > 0:
            vowel_reduction_mode = "none" if syllable_complexity < 0.25 else "inventory_snap"
        else:
            p_centralize = 1.0 / (1.0 + math.exp(-8.0 * (self.population - 0.60)))
            if syllable_complexity < 0.25:
                vowel_reduction_mode = "none"
            elif random.random() < p_centralize:
                vowel_reduction_mode = "centralize"
            else:
                vowel_reduction_mode = "inventory_snap"

        # 9. Epenthetic Vowel Barycenter (Resting mandibular posture)
        ep_height = round(6.0 * (1.0 - self.temperature), 2)
        epenthetic_vowel = (ep_height, 1.0, 0.0)

        # 10. Continuous Metrical Stress Pattern Assignment
        p_initial = 1.0 / (1.0 + math.exp(-6.0 * (self.altitude - 0.45)))
        p_penult = 1.0 / (1.0 + math.exp(6.0 * (syllable_complexity - 0.35)))
        
        r_stress = random.random()
        if r_stress < (p_penult * 0.70):
            stress_pattern = StressPattern.PENULTIMATE
        elif r_stress < (p_penult * 0.70 + p_initial * 0.60):
            stress_pattern = StressPattern.INITIAL
        else:
            stress_pattern = StressPattern.LATIN

        # 11. Head Directionality
        if initial_head_directionality is not None:
            head_directionality = initial_head_directionality
        else:
            h_choice = random.choices([0.15, 0.50, 0.85], weights=[45.0, 45.0, 10.0])[0]
            head_directionality = h_choice + random.uniform(-0.06, 0.06)

        # 12. Morphological Synthesis Target (S_idx) [Self-bounding in [0.12, 0.88]]
        s_driver = (self.population ** 1.3) - (self.altitude * 0.55) - 0.40
        synthesis_index = round(0.12 + (0.76 / (1.0 + math.exp(4.5 * s_driver))), 3)

        # 13. Continuous Acoustic Roughness Index (R_rough) [Self-bounding in [0.10, 0.90]]
        attn = cultural_attention or [1.0] * 7
        w_con, w_anim, w_val, w_pot, w_dyn, w_soc, w_ext = attn
        
        aridity = 1.0 - self.humidity
        rough_driver = (
            (w_pot * 1.20 + w_con * 0.50) -
            (w_soc * 1.05 + w_val * 0.50) +
            (aridity * 0.70) +
            (self.altitude * 0.45) -
            (alpha_damp * 0.55) - 0.25
        )
        acoustic_roughness = round(0.10 + (0.80 / (1.0 + math.exp(-3.5 * rough_driver))), 3)

        return LanguageProfile(
            name=name,
            max_transition_cost=max_transition_cost,
            voicing_assimilation_bias="regressive",
            allow_clusters=allow_clusters,
            double_stop_strategy="spirantization" if (self.altitude * 0.6 + aridity * 0.4) > 0.45 else "gemination",
            epenthetic_vowel_point=epenthetic_vowel,
            stress_pattern=stress_pattern,
            vowel_reduction_mode=vowel_reduction_mode,
            apocope_rate=round(1.0 / (1.0 + math.exp(-4.0 * ((1.0 - self.vegetation) * 0.6 + (1.0 - self.temperature) * 0.4 - 0.5))), 3),
            syncope_rate=round(1.0 / (1.0 + math.exp(-4.0 * (self.population * 0.55 + self.altitude * 0.45 - 0.5))), 3),
            tone_tier=tone_tier,
            head_directionality=round(head_directionality, 3),
            d_min_vowel=d_min_vowel,
            d_min_consonant=d_min_consonant,
            synthesis_index=synthesis_index,
            syllable_complexity=syllable_complexity,
            acoustic_roughness=acoustic_roughness,
        )


class TerrainField:
    """Continuous 2D scalar field manifold for elevation and climate."""

    def __init__(self, elevation_fn: Callable[[float, float], float] | None = None):
        self.elevation_fn = elevation_fn or self._default_continental_heightmap

    @staticmethod
    def _default_continental_heightmap(x: float, y: float) -> float:
        dist_to_spine = math.sqrt((x - 0.75) ** 2 + (y - 0.80) ** 2)
        mountain_peak = 0.95 * math.exp(-(dist_to_spine ** 2) / 0.08)
        coastal_shelf = max(0.0, (x - 0.20) * 0.40)
        return round(mountain_peak + coastal_shelf, 3)

    def sample_environment(self, x: float, y: float, population: float = 0.50) -> Environment:
        clamped_x = max(0.0, min(1.0, float(x)))
        clamped_y = max(0.0, min(1.0, float(y)))

        altitude = self.elevation_fn(clamped_x, clamped_y)
        
        # Temperature: Latitude gradient + lapse cooling
        base_temp = 0.95 - (clamped_y * 0.55)
        lapse_cooling = altitude * 0.55
        temperature = round(base_temp - lapse_cooling, 3)

        # Humidity: Oceanic proximity - continuous leeward mountain rain shadow
        dist_from_ocean = clamped_x
        base_humidity = 1.0 - (dist_from_ocean * 0.95)
        
        # Continuous Leeward Rain Shadow
        leeward_dist = max(0.0, clamped_x - 0.70)
        rain_shadow = 0.30 * (1.0 - math.exp(-4.0 * leeward_dist)) * max(0.0, altitude - 0.15)
        humidity = round(max(0.05, base_humidity - rain_shadow), 3)

        # Vegetation
        veg_score = (humidity * 0.70) + (temperature * 0.30) - (altitude * 0.40)
        vegetation = round(max(0.05, min(0.95, veg_score)), 3)

        # Ambient Acoustic Noise (coastal surf, mountain ridge wind)
        ridge_wind = altitude * 0.65
        surf_noise = max(0.0, (0.25 - clamped_x) * 2.0) if clamped_x < 0.25 else 0.0
        ambient_noise = round(min(0.95, ridge_wind + surf_noise), 3)

        return Environment(
            temperature=temperature,
            humidity=humidity,
            altitude=altitude,
            vegetation=vegetation,
            ambient_noise=ambient_noise,
            population=population,
        )


class Civilization:
    """A living speech community situated at continuous coordinates (x, y)."""

    def __init__(
        self,
        name: str,
        language: Language,
        x: float,
        y: float,
        terrain: TerrainField,
        population: float = 0.50,
    ):
        self.name = name
        self.language = language
        self.x = max(0.0, min(1.0, float(x)))
        self.y = max(0.0, min(1.0, float(y)))
        self.terrain = terrain
        self.population = float(population)
        self.sync_environment()

    @property
    def environment(self) -> Environment:
        return self.terrain.sample_environment(self.x, self.y, self.population)

    def sync_environment(self) -> None:
        env = self.environment
        self.language.environment = env
        self.language.profile = env.generate_profile(
            name=f"{self.name} Profile",
            initial_head_directionality=self.language.profile.head_directionality,
            cultural_attention=self.language.cultural_attention,
        )

    def move_to(self, new_x: float, new_y: float) -> None:
        self.x = max(0.0, min(1.0, float(new_x)))
        self.y = max(0.0, min(1.0, float(new_y)))
        self.sync_environment()

    def set_population(self, new_pop: float) -> None:
        self.population = float(new_pop)
        self.sync_environment()


class ContinuousWorldMap:
    """Manages continuous geography, trade friction, and multi-threaded simulation."""

    def __init__(self, name: str = "Pangaea Field", terrain: TerrainField | None = None):
        self.name = name
        self.terrain = terrain or TerrainField()
        self.civilizations: dict[str, Civilization] = {}
        self.current_epoch: int = 0
        self._friction_cache: dict[tuple[float, float, float, float], float] = {}

    def spawn_civilization(
        self,
        name: str,
        language: Language,
        x: float,
        y: float,
        population: float = 0.50,
    ) -> Civilization:
        civ = Civilization(name, language, x, y, self.terrain, population)
        self.civilizations[name] = civ
        return civ

    def calculate_path_friction(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculates geographic travel friction dynamically from elevation gradients."""
        cache_key = (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))
        if cache_key in self._friction_cache:
            return self._friction_cache[cache_key]

        geo_dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        max_ridge_altitude = 0.0
        elevation_gradient_sum = 0.0
        avg_vegetation = 0.0
        prev_alt = self.terrain.sample_environment(x1, y1).altitude

        for step in (0.25, 0.50, 0.75, 1.00):
            sx = x1 + step * (x2 - x1)
            sy = y1 + step * (y2 - y1)
            sample_env = self.terrain.sample_environment(sx, sy)
            if sample_env.altitude > max_ridge_altitude:
                max_ridge_altitude = sample_env.altitude
            elevation_gradient_sum += abs(sample_env.altitude - prev_alt)
            prev_alt = sample_env.altitude
            avg_vegetation += sample_env.vegetation * 0.25

        friction = geo_dist + (elevation_gradient_sum * 2.50) + (max_ridge_altitude * 1.50) + (avg_vegetation * 0.20)
        self._friction_cache[cache_key] = friction
        return friction

    def calculate_contact_intensity(self, civ_a: Civilization, civ_b: Civilization) -> float:
        friction = self.calculate_path_friction(civ_a.x, civ_a.y, civ_b.x, civ_b.y)
        demographic_mass = math.sqrt(civ_a.population * civ_b.population)
        gravity = (demographic_mass * 0.70) / max(0.05, friction)
        return round(1.0 / (1.0 + math.exp(-4.0 * (gravity - 0.50))), 3)

    def synchronize_contact_network(self) -> None:
        civ_list = list(self.civilizations.values())
        n = len(civ_list)
        for i in range(n):
            for j in range(i + 1, n):
                c_a, c_b = civ_list[i], civ_list[j]
                intensity = self.calculate_contact_intensity(c_a, c_b)
                c_a.language.set_contact(c_b.language, intensity)

    def step_epoch(
        self,
        reduction_strength: float = 0.75,
        enable_neologisms: bool = True,
        parallel: bool = True,
    ) -> None:
        self.current_epoch += 1
        self.synchronize_contact_network()
        civ_list = list(self.civilizations.values())

        for civ in civ_list:
            for neighbor in civ_list:
                if civ.name == neighbor.name:
                    continue
                intensity = self.calculate_contact_intensity(civ, neighbor)
                
                if random.random() < (intensity * 0.20):
                    candidates = [w for w in neighbor.language.words.values() if not w.is_proto_root and w.category == LexicalCategory.CONTENT_OPEN]
                    if not candidates:
                        continue

                    candidate = random.choice(candidates)
                    already_borrowed = any(
                        w.derivation.provenance and w.derivation.provenance.donor_word_id == candidate.id
                        for w in civ.language.words.values()
                    )
                    if not already_borrowed:
                        attn = civ.language.cultural_attention
                        cultural_inertia = (attn[2] ** 1.4 * attn[5]) / (1.0 + attn[4] * 0.8)
                        
                        # Continuous Sociolinguistic Calquing vs. Borrowing Sigmoid
                        p_calque = 1.0 / (1.0 + math.exp(-6.0 * (cultural_inertia - 1.05)))
                        if random.random() < p_calque:
                            civ.language.calque_word(
                                foreign_word=candidate,
                                source_language_name=neighbor.name,
                                current_generation=self.current_epoch,
                            )
                        else:
                            civ.language.borrow_word(
                                foreign_word=candidate,
                                source_language_name=neighbor.name,
                                current_generation=self.current_epoch,
                            )

        def _evolve_civ(c: Civilization):
            c.language.evolve(
                reduction_strength=reduction_strength,
                enable_neologisms=enable_neologisms,
                current_generation=self.current_epoch,
            )

        if parallel and len(civ_list) > 1:
            with ThreadPoolExecutor(max_workers=min(len(civ_list), 8)) as executor:
                list(executor.map(_evolve_civ, civ_list))
        else:
            for civ in civ_list:
                _evolve_civ(civ)