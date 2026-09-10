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
        morphosyntactic targets. Fully supports extreme worlds (T=2.00, Pop=5.00, Alt=3.00).
        """
        # 1. Atmospheric Air Density (Ideal Gas Law Exponential: strictly real & positive for all T, Alt)
        rho_air = math.exp(-2.40 * max(0.0, self.altitude)) * math.exp(-0.25 * (self.temperature - 0.50))

        # 2. Vocal Fold Compliance (T in [-inf, +inf], H in [-inf, +inf])
        mu_vocal = (1.0 / (1.0 + math.exp(-6.0 * (self.humidity - 0.45)))) * math.exp(-2.0 * ((self.temperature - 0.60) ** 2))

        # 3. High-Frequency Spectral Damping
        alpha_damp = 1.0 / (1.0 + math.exp(-5.0 * ((self.vegetation * 0.65 + self.humidity * 0.35) - 0.50)))

        # 4. Continuous Syllable Complexity (Self-bounding in [0.10, 0.90] for any inputs)
        s_comp_driver = alpha_damp + (max(0.0, self.ambient_noise) * 0.35) - ((1.0 - rho_air) * 0.60) - 0.40
        syllable_complexity = round(0.10 + (0.80 / (1.0 + math.exp(5.0 * s_comp_driver))), 3)
        allow_clusters = (syllable_complexity >= 0.28)
        max_transition_cost = round(2.0 + (3.80 * syllable_complexity), 2)

        # 5. Continuous Vowel Dispersion (Vapor Pressure Deficit supports T > 1.0 without crashing on T < 0)
        aridity = max(0.0, 1.0 - self.humidity)
        vpd = (max(0.0, self.temperature) ** 1.2) * aridity * 1.50
        v_driver = vpd + (max(0.0, self.ambient_noise) * 0.15) - (alpha_damp * 0.15) - 0.35
        d_min_vowel = round(0.12 + (0.22 / (1.0 + math.exp(-6.0 * v_driver))), 3)

        # 6. Continuous Consonant Dispersion Threshold
        d_min_consonant = round(0.22 + (0.12 / (1.0 + math.exp(-5.0 * (rho_air - 0.60)))), 3)

        # 7. Continuous Haudricourt Tonogenesis Capacity
        tonal_driver = (mu_vocal * 1.40) + (self.humidity * 0.80) - (max(0.0, self.population) * 0.80) - 1.20
        raw_tone_tier = 8.0 / (1.0 + math.exp(-4.5 * tonal_driver))

        if raw_tone_tier < 1.20:
            tone_tier = 0
        else:
            tone_tier = int(round(raw_tone_tier / 2.0) * 2)

        # 8. Vowel Reduction Mode
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

        # 9. Epenthetic Vowel Barycenter (Bounded to vocal tract height [0, 6])
        ep_height = round(max(0.0, min(6.0, 6.0 * (1.0 - self.temperature))), 2)
        epenthetic_vowel = (ep_height, 1.0, 0.0)

        # 10. Metrical Stress Pattern
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

        # 12. Morphological Synthesis Target (Protected against Pop < 0)
        s_driver = (max(0.0, self.population) ** 1.3) - (self.altitude * 0.55) - 0.40
        synthesis_index = round(0.12 + (0.76 / (1.0 + math.exp(4.5 * s_driver))), 3)

        # 13. Acoustic Roughness (Self-bounding in [0.10, 0.90])
        attn = cultural_attention or [1.0] * 7
        w_con, w_anim, w_val, w_pot, w_dyn, w_soc, w_ext = attn
        
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
        """
        Realistic Continental & Oceanic Geography:
        - Western Ocean (x < 0.20)
        - Central Plains & Mountain Spine (x ~ 0.55 - 0.72, y ~ 0.75)
        - Eastern Inland Sea (x ~ 0.75 - 0.82, Alt < 0.20: Blue Water!)
        - Eastern Volcanic Island Arc / Archipelago (x ~ 0.84 - 0.88, Alt ~ 0.30)
        - Deep Pacific Ocean (x > 0.92, Blue Water!)
        """
        # Central mountain spine
        dist_to_spine = math.sqrt((x - 0.65) ** 2 + ((y - 0.75) * 1.4) ** 2)
        mountain_peak = 0.95 * math.exp(-(dist_to_spine ** 2) / 0.06)

        # Continental body (rises in center, drops on both coasts)
        continental_mass = 0.45 * math.sin(max(0.0, min(1.0, (x - 0.18) / 0.60)) * math.pi)

        # Eastern Inland Sea Trench (Drops elevation below sea level between x=0.75 and 0.82)
        sea_trench = 0.35 * math.exp(-((x - 0.78) ** 2) / 0.003)

        # Eastern Island Arc / Archipelago (Japan / Insular Arc at x ~ 0.85)
        dist_to_island_arc = math.sqrt(((x - 0.85) * 1.8) ** 2 + ((y - 0.50) * 0.8) ** 2)
        island_arc = 0.38 * math.exp(-(dist_to_island_arc ** 2) / 0.04)

        raw_elevation = mountain_peak + continental_mass - sea_trench + island_arc
        return round(max(0.05, raw_elevation), 3)

    def sample_environment(self, x: float, y: float, population: float = 0.50) -> Environment:
        clamped_x = max(0.0, min(1.0, float(x)))
        clamped_y = max(0.0, min(1.0, float(y)))

        altitude = self.elevation_fn(clamped_x, clamped_y)
        
        # Temperature: Latitude gradient + lapse cooling (Can drop below 0.0 in polar peaks!)
        base_temp = 0.95 - (clamped_y * 0.55)
        lapse_cooling = altitude * 0.55
        temperature = round(base_temp - lapse_cooling, 3)

       # True Oceanic Proximity: Distance to nearest water body (West Ocean, Inland Sea, or East Ocean)
        dist_to_water = min(clamped_x, abs(clamped_x - 0.78), 1.0 - clamped_x)
        base_humidity = 1.0 - (dist_to_water * 1.40)

        # Rain shadow applies only to leeward continental plains (x between 0.68 and 0.75)
        is_leeward_plain = (0.68 <= clamped_x <= 0.75 and altitude < 0.30)
        rain_shadow = 0.25 if is_leeward_plain else 0.0

        # Island arc and coastal zones remain richly maritime
        is_island = (clamped_x >= 0.82 and altitude >= 0.22)
        min_hum = 0.65 if is_island else (0.45 if dist_to_water < 0.12 else 0.05)
        humidity = round(max(min_hum, base_humidity - rain_shadow), 3)

        # Vegetation
        veg_score = (humidity * 0.70) + (max(0.0, temperature) * 0.30) - (altitude * 0.40)
        vegetation = round(max(0.01, min(0.99, veg_score)), 3)

        # Ambient Noise
        ridge_wind = altitude * 0.65
        surf_noise = max(0.0, (0.25 - clamped_x) * 2.0) if clamped_x < 0.25 else 0.0
        ambient_noise = round(min(0.99, ridge_wind + surf_noise), 3)

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