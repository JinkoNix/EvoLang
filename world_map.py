"""
world_map.py — Continuous 2D Physical Geography, Biophysical Channels, and Demographic Contact.
Features pure self-bounding logistic response curves (Weber-Fechner biophysics) with zero hard clamps.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from collections import Counter
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
    """
    Abiotic physical parameters at coordinate (x, y).
    Accessing .vegetation dynamically computes emergent biomass if not manually overridden.
    """
    temperature: float = 0.50
    humidity: float = 0.50
    altitude: float = 0.20
    ambient_noise: float = 0.20
    _vegetation: float | None = field(default=None, repr=False)
    _population: float | None = field(default=None, repr=False)

    def __init__(
        self,
        temperature: float = 0.50,
        humidity: float = 0.50,
        altitude: float = 0.20,
        ambient_noise: float = 0.20,
        vegetation: float | None = None,
        population: float | None = None,
    ):
        self.temperature = float(temperature)
        self.humidity = float(humidity)
        self.altitude = float(altitude)
        self.ambient_noise = float(ambient_noise)
        self._vegetation = float(vegetation) if vegetation is not None else None
        self._population = float(population) if population is not None else None

    @property
    def vegetation(self) -> float:
        if self._vegetation is not None:
            return self._vegetation
        return self.emergent_vegetation(population=self._population or 0.0)

    @vegetation.setter
    def vegetation(self, val: float | None):
        self._vegetation = float(val) if val is not None else None

    @property
    def population(self) -> float:
        return self._population if self._population is not None else 0.50

    @population.setter
    def population(self, val: float | None):
        self._population = float(val) if val is not None else None

    def emergent_vegetation(
        self,
        population: float = 0.0,
        cultural_attention: Sequence[float] | None = None,
    ) -> float:
        if self._vegetation is not None:
            return self._vegetation

        f_temp = 1.0 / (1.0 + math.exp(-8.0 * (self.temperature - 0.22)))
        f_hum = 1.0 / (1.0 + math.exp(-7.0 * (self.humidity - 0.20)))
        f_alt = math.exp(-2.80 * (max(0.0, self.altitude - 0.30) ** 2))
        climax_veg = f_temp * f_hum * f_alt

        attn = cultural_attention or [1.0] * 7
        w_con, w_anim, w_val, w_pot = attn[0], attn[1], attn[2], attn[3]
        deforestation_rate = (w_con * 0.70 + w_pot * 0.50) / max(0.50, 1.0 + w_anim * 0.80 + w_val * 0.80)
        
        depletion = math.exp(-0.70 * deforestation_rate * (max(0.0, population) ** 1.3))
        return round(max(0.01, min(0.99, climax_veg * depletion)), 3)

    def generate_profile(
        self,
        name: str = "Adapted Profile",
        initial_head_directionality: float | None = None,
        cultural_attention: Sequence[float] | None = None,
        population: float = 0.50,
        vegetation: float | None = None,
    ) -> LanguageProfile:
        """
        Layer 2: Derives continuous acoustic impedance channels and morphosyntax.
        Reads population and emergent vegetation explicitly from the speech community.
        """
        pop = max(0.0, float(population))
        veg = vegetation if vegetation is not None else self.emergent_vegetation(population=pop, cultural_attention=cultural_attention)

        # 1. Atmospheric Air Density
        rho_air = math.exp(-2.40 * max(0.0, self.altitude)) * math.exp(-0.25 * (self.temperature - 0.50))

        # 2. Vocal Fold Compliance
        mu_vocal = (1.0 / (1.0 + math.exp(-6.0 * (self.humidity - 0.45)))) * math.exp(-2.0 * ((self.temperature - 0.60) ** 2))

        # 3. High-Frequency Spectral Damping (Uses emergent vegetation)
        alpha_damp = 1.0 / (1.0 + math.exp(-5.0 * ((veg * 0.65 + self.humidity * 0.35) - 0.50)))

        # 4. Continuous Syllable Complexity
        s_comp_driver = alpha_damp + (max(0.0, self.ambient_noise) * 0.35) - ((1.0 - rho_air) * 0.60) - 0.40
        syllable_complexity = round(0.10 + (0.80 / (1.0 + math.exp(5.0 * s_comp_driver))), 3)
        allow_clusters = (syllable_complexity >= 0.28)
        max_transition_cost = round(2.0 + (3.80 * syllable_complexity), 2)

        # 5. Continuous Vowel Dispersion (VPD)
        aridity = max(0.0, 1.0 - self.humidity)
        vpd = (max(0.0, self.temperature) ** 1.2) * aridity * 1.50
        v_driver = vpd + (max(0.0, self.ambient_noise) * 0.15) - (alpha_damp * 0.15) - 0.35
        d_min_vowel = round(0.12 + (0.22 / (1.0 + math.exp(-6.0 * v_driver))), 3)

        # 6. Continuous Consonant Dispersion Threshold
        d_min_consonant = round(0.22 + (0.12 / (1.0 + math.exp(-5.0 * (rho_air - 0.60)))), 3)

        # 7. Continuous Tonogenesis
        tonal_driver = (mu_vocal * 1.40) + (self.humidity * 0.80) - (pop * 0.80) - 1.20
        raw_tone_tier = 8.0 / (1.0 + math.exp(-4.5 * tonal_driver))
        tone_tier = 0 if raw_tone_tier < 1.20 else int(round(raw_tone_tier / 2.0) * 2)

        # 8. Vowel Reduction Mode
        if tone_tier > 0:
            vowel_reduction_mode = "none" if syllable_complexity < 0.25 else "inventory_snap"
        else:
            p_centralize = 1.0 / (1.0 + math.exp(-8.0 * (pop - 0.60)))
            if syllable_complexity < 0.25:
                vowel_reduction_mode = "none"
            elif random.random() < p_centralize:
                vowel_reduction_mode = "centralize"
            else:
                vowel_reduction_mode = "inventory_snap"

        # 9. Epenthetic Vowel Barycenter
        ep_height = round(max(0.0, min(6.0, 6.0 * (1.0 - self.temperature))), 2)
        epenthetic_vowel = (ep_height, 1.0, 0.0)

        # 10. Metrical Stress
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

        # 12. Synthesis Index
        s_driver = (pop ** 1.3) - (self.altitude * 0.55) - 0.40
        synthesis_index = round(0.12 + (0.76 / (1.0 + math.exp(4.5 * s_driver))), 3)

        # 13. Acoustic Roughness
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
            apocope_rate=round(1.0 / (1.0 + math.exp(-4.0 * ((1.0 - veg) * 0.6 + (1.0 - self.temperature) * 0.4 - 0.5))), 3),
            syncope_rate=round(1.0 / (1.0 + math.exp(-4.0 * (pop * 0.55 + self.altitude * 0.45 - 0.5))), 3),
            tone_tier=tone_tier,
            head_directionality=round(head_directionality, 3),
            d_min_vowel=d_min_vowel,
            d_min_consonant=d_min_consonant,
            synthesis_index=synthesis_index,
            syllable_complexity=syllable_complexity,
            acoustic_roughness=acoustic_roughness,
        )


class TerrainField:
    """Continuous 2D scalar field manifold for elevation, climate, and emergent ecology."""

    def __init__(self, elevation_fn: Callable[[float, float], float] | None = None):
        self.elevation_fn = elevation_fn or self._default_continental_heightmap

    @staticmethod
    def _default_continental_heightmap(x: float, y: float) -> float:
        """
        Calibrated Planetary Geography:
        - Western Ocean (x < 0.18, Alt < 0.0: Deep Blue)
        - Continental Landmass (0.20 <= x <= 0.72, Alt > 0.0)
        - Central Mountain Peak (x=0.66, y=0.75, Alt ~ 0.92)
        - Eastern Inland Sea Trench (0.75 <= x <= 0.82, Alt < 0.0: Deep Blue)
        - Eastern Island Arc (0.83 <= x <= 0.88, Alt ~ 0.28: Dry Land Island)
        - Deep Pacific Ocean (x > 0.90, Alt < 0.0: Deep Blue)
        """
        # Central mountain spine
        dist_to_spine = math.sqrt(((x - 0.66) * 1.6) ** 2 + ((y - 0.75) * 1.3) ** 2)
        mountain_peak = 0.92 * math.exp(-(dist_to_spine ** 2) / 0.035)

        # Continental mass (Mainland: rises at 0.20, ends at 0.72)
        in_continent = max(0.0, min(1.0, (x - 0.18) / 0.54))
        continental_mass = 0.35 * math.sin(in_continent * math.pi) if x <= 0.74 else 0.0

        # Eastern Sea Trench (Drops to negative depths!)
        sea_trench = 0.45 * math.exp(-((x - 0.78) ** 2) / 0.002)

        # Eastern Island Arc (Japan/Ryukyu: peaks at x=0.85, y=0.50)
        dist_to_island = math.sqrt(((x - 0.855) * 2.2) ** 2 + ((y - 0.50) * 0.9) ** 2)
        island_arc = 0.34 * math.exp(-(dist_to_island ** 2) / 0.025)

        # Western and Eastern Ocean Shelf Drops
        west_drop = max(0.0, (0.19 - x) * 2.5)
        east_drop = max(0.0, (x - 0.89) * 3.5)

        raw_elevation = mountain_peak + continental_mass - sea_trench + island_arc - west_drop - east_drop
        return round(raw_elevation, 3)

    def sample_environment(
        self,
        x: float,
        y: float,
        epoch: int = 0,
        **kwargs,
    ) -> Environment:
        """Samples planetary parameters with true sea-level physics and maritime thermal moderation."""
        clamped_x = max(0.0, min(1.0, float(x)))
        clamped_y = max(0.0, min(1.0, float(y)))

        altitude = self.elevation_fn(clamped_x, clamped_y)
        climate_cycle = 0.05 * math.sin(2.0 * math.pi * epoch / 300.0)

        # 1. True Distance to Ocean Water (Shoreline datum Alt = 0.0)
        dist_to_coast = min(clamped_x, abs(clamped_x - 0.78), 1.0 - clamped_x)

        # 2. Temperature: Latitude gradient + lapse cooling + Maritime Thermal Moderation
        # Marine air has high heat capacity (cp ~ 4.184 J/gK), moderating coastal temperatures toward 0.50!
        base_temp = 0.95 - (clamped_y * 0.55) + climate_cycle
        lapse_cooling = max(0.0, altitude) * 0.55
        continental_temp = base_temp - lapse_cooling
        
        # Marine moderation factor (pulls toward 0.50 near water)
        marine_moderation = 0.35 * max(0.0, 1.0 - dist_to_coast * 3.5)
        temperature = round(continental_temp + marine_moderation * (0.50 - continental_temp), 3)

        # 3. Humidity
        base_humidity = 1.0 - (dist_to_coast * 1.30) + climate_cycle
        is_leeward_plain = (0.68 <= clamped_x <= 0.75 and altitude < 0.30)
        rain_shadow = 0.25 if is_leeward_plain else 0.0
        is_island = (clamped_x >= 0.82 and altitude > 0.0)
        min_hum = 0.65 if is_island else (0.45 if dist_to_coast < 0.10 else 0.05)
        humidity = round(max(min_hum, base_humidity - rain_shadow), 3)

        # 4. Ambient Noise
        ridge_wind = max(0.0, altitude) * 0.65
        surf_noise = max(0.0, (0.15 - dist_to_coast) * 3.0) if dist_to_coast < 0.15 else 0.0
        ambient_noise = round(min(0.99, ridge_wind + surf_noise), 3)

        env = Environment(
            temperature=temperature,
            humidity=humidity,
            altitude=altitude,
            ambient_noise=ambient_noise,
        )

        # Terrestrial vegetation is strictly ZERO on open water (Alt <= 0.0)
        if altitude <= 0.0:
            env.vegetation = 0.0

        return env


class Civilization:
    """A living speech community situated at continuous coordinates with dynamic demographics."""

    def __init__(
        self,
        name: str,
        language: Language,
        x: float,
        y: float,
        terrain: TerrainField,
        population: float = 0.50,
        lineage: str | None = None,
    ):
        self.name = name
        self.language = language
        self.x = max(0.02, min(0.98, float(x)))
        self.y = max(0.02, min(0.98, float(y)))
        self.vx = 0.0  # Momentum vector (eliminates left-right jitter!)
        self.vy = 0.0
        self.terrain = terrain
        self.population = float(population)
        self.lineage = lineage or name
        self.food_capacity = 0.65
        self.food_deficit = 0.00
        self.territory_radius = 0.08
        self.sync_environment()

    def _calculate_food_and_territory(self, env: Environment, veg: float) -> None:
        attn = self.language.cultural_attention
        w_dyn, w_ext = attn[4], attn[6]

        # Arable Yield: Thrives in lowlands (Alt <= 0.25)
        agri_dist = ((env.humidity - 0.55) / 0.22) ** 2 + ((veg - 0.45) / 0.22) ** 2 + ((env.temperature - 0.55) / 0.22) ** 2
        altitude_penalty = max(0.0, 1.0 - (env.altitude / 0.65) ** 1.5)
        arable_yield = math.exp(-agri_dist) * altitude_penalty

        # Sustaining Food Capacity (Permits populations of 0.70 - 1.10 in lowlands!)
        self.food_capacity = round(
            0.25 + (0.95 * arable_yield * (1.0 + 0.35 * (w_dyn - 1.0))) + (0.30 * veg), 
            3
        )
        self.food_deficit = round(max(0.0, self.population - self.food_capacity), 3)

        base_radius = 0.05 + 0.08 * math.sqrt(max(0.01, self.population)) * (w_ext / 1.0)
        hunger_push = 1.0 + 0.80 * self.food_deficit
        altitude_choke = max(0.40, 1.0 - env.altitude * 0.50)
        self.territory_radius = round(base_radius * hunger_push * altitude_choke, 3)

    def step_migration(self) -> None:
        """Smooth directional migration with momentum; strictly blocked from ocean water (Alt <= 0.0)."""
        attn = self.language.cultural_attention
        w_dyn, w_ext = attn[4], attn[6]
        mobility = 0.015 * (w_dyn / 1.0) * (w_ext / 1.0) / (1.0 + self.language.cultural_inertia * 0.40)

        def evaluate_location(eval_x: float, eval_y: float) -> float:
            env = self.terrain.sample_environment(eval_x, eval_y)
            # Strict shoreline boundary: people cannot step into water and drown!
            if env.altitude <= 0.0:
                return -9999.0
            
            agri_dist = ((env.humidity - 0.55) / 0.22) ** 2 + ((env.vegetation - 0.45) / 0.22) ** 2
            return math.exp(-agri_dist) * max(0.0, 1.0 - env.altitude * 1.50) - (env.altitude * 0.40)

        current_val = evaluate_location(self.x, self.y)
        grad_x, grad_y = 0.0, 0.0
        probe_dist = 0.02

        for dx, dy in [(-probe_dist, 0), (probe_dist, 0), (0, -probe_dist), (0, probe_dist)]:
            val = evaluate_location(max(0.02, min(0.98, self.x + dx)), max(0.02, min(0.98, self.y + dy)))
            diff = val - current_val
            if dx != 0: grad_x += diff * (1.0 if dx > 0 else -1.0)
            if dy != 0: grad_y += diff * (1.0 if dy > 0 else -1.0)

        # Physical momentum vector
        self.vx = 0.80 * self.vx + 0.20 * grad_x * mobility
        self.vy = 0.80 * self.vy + 0.20 * grad_y * mobility

        # Target next position
        next_x = max(0.02, min(0.98, self.x + self.vx))
        next_y = max(0.02, min(0.98, self.y + self.vy))
        
        # Verify next position is dry land
        next_env = self.terrain.sample_environment(next_x, next_y)
        if next_env.altitude > 0.0:
            self.x = next_x
            self.y = next_y
        else:
            # Bounce/deflect away from the coast
            self.vx *= -0.5
            self.vy *= -0.5

    @property
    def environment(self) -> Environment:
        return self.terrain.sample_environment(self.x, self.y)

    def sync_environment(self, epoch: int = 0) -> None:
        env = self.terrain.sample_environment(self.x, self.y, epoch=epoch)
        self.language.environment = env
        self.language.population = self.population
        
        veg = env.emergent_vegetation(population=self.population, cultural_attention=self.language.cultural_attention)
        self.language.profile = env.generate_profile(
            name=f"{self.name} Profile",
            initial_head_directionality=self.language.profile.head_directionality,
            cultural_attention=self.language.cultural_attention,
            population=self.population,
            vegetation=veg,
        )
        self._calculate_food_and_territory(env, veg)

    def step_demographics(self, delta_time: float = 1.0) -> None:
        """Executes Verhulst Logistic Population Growth coupled to Food Capacity."""
        attn = self.language.cultural_attention
        w_anim = attn[1]
        w_val = attn[2]
        w_pot = attn[3]

        # Nature-revering cultures moderate fertility; martial/materialist cultures boom
        fertility_bias = 0.04 * (1.0 + 0.25 * (w_pot - (w_anim + w_val) * 0.5))
        
        # Logistic demographic growth
        capacity = max(0.10, self.food_capacity)
        d_pop = fertility_bias * self.population * (1.0 - (self.population / capacity))

        # Famine Attrition if sustained severe deficit
        if self.food_deficit > 0.15:
            d_pop -= 0.05 * self.food_deficit

        self.population = round(max(0.04, self.population + d_pop * delta_time), 3)



class ContinuousWorldMap:
    """Manages continuous geography, geopolitical trade, warfare, and speciation."""

    def __init__(self, name: str = "Pangaea Field", terrain: TerrainField | None = None):
        self.name = name
        self.terrain = terrain or TerrainField()
        self.civilizations: dict[str, Civilization] = {}
        self.current_epoch: int = 0
        self._friction_cache: dict[tuple[float, float, float, float], float] = {}
        self.active_trade_routes: list[tuple[str, str]] = []
        self.active_conflicts: list[tuple[str, str]] = []

    def spawn_civilization(
        self,
        name: str,
        language: Language,
        x: float,
        y: float,
        population: float = 0.50,
        lineage: str | None = None,
    ) -> Civilization:
        civ = Civilization(name, language, x, y, self.terrain, population, lineage=lineage)
        self.civilizations[name] = civ
        return civ

    def calculate_path_friction(self, x1: float, y1: float, x2: float, y2: float) -> float:
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

    def step_epoch(
        self,
        reduction_strength: float = 0.75,
        enable_neologisms: bool = True,
        parallel: bool = False,
    ) -> None:
        import heapq
        self.current_epoch += 1
        self.active_trade_routes.clear()
        self.active_conflicts.clear()
        civ_list = list(self.civilizations.values())

        # 1. Update Demographics, Ecosystems, and Migration
        for civ in civ_list:
            civ.sync_environment(epoch=self.current_epoch)
            civ.step_demographics()
            civ.step_migration()

        # 2. Dijkstra Cost-Distance Cellular Expansion (NO DIAMONDS!)
        # Territory flows down lowlands and arable valleys, stopping at mountain ridges and coastlines
        res = 50
        grid_claims: dict[tuple[int, int], str] = {}
        grid_density: dict[tuple[int, int], float] = {}

        for civ in civ_list:
            c_col = max(0, min(res - 1, int(round(civ.x * (res - 1)))))
            c_row = max(0, min(res - 1, int(round(civ.y * (res - 1)))))
            
            w_ext = civ.language.cultural_attention[6]
            max_cells = int(round(15 + (civ.population * 55) * (w_ext / 1.0)))

            # Priority Queue: (accumulated_cost, row, col)
            pq = [(0.0, c_row, c_col)]
            visited = {(c_row, c_col): 0.0}
            claimed = 0

            while pq and claimed < max_cells:
                cost, r, c = heapq.heappop(pq)
                gx = c / float(res - 1)
                gy = r / float(res - 1)
                cell_env = self.terrain.sample_environment(gx, gy)

                # Strictly impassable ocean water
                if cell_env.altitude <= 0.0:
                    continue

                grid_claims[(r, c)] = civ.name
                
                # Non-uniform density: core is dense (1.0), frontier thins out exponentially
                dist_to_center = math.sqrt((gx - civ.x)**2 + (gy - civ.y)**2)
                density = civ.population * math.exp(-dist_to_center * 8.0)
                grid_density[(r, c)] = round(max(0.08, min(1.0, density)), 2)
                claimed += 1

                # Expand to 4 neighbors with terrain cost
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < res and 0 <= nc < res:
                        ngx = nc / float(res - 1)
                        ngy = nr / float(res - 1)
                        nenv = self.terrain.sample_environment(ngx, ngy)

                        if nenv.altitude <= 0.0:
                            continue

                        # Terrain friction: lowlands/arable valleys are cheap (1.0), mountains are steep (5.0-15.0!)
                        agri_dist = ((nenv.humidity - 0.55) / 0.22) ** 2 + ((nenv.vegetation - 0.45) / 0.22) ** 2
                        arable_bonus = math.exp(-agri_dist)
                        slope_cost = 1.0 + 8.0 * (max(0.0, nenv.altitude - 0.25) ** 2) - 0.40 * arable_bonus
                        new_cost = cost + max(0.50, slope_cost)

                        if (nr, nc) not in visited or new_cost < visited[(nr, nc)]:
                            visited[(nr, nc)] = new_cost
                            heapq.heappush(pq, (new_cost, nr, nc))

        self.territory_grid = grid_claims
        self.density_grid = grid_density

        # 3. Lineage-Aware Geopolitical Contact & Conflict Spectrum
        n = len(civ_list)
        for i in range(n):
            for j in range(i + 1, n):
                civ_a, civ_b = civ_list[i], civ_list[j]
                dist_centers = math.sqrt((civ_a.x - civ_b.x) ** 2 + (civ_a.y - civ_b.y) ** 2)
                
                if dist_centers < 0.25:
                    is_kin = (civ_a.lineage == civ_b.lineage)
                    pot_a = civ_a.language.cultural_attention[3] * civ_a.population
                    pot_b = civ_b.language.cultural_attention[3] * civ_b.population
                    power_ratio = pot_a / max(0.01, pot_b)

                    # Kinship dampens civil wars: colonies don't fight parent without extreme famine!
                    war_threshold = 1.85 if is_kin else 1.25
                    has_conflict = (power_ratio > war_threshold or power_ratio < (1.0 / war_threshold)) and (
                        civ_a.food_deficit > (0.15 if is_kin else 0.02) or civ_b.food_deficit > (0.15 if is_kin else 0.02)
                    )

                    if has_conflict:
                        self.active_conflicts.append((civ_a.name, civ_b.name))
                        victor = civ_a if power_ratio > 1.0 else civ_b
                        conquered = civ_b if power_ratio > 1.0 else civ_a

                        candidates = [w for w in victor.language.words.values() if w.category == LexicalCategory.CONTENT_OPEN and w.usage_frequency >= 2.0]
                        if candidates and random.random() < 0.35:
                            loan = random.choice(candidates)
                            conquered.language.borrow_word(loan, source_language_name=victor.name, current_generation=self.current_epoch)

                        annex_pop = min(0.03, conquered.population * 0.08)
                        conquered.population = max(0.04, conquered.population - annex_pop)
                        victor.population += annex_pop * 0.50
                    else:
                        self.active_trade_routes.append((civ_a.name, civ_b.name))
                        intensity = 1.0 / (1.0 + math.exp(-4.0 * (0.25 - dist_centers) * 5.0))
                        civ_a.language.set_contact(civ_b.language, intensity)

                        if random.random() < (intensity * 0.25):
                            cand_pool = [w for w in civ_b.language.words.values() if w.category == LexicalCategory.CONTENT_OPEN]
                            if cand_pool:
                                target_w = random.choice(cand_pool)
                                p_calque = 1.0 / (1.0 + math.exp(-6.0 * (civ_a.language.cultural_inertia - 1.05)))
                                if random.random() < p_calque:
                                    civ_a.language.calque_word(target_w, source_language_name=civ_b.name, current_generation=self.current_epoch)
                                else:
                                    civ_a.language.borrow_word(target_w, source_language_name=civ_b.name, current_generation=self.current_epoch)

        # 4. Frontier Colony Nucleation (NO TELEPORTATION!)
        # Buds off from an outer claimed cell on the frontier of the parent territory
        if len(self.civilizations) < 6 and self.current_epoch % 25 == 0:
            for civ in civ_list:
                attn = civ.language.cultural_attention
                w_dyn = attn[4]
                w_ext = attn[6]
                
                saturation = civ.population / max(0.10, civ.food_capacity)
                p_fork = 0.22 * (w_dyn / 1.0) * (w_ext / 1.0) * max(0.0, saturation - 0.78)

                if saturation >= 0.80 and random.random() < p_fork:
                    # Find peripheral claimed cells at the edge of the territory
                    civ_claimed_coords = [
                        (c / float(res - 1), r / float(res - 1))
                        for (r, c), owner in grid_claims.items()
                        if owner == civ.name
                    ]
                    
                    # Sort cells by distance from capital (frontier nodes)
                    frontier_candidates = sorted(
                        civ_claimed_coords,
                        key=lambda pt: math.sqrt((pt[0] - civ.x)**2 + (pt[1] - civ.y)**2),
                        reverse=True
                    )

                    # Select an outer frontier cell that has dry land and is not too close to other cities
                    for cand_x, cand_y in frontier_candidates[:8]:
                        too_close = any(math.sqrt((c.x - cand_x)**2 + (c.y - cand_y)**2) < 0.12 for c in self.civilizations.values())
                        cand_env = self.terrain.sample_environment(cand_x, cand_y)

                        if not too_close and cand_env.altitude > 0.0:
                            daughter_num = len(self.civilizations) + 1
                            daughter_name = f"{civ.name}-Colony{daughter_num}"
                            daughter_lang = civ.language.fork(daughter_name, environment=cand_env)
                            
                            colonist_pop = round(civ.population * 0.28, 3)
                            civ.population = round(civ.population - colonist_pop, 3)

                            self.spawn_civilization(
                                name=daughter_name,
                                language=daughter_lang,
                                x=cand_x,
                                y=cand_y,
                                population=colonist_pop,
                                lineage=civ.lineage
                            )
                            self.event_log.append(f"Gen {self.current_epoch}: {civ.name} nucleated frontier colony '{daughter_name}' at ({cand_x:.2f}, {cand_y:.2f})!")
                            break
                    break

        # 5. Language Evolution
        for civ in list(self.civilizations.values()):
            civ.language.evolve(
                reduction_strength=reduction_strength,
                enable_neologisms=enable_neologisms,
                current_generation=self.current_epoch,
            )