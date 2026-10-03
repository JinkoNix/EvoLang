"""
world_map.py — Continuous 2D Physical Geography, Strategic Urban Fortress Dynamics, and Speciation.
Features:
- Organic Domestic Urbanization: Empires build secondary provincial cities in their own fertile valleys.
- True Frontier & Island Expeditions: Pioneers colonize UNCLAIMED wilderness or offshore islands.
- Speciation by Geographic Isolation: Outposts only fork into new languages if isolated across water or mountain ranges.
- City-to-City Logistical Warfare: Frontier sieges batter walls progressively; armistices enforce peace.
- Topographic Power Voronoi: Pure Euclidean geometry; mountains and sea trenches constrain expansion.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from collections import Counter
from typing import Sequence, Callable

import numpy as np

from language import Language, LanguageProfile
from syllable import StressPattern
from semantics import SemanticVector, DerivationType
from word import Word, LexicalCategory


@dataclass
class SettlementNode:
    """A permanent urban center with discrete local population, fortifications, and history."""
    name: str
    x: float
    y: float
    local_pop: float = 0.05
    fortification_level: float = 1.0
    local_infrastructure: float = 1.0
    is_capital: bool = False
    is_permanent: bool = True
    founder: str = ""
    founded_epoch: int = 0
    history: list[str] = field(default_factory=list)

    @property
    def population_share(self) -> float:
        return self.local_pop


@dataclass
class RoadEdge:
    """A direct transport connection between two cities with wear/quality level."""
    city_a: str
    city_b: str
    x0: float
    y0: float
    x1: float
    y1: float
    level: float = 1.0
    is_naval: bool = False


@dataclass
class EconomyProfile:
    """Normalized 100% breakdown of subsistence resource streams."""
    grain: float = 0.0
    livestock: float = 0.0
    marine: float = 0.0
    minerals: float = 0.0
    total_yield: float = 0.0


@dataclass
class Environment:
    temperature: float = 0.50
    humidity: float = 0.50
    altitude: float = 0.20
    ambient_noise: float = 0.20
    season_phase: float = 0.0
    _vegetation: float | None = field(default=None, repr=False)
    _population: float | None = field(default=None, repr=False)

    def __init__(
        self,
        temperature: float = 0.50,
        humidity: float = 0.50,
        altitude: float = 0.20,
        ambient_noise: float = 0.20,
        season_phase: float = 0.0,
        vegetation: float | None = None,
        population: float | None = None,
    ):
        self.temperature = float(temperature)
        self.humidity = float(humidity)
        self.altitude = float(altitude)
        self.ambient_noise = float(ambient_noise)
        self.season_phase = float(season_phase)
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
        seasonal_growth = 1.0 + 0.25 * self.season_phase
        climax_veg = f_temp * f_hum * f_alt * seasonal_growth

        attn = cultural_attention or [1.0] * 7
        w_con, w_pot = attn[0], attn[3]
        deforestation_rate = (w_con * 0.70 + w_pot * 0.50)
        depletion = math.exp(-0.70 * deforestation_rate * (max(0.0, population) ** 1.3))
        return round(max(0.01, min(0.99, climax_veg * depletion)), 3)

    def generate_profile(
        self,
        name: str = "Adapted Profile",
        initial_head_directionality: float | None = None,
        cultural_attention: Sequence[float] | None = None,
        population: float = 0.50,
        vegetation: float | None = None,
        parent_bias: Sequence[float] | None = None,
    ) -> LanguageProfile:
        pop = max(0.0, float(population))
        veg = vegetation if vegetation is not None else self.emergent_vegetation(population=pop, cultural_attention=cultural_attention)

        rho_air = math.exp(-2.40 * max(0.0, self.altitude)) * math.exp(-0.25 * (self.temperature - 0.50))
        alpha_damp = 1.0 / (1.0 + math.exp(-5.0 * ((veg * 0.65 + self.humidity * 0.35) - 0.50)))

        s_comp_driver = alpha_damp + (max(0.0, self.ambient_noise) * 0.35) - ((1.0 - rho_air) * 0.60) - 0.40
        syllable_complexity = round(0.10 + (0.80 / (1.0 + math.exp(5.0 * s_comp_driver))), 3)
        max_transition_cost = round(2.0 + (3.80 * syllable_complexity), 2)

        aridity = max(0.0, 1.0 - self.humidity)
        vpd = (max(0.0, self.temperature) ** 1.2) * aridity * 1.50
        v_driver = vpd + (max(0.0, self.ambient_noise) * 0.15) - (alpha_damp * 0.15) - 0.35
        d_min_vowel = round(0.12 + (0.22 / (1.0 + math.exp(-6.0 * v_driver))), 3)
        d_min_consonant = round(0.22 + (0.12 / (1.0 + math.exp(-5.0 * (rho_air - 0.60)))), 3)

        p_centralize = 1.0 / (1.0 + math.exp(-8.0 * (pop - 0.60)))
        vowel_reduction_mode = "none" if syllable_complexity < 0.25 else ("centralize" if random.random() < p_centralize else "inventory_snap")

        ep_height = round(max(0.0, min(6.0, 6.0 * (1.0 - self.temperature))), 2)
        epenthetic_vowel = (ep_height, 1.0, 0.0)
        stress_pattern = StressPattern.NATURAL_WEIGHT

        head_directionality = initial_head_directionality if initial_head_directionality is not None else (
            random.choices([0.20, 0.50, 0.80], weights=[45.0, 45.0, 10.0])[0] + random.uniform(-0.06, 0.06)
        )

        s_driver = (pop ** 1.3) - (self.altitude * 0.55) - 0.40
        synthesis_index = round(0.12 + (0.76 / (1.0 + math.exp(4.5 * s_driver))), 3)

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

        if parent_bias is not None:
            aesthetic_bias = tuple(round(max(-0.60, min(0.60, b + random.gauss(0.0, 0.03))), 3) for b in parent_bias[:6])
        else:
            aesthetic_bias = tuple(round(random.uniform(-0.35, 0.35), 3) for _ in range(6))

        alignment = "ergative_absolutive" if w_pot > (w_soc * 1.15) else "nominative_accusative"

        return LanguageProfile(
            name=name,
            max_transition_cost=max_transition_cost,
            voicing_assimilation_bias="regressive",
            double_stop_strategy="spirantization" if (self.altitude * 0.6 + aridity * 0.4) > 0.45 else "gemination",
            epenthetic_vowel_point=epenthetic_vowel,
            stress_pattern=stress_pattern,
            vowel_reduction_mode=vowel_reduction_mode,
            apocope_rate=round(1.0 / (1.0 + math.exp(-4.0 * ((1.0 - veg) * 0.6 + (1.0 - self.temperature) * 0.4 - 0.5))), 3),
            syncope_rate=round(1.0 / (1.0 + math.exp(-4.0 * (pop * 0.55 + self.altitude * 0.45 - 0.5))), 3),
            head_directionality=round(head_directionality, 3),
            d_min_vowel=d_min_vowel,
            d_min_consonant=d_min_consonant,
            synthesis_index=synthesis_index,
            syllable_complexity=syllable_complexity,
            acoustic_roughness=acoustic_roughness,
            alignment=alignment,
            aesthetic_bias=aesthetic_bias,
        )


class TerrainField:
    def __init__(self, elevation_fn: Callable[[float, float], float] | None = None):
        self.elevation_fn = elevation_fn or self._default_continental_heightmap

    @staticmethod
    def _default_continental_heightmap(x: float, y: float) -> float:
        dist_to_spine = math.sqrt(((x - 0.66) * 1.6) ** 2 + ((y - 0.75) * 1.3) ** 2)
        mountain_peak = 0.92 * math.exp(-(dist_to_spine ** 2) / 0.035)

        in_continent = max(0.0, min(1.0, (x - 0.18) / 0.54))
        continental_mass = 0.35 * math.sin(in_continent * math.pi) if x <= 0.74 else 0.0

        sea_trench = 0.45 * math.exp(-((x - 0.78) ** 2) / 0.002)

        dist_to_island = math.sqrt(((x - 0.855) * 2.2) ** 2 + ((y - 0.50) * 0.9) ** 2)
        island_arc = 0.34 * math.exp(-(dist_to_island ** 2) / 0.025)

        west_drop = max(0.0, (0.19 - x) * 2.5)
        east_drop = max(0.0, (x - 0.89) * 3.5)

        raw_elevation = mountain_peak + continental_mass - sea_trench + island_arc - west_drop - east_drop
        return round(raw_elevation, 3)

    def sample_environment(self, x: float, y: float, epoch: int = 0) -> Environment:
        clamped_x = max(0.0, min(1.0, float(x)))
        clamped_y = max(0.0, min(1.0, float(y)))
        altitude = self.elevation_fn(clamped_x, clamped_y)

        season_phase = math.sin(2.0 * math.pi * (epoch % 12) / 12.0)
        macro_climate_pulse = 0.14 * math.sin(2.0 * math.pi * epoch / 160.0) + 0.06 * math.sin(2.0 * math.pi * epoch / 45.0)

        dist_to_coast = min(clamped_x, abs(clamped_x - 0.78), 1.0 - clamped_x)
        base_temp = 0.95 - (clamped_y * 0.55) + (season_phase * 0.10) + (macro_climate_pulse * 0.06)
        lapse_cooling = max(0.0, altitude) * 0.55
        continental_temp = base_temp - lapse_cooling
        marine_moderation = 0.35 * max(0.0, 1.0 - dist_to_coast * 3.5)
        temperature = round(continental_temp + marine_moderation * (0.50 - continental_temp), 3)

        base_humidity = 1.0 - (dist_to_coast * 1.30) + (macro_climate_pulse * 0.14)
        is_leeward_plain = (0.68 <= clamped_x <= 0.75 and altitude < 0.30)
        rain_shadow = 0.25 if is_leeward_plain else 0.0
        is_island = (clamped_x >= 0.82 and altitude > 0.0)
        min_hum = 0.65 if is_island else (0.45 if dist_to_coast < 0.10 else 0.08)
        humidity = round(max(min_hum, base_humidity - rain_shadow), 3)

        ridge_wind = max(0.0, altitude) * 0.65
        surf_noise = max(0.0, (0.15 - dist_to_coast) * 3.0) if dist_to_coast < 0.15 else 0.0
        ambient_noise = round(min(0.99, ridge_wind + surf_noise), 3)

        env = Environment(
            temperature=temperature,
            humidity=humidity,
            altitude=altitude,
            ambient_noise=ambient_noise,
            season_phase=round(season_phase, 2),
        )

        if altitude <= 0.0:
            env.vegetation = 0.0

        return env


class Civilization:
    """A living speech community situated at continuous coordinates with dynamic demographics and settlements."""

    def __init__(
        self,
        name: str,
        language: Language,
        x: float,
        y: float,
        terrain: TerrainField,
        population: float = 0.50,
        lineage: str | None = None,
        founded_epoch: int = 0,
    ):
        self.name = name
        self.language = language
        self.x = max(0.02, min(0.98, float(x)))
        self.y = max(0.02, min(0.98, float(y)))
        self.vx = 0.0
        self.vy = 0.0
        self.terrain = terrain
        self.population = float(population)
        self.lineage = lineage or name
        self.founded_epoch = founded_epoch
        self.food_capacity = 0.65
        self.food_deficit = 0.00
        self.territory_radius = 0.08
        self.claimed_cells_count = 16
        self.infrastructure_level = 0.8
        
        self.economy = EconomyProfile()
        
        # Primordial capital starts with permanent stone status and Level 1.5 walls
        self.settlements: list[SettlementNode] = [
            SettlementNode(
                name=f"{name}", x=self.x, y=self.y,
                local_pop=float(population),
                fortification_level=1.5,
                local_infrastructure=1.0,
                is_capital=True,
                is_permanent=True,
                founder=name,
                founded_epoch=founded_epoch,
                history=[f"Founded at Epoch {founded_epoch} as primordial seat of {name}."]
            )
        ]
        self.sync_environment()

    def update_economic_production(
        self,
        cell_breakdowns: list[dict],
        world_map: ContinuousWorldMap | None = None
    ) -> None:
        self.claimed_cells_count = len(cell_breakdowns)
        total_grain = 0.0
        total_herd = 0.0
        total_marine = 0.0
        total_minerals = 0.0

        self.infrastructure_level = min(25.0, self.infrastructure_level + (self.population * 0.05))

        for c in cell_breakdowns:
            alt, hum, temp, veg = c["alt"], c["hum"], c["temp"], c["veg"]
            dist_coast = c["dist_coast"]

            if alt <= 0.0:
                depth_yield = 1.60 + min(1.20, abs(alt) * 3.5)
                naval_efficiency = (0.75 + 0.08 * math.log(1.0 + self.infrastructure_level))
                total_marine += depth_yield * naval_efficiency
            else:
                agri_dist = ((hum - 0.55) / 0.22) ** 2 + ((veg - 0.45) / 0.22) ** 2 + ((temp - 0.55) / 0.22) ** 2
                raw_arable = math.exp(-agri_dist) * max(0.0, 1.0 - alt * 1.3)
                total_grain += raw_arable * (0.80 + 0.45 * math.log(1.0 + self.infrastructure_level))

                pastoral_dist = ((veg - 0.30) / 0.18) ** 2 + ((hum - 0.38) / 0.20) ** 2
                total_herd += math.exp(-pastoral_dist) * max(0.0, 1.0 - alt * 0.8)

                if dist_coast < 0.14:
                    coastal_proximity = (0.14 - dist_coast) / 0.14
                    total_marine += (coastal_proximity * 1.80) * (0.80 + hum * 0.40)

                if alt > 0.30:
                    total_minerals += (alt - 0.30) * 2.5
                elif veg > 0.70:
                    total_minerals += 0.10

        n_cells = max(1, self.claimed_cells_count)
        g_val = total_grain / n_cells
        h_val = total_herd / n_cells
        m_val = total_marine / n_cells
        min_val = total_minerals / n_cells

        tot_raw = g_val + h_val + m_val + min_val or 1.0
        self.economy = EconomyProfile(
            grain=round((g_val / tot_raw) * 100.0, 1),
            livestock=round((h_val / tot_raw) * 100.0, 1),
            marine=round((m_val / tot_raw) * 100.0, 1),
            minerals=round((min_val / tot_raw) * 100.0, 1),
            total_yield=round(tot_raw, 3)
        )

        w = list(self.language.cultural_attention)
        tech_multiplier = (w[0] * 0.30 + w[3] * 0.35 + self.language.cultural_inertia * 0.35)
        food_calories = total_grain * 1.5 + total_marine * 1.7 + total_herd * 1.1
        land_carrying = 0.55 * tech_multiplier * math.log(1.0 + food_calories + min(1.2, total_minerals * 0.10))
        self.food_capacity = round(max(0.25, 0.40 + land_carrying), 3)
        self.food_deficit = round(max(0.0, self.population - self.food_capacity), 3)

        # 1. DAMPED PSYCHOLOGY (Using tanh to prevent Sociality & Dynamism explosion)
        pop_eff = math.tanh(max(0.0, self.population) / 1.8)
        deficit_ratio = (self.food_deficit / max(0.20, self.food_capacity)) if self.food_deficit > 0.0 else 0.0

        if deficit_ratio > 0.02:
            w[2] = max(0.30, w[2] - deficit_ratio * 0.04)  # Valence drops gradually
            w[4] = min(2.20, w[4] + deficit_ratio * 0.03)  # Dynamism rises
            w[3] = min(2.20, w[3] + deficit_ratio * 0.03)  # Potency rises
            w[5] = max(0.30, w[5] - deficit_ratio * 0.02)  # Sociality frays
        else:
            w[0] += (g_val * 0.04 + min_val * 0.03)
            w[1] += (h_val * 0.04)
            w[4] += (m_val * 0.05)
            w[5] += pop_eff * 0.015  # Bounded social scaling!
            w[6] += (m_val * 0.05 + h_val * 0.02)

        # 1B. THERMOSTATIC EQUILIBRIUM: 1% pull toward baseline 1.0 prevents permanent 2.50 pegging
        for idx in range(7):
            w[idx] += 0.010 * (1.0 - w[idx])

        tot_w = sum(w) or 7.0
        self.language.cultural_attention = [round(max(0.30, min(2.20, 7.0 * (v / tot_w))), 3) for v in w]

        # 2. SCALED MAINTENANCE ENGINE: Towns of 3k-10k sustain themselves
        ENTROPY_DECAY = 0.025
        for s in self.settlements:
            maintenance = min(0.06, (s.local_pop / 0.04) * 0.030)
            net_repair = maintenance - ENTROPY_DECAY
            # Walls can repair up from 0.0
            s.fortification_level = round(min(5.0, max(0.0, s.fortification_level + net_repair)), 2)
            s.local_infrastructure = round(min(15.0, max(0.5, s.local_infrastructure + net_repair * 0.5)), 2)

        # 3. CONURBATION (Merging adjacent cities)
        current_epoch = getattr(world_map, "current_epoch", self.founded_epoch)
        merge_dist = 0.035
        merged_any = True
        while merged_any and len(self.settlements) > 1:
            merged_any = False
            for i in range(len(self.settlements)):
                for j in range(i + 1, len(self.settlements)):
                    s1, s2 = self.settlements[i], self.settlements[j]
                    dist = math.sqrt((s1.x - s2.x)**2 + (s1.y - s2.y)**2)
                    if dist < merge_dist:
                        keeper = s1 if s1.local_pop >= s2.local_pop else s2
                        absorbed = s2 if keeper is s1 else s1
                        keeper.local_pop = round(keeper.local_pop + absorbed.local_pop, 3)
                        keeper.fortification_level = round(max(keeper.fortification_level, absorbed.fortification_level) + 0.20, 2)
                        keeper.history.append(f"Epoch {current_epoch}: Conurbated with neighboring {absorbed.name}.")
                        base_name = keeper.name.replace("Greater ", "").split()[0]
                        if keeper.local_pop >= 0.40:
                            keeper.name = f"Greater {base_name}"
                        self.settlements.remove(absorbed)
                        merged_any = True
                        break
                if merged_any:
                    break

        # 4. DOMESTIC URBANIZATION (Empires build secondary cities in outer valleys)
        has_economy = (self.economy.grain >= 10.0 or self.economy.marine >= 15.0 or self.economy.minerals >= 15.0)
        is_sedentary = (has_economy and self.infrastructure_level >= 0.7)

        # 1 city per ~25,000 citizens (allows realms of 60k-100k to have 2-4 cities)
        pop_per_city = 0.25 + 0.02 * self.infrastructure_level
        max_allowed_cities = min(8, 1 + int(self.population / pop_per_city)) if is_sedentary else 1

        if is_sedentary and len(self.settlements) < max_allowed_cities and cell_breakdowns:
            candidate_valleys = []
            for c in cell_breakdowns:
                if c["alt"] >= 0.06:
                    dist_to_cities = [math.hypot(s.x - c["gx"], s.y - c["gy"]) for s in self.settlements]
                    min_dist = min(dist_to_cities) if dist_to_cities else 1.0
                    
                    # Safe distance from existing towns (0.065 to 0.25)
                    if 0.065 <= min_dist <= 0.25:
                        agri_dist = ((c["hum"] - 0.55) / 0.22) ** 2 + ((c["veg"] - 0.45) / 0.22) ** 2
                        arable_score = math.exp(-agri_dist)
                        port_score = 1.2 if c["dist_coast"] < 0.08 else 0.0
                        mineral_score = (c["alt"] - 0.30) * 1.2 if c["alt"] > 0.30 else 0.0
                        candidate_valleys.append((arable_score + port_score + mineral_score, c["gx"], c["gy"]))

            if candidate_valleys:
                candidate_valleys.sort(key=lambda item: item[0], reverse=True)
                _, bx, by = candidate_valleys[0]
                
                # Any city with at least 4,000 citizens can sponsor a new town
                eligible_cities = [s for s in self.settlements if s.local_pop >= 0.04]
                if eligible_cities:
                    nearest_city = min(eligible_cities, key=lambda s: math.hypot(s.x - bx, s.y - by))
                    seed_pop = round(min(0.04, max(0.015, nearest_city.local_pop * 0.15)), 3)
                    nearest_city.local_pop = round(nearest_city.local_pop - seed_pop, 3)

                    town_number = len(self.settlements) + 1
                    new_city = SettlementNode(
                        name=f"{self.name} City-{town_number}",
                        x=bx, y=by, local_pop=seed_pop,
                        fortification_level=1.0,
                        local_infrastructure=1.0,
                        is_capital=False, is_permanent=True,
                        founder=self.name,
                        founded_epoch=current_epoch,
                        history=[f"Founded at Epoch {current_epoch} as domestic province of {self.name}."]
                    )
                    self.settlements.append(new_city)
                    if world_map is not None:
                        is_naval = self.terrain.sample_environment((nearest_city.x + bx) * 0.5, (nearest_city.y + by) * 0.5).altitude <= 0.0
                        world_map._ensure_road_edge(nearest_city, new_city, is_naval=is_naval, traffic_boost=0.30)
                        world_map.event_log.append(f"Urban Expansion: '{self.name}' established domestic city '{new_city.name}'!")

        # 5. Inter-City Migration
        if len(self.settlements) > 1:
            for s in self.settlements:
                local_capacity = 0.40 + 0.12 * s.local_infrastructure
                if s.local_pop > local_capacity:
                    surplus = (s.local_pop - local_capacity) * 0.12
                    destination = min(
                        (other for other in self.settlements if other != s),
                        key=lambda o: math.hypot(o.x - s.x, o.y - s.y)
                    )
                    s.local_pop = round(s.local_pop - surplus, 3)
                    destination.local_pop = round(destination.local_pop + surplus, 3)

        self.population = round(sum(s.local_pop for s in self.settlements), 3)

    def step_demographics(self, delta_time: float = 1.0) -> None:
        # GEOGRAPHICALLY DIVERSIFIED CITY GROWTH:
        # River valleys grow large; alpine mountain outposts stay small!
        attn = self.language.cultural_attention
        w_pot = attn[3]
        w_dyn = attn[4]

        infra_fertility = 0.18 * math.tanh(self.infrastructure_level / 5.0)
        base_fertility = (0.045 + infra_fertility) * (1.0 + 0.15 * (w_pot - 1.0) + 0.10 * (w_dyn - 1.0))
        deficit_ratio = (self.food_deficit / max(0.10, self.population)) if self.food_deficit > 0.0 else 0.0

        for s in self.settlements:
            s_env = self.terrain.sample_environment(s.x, s.y)
            # Local environmental fertility factor
            fert = math.exp(-(((s_env.humidity - 0.55)/0.22)**2 + ((s_env.vegetation - 0.45)/0.22)**2)) * max(0.1, 1.0 - s_env.altitude * 1.2)
            local_cap = 0.20 + 1.30 * fert + 0.12 * s.local_infrastructure

            # Frontier boom: New outposts with plenty of open land grow 60% faster!
            is_frontier_boom = (s.local_pop < local_cap * 0.40)
            fertility_multiplier = 1.60 if is_frontier_boom else 1.00

            d_local = base_fertility * fertility_multiplier * s.local_pop * (1.0 - (s.local_pop / max(0.15, local_cap)))
            
            if deficit_ratio > 0.0:
                d_local -= s.local_pop * min(0.20, deficit_ratio * 0.15)
            
            s.local_pop = round(max(0.002, s.local_pop + d_local * delta_time), 3)

        self.population = round(sum(s.local_pop for s in self.settlements), 3)

    def _calculate_food_and_territory(self, env: Environment, veg: float) -> None:
        attn = self.language.cultural_attention
        w_dyn, w_ext = attn[4], attn[6]
        base_radius = 0.05 + 0.07 * math.sqrt(max(0.01, self.population)) * (w_ext / 1.0)
        hunger_push = 1.0 + 0.50 * self.food_deficit
        altitude_choke = max(0.40, 1.0 - env.altitude * 0.50)
        self.territory_radius = round(base_radius * hunger_push * altitude_choke, 3)

    def step_migration(self) -> None:
        has_permanent_cities = any(getattr(s, "is_permanent", False) for s in self.settlements)
        if has_permanent_cities or self.infrastructure_level >= 1.0:
            return

        attn = self.language.cultural_attention
        w_dyn = attn[4]
        mobility = 0.012 * (w_dyn / 1.0) / (1.0 + self.language.cultural_inertia * 0.40)

        def evaluate_location(eval_x: float, eval_y: float) -> float:
            env = self.terrain.sample_environment(eval_x, eval_y)
            if env.altitude <= 0.06: return -9999.0
            pastoral_dist = ((env.vegetation - 0.30) / 0.18) ** 2 + ((env.humidity - 0.38) / 0.20) ** 2
            return math.exp(-pastoral_dist) * max(0.0, 1.0 - env.altitude * 0.8)

        current_val = evaluate_location(self.x, self.y)
        grad_x, grad_y = 0.0, 0.0
        probe_dist = 0.02

        for dx, dy in [(-probe_dist, 0), (probe_dist, 0), (0, -probe_dist), (0, probe_dist)]:
            val = evaluate_location(max(0.02, min(0.98, self.x + dx)), max(0.02, min(0.98, self.y + dy)))
            diff = val - current_val
            if dx != 0: grad_x += diff * (1.0 if dx > 0 else -1.0)
            if dy != 0: grad_y += diff * (1.0 if dy > 0 else -1.0)

        self.vx = 0.80 * self.vx + 0.20 * grad_x * mobility
        self.vy = 0.80 * self.vy + 0.20 * grad_y * mobility

        next_x = max(0.02, min(0.98, self.x + self.vx))
        next_y = max(0.02, min(0.98, self.y + self.vy))
        
        next_env = self.terrain.sample_environment(next_x, next_y)
        if next_env.altitude >= 0.06:
            self.x = next_x
            self.y = next_y
            if self.settlements and not self.settlements[0].is_permanent:
                self.settlements[0].x = next_x
                self.settlements[0].y = next_y
        else:
            self.vx *= -0.5
            self.vy *= -0.5

    @property
    def environment(self) -> Environment:
        return self.terrain.sample_environment(self.x, self.y)

    def sync_environment(self, epoch: int = 0) -> None:
        env = self.terrain.sample_environment(self.x, self.y, epoch=epoch)
        self.language.environment = env
        self.language.population = self.population
        
        veg = env.emergent_vegetation(
            population=self.population, 
            cultural_attention=self.language.cultural_attention
        )
        
        target_profile = env.generate_profile(
            name=f"{self.name} Profile",
            initial_head_directionality=self.language.profile.head_directionality,
            cultural_attention=self.language.cultural_attention,
            population=self.population,
            vegetation=veg,
            parent_bias=self.language.profile.aesthetic_bias,
        )
        
        curr = self.language.profile
        alpha = 0.05
        
        smoothed_complexity = round((1.0 - alpha) * curr.syllable_complexity + alpha * target_profile.syllable_complexity, 3)
        aridity = max(0.0, 1.0 - env.humidity)

        w_ext = self.language.cultural_attention[6]
        w_soc = self.language.cultural_attention[5]
        h_drift = 0.008 * (w_ext - w_soc)
        smoothed_head_dir = max(0.10, min(0.90, curr.head_directionality + h_drift))

        w_pot = self.language.cultural_attention[3]
        target_alignment = "ergative_absolutive" if w_pot > (w_soc * 1.15) else "nominative_accusative"

        self.language.profile = curr._replace(
            max_transition_cost=round((1.0 - alpha) * curr.max_transition_cost + alpha * target_profile.max_transition_cost, 2),
            apocope_rate=round((1.0 - alpha) * curr.apocope_rate + alpha * target_profile.apocope_rate, 3),
            syncope_rate=round((1.0 - alpha) * curr.syncope_rate + alpha * target_profile.syncope_rate, 3),
            synthesis_index=round((1.0 - alpha) * curr.synthesis_index + alpha * target_profile.synthesis_index, 3),
            syllable_complexity=smoothed_complexity,
            head_directionality=round(smoothed_head_dir, 3),
            alignment=target_alignment,
            double_stop_strategy="spirantization" if (env.altitude * 0.6 + aridity * 0.4) > 0.45 else "gemination",
            acoustic_roughness=round((1.0 - alpha) * curr.acoustic_roughness + alpha * target_profile.acoustic_roughness, 3),
            d_min_vowel=round((1.0 - alpha) * curr.d_min_vowel + alpha * target_profile.d_min_vowel, 3),
            d_min_consonant=round((1.0 - alpha) * curr.d_min_consonant + alpha * target_profile.d_min_consonant, 3),
        )
        self._calculate_food_and_territory(env, veg)


class ContinuousWorldMap:
    def __init__(self, name: str = "Pangaea Field", terrain: TerrainField | None = None):
        self.name = name
        self.terrain = terrain or TerrainField()
        self.civilizations: dict[str, Civilization] = {}
        self.dormant_classical_languages: dict[str, Language] = {}
        self.abandoned_ruins: list[SettlementNode] = []
        self.active_expeditions: list[dict] = []
        self.road_network: list[RoadEdge] = []
        self.active_truces: dict[tuple[str, str], int] = {}
        self.current_epoch: int = 0
        self._friction_cache: dict[tuple[float, float, float, float], float] = {}
        self.active_trade_routes: list[tuple[str, str]] = []
        self.active_conflicts: list[tuple[str, str]] = []
        self.event_log: list[str] = []
        self.territory_grid: dict[tuple[int, int], str] = {}
        self.density_grid: dict[tuple[int, int], float] = {}
        self._precompute_static_terrain_env()

    def _precompute_static_terrain_env(self, resolution: int = 50):
        self._env_cache: list[list[dict]] = []
        for r in range(resolution):
            row = []
            gy = r / float(resolution - 1)
            for c in range(resolution):
                gx = c / float(resolution - 1)
                env = self.terrain.sample_environment(gx, gy, epoch=0)
                row.append({
                    "alt": env.altitude,
                    "hum": env.humidity,
                    "temp": env.temperature,
                    "climax_veg": env.emergent_vegetation(population=0.0)
                })
            self._env_cache.append(row)

    def spawn_civilization(
        self,
        name: str,
        language: Language,
        x: float,
        y: float,
        population: float = 0.50,
        lineage: str | None = None,
        founded_epoch: int = 0,
    ) -> Civilization:
        civ = Civilization(name, language, x, y, self.terrain, population, lineage=lineage, founded_epoch=founded_epoch)
        self.civilizations[name] = civ
        return civ

    def _ensure_road_edge(self, s1: SettlementNode, s2: SettlementNode, is_naval: bool = False, traffic_boost: float = 0.25):
        if not is_naval and math.hypot(s1.x - s2.x, s1.y - s2.y) > 0.22:
            return

        for edge in self.road_network:
            if (edge.city_a == s1.name and edge.city_b == s2.name) or (edge.city_a == s2.name and edge.city_b == s1.name):
                edge.level = min(3.0, round(edge.level + traffic_boost, 2))
                return

        self.road_network.append(
            RoadEdge(
                city_a=s1.name, city_b=s2.name,
                x0=s1.x, y0=s1.y, x1=s2.x, y1=s2.y,
                level=1.0 + traffic_boost, is_naval=is_naval
            )
        )

    def step_epoch(
        self,
        reduction_strength: float = 0.75,
        enable_neologisms: bool = True,
        parallel: bool = False,
    ) -> None:
        self.current_epoch += 1
        self.active_trade_routes.clear()
        self.active_conflicts.clear()
        civ_list = list(self.civilizations.values())
        res = len(self._env_cache)

        # 1. Demographics, Extinction & True Abandonment (No more fake desertions!)
        dead_civs = []
        for civ in civ_list:
            if civ.population <= 0.005 or not civ.settlements:
                dead_civs.append(civ.name)
                continue
            civ.sync_environment(epoch=self.current_epoch)
            civ.step_demographics()
            civ.step_migration()

            retained_settlements = []
            for s in civ.settlements:
                # A city ONLY deserts if it has literally zero inhabitants (< 100 people)!
                if s.local_pop <= 0.001:
                    s.history.append(f"Epoch {self.current_epoch}: Deserted due to zero population; sovereignty lost from {civ.name}.")
                    self.event_log.append(f"Desertion: '{s.name}' was abandoned! Sovereignty lost.")
                    s._deserted_epoch = self.current_epoch
                    self.abandoned_ruins.append(s)
                else:
                    retained_settlements.append(s)
            civ.settlements = retained_settlements

        # When a civilization goes extinct, its cities STAY ON THE MAP AS ANCIENT RUINS!
        for dname in dead_civs:
            dead_civ = self.civilizations.get(dname)
            if dead_civ:
                for s in dead_civ.settlements:
                    s.history.append(f"Epoch {self.current_epoch}: Realm {dname} collapsed; city left as ancient ruins.")
                    s._deserted_epoch = self.current_epoch
                    self.abandoned_ruins.append(s)
                del self.civilizations[dname]
                self.event_log.append(f"Epoch {self.current_epoch}: Civilization '{dname}' collapsed; its monuments remain as ruins.")
        civ_list = [c for c in self.civilizations.values() if c.population > 0.005 and c.settlements]

        # 1B. Road Entropy Decay & Orphaned Road Pruning
        living_city_names = {s.name for civ in civ_list for s in civ.settlements}
        retained_roads = []
        for edge in self.road_network:
            a_alive = (edge.city_a in living_city_names)
            b_alive = (edge.city_b in living_city_names)

            if not a_alive and not b_alive:
                continue
            elif not a_alive or not b_alive:
                edge.level = round(edge.level - 0.12, 3)
            else:
                edge.level = round(edge.level - 0.035, 3)

            if edge.level >= 0.10:
                retained_roads.append(edge)
        self.road_network = retained_roads

        # =====================================================================
        # 2. LOGARITHMIC POWER FIELD (Guaranteed Seeding for every civ!)
        # =====================================================================
        grid_claims: dict[tuple[int, int], str] = {}
        grid_density: dict[tuple[int, int], float] = {}
        civ_claimed_cells_data: dict[str, list[dict]] = {c.name: [] for c in civ_list}

        active_city_list: list[tuple[SettlementNode, Civilization]] = []
        for civ in civ_list:
            if not civ.settlements: continue
            primary_s = max(civ.settlements, key=lambda s: s.local_pop)
            active_city_list.append((primary_s, civ))
            for s in civ.settlements:
                if s != primary_s and s.local_pop >= 0.015:
                    active_city_list.append((s, civ))

        if active_city_list:
            city_x = np.array([item[0].x for item in active_city_list], dtype=np.float32)
            city_y = np.array([item[0].y for item in active_city_list], dtype=np.float32)
            
            city_power = np.array([
                1.0 + 0.35 * math.log(1.0 + item[0].local_pop * 10.0) * (item[1].language.cultural_attention[6] / 1.0)
                for item in active_city_list
            ], dtype=np.float32)

            y_coords = np.linspace(0.0, 1.0, res, dtype=np.float32)
            x_coords = np.linspace(0.0, 1.0, res, dtype=np.float32)
            grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')

            friction_arr = np.zeros((res, res), dtype=np.float32)
            for r in range(res):
                for c in range(res):
                    alt = self._env_cache[r][c]["alt"]
                    if alt <= 0.0:
                        friction_arr[r, c] = 3.5 + min(12.0, abs(alt) * 30.0)
                    else:
                        friction_arr[r, c] = 1.0 + 5.5 * (max(0.0, alt - 0.25) ** 2)

            dx = grid_x[None, :, :] - city_x[:, None, None]
            dy = grid_y[None, :, :] - city_y[:, None, None]
            dist = np.sqrt(dx*dx + dy*dy)

            trench_mask = ((grid_x[None, :, :] >= 0.80) & (city_x[:, None, None] < 0.75)) | \
                          ((grid_x[None, :, :] < 0.75) & (city_x[:, None, None] >= 0.80))
            effective_friction = np.where(trench_mask, 999.0, friction_arr[None, :, :])

            weighted_cost = (dist * effective_friction) / (city_power[:, None, None])

            min_dist_to_any = np.min(dist, axis=0)
            closest_city_idx = np.argmin(dist, axis=0)
            winning_indices = np.argmin(weighted_cost, axis=0)

            # Inalienable home basin: d <= 0.06 is strictly owned by the local city
            in_home_basin = (min_dist_to_any <= 0.06)
            winning_indices = np.where(in_home_basin, closest_city_idx, winning_indices)

            max_reach = 0.24

            for r in range(res):
                for c in range(res):
                    if min_dist_to_any[r, c] > max_reach:
                        continue

                    best_k = winning_indices[r, c]
                    winning_node, winning_civ = active_city_list[best_k]
                    cname = winning_civ.name

                    grid_claims[(r, c)] = cname
                    gx, gy = c / float(res - 1), r / float(res - 1)
                    cell_env = self.terrain.sample_environment(gx, gy, epoch=self.current_epoch)
                    dist_coast = min(gx, abs(gx - 0.78), 1.0 - gx)

                    civ_claimed_cells_data[cname].append({
                        "gx": gx, "gy": gy,
                        "alt": cell_env.altitude, "hum": cell_env.humidity,
                        "temp": cell_env.temperature, "veg": cell_env.vegetation,
                        "dist_coast": dist_coast,
                    })

        for (r, c), cname in grid_claims.items():
            civ_obj = self.civilizations.get(cname)
            if not civ_obj:
                continue
            gx, gy = c / float(res - 1), r / float(res - 1)
            
            total_urban_density = sum(
                s.local_pop * math.exp(-math.sqrt((gx - s.x)**2 + (gy - s.y)**2) * 55.0)
                for s in civ_obj.settlements
            )
            rural_baseline = min(0.025, civ_obj.population * 0.006)
            combined_density = min(1.0, total_urban_density + rural_baseline)
            grid_density[(r, c)] = round(combined_density, 3)

        self.territory_grid = grid_claims
        self.density_grid = grid_density

        for civ in civ_list:
            civ.update_economic_production(civ_claimed_cells_data.get(civ.name, []), world_map=self)

        # 3. Reoccupation of Abandoned Ruins (With 3-epoch cooldown!)
        if self.abandoned_ruins:
            retained_ruins = []
            for ruin in self.abandoned_ruins:
                # Cooldown prevents same-tick desertion-reclaim ping-pong
                if (self.current_epoch - getattr(ruin, "_deserted_epoch", 0)) < 3:
                    retained_ruins.append(ruin)
                    continue

                r_c = max(0, min(res - 1, int(round(ruin.x * (res - 1)))))
                r_r = max(0, min(res - 1, int(round(ruin.y * (res - 1)))))
                owner_name = grid_claims.get((r_r, r_c))
                reoccupied = False

                if owner_name:
                    new_owner = self.civilizations.get(owner_name)
                    if new_owner and new_owner.population >= 0.25:
                        ruin.name = f"{ruin.name.split()[0]} ({new_owner.name} Reclaimed)"
                        ruin.local_pop = 0.03
                        ruin.fortification_level = 1.0
                        ruin.history.append(f"Epoch {self.current_epoch}: Reoccupied and restored by {new_owner.name}.")
                        new_owner.settlements.append(ruin)
                        self.event_log.append(f"Restoration: '{new_owner.name}' reoccupied the ancient ruins of '{ruin.name}'!")
                        reoccupied = True

                if not reoccupied:
                    ruin_age = getattr(ruin, "_ruin_decay_age", 0) + 1
                    ruin._ruin_decay_age = ruin_age
                    if ruin_age < 30:
                        retained_ruins.append(ruin)
                    else:
                        self.event_log.append(f"Oblivion: The ancient stone ruins of '{ruin.name}' crumbled into the landscape forever.")
            self.abandoned_ruins = retained_ruins

        # 4. City-to-City Trade & Conflict
        for civ in civ_list:
            if len(civ.settlements) > 1 and civ.infrastructure_level >= 1.0:
                for idx, s in enumerate(civ.settlements[1:], start=1):
                    sister_city = min(
                        civ.settlements[:idx],
                        key=lambda o: math.hypot(o.x - s.x, o.y - s.y)
                    )
                    self._ensure_road_edge(s, sister_city, is_naval=False, traffic_boost=0.15)

        # 4A. BORDER NEIGHBORS (Dry land only!)
        border_neighbors: dict[str, set[str]] = {c.name: set() for c in civ_list}
        for (r, c), owner in grid_claims.items():
            if self._env_cache[r][c]["alt"] <= 0.0:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < res and 0 <= nc < res and (nr, nc) in grid_claims:
                    if self._env_cache[nr][nc]["alt"] > 0.0:
                        adj_owner = grid_claims[(nr, nc)]
                        if adj_owner != owner:
                            border_neighbors[owner].add(adj_owner)

        # 4B. CLEAN LINEAGE CHECK: Sister colonies are NEVER foreign threats!
        for civ in civ_list:
            foreign_threats = []
            for other_name in border_neighbors.get(civ.name, set()):
                other_civ = self.civilizations.get(other_name)
                if other_civ and other_civ.lineage != civ.lineage:
                    foreign_threats.append(other_name)

            if len(foreign_threats) >= 2:
                w = civ.language.cultural_attention
                w[3] = min(2.20, round(w[3] + 0.025 * len(foreign_threats), 3))
                w[2] = max(0.30, round(w[2] - 0.020, 3))
                w[5] = min(2.20, round(w[5] + 0.015, 3))
                tot = sum(w) or 7.0
                civ.language.cultural_attention = [round(7.0 * (v / tot), 3) for v in w]

        total_world_pop = sum(c.population for c in civ_list) or 1.0

        # 4C. BILATERAL TRADE & CONFLICT
        n = len(civ_list)
        for i in range(n):
            for j in range(i + 1, n):
                civ_a, civ_b = civ_list[i], civ_list[j]
                if civ_a.population <= 0.005 or civ_b.population <= 0.005: continue
                if not civ_a.settlements or not civ_b.settlements: continue

                closest_pair = min(
                    ((s_a, s_b) for s_a in civ_a.settlements for s_b in civ_b.settlements),
                    key=lambda pair: math.sqrt((pair[0].x - pair[1].x)**2 + (pair[0].y - pair[1].y)**2)
                )
                city_a, city_b = closest_pair
                dist_cities = math.sqrt((city_a.x - city_b.x)**2 + (city_a.y - city_b.y)**2)

                shares_border = (civ_b.name in border_neighbors.get(civ_a.name, set()))
                has_direct_route = any(
                    (e.city_a in [s.name for s in civ_a.settlements] and e.city_b in [s.name for s in civ_b.settlements]) or
                    (e.city_b in [s.name for s in civ_a.settlements] and e.city_a in [s.name for s in civ_b.settlements])
                    for e in self.road_network
                )

                if not shares_border and not has_direct_route and dist_cities > 0.14:
                    continue

                truce_key = tuple(sorted([civ_a.name, civ_b.name]))
                has_active_truce = (self.active_truces.get(truce_key, 0) > self.current_epoch)

                is_kin = (civ_a.lineage == civ_b.lineage)
                pot_a = civ_a.language.cultural_attention[3] * civ_a.population
                pot_b = civ_b.language.cultural_attention[3] * civ_b.population
                power_ratio = pot_a / max(0.01, pot_b)

                attacker = civ_a if power_ratio >= 1.0 else civ_b
                defender = civ_b if attacker is civ_a else civ_a

                w_soc_att = attacker.language.cultural_attention[5]
                w_pot_att = attacker.language.cultural_attention[3]
                w_dyn_att = attacker.language.cultural_attention[4]
                w_val_att = attacker.language.cultural_attention[2]

                cultural_belligerence = (w_pot_att * 1.30 + w_dyn_att * 0.85) / max(0.5, (w_soc_att * 0.90 + w_val_att * 0.70))
                kinship_bonus = 0.65 if is_kin else 0.0
                war_threshold = round(max(1.15, 1.45 + kinship_bonus - 0.20 * (cultural_belligerence - 1.0)), 2)

                defender_age = self.current_epoch - defender.founded_epoch
                is_predatory = (cultural_belligerence >= 1.35 and power_ratio > war_threshold and not is_kin)

                # War is blocked by armistice, directing energy into peaceful trade!
                has_conflict = not has_active_truce and (power_ratio > war_threshold or power_ratio < (1.0 / war_threshold)) and (
                    civ_a.food_deficit > 0.010 or civ_b.food_deficit > 0.010 or (is_predatory and defender_age >= 20)
                )

                if has_conflict:
                    self.active_conflicts.append((civ_a.name, civ_b.name))
                    victor = attacker
                    conquered = defender
                    if not conquered.settlements or not victor.settlements:
                        continue

                    # 1. SIEGE LOGISTICS & SIEGECRAFT
                    target_town = min(conquered.settlements, key=lambda s: math.sqrt((s.x - victor.x)**2 + (s.y - victor.y)**2))
                    dist_campaign = math.sqrt((target_town.x - victor.x)**2 + (target_town.y - victor.y)**2)
                    
                    mid_x = (victor.x + target_town.x) * 0.5
                    mid_y = (victor.y + target_town.y) * 0.5
                    is_amphibious = (self.terrain.sample_environment(mid_x, mid_y).altitude <= 0.0) or (target_town.x >= 0.80 and victor.x < 0.75)

                    if is_amphibious and victor.economy.marine < 15.0:
                        continue

                    logistics_efficiency = math.exp(-dist_campaign / 0.35)
                    if is_amphibious:
                        naval_transport = min(1.0, victor.economy.marine / 40.0)
                        logistics_efficiency *= (0.20 * naval_transport)

                    # High infrastructure brings siege engines (catapults, rams)
                    siege_engineering = 1.0 + 0.18 * math.log(1.0 + victor.infrastructure_level)
                    mobilization_rate = 0.10 + 0.04 * (victor.language.cultural_attention[3] / 1.0)
                    assault_power = (victor.population * mobilization_rate) * victor.language.cultural_attention[3] * 1.40 * siege_engineering * logistics_efficiency

                    island_defense_multiplier = 2.2 if is_amphibious else 1.0
                    is_last_city = (len(conquered.settlements) == 1)
                    capital_mult = 1.8 if is_last_city else 1.4
                    
                    base_defense = target_town.local_pop * conquered.language.cultural_attention[3] * max(0.20, target_town.fortification_level) * capital_mult * island_defense_multiplier

                    coalition_bonus = 0.0
                    if (victor.population / total_world_pop) >= 0.35:
                        coalition_allies = [c for c in civ_list if c != victor and c != conquered and c.population >= 0.20]
                        raw_coalition = sum(c.population * 0.12 * c.language.cultural_attention[3] for c in coalition_allies)
                        coalition_bonus = min(base_defense * 1.5, raw_coalition)

                    city_defense = base_defense + coalition_bonus

                    # Direct breach or surrender when defenses crumble (<= 0.20)
                    if assault_power >= city_defense or target_town.fortification_level <= 0.20:
                        conquered.settlements.remove(target_town)
                        
                        nearest_victor_city = min(
                            victor.settlements,
                            key=lambda s: math.hypot(s.x - target_town.x, s.y - target_town.y)
                        )
                        self._ensure_road_edge(nearest_victor_city, target_town, is_naval=is_amphibious, traffic_boost=0.50)

                        if is_last_city:
                            target_town.name = f"{conquered.name} ({victor.name} Annex)"
                            target_town.fortification_level = max(0.8, target_town.fortification_level * 0.65)
                            target_town.local_pop = round(target_town.local_pop * 0.75, 3)
                            target_town.history.append(f"Epoch {self.current_epoch}: Capital fallen to {victor.name}; realm annexed.")

                            alpha = min(0.48, max(0.06, conquered.population / max(0.01, victor.population + conquered.population)))
                            w_vic, w_conq = victor.language.cultural_attention, conquered.language.cultural_attention
                            blended_w = [(1.0 - alpha) * w_vic[k] + alpha * w_conq[k] for k in range(7)]
                            tot = sum(blended_w) or 7.0
                            victor.language.cultural_attention = [round(7.0 * (val / tot), 3) for val in blended_w]

                            pv, pc = victor.language.profile, conquered.language.profile
                            victor.language.profile = pv._replace(
                                head_directionality=round((1.0 - alpha) * pv.head_directionality + alpha * pc.head_directionality, 3),
                                synthesis_index=round((1.0 - alpha) * pv.synthesis_index + alpha * pc.synthesis_index, 3),
                                syllable_complexity=round((1.0 - alpha) * pv.syllable_complexity + alpha * pc.syllable_complexity, 3),
                            )

                            core_vocab = sorted([w for w in conquered.language.words.values() if w.category == LexicalCategory.CONTENT_OPEN], key=lambda w: w.usage_frequency, reverse=True)
                            for loan in core_vocab[:int(round(8 + 35 * alpha))]:
                                victor.language.borrow_word(loan, source_language_name=conquered.name, current_generation=self.current_epoch)

                            victor.language.rebuild_inventory()
                            victor.language._cached_grammar_paradigm = None
                            self.dormant_classical_languages[conquered.name] = conquered.language

                            conquered.population = 0.0
                            self.event_log.append(f"HISTORIC ANNEXATION: '{victor.name}' conquered '{conquered.name}' (Substrate: {alpha*100:.1f}%)!")
                        else:
                            target_town.name = f"{target_town.name.split()[0]} ({victor.name} Province)"
                            target_town.fortification_level = max(0.6, target_town.fortification_level * 0.50)
                            target_town.local_pop = round(target_town.local_pop * 0.70, 3)
                            target_town.history.append(f"Epoch {self.current_epoch}: Sacked and annexed by {victor.name}.")
                            self.event_log.append(f"Conquest: '{victor.name}' breached and captured '{target_town.name}'!")

                        victor.settlements.append(target_town)
                        victor.infrastructure_level = min(25.0, victor.infrastructure_level + 0.50)
                        
                        # Armistice follows decisive capture
                        self.active_truces[truce_key] = self.current_epoch + 25
                    else:
                        # Structural wall damage
                        damage = round(min(target_town.fortification_level, 0.40 + 0.05 * victor.language.cultural_attention[3]), 2)
                        target_town.fortification_level = round(max(0.0, target_town.fortification_level - damage), 2)
                        # Cannot drop below 1,000 citizens from siege attrition
                        target_town.local_pop = round(max(0.010, target_town.local_pop * 0.90), 3)

                        if target_town.fortification_level == 0.0:
                            self.event_log.append(f"WALLS BREACHED: Garrison at '{target_town.name}' held, but stone walls crumbled into rubble!")
                        else:
                            self.event_log.append(f"Siege: Walls of '{target_town.name}' battered down to Level {target_town.fortification_level}!")

                    for p in (victor, conquered):
                        if p.population > 0.005:
                            w = p.language.cultural_attention
                            w[3] = round(max(0.40, min(2.20, w[3] * 0.98)), 3)
                            w[2] = round(min(2.00, max(0.30, w[2] + 0.015)), 3)
                            tot = sum(w) or 7.0
                            p.language.cultural_attention = [round(7.0 * (v / tot), 3) for v in w]

                else:
                    # =============================================================
                    # PEACEFUL COMMERCE (Truce or Peace channels Trade!)
                    # =============================================================
                    self.active_trade_routes.append((civ_a.name, civ_b.name))
                    mid_x, mid_y = (city_a.x + city_b.x) * 0.5, (city_a.y + city_b.y) * 0.5
                    is_crossing_water = self.terrain.sample_environment(mid_x, mid_y).altitude <= 0.0
                    self._ensure_road_edge(city_a, city_b, is_naval=is_crossing_water, traffic_boost=0.20)

                    intensity = 1.0 / (1.0 + math.exp(-4.0 * (0.25 - dist_cities) * 6.0))
                    civ_a.language.set_contact(civ_b.language, intensity)

                    infra_diff = civ_a.infrastructure_level - civ_b.infrastructure_level
                    if abs(infra_diff) > 0.5:
                        transfer = intensity * 0.04 * abs(infra_diff)
                        if infra_diff > 0: civ_b.infrastructure_level = min(25.0, round(civ_b.infrastructure_level + transfer, 2))
                        else: civ_a.infrastructure_level = min(25.0, round(civ_a.infrastructure_level + transfer, 2))

                    for partner in (civ_a, civ_b):
                        w = partner.language.cultural_attention
                        w[4] = min(2.20, round(w[4] + 0.015, 3))
                        w[5] = min(2.20, round(w[5] + 0.015, 3))
                        w[2] = min(2.00, round(w[2] + 0.010, 3))
                        tot = sum(w) or 7.0
                        partner.language.cultural_attention = [round(7.0 * (v / tot), 3) for v in w]

                    if random.random() < (intensity * 0.25):
                        cand_pool = [w for w in civ_b.language.words.values() if w.category == LexicalCategory.CONTENT_OPEN]
                        if cand_pool:
                            target_w = random.choice(cand_pool)
                            p_calque = 1.0 / (1.0 + math.exp(-6.0 * (civ_a.language.cultural_inertia - 1.05)))
                            if random.random() < p_calque:
                                civ_a.language.calque_word(target_w, source_language_name=civ_b.name, current_generation=self.current_epoch)
                            else:
                                civ_a.language.borrow_word(target_w, source_language_name=civ_b.name, current_generation=self.current_epoch)

        # 5. Dormant Classical Liturgical Revival
        if self.dormant_classical_languages and random.random() < 0.15:
            scholar_civ = random.choice(civ_list)
            for dead_name, dead_lang in self.dormant_classical_languages.items():
                if dead_name.split("-")[0] in scholar_civ.lineage or random.random() < 0.08:
                    sacred_cands = [w for w in dead_lang.words.values() if w.vector.valence >= 0.70 or w.vector.potency >= 0.75]
                    if sacred_cands:
                        relic = random.choice(sacred_cands)
                        scholar_civ.language.borrow_word(relic, source_language_name=f"Classical {dead_name}", current_generation=self.current_epoch)
                        self.event_log.append(f"Scholarship: '{scholar_civ.name}' revived archaic root from extinct '{dead_name}'!")
                        break

        # =====================================================================
        # 6. PIONEER CONVOYS: OUTWARD TO UNCLAIMED NEUTRAL WILDERNESS
        # =====================================================================
        remaining_expeditions = []
        for exp in self.active_expeditions:
            exp["progress"] += exp["speed"]
            if exp["progress"] >= 1.0:
                tx, ty = exp["target"]
                col_pop = exp["pop"]
                d_name = exp["name"]
                parent_lineage = exp["lineage"]
                parent_civ = self.civilizations.get(exp["civ"])

                # If overseas or remote (dist >= 0.25): FORKS a new civilization!
                # If close to home: adds a DOMESTIC CITY to the parent empire!
                is_remote_speciation = exp.get("is_naval", False) or exp.get("dist", 0.0) >= 0.25

                if is_remote_speciation or parent_civ is None:
                    d_lang = exp["lang"]
                    self.spawn_civilization(
                        name=d_name, language=d_lang,
                        x=tx, y=ty,
                        population=col_pop, lineage=parent_lineage,
                        founded_epoch=self.current_epoch
                    )
                    self.event_log.append(
                        f"Colony Established: {exp['civ']} pioneers founded new realm '{d_name}' with {int(col_pop*100000):,} citizens!"
                    )
                else:
                    # Stays as an allied provincial city of the mother empire!
                    new_town = SettlementNode(
                        name=f"{parent_civ.name} Frontier City",
                        x=tx, y=ty, local_pop=col_pop,
                        fortification_level=1.0, local_infrastructure=1.0,
                        is_capital=False, is_permanent=True,
                        founder=parent_civ.name, founded_epoch=self.current_epoch,
                        history=[f"Founded at Epoch {self.current_epoch} by pioneer expedition from {exp['civ']}."]
                    )
                    parent_civ.settlements.append(new_town)
                    parent_civ.population = round(sum(s.local_pop for s in parent_civ.settlements), 3)
                    self.event_log.append(
                        f"Colony Established: {exp['civ']} pioneers expanded the realm, founding '{new_town.name}'!"
                    )
            else:
                remaining_expeditions.append(exp)
        self.active_expeditions = remaining_expeditions

        if (len(self.civilizations) + len(self.active_expeditions)) < 40 and (self.current_epoch % 12 == 0):
            for civ in civ_list:
                if civ.population <= 0.005 or not civ.settlements: continue
                attn = civ.language.cultural_attention
                w_dyn = attn[4]
                w_ext = attn[6]
                saturation = civ.population / max(0.20, civ.food_capacity)

                if saturation >= 0.35 and random.random() < (0.40 * (w_dyn / 1.0) * (w_ext / 1.0)):
                    viable_cities = [s for s in civ.settlements if s.local_pop >= 0.04]
                    if not viable_cities: continue
                    departure_city = max(viable_cities, key=lambda s: s.local_pop)

                    # PIONEERS SEEK UNCLAIMED FRONTIER OR REMOTE COASTS/ISLANDS
                    pioneer_candidates = []
                    for r_i in range(res):
                        for c_i in range(res):
                            cell_owner = grid_claims.get((r_i, c_i))
                            # Never settle inside another rival foreign empire
                            if cell_owner is not None and cell_owner != civ.name:
                                continue

                            cand_x = c_i / float(res - 1)
                            cand_y = r_i / float(res - 1)
                            c_env = self.terrain.sample_environment(cand_x, cand_y, epoch=self.current_epoch)

                            if c_env.altitude >= 0.06:
                                dist_to_dep = math.hypot(departure_city.x - cand_x, departure_city.y - cand_y)
                                
                                # Search frontier: distance between 0.08 and 0.28
                                if 0.08 <= dist_to_dep <= 0.28:
                                    min_dist_cities = min(
                                        (math.hypot(s.x - cand_x, s.y - cand_y) for c in self.civilizations.values() for s in c.settlements),
                                        default=1.0
                                    )
                                    # Must not crowd existing cities (>= 0.075)
                                    if min_dist_cities >= 0.075:
                                        agri_dist = ((c_env.humidity - 0.55) / 0.22) ** 2 + ((c_env.vegetation - 0.45) / 0.22) ** 2
                                        fertility = math.exp(-agri_dist)
                                        # Bonus for uncolonized / offshore land
                                        island_bonus = 1.5 if (cand_x >= 0.80 or cell_owner is None) else 1.0
                                        pioneer_candidates.append((fertility * island_bonus, cand_x, cand_y, c_env, dist_to_dep))

                    if pioneer_candidates:
                        pioneer_candidates.sort(key=lambda item: item[0], reverse=True)
                        _, target_x, target_y, target_env, travel_dist = pioneer_candidates[0]

                        pioneer_seedling = round(min(0.06, max(0.015, departure_city.local_pop * 0.15)), 3)
                        departure_city.local_pop = round(departure_city.local_pop - pioneer_seedling, 3)
                        civ.population = round(sum(s.local_pop for s in civ.settlements), 3)

                        d_count = sum(1 for c in self.civilizations.values() if c.lineage == civ.lineage)
                        d_name = f"{civ.lineage}-{d_count + 1}"
                        d_lang = civ.language.fork(d_name, environment=target_env)

                        mid_voyage_x, mid_voyage_y = (departure_city.x + target_x) * 0.5, (departure_city.y + target_y) * 0.5
                        is_island_colony = self.terrain.sample_environment(mid_voyage_x, mid_voyage_y).altitude <= 0.0

                        self.active_expeditions.append({
                            "name": d_name, "civ": civ.name, "lineage": civ.lineage,
                            "origin": [departure_city.x, departure_city.y], "target": [target_x, target_y],
                            "pop": pioneer_seedling, "lang": d_lang, "progress": 0.0,
                            "speed": 0.25, "is_naval": is_island_colony, "dist": travel_dist
                        })

                        self._ensure_road_edge(departure_city, SettlementNode(d_name, target_x, target_y), is_naval=is_island_colony, traffic_boost=0.15)

                        mode_str = "navigating sea lanes" if is_island_colony else "marching along frontier roads"
                        self.event_log.append(
                            f"Expedition: '{civ.name}' pioneers departed from {departure_city.name} "
                            f"and are {mode_str} toward ({target_x:.2f}, {target_y:.2f}) with {int(pioneer_seedling*100000):,} settlers!"
                        )
                        break

        # 7. Natural Urban Speciation
        for civ in list(self.civilizations.values()):
            if len(civ.settlements) >= 3 and civ.population >= 0.80:
                primary_city = civ.settlements[0]
                for s in list(civ.settlements[1:]):
                    dist_to_capital = math.hypot(s.x - primary_city.x, s.y - primary_city.y)
                    if dist_to_capital >= 0.32 and s.local_pop >= 0.15 and random.random() < 0.04:
                        rebel_name = f"{civ.lineage}-{len(self.civilizations) + 1}"
                        rebel_env = self.terrain.sample_environment(s.x, s.y, epoch=self.current_epoch)
                        rebel_lang = civ.language.fork(rebel_name, environment=rebel_env)

                        civ.settlements.remove(s)
                        civ.population = round(sum(other.local_pop for other in civ.settlements), 3)

                        rebel_civ = self.spawn_civilization(
                            name=rebel_name, language=rebel_lang,
                            x=s.x, y=s.y, population=s.local_pop,
                            lineage=civ.lineage, founded_epoch=self.current_epoch
                        )
                        rebel_civ.infrastructure_level = max(1.0, s.local_infrastructure)
                        s.name = f"{rebel_name} (Seat)"
                        s.is_capital = True
                        rebel_civ.settlements = [s]

                        self.event_log.append(
                            f"Linguistic Speciation: Distant city '{s.name}' diverged in dialect and declared independence as '{rebel_name}'!"
                        )
                        break

        # 8. Language Evolution
        for civ in list(self.civilizations.values()):
            if civ.population > 0.005 and civ.settlements:
                civ.language.evolve(
                    reduction_strength=reduction_strength,
                    enable_neologisms=enable_neologisms,
                    current_generation=self.current_epoch,
                )