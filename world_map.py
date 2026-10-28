"""
world_map.py — Continuous 2D Physical Geography, Orographic Climate, and Speciation.
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

        if self.altitude <= 0.0:
            return 0.0

        temp_effective = max(0.0, self.temperature)
        pet = max(0.08, (temp_effective ** 1.3) * 1.40)
        aridity_index = self.humidity / pet

        if aridity_index < 0.12:
            climax_veg = 0.02 + 0.15 * aridity_index
        elif aridity_index < 0.40:
            climax_veg = 0.15 + 0.50 * (aridity_index - 0.12)
        elif aridity_index < 0.85:
            climax_veg = 0.35 + 0.65 * (aridity_index - 0.40)
        else:
            climax_veg = min(0.95, 0.65 + 0.30 * min(1.0, aridity_index - 0.85))

        treeline_factor = math.exp(-3.50 * (max(0.0, self.altitude - 0.35) ** 2))
        thermal_growth = 1.0 / (1.0 + math.exp(-8.0 * (self.temperature - 0.22)))
        climax_veg *= treeline_factor * thermal_growth * (1.0 + 0.15 * self.season_phase)

        attn = cultural_attention or [1.0] * 7
        deforestation_rate = attn[0] * 0.60 + attn[3] * 0.40
        depletion = math.exp(-0.55 * deforestation_rate * (max(0.0, population) ** 1.25))

        return round(max(0.01, min(0.98, climax_veg * depletion)), 3)

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

        # 1. Barometric Density and Viscous-Thermal Air Dissipation
        rho_air = math.exp(-0.90 * max(0.0, self.altitude)) * math.exp(-0.25 * (self.temperature - 0.50))
        
        # 2. Acoustic Adaptation: Foliage scattering + warm vapor acoustic absorption
        canopy_absorption = veg * 0.60 + self.humidity * 0.40
        thermal_viscous_damping = math.tanh(max(0.0, self.temperature) * 1.5) * self.humidity
        total_acoustic_damping = 0.55 * canopy_absorption + 0.45 * thermal_viscous_damping

        # 3. Alpine & Arid Channel Opening
        aridity = max(0.0, 1.0 - self.humidity)
        cold_steppe = max(0.0, 0.45 - self.temperature) * 1.5
        altitude_drive = max(0.0, self.altitude) * 1.4

        # Continuous Complexity Driver: maps into [0.12, 0.85]
        s_comp_driver = (altitude_drive + aridity * 0.80 + cold_steppe * 0.70) - (total_acoustic_damping * 2.20)
        syllable_complexity = round(0.12 + (0.73 / (1.0 + math.exp(-4.5 * s_comp_driver))), 3)
        max_transition_cost = round(1.80 + (3.60 * syllable_complexity), 2)

        # 4. Phonemic Separation Wells
        channel_noise = (self.ambient_noise * 0.35 + aridity * 0.30) / max(0.20, rho_air)
        d_min_vowel = round(max(0.18, min(0.38, 0.22 + 0.16 * channel_noise)), 3)
        d_min_consonant = round(0.20 + (0.15 / (1.0 + math.exp(-4.5 * (rho_air - 0.60)))), 3)

        p_centralize = 1.0 / (1.0 + math.exp(-8.0 * (pop - 0.60)))
        vowel_reduction_mode = "none" if syllable_complexity < 0.28 else ("centralize" if random.random() < p_centralize else "inventory_snap")

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
            (total_acoustic_damping * 0.75) - 0.25
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
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.wind_dir = np.array([0.85, -0.15], dtype=np.float32)
        self.wind_dir /= np.linalg.norm(self.wind_dir)

    def elevation_at(self, x: float, y: float, epoch: int = 0) -> float:
        c1 = math.sin(x * 3.8 + 0.4) * math.cos(y * 3.2 - 0.2)
        c2 = 0.50 * math.sin(x * 7.5 - 1.2) * math.sin(y * 6.8 + 0.8)
        c3 = 0.25 * math.cos(x * 14.0 + y * 12.0)
        continental_plate = 0.42 * (c1 + c2 + c3)

        spine_dist = math.sqrt(((x - 0.62) * 1.7) ** 2 + ((y - 0.70) * 1.3) ** 2)
        mountain_chain = 0.88 * math.exp(-(spine_dist ** 2) / 0.028)

        trench = 0.40 * math.exp(-((x - 0.76) ** 2) / 0.003)
        island_dist = math.sqrt(((x - 0.84) * 2.2) ** 2 + ((y - 0.48) * 1.1) ** 2)
        island_arc = 0.36 * math.exp(-(island_dist ** 2) / 0.020)

        west_shelf = max(0.0, (0.16 - x) * 2.8)
        east_abyss = max(0.0, (x - 0.90) * 3.6)

        base_z = continental_plate + mountain_chain - trench + island_arc - west_shelf - east_abyss
        sea_level_pulse = 0.05 * math.sin(2.0 * math.pi * epoch / 200.0)
        return float(np.clip(base_z - sea_level_pulse, -1.0, 1.2))

    def sample_environment(self, x: float, y: float, epoch: int = 0) -> Environment:
        cx = max(0.0, min(1.0, float(x)))
        cy = max(0.0, min(1.0, float(y)))
        altitude = self.elevation_at(cx, cy, epoch=epoch)

        season_phase = math.sin(2.0 * math.pi * (epoch % 12) / 12.0)
        pluvial_pulse = 0.12 * math.sin(2.0 * math.pi * epoch / 180.0)

        base_temp = 0.96 - (cy * 0.58) + (season_phase * 0.08)
        lapse_cooling = max(0.0, altitude) * 0.55
        temperature = round(base_temp - lapse_cooling, 3)

        dist_to_coast = min(cx, abs(cx - 0.76), 1.0 - cx)
        ocean_moisture = 1.0 - min(0.85, dist_to_coast * 1.45)

        eps = 0.02
        z_upwind = self.elevation_at(cx - self.wind_dir[0] * eps, cy - self.wind_dir[1] * eps, epoch=epoch)
        slope_lift = (altitude - z_upwind) / eps

        if slope_lift > 0.0:
            orographic_effect = min(0.35, slope_lift * 0.06)
        else:
            orographic_effect = max(-0.55, slope_lift * 0.12)

        hadley_subsidence = 0.22 * math.exp(-((cy - 0.32) ** 2) / 0.018)
        raw_humidity = ocean_moisture + orographic_effect - hadley_subsidence + pluvial_pulse
        min_hum = 0.55 if altitude <= 0.0 else (0.40 if dist_to_coast < 0.08 else 0.04)
        humidity = round(max(min_hum, min(0.98, raw_humidity)), 3)

        ambient_noise = round(min(0.99, max(0.0, altitude) * 0.60 + max(0.0, 0.12 - dist_to_coast) * 2.5), 3)

        env = Environment(
            temperature=temperature,
            humidity=humidity,
            altitude=round(altitude, 3),
            ambient_noise=ambient_noise,
            season_phase=round(season_phase, 2),
        )
        if altitude <= 0.0:
            env.vegetation = 0.0

        return env


class Civilization:
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
        self.terrain = terrain

        # Clamp and guarantee terrestrial placement (never underwater)
        cx = max(0.04, min(0.96, float(x)))
        cy = max(0.04, min(0.96, float(y)))
        
        if self.terrain.elevation_at(cx, cy, epoch=founded_epoch) <= 0.04:
            best_d = 999.0
            best_pt = (cx, cy)
            for r_step in range(-12, 13):
                for c_step in range(-12, 13):
                    tx = max(0.04, min(0.96, cx + c_step * 0.02))
                    ty = max(0.04, min(0.96, cy + r_step * 0.02))
                    if self.terrain.elevation_at(tx, ty, epoch=founded_epoch) >= 0.08:
                        d = math.hypot(tx - cx, ty - cy)
                        if d < best_d:
                            best_d = d
                            best_pt = (tx, ty)
            cx, cy = best_pt

        self.x = cx
        self.y = cy
        self.vx = 0.0
        self.vy = 0.0
        self.population = float(population)
        self.lineage = lineage or name
        self.founded_epoch = founded_epoch
        self.food_capacity = 0.65
        self.food_deficit = 0.00
        self.territory_radius = 0.08
        self.claimed_cells_count = 16
        self.infrastructure_level = 0.8
        self.economy = EconomyProfile()

        self.settlements: list[SettlementNode] = [
            SettlementNode(
                name=f"{name}",
                x=self.x,
                y=self.y,
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
                naval_eff = 0.75 + 0.08 * math.log(1.0 + self.infrastructure_level)
                total_marine += depth_yield * naval_eff
            else:
                agri_dist = ((hum - 0.55) / 0.22) ** 2 + ((veg - 0.45) / 0.22) ** 2 + ((temp - 0.55) / 0.22) ** 2
                raw_arable = math.exp(-agri_dist) * max(0.0, 1.0 - alt * 1.3)
                total_grain += raw_arable * (0.80 + 0.45 * math.log(1.0 + self.infrastructure_level))

                pastoral_dist = ((veg - 0.30) / 0.18) ** 2 + ((hum - 0.38) / 0.20) ** 2
                total_herd += math.exp(-pastoral_dist) * max(0.0, 1.0 - alt * 0.8)

                if dist_coast < 0.14:
                    coastal_prox = (0.14 - dist_coast) / 0.14
                    total_marine += (coastal_prox * 1.80) * (0.80 + hum * 0.40)

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
            total_yield=round(tot_raw, 3),
        )

        w = list(self.language.cultural_attention)
        tech_multiplier = (w[0] * 0.30 + w[3] * 0.35 + self.language.cultural_inertia * 0.35)
        food_calories = total_grain * 1.5 + total_marine * 1.7 + total_herd * 1.1
        land_carrying = 0.55 * tech_multiplier * math.log(1.0 + food_calories + min(1.2, total_minerals * 0.10))
        self.food_capacity = round(max(0.25, 0.40 + land_carrying), 3)
        self.food_deficit = round(max(0.0, self.population - self.food_capacity), 3)

        pop_eff = math.tanh(max(0.0, self.population) / 1.8)
        deficit_ratio = (self.food_deficit / max(0.20, self.food_capacity)) if self.food_deficit > 0.0 else 0.0

        if deficit_ratio > 0.02:
            w[2] = max(0.30, w[2] - deficit_ratio * 0.04)
            w[4] = min(2.20, w[4] + deficit_ratio * 0.03)
            w[3] = min(2.20, w[3] + deficit_ratio * 0.03)
            w[5] = max(0.30, w[5] - deficit_ratio * 0.02)
        else:
            w[0] += (g_val * 0.04 + min_val * 0.03)
            w[1] += (h_val * 0.04)
            w[4] += (m_val * 0.05)
            w[5] += pop_eff * 0.015
            w[6] += (m_val * 0.05 + h_val * 0.02)

        for idx in range(7):
            w[idx] += 0.010 * (1.0 - w[idx])

        tot_w = sum(w) or 7.0
        self.language.cultural_attention = [round(max(0.30, min(2.20, 7.0 * (v / tot_w))), 3) for v in w]

        for s in self.settlements:
            maintenance = min(0.06, (s.local_pop / 0.04) * 0.030)
            net_repair = maintenance - 0.025
            s.fortification_level = round(min(5.0, max(0.0, s.fortification_level + net_repair)), 2)
            s.local_infrastructure = round(min(15.0, max(0.5, s.local_infrastructure + net_repair * 0.5)), 2)

        current_epoch = getattr(world_map, "current_epoch", self.founded_epoch)
        retained_settlements = []
        absorbed_names = set()

        for i, s1 in enumerate(self.settlements):
            if s1.name in absorbed_names:
                continue
            for s2 in self.settlements[i + 1:]:
                if s2.name in absorbed_names:
                    continue
                if math.hypot(s1.x - s2.x, s1.y - s2.y) < 0.035:
                    if s1.local_pop >= s2.local_pop:
                        s1.local_pop = round(s1.local_pop + s2.local_pop, 3)
                        s1.fortification_level = round(max(s1.fortification_level, s2.fortification_level) + 0.20, 2)
                        s1.history.append(f"Epoch {current_epoch}: Absorbed {s2.name}.")
                        absorbed_names.add(s2.name)
                    else:
                        s2.local_pop = round(s2.local_pop + s1.local_pop, 3)
                        s2.fortification_level = round(max(s1.fortification_level, s2.fortification_level) + 0.20, 2)
                        s2.history.append(f"Epoch {current_epoch}: Absorbed {s1.name}.")
                        absorbed_names.add(s1.name)
                        break

            if s1.name not in absorbed_names:
                retained_settlements.append(s1)

        self.settlements = retained_settlements

        has_economy = (self.economy.grain >= 10.0 or self.economy.marine >= 15.0 or self.economy.minerals >= 15.0)
        is_sedentary = (has_economy and self.infrastructure_level >= 0.7)
        max_cities = min(8, 1 + int(self.population / (0.25 + 0.02 * self.infrastructure_level))) if is_sedentary else 1

        if is_sedentary and len(self.settlements) < max_cities and cell_breakdowns:
            candidate_valleys = []
            for c in cell_breakdowns:
                if c["alt"] >= 0.06:
                    dist_to_existing = min(math.hypot(s.x - c["gx"], s.y - c["gy"]) for s in self.settlements)
                    if 0.065 <= dist_to_existing <= 0.25:
                        agri_d = ((c["hum"] - 0.55) / 0.22) ** 2 + ((c["veg"] - 0.45) / 0.22) ** 2
                        arable_score = math.exp(-agri_d)
                        port_score = 1.2 if c["dist_coast"] < 0.08 else 0.0
                        candidate_valleys.append((arable_score + port_score, c["gx"], c["gy"]))

            if candidate_valleys:
                candidate_valleys.sort(key=lambda item: item[0], reverse=True)
                _, bx, by = candidate_valleys[0]
                eligible_cities = [s for s in self.settlements if s.local_pop >= 0.04]
                if eligible_cities:
                    sponsor = min(eligible_cities, key=lambda s: math.hypot(s.x - bx, s.y - by))
                    seed_pop = round(min(0.04, max(0.015, sponsor.local_pop * 0.15)), 3)
                    sponsor.local_pop = round(sponsor.local_pop - seed_pop, 3)

                    new_city = SettlementNode(
                        name=f"{self.name} City-{len(self.settlements) + 1}",
                        x=bx,
                        y=by,
                        local_pop=seed_pop,
                        fortification_level=1.0,
                        local_infrastructure=1.0,
                        is_capital=False,
                        is_permanent=True,
                        founder=self.name,
                        founded_epoch=current_epoch,
                        history=[f"Founded at Epoch {current_epoch} as domestic province."]
                    )
                    self.settlements.append(new_city)
                    if world_map is not None:
                        is_naval = self.terrain.elevation_at((sponsor.x + bx) * 0.5, (sponsor.y + by) * 0.5) <= 0.0
                        world_map._ensure_road_edge(sponsor, new_city, is_naval=is_naval, traffic_boost=0.30)
                        world_map.event_log.append(f"Urban Expansion: '{self.name}' founded '{new_city.name}'.")

        self.population = round(sum(s.local_pop for s in self.settlements), 3)

    def step_demographics(self, delta_time: float = 1.0) -> None:
        attn = self.language.cultural_attention
        base_fertility = (0.045 + 0.18 * math.tanh(self.infrastructure_level / 5.0)) * (1.0 + 0.15 * (attn[3] - 1.0))
        deficit_ratio = (self.food_deficit / max(0.10, self.population)) if self.food_deficit > 0.0 else 0.0

        for s in self.settlements:
            s_env = self.terrain.sample_environment(s.x, s.y)
            fert = math.exp(-(((s_env.humidity - 0.55) / 0.22) ** 2 + ((s_env.vegetation - 0.45) / 0.22) ** 2)) * max(0.1, 1.0 - s_env.altitude * 1.2)
            local_cap = 0.20 + 1.30 * fert + 0.12 * s.local_infrastructure
            fertility_mult = 1.60 if s.local_pop < local_cap * 0.40 else 1.00

            d_local = base_fertility * fertility_mult * s.local_pop * (1.0 - (s.local_pop / max(0.15, local_cap)))
            if deficit_ratio > 0.0:
                d_local -= s.local_pop * min(0.20, deficit_ratio * 0.15)

            s.local_pop = round(max(0.002, s.local_pop + d_local * delta_time), 3)

        self.population = round(sum(s.local_pop for s in self.settlements), 3)

    def step_migration(self) -> None:
        if any(s.is_permanent for s in self.settlements) or self.infrastructure_level >= 1.0:
            return

        w_dyn = self.language.cultural_attention[4]
        mobility = 0.012 * (w_dyn / 1.0) / (1.0 + self.language.cultural_inertia * 0.40)

        def eval_loc(ex: float, ey: float) -> float:
            env = self.terrain.sample_environment(ex, ey)
            if env.altitude <= 0.06:
                return -999.0
            p_dist = ((env.vegetation - 0.30) / 0.18) ** 2 + ((env.humidity - 0.38) / 0.20) ** 2
            return math.exp(-p_dist) * max(0.0, 1.0 - env.altitude * 0.8)

        cur_val = eval_loc(self.x, self.y)
        grad_x = eval_loc(min(0.98, self.x + 0.02), self.y) - cur_val
        grad_y = eval_loc(self.x, min(0.98, self.y + 0.02)) - cur_val

        self.vx = 0.80 * self.vx + 0.20 * grad_x * mobility
        self.vy = 0.80 * self.vy + 0.20 * grad_y * mobility

        nx = max(0.02, min(0.98, self.x + self.vx))
        ny = max(0.02, min(0.98, self.y + self.vy))

        if self.terrain.elevation_at(nx, ny) >= 0.06:
            self.x, self.y = nx, ny
            if self.settlements and not self.settlements[0].is_permanent:
                self.settlements[0].x, self.settlements[0].y = nx, ny

    @property
    def environment(self) -> Environment:
        return self.terrain.sample_environment(self.x, self.y)

    def sync_environment(self, epoch: int = 0) -> None:
        env = self.terrain.sample_environment(self.x, self.y, epoch=epoch)
        self.language.environment = env
        self.language.population = self.population

        veg = env.emergent_vegetation(population=self.population, cultural_attention=self.language.cultural_attention)
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
        aridity = max(0.0, 1.0 - env.humidity)

        # Measured synthesis index directly reflects lexicon morphemic state
        measured_s_idx = self.language.synthesis_index

        # Damped Restorative Word Order Equilibrium: prevents saturation at 0.90
        # h oscillates around cultural attention balance with a restoring potential toward stability
        h_force = 0.012 * (self.language.cultural_attention[6] - self.language.cultural_attention[5])
        h_restore = -0.04 * (curr.head_directionality - 0.50)
        smoothed_head_dir = round(max(0.15, min(0.85, curr.head_directionality + h_force + h_restore)), 3)

        self.language.profile = curr._replace(
            max_transition_cost=round((1.0 - alpha) * curr.max_transition_cost + alpha * target_profile.max_transition_cost, 2),
            apocope_rate=round((1.0 - alpha) * curr.apocope_rate + alpha * target_profile.apocope_rate, 3),
            syncope_rate=round((1.0 - alpha) * curr.syncope_rate + alpha * target_profile.syncope_rate, 3),
            synthesis_index=measured_s_idx,
            syllable_complexity=round((1.0 - alpha) * curr.syllable_complexity + alpha * target_profile.syllable_complexity, 3),
            head_directionality=smoothed_head_dir,
            alignment="ergative_absolutive" if self.language.cultural_attention[3] > (self.language.cultural_attention[5] * 1.15) else "nominative_accusative",
            double_stop_strategy="spirantization" if (env.altitude * 0.6 + aridity * 0.4) > 0.45 else "gemination",
            acoustic_roughness=round((1.0 - alpha) * curr.acoustic_roughness + alpha * target_profile.acoustic_roughness, 3),
            d_min_vowel=round((1.0 - alpha) * curr.d_min_vowel + alpha * target_profile.d_min_vowel, 3),
            d_min_consonant=round((1.0 - alpha) * curr.d_min_consonant + alpha * target_profile.d_min_consonant, 3),
        )

        base_radius = 0.05 + 0.07 * math.sqrt(max(0.01, self.population)) * (self.language.cultural_attention[6] / 1.0)
        self.territory_radius = round(base_radius * (1.0 + 0.50 * self.food_deficit) * max(0.40, 1.0 - env.altitude * 0.50), 3)


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
        self.active_trade_routes: list[tuple[str, str]] = []
        self.active_conflicts: list[tuple[str, str]] = []
        self.event_log: list[str] = []
        self.territory_grid: dict[tuple[int, int], str] = {}
        self.density_grid: dict[tuple[int, int], float] = {}

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
        res = 50

        dead_civ_names = []
        for civ in list(self.civilizations.values()):
            if civ.population <= 0.005 or not civ.settlements:
                dead_civ_names.append(civ.name)
                continue

            civ.sync_environment(epoch=self.current_epoch)
            civ.step_demographics()
            civ.step_migration()

            surviving_settlements = []
            for s in civ.settlements:
                if s.local_pop <= 0.001:
                    s.history.append(f"Epoch {self.current_epoch}: Abandoned.")
                    self.event_log.append(f"Desertion: '{s.name}' was abandoned.")
                    setattr(s, "_deserted_epoch", self.current_epoch)
                    self.abandoned_ruins.append(s)
                else:
                    surviving_settlements.append(s)
            civ.settlements = surviving_settlements

        for dname in dead_civ_names:
            dead_civ = self.civilizations.pop(dname, None)
            if dead_civ:
                for s in dead_civ.settlements:
                    setattr(s, "_deserted_epoch", self.current_epoch)
                    self.abandoned_ruins.append(s)
                self.event_log.append(f"Epoch {self.current_epoch}: Civilization '{dname}' collapsed.")

        civ_list = [c for c in self.civilizations.values() if c.population > 0.005 and c.settlements]

        grid_claims: dict[tuple[int, int], str] = {}
        grid_density: dict[tuple[int, int], float] = {}
        civ_claimed_cells: dict[str, list[dict]] = {c.name: [] for c in civ_list}

        active_cities: list[tuple[SettlementNode, Civilization]] = [
            (s, c) for c in civ_list for s in c.settlements if s.local_pop >= 0.015 or s.is_capital
        ]

        if active_cities:
            city_coords = np.array([[item[0].x, item[0].y] for item in active_cities], dtype=np.float32)
            city_powers = np.array([
                1.0 + 0.35 * math.log(1.0 + item[0].local_pop * 10.0) * (item[1].language.cultural_attention[6] / 1.0)
                for item in active_cities
            ], dtype=np.float32)

            y_grid, x_grid = np.meshgrid(
                np.linspace(0.0, 1.0, res, dtype=np.float32),
                np.linspace(0.0, 1.0, res, dtype=np.float32),
                indexing='ij'
            )

            elev_grid = np.array([
                [self.terrain.elevation_at(x_grid[r, c], y_grid[r, c], epoch=self.current_epoch) for c in range(res)]
                for r in range(res)
            ], dtype=np.float32)

            water_mask = elev_grid <= 0.0
            friction = np.where(
                water_mask,
                3.5 + np.clip(-elev_grid * 30.0, 0.0, 12.0),
                1.0 + 5.5 * (np.maximum(0.0, elev_grid - 0.25) ** 2)
            )

            dx = x_grid[None, :, :] - city_coords[:, 0, None, None]
            dy = y_grid[None, :, :] - city_coords[:, 1, None, None]
            dists = np.sqrt(dx * dx + dy * dy)

            costs = (dists * friction[None, :, :]) / city_powers[:, None, None]
            min_dist = np.min(dists, axis=0)
            winner_idx = np.argmin(costs, axis=0)

            home_mask = min_dist <= 0.06
            closest_idx = np.argmin(dists, axis=0)
            winner_idx = np.where(home_mask, closest_idx, winner_idx)

            city_pops = np.array([item[0].local_pop for item in active_cities], dtype=np.float32)
            urban_density_all = np.sum(city_pops[:, None, None] * np.exp(-dists * 55.0), axis=0)

            for r in range(res):
                for c in range(res):
                    if min_dist[r, c] > 0.24:
                        continue
                    winning_node, winning_civ = active_cities[winner_idx[r, c]]
                    cname = winning_civ.name
                    grid_claims[(r, c)] = cname

                    gx, gy = float(x_grid[r, c]), float(y_grid[r, c])
                    cell_env = self.terrain.sample_environment(gx, gy, epoch=self.current_epoch)
                    dist_c = min(gx, abs(gx - 0.76), 1.0 - gx)

                    civ_claimed_cells[cname].append({
                        "gx": gx, "gy": gy,
                        "alt": cell_env.altitude, "hum": cell_env.humidity,
                        "temp": cell_env.temperature, "veg": cell_env.vegetation,
                        "dist_coast": dist_c,
                    })

                    dens = float(urban_density_all[r, c]) + min(0.025, winning_civ.population * 0.006)
                    grid_density[(r, c)] = round(min(1.0, dens), 3)

        self.territory_grid = grid_claims
        self.density_grid = grid_density

        for civ in civ_list:
            civ.update_economic_production(civ_claimed_cells.get(civ.name, []), world_map=self)

        border_neighbors: dict[str, set[str]] = {c.name: set() for c in civ_list}
        for (r, c), owner in grid_claims.items():
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in grid_claims:
                    adj = grid_claims[(nr, nc)]
                    if adj != owner:
                        border_neighbors[owner].add(adj)

        n_civs = len(civ_list)
        for i in range(n_civs):
            for j in range(i + 1, n_civs):
                ca, cb = civ_list[i], civ_list[j]
                if not ca.settlements or not cb.settlements:
                    continue

                closest_pair = min(
                    ((sa, sb) for sa in ca.settlements for sb in cb.settlements),
                    key=lambda p: math.hypot(p[0].x - p[1].x, p[0].y - p[1].y)
                )
                city_a, city_b = closest_pair
                dist_ab = math.hypot(city_a.x - city_b.x, city_a.y - city_b.y)

                shares_border = (cb.name in border_neighbors.get(ca.name, set()))
                if not shares_border and dist_ab > 0.16:
                    continue

                truce_key = tuple(sorted([ca.name, cb.name]))
                has_truce = (self.active_truces.get(truce_key, 0) > self.current_epoch)

                pot_a = ca.language.cultural_attention[3] * ca.population
                pot_b = cb.language.cultural_attention[3] * cb.population
                p_ratio = pot_a / max(0.01, pot_b)

                attacker, defender = (ca, cb) if p_ratio >= 1.0 else (cb, ca)
                is_kin = (ca.lineage == cb.lineage)
                war_thresh = 1.45 + (0.65 if is_kin else 0.0)

                has_conflict = not has_truce and (p_ratio > war_thresh or p_ratio < (1.0 / war_thresh)) and (
                    ca.food_deficit > 0.01 or cb.food_deficit > 0.01
                )

                if has_conflict and attacker.settlements and defender.settlements:
                    self.active_conflicts.append((ca.name, cb.name))
                    target = min(defender.settlements, key=lambda s: math.hypot(s.x - attacker.x, s.y - attacker.y))
                    dist_camp = math.hypot(target.x - attacker.x, target.y - attacker.y)

                    logistics = math.exp(-dist_camp / 0.35)
                    assault = (attacker.population * 0.12) * attacker.language.cultural_attention[3] * logistics * 1.4
                    defense = target.local_pop * defender.language.cultural_attention[3] * max(0.20, target.fortification_level) * (1.8 if len(defender.settlements) == 1 else 1.4)

                    if assault >= defense or target.fortification_level <= 0.20:
                        defender.settlements.remove(target)
                        attacker.settlements.append(target)
                        target.name = f"{target.name.split()[0]} ({attacker.name})"
                        target.fortification_level = max(0.6, target.fortification_level * 0.60)
                        target.history.append(f"Epoch {self.current_epoch}: Captured by {attacker.name}.")
                        self.active_truces[truce_key] = self.current_epoch + 25

                        if not defender.settlements:
                            defender.population = 0.0
                            self.dormant_classical_languages[defender.name] = defender.language
                            self.event_log.append(f"HISTORIC CONQUEST: '{attacker.name}' annexed '{defender.name}'.")
                        else:
                            self.event_log.append(f"Siege: '{attacker.name}' breached and captured '{target.name}'.")
                    else:
                        target.fortification_level = round(max(0.0, target.fortification_level - 0.35), 2)
                        target.local_pop = round(max(0.01, target.local_pop * 0.92), 3)

                elif not has_conflict:
                    self.active_trade_routes.append((ca.name, cb.name))
                    mid_z = self.terrain.elevation_at((city_a.x + city_b.x) * 0.5, (city_a.y + city_b.y) * 0.5)
                    self._ensure_road_edge(city_a, city_b, is_naval=(mid_z <= 0.0), traffic_boost=0.20)
                    intensity = 1.0 / (1.0 + math.exp(-4.0 * (0.25 - dist_ab) * 6.0))
                    ca.language.set_contact(cb.language, intensity)

                    if random.random() < (intensity * 0.20):
                        pool = [w for w in cb.language.words.values() if w.category == LexicalCategory.CONTENT_OPEN]
                        if pool:
                            ca.language.borrow_word(random.choice(pool), source_language_name=cb.name, current_generation=self.current_epoch)

        remaining_exp = []
        for exp in self.active_expeditions:
            exp["progress"] += exp["speed"]
            if exp["progress"] >= 1.0:
                tx, ty = exp["target"]
                col_pop = exp["pop"]
                d_name = exp["name"]
                parent_civ = self.civilizations.get(exp["civ"])

                if exp.get("is_naval", False) or exp.get("dist", 0.0) >= 0.25 or parent_civ is None:
                    d_lang = exp["lang"]
                    self.spawn_civilization(
                        name=d_name, language=d_lang, x=tx, y=ty,
                        population=col_pop, lineage=exp["lineage"],
                        founded_epoch=self.current_epoch
                    )
                    self.event_log.append(f"Speciation: {exp['civ']} pioneers founded new realm '{d_name}'.")
                else:
                    new_town = SettlementNode(
                        name=f"{parent_civ.name} Outpost",
                        x=tx, y=ty, local_pop=col_pop,
                        fortification_level=1.0, local_infrastructure=1.0,
                        is_capital=False, is_permanent=True,
                        founder=parent_civ.name, founded_epoch=self.current_epoch,
                        history=[f"Founded by {exp['civ']} pioneers."]
                    )
                    parent_civ.settlements.append(new_town)
                    parent_civ.population = round(sum(s.local_pop for s in parent_civ.settlements), 3)
            else:
                remaining_exp.append(exp)
        self.active_expeditions = remaining_exp

        if (len(self.civilizations) + len(self.active_expeditions)) < 35 and (self.current_epoch % 10 == 0):
            for civ in civ_list:
                if civ.population < 0.35 or (civ.population / max(0.20, civ.food_capacity)) < 0.40:
                    continue

                sponsor = max(civ.settlements, key=lambda s: s.local_pop)
                if sponsor.local_pop < 0.05:
                    continue

                unclaimed_r, unclaimed_c = np.where(
                    np.array([[grid_claims.get((r, c)) is None for c in range(res)] for r in range(res)])
                )
                if len(unclaimed_r) == 0:
                    continue

                coords_unclaimed = np.column_stack((unclaimed_c / (res - 1.0), unclaimed_r / (res - 1.0)))
                dists_to_sponsor = np.hypot(coords_unclaimed[:, 0] - sponsor.x, coords_unclaimed[:, 1] - sponsor.y)
                valid_indices = np.where((dists_to_sponsor >= 0.08) & (dists_to_sponsor <= 0.28))[0]

                if len(valid_indices) > 0:
                    candidate_scores = []
                    for idx_val in valid_indices:
                        cand_x = float(coords_unclaimed[idx_val, 0])
                        cand_y = float(coords_unclaimed[idx_val, 1])
                        c_env = self.terrain.sample_environment(cand_x, cand_y, epoch=self.current_epoch)
                        
                        if c_env.altitude >= 0.06:
                            fertility = math.exp(-(((c_env.humidity - 0.55) / 0.22) ** 2 + ((c_env.vegetation - 0.45) / 0.22) ** 2))
                            
                            # Physical ray-march: verify if travel path crosses marine water (altitude <= 0.0)
                            t_samples = np.linspace(0.2, 0.8, 4)
                            path_altitudes = [
                                self.terrain.elevation_at(sponsor.x + t * (cand_x - sponsor.x), sponsor.y + t * (cand_y - sponsor.y), epoch=self.current_epoch)
                                for t in t_samples
                            ]
                            is_overseas = any(z <= 0.0 for z in path_altitudes)
                            marine_boost = 1.45 if is_overseas else 1.0
                            
                            candidate_scores.append((idx_val, fertility * marine_boost, cand_x, cand_y, c_env, is_overseas))

                    if candidate_scores:
                        candidate_scores.sort(key=lambda item: item[1], reverse=True)
                        top_picks = candidate_scores[:min(4, len(candidate_scores))]
                        pick_idx, _, tx, ty, t_env, is_naval_route = random.choice(top_picks)

                        p_pop = round(min(0.05, sponsor.local_pop * 0.15), 3)
                        sponsor.local_pop = round(sponsor.local_pop - p_pop, 3)
                        d_name = f"{civ.lineage}-{len(self.civilizations) + 1}"
                        d_lang = civ.language.fork(d_name, environment=t_env)

                        self.active_expeditions.append({
                            "name": d_name, "civ": civ.name, "lineage": civ.lineage,
                            "origin": [sponsor.x, sponsor.y], "target": [tx, ty],
                            "pop": p_pop, "lang": d_lang, "progress": 0.0,
                            "speed": 0.25, "is_naval": is_naval_route, "dist": float(dists_to_sponsor[pick_idx]),
                        })
                        break

        for civ in civ_list:
            if civ.population > 0.005 and civ.settlements:
                civ.language.evolve(
                    reduction_strength=reduction_strength,
                    enable_neologisms=enable_neologisms,
                    current_generation=self.current_epoch,
                    infrastructure=civ.infrastructure_level,
                )