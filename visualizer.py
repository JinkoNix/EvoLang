"""
visualizer.py — Browser Visualizer Server for EvoLang.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from language import Language
from world_map import ContinuousWorldMap, Environment, TerrainField
from grammar import GrammarEngine
from translator import SemanticTranslator
from energy import BiomechanicalCostModel
from syllable import Tone


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


class WorldSimulationState:
    def __init__(self):
        self.terrain = TerrainField()
        self.world = ContinuousWorldMap(name="Pangaea Field", terrain=self.terrain)
        self.event_log: list[str] = []
        self._init_default_world()
        self._precompute_terrain_grid()

    def _precompute_terrain_grid(self, resolution: int = 50):
        self.terrain_grid = []
        for row in range(resolution):
            y = row / float(resolution - 1)
            grid_row = []
            for col in range(resolution):
                x = col / float(resolution - 1)
                env = self.terrain.sample_environment(x, y)
                climax_veg = env.emergent_vegetation(population=0.0)
                grid_row.append({
                    "alt": round(env.altitude, 2),
                    "temp": round(env.temperature, 2),
                    "hum": round(env.humidity, 2),
                    "veg": round(climax_veg, 2),
                })
            self.terrain_grid.append(grid_row)

    def _init_default_world(self):
        best_cradle = (0.50, 0.50)
        best_score = -1.0

        for r in range(10, 40):
            for c in range(10, 40):
                cx, cy = c / 50.0, r / 50.0
                env = self.terrain.sample_environment(cx, cy)
                if 0.12 <= env.altitude <= 0.32:
                    veg = env.emergent_vegetation(population=0.0)
                    score = veg * math.exp(-(((env.temperature - 0.55) / 0.25) ** 2 + ((env.humidity - 0.55) / 0.25) ** 2))
                    if score > best_score:
                        best_score = score
                        best_cradle = (cx, cy)

        cradle_x, cradle_y = best_cradle
        env_cradle = self.terrain.sample_environment(cradle_x, cradle_y)
        prof_cradle = env_cradle.generate_profile(
            name="Proto-Cradle Profile",
            initial_head_directionality=0.50,
            population=0.035
        )
        lang_cradle = Language("Proto-Cradle", profile=prof_cradle, environment=env_cradle, population=0.035)
        lang_cradle.populate_default_phonemes()
        lang_cradle.generate_lexicon()

        self.world.spawn_civilization(
            "Proto-Cradle",
            lang_cradle,
            x=cradle_x,
            y=cradle_y,
            population=0.035,
            lineage="Cradle",
            founded_epoch=0
        )
        self.event_log.append(f"World initialized with primordial seat 'Proto-Cradle' on emerged soil (Alt: {env_cradle.altitude:.2f}).")

    def step(self, epochs: int = 1):
        for _ in range(epochs):
            self.world.step_epoch(reduction_strength=0.75, enable_neologisms=True, parallel=False)
            if self.world.current_epoch % 20 == 0:
                self.event_log.append(f"Epoch {self.world.current_epoch}: Regional adaptation active.")
            if len(self.event_log) > 40:
                self.event_log = self.event_log[-40:]

    def get_summary_state(self) -> dict:
        civs_data = []
        resolution = len(self.terrain_grid)
        total_map_cells = resolution * resolution
        claims = getattr(self.world, "territory_grid", {})
        densities = getattr(self.world, "density_grid", {})
        cell_counts = Counter(claims.values())

        for name, civ in self.world.civilizations.items():
            stats = civ.language.get_phoneme_distribution_stats()
            attn = civ.language.cultural_attention
            env = civ.environment
            actual_veg = env.emergent_vegetation(population=civ.population, cultural_attention=attn)

            paradigm = GrammarEngine.discover_grammar_system(civ.language)
            alignment = getattr(civ.language.profile, "alignment", GrammarEngine.get_alignment_system(civ.language))

            cached_ui = getattr(civ, "_cached_ui_paradigm", None)
            paradigm_age = getattr(civ, "_cached_ui_epoch", -1)

            if cached_ui is None or (self.world.current_epoch - paradigm_age) >= 5:
                noun_classes_data = []
                living_nouns = [w for w in civ.language.words.values() if GrammarEngine.get_part_of_speech(w) == "Noun"]
                sample_w = living_nouns[0] if living_nouns else list(civ.language.words.values())[0]
                decl = GrammarEngine.get_noun_declension_table(sample_w, civ.language, paradigm)
                noun_classes_data.append({
                    "id": 1, "label": "Universal Noun Declension", "marker": "Ø",
                    "sample_form": sample_w.form,
                    "sample_gloss": SemanticTranslator.translate(sample_w, civ.language),
                    "declension": decl
                })

                verb_classes_data = []
                living_verbs = [w for w in civ.language.words.values() if GrammarEngine.get_part_of_speech(w) == "Verb"]
                sample_v = living_verbs[0] if living_verbs else list(civ.language.words.values())[0]
                conj = GrammarEngine.get_verb_conjugation_table(sample_v, civ.language, paradigm)
                v_cls = paradigm.verb_classes[0] if paradigm.verb_classes else None
                verb_classes_data.append({
                    "id": 1,
                    "label": v_cls.label if v_cls else "Regular Verb",
                    "strategy": v_cls.morph_strategy if v_cls else "analytic",
                    "aspect": v_cls.aspect_label if v_cls else "Aspectual",
                    "thematic": "Ø",
                    "sample_form": sample_v.form,
                    "sample_gloss": SemanticTranslator.translate(sample_v, civ.language),
                    "conjugation": conj
                })

                civ._cached_ui_paradigm = (noun_classes_data, verb_classes_data, sample_w, sample_v)
                civ._cached_ui_epoch = self.world.current_epoch
            else:
                noun_classes_data, verb_classes_data, sample_w, sample_v = cached_ui

            sorted_words = sorted(civ.language.words.values(), key=lambda w: w.usage_frequency, reverse=True)[:60]
            lexicon_sample = [
                {
                    "id": w.id, "form": w.form, "plain": w.plain_form,
                    "pos": GrammarEngine.get_part_of_speech(w),
                    "gloss": SemanticTranslator.translate(w, civ.language),
                    "freq": round(w.usage_frequency, 2),
                    "age": self.world.current_epoch - w.generation_born
                }
                for w in sorted_words
            ]

            active_tones = Counter()
            for w in civ.language.words.values():
                for syl in w.syllables:
                    s_tone = getattr(syl, 'tone', Tone.NONE)
                    if s_tone != Tone.NONE:
                        active_tones[s_tone.value] += 1

            tone_desc = "Non-Tonal (Stress-timed)" if not active_tones else f"Emergent {len(active_tones)}-Tone System ({' '.join(t[0] for t in active_tones.most_common())})"

            mean_cost = round(getattr(civ.language, "cached_mean_cost", 1.20), 2)
            instability = round(getattr(civ.language, "systemic_instability", 0.0), 2)
            active_series = sorted(list(civ.language.get_active_feature_series())) if hasattr(civ.language, "get_active_feature_series") else []

            words_list = list(civ.language.words.values())
            sample_word_obj = words_list[len(words_list) // 2] if words_list else None
            sample_word_form = sample_word_obj.form if sample_word_obj else "N/A"
            sample_word_gloss = SemanticTranslator.translate(sample_word_obj, civ.language) if sample_word_obj else "N/A"

            occupied_cells = cell_counts.get(name, 0)
            map_share_pct = round((occupied_cells / float(total_map_cells)) * 100, 1)

            settlement_list = [
                {
                    "name": s.name,
                    "x": s.x, "y": s.y,
                    "is_capital": s.is_capital,
                    "pop": round(getattr(s, "local_pop", getattr(s, "population_share", 0.05)), 3),
                    "fort": round(s.fortification_level, 2),
                    "founder": getattr(s, "founder", civ.name),
                    "founded_epoch": getattr(s, "founded_epoch", 0),
                    "history": getattr(s, "history", [])
                }
                for s in getattr(civ, "settlements", [])
            ]

            tension_val = round(getattr(civ.language, "sociolinguistic_tension", 0.0), 3)
            standard_status = "Classical Norm" if tension_val < 0.20 else ("Diglossia" if tension_val < 0.45 else "Reform Due")

            civs_data.append({
                "name": name,
                "x": civ.x, "y": civ.y,
                "population": round(civ.population, 3),
                "alt": round(env.altitude, 2), "hum": round(env.humidity, 2), "temp": round(env.temperature, 2),
                "veg": round(actual_veg, 2),
                "season": "Summer" if env.season_phase > 0.35 else ("Winter" if env.season_phase < -0.35 else "Equinox"),
                "economy": {
                    "grain": round(civ.economy.grain, 1),
                    "livestock": round(civ.economy.livestock, 1),
                    "marine": round(civ.economy.marine, 1),
                    "minerals": round(civ.economy.minerals, 1),
                },
                "infrastructure": round(civ.infrastructure_level, 2),
                "is_sedentary": (civ.infrastructure_level >= 1.50),
                "word_count": len(civ.language.words),
                "cv_ratio": stats.get("cv_ratio", 1.0),
                "entropy": stats.get("entropy_evenness", 1.0),
                "mean_cost": mean_cost,
                "instability": instability,
                "sociolinguistic_tension": tension_val,
                "standard_status": standard_status,
                "active_series": active_series,
                "word_order": civ.language.word_order,
                "head_dir": round(civ.language.profile.head_directionality, 2),
                "synthesis_index": round(civ.language.profile.synthesis_index, 2),
                "syllable_complexity": round(civ.language.profile.syllable_complexity, 2),
                "allow_clusters": civ.language.profile.allow_clusters,
                "alignment": alignment,
                "tone_desc": tone_desc,
                "number_system": paradigm.number_system,
                "article_system": paradigm.article_system,
                "numeral_system": paradigm.numeral_chain.name,
                "cultural_inertia": round(civ.language.cultural_inertia, 2),
                "attention": [round(w, 2) for w in attn],
                "top_consonants": list(stats.get("consonants", {}).items())[:6],
                "top_vowels": list(stats.get("vowels", {}).items())[:6],
                "vowels": sorted(list(set(p.ipa for p in civ.language.vowels))),
                "consonants": sorted(list(set(p.ipa for p in civ.language.consonants))),
                "sample_word": sample_word_form,
                "sample_gloss": sample_word_gloss,
                "sample_noun": {"form": sample_w.form, "gloss": SemanticTranslator.translate(sample_w, civ.language)},
                "sample_verb": {"form": sample_v.form, "gloss": SemanticTranslator.translate(sample_v, civ.language)},
                "noun_classes": noun_classes_data,
                "verb_classes": verb_classes_data,
                "lexicon": lexicon_sample,
                "territory_radius": getattr(civ, "territory_radius", 0.08),
                "food_capacity": getattr(civ, "food_capacity", 0.50),
                "food_deficit": getattr(civ, "food_deficit", 0.00),
                "lineage": getattr(civ, "lineage", name),
                "occupied_cells": occupied_cells,
                "map_share_pct": map_share_pct,
                "settlements": settlement_list,
            })

        live_terrain = []
        for r in range(resolution):
            grid_row = []
            for c in range(resolution):
                base_cell = self.terrain_grid[r][c]
                alt = base_cell["alt"]
                climax_veg = base_cell["veg"]
                cell_owner = claims.get((r, c), None)
                cell_density = densities.get((r, c), 0.0)

                if alt <= 0.0:
                    grid_row.append({"alt": alt, "temp": base_cell["temp"], "hum": base_cell["hum"], "veg": 0.0, "owner": cell_owner, "density": 0.0})
                else:
                    actual_veg = (climax_veg * math.exp(-cell_density * 2.20)) if cell_owner else climax_veg
                    grid_row.append({
                        "alt": alt, "temp": base_cell["temp"], "hum": base_cell["hum"],
                        "veg": round(max(0.04, actual_veg), 2),
                        "owner": cell_owner, "density": cell_density,
                    })
            live_terrain.append(grid_row)

        expeditions_data = [
            {
                "name": str(exp.get("name", "")),
                "civ": str(exp.get("civ", "")),
                "lineage": str(exp.get("lineage", "")),
                "origin": [float(exp["origin"][0]), float(exp["origin"][1])],
                "target": [float(exp["target"][0]), float(exp["target"][1])],
                "pop": round(float(exp.get("pop", 0.1)), 3),
                "progress": round(float(exp.get("progress", 0.0)), 2),
                "is_naval": bool(exp.get("is_naval", False)),
            }
            for exp in getattr(self.world, "active_expeditions", [])
        ]

        roads_data = [
            {
                "x0": round(float(e.x0), 3), "y0": round(float(e.y0), 3),
                "x1": round(float(e.x1), 3), "y1": round(float(e.y1), 3),
                "level": round(float(e.level), 2),
                "is_naval": bool(e.is_naval)
            }
            for e in getattr(self.world, "road_network", [])
        ]

        ruins_data = [
            {
                "name": str(r.name),
                "x": float(r.x),
                "y": float(r.y),
                "founder": str(getattr(r, "founder", "Unknown")),
                "epoch": int(getattr(r, "founded_epoch", 0))
            }
            for r in getattr(self.world, "abandoned_ruins", [])
        ]

        return {
            "epoch": self.world.current_epoch,
            "civs": civs_data,
            "trade_routes": self.world.active_trade_routes,
            "conflicts": self.world.active_conflicts,
            "expeditions": expeditions_data,
            "roads": roads_data,
            "ruins": ruins_data,
            "events": self.world.event_log[-20:],
            "terrain": live_terrain,
        }


STATE = WorldSimulationState()


class EvoLangHandler(BaseHTTPRequestHandler):
    def _set_json_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._set_json_headers()
            self.wfile.write(json.dumps(STATE.get_summary_state(), default=str).encode("utf-8"))
        elif parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()

            base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
            html_path = os.path.join(base_dir, "index.html")
            if not os.path.exists(html_path):
                html_path = os.path.join(os.getcwd(), "index.html")

            if os.path.exists(html_path):
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"<h1>index.html not found. Place it in the project directory.</h1>")
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/step":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            payload = json.loads(body) if body else {}
            epochs_to_step = int(payload.get("epochs", 1))
            STATE.step(epochs_to_step)
            self._set_json_headers()
            self.wfile.write(json.dumps(STATE.get_summary_state(), default=str).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")


def start_server(port: int = 8008):
    server_address = ("", port)
    httpd = ReusableHTTPServer(server_address, EvoLangHandler)
    print(f"\n==================================================================")
    print(f"  EVOLANG BIOMECHANICAL & SETTLEMENT OBSERVER RUNNING")
    print(f"  Open your browser at: http://localhost:{port}")
    print(f"==================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualizer server...")
        httpd.server_close()


if __name__ == "__main__":
    start_server()