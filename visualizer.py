"""
visualizer.py — Minimalist Real-Time Browser Visualizer for EvoLang.
Uses Python's standard library (zero pip dependencies required).
Run with: python visualizer.py
Then open: http://localhost:8000
"""

from __future__ import annotations

import json
import mimetypes
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from language import Language
from world_map import ContinuousWorldMap, Environment, TerrainField


# =====================================================================
# Global Simulation State
# =====================================================================

class WorldSimulationState:
    def __init__(self):
        self.terrain = TerrainField()
        self.world = ContinuousWorldMap(name="Pangaea Field", terrain=self.terrain)
        self.event_log: list[str] = []
        self._init_default_world()
        self._precompute_terrain_grid()

    def _precompute_terrain_grid(self, resolution: int = 50):
        """Precomputes a low-res terrain matrix for fast canvas background rendering."""
        self.terrain_grid = []
        for row in range(resolution):
            y = row / float(resolution - 1)
            grid_row = []
            for col in range(resolution):
                x = col / float(resolution - 1)
                env = self.terrain.sample_environment(x, y)
                grid_row.append({
                    "alt": round(env.altitude, 2),
                    "temp": round(env.temperature, 2),
                    "hum": round(env.humidity, 2),
                    "veg": round(env.vegetation, 2),
                })
            self.terrain_grid.append(grid_row)

    def _init_default_world(self):
        """Spawns 3 diverse archetype civilizations across the map."""
        # 1. Romano-Italic in Mediterranean Plains (x=0.45, y=0.55)
        env_it = self.terrain.sample_environment(0.45, 0.55, population=0.50)
        prof_it = env_it.generate_profile(name="Romano-Italic Profile", initial_head_directionality=0.50)
        lang_it = Language("Romano-Italic", profile=prof_it, environment=env_it)
        lang_it.populate_default_phonemes()
        lang_it.generate_lexicon()
        self.world.spawn_civilization("Romano-Italic", lang_it, x=0.45, y=0.55, population=0.50)

        # 2. Yamato-Japonic in Maritime Coast (x=0.85, y=0.40)
        env_jp = self.terrain.sample_environment(0.85, 0.40, population=0.40)
        prof_jp = env_jp.generate_profile(name="Yamato-Japonic Profile", initial_head_directionality=0.18)
        lang_jp = Language("Yamato-Japonic", profile=prof_jp, environment=env_jp)
        lang_jp.populate_default_phonemes()
        lang_jp.generate_lexicon()
        self.world.spawn_civilization("Yamato-Japonic", lang_jp, x=0.85, y=0.40, population=0.40)

        # 3. Alpine-Highlands in Mountain Ridge (x=0.65, y=0.80)
        env_alp = self.terrain.sample_environment(0.65, 0.80, population=0.25)
        prof_alp = env_alp.generate_profile(name="Alpine-Highlands Profile", initial_head_directionality=0.55)
        lang_alp = Language("Alpine-Highlands", profile=prof_alp, environment=env_alp)
        lang_alp.populate_default_phonemes()
        lang_alp.generate_lexicon()
        self.world.spawn_civilization("Alpine-Highlands", lang_alp, x=0.65, y=0.80, population=0.25)

        self.event_log.append("World initialized with 3 primordial civilizations.")

    def step(self, epochs: int = 1):
        for _ in range(epochs):
            self.world.step_epoch(reduction_strength=0.75, enable_neologisms=True, parallel=False)
            if self.world.current_epoch % 25 == 0:
                self.event_log.append(f"Epoch {self.world.current_epoch}: Climatic sound shifts and lexical pruning complete.")
            if len(self.event_log) > 40:
                self.event_log = self.event_log[-40:]

    def get_summary_state(self) -> dict:
        civs_data = []
        for name, civ in self.world.civilizations.items():
            stats = civ.language.get_phoneme_distribution_stats()
            attn = civ.language.cultural_attention
            env = civ.environment

            from grammar import GrammarEngine
            from translator import SemanticTranslator
            paradigm = GrammarEngine.discover_grammar_system(civ.language)
            alignment = GrammarEngine.get_alignment_system(civ.language)

            # 1. Full Noun Classes & Declension Tables
            noun_classes_data = []
            living_nouns = [w for w in civ.language.words.values() if GrammarEngine.get_part_of_speech(w) == "Noun"]
            
            if paradigm.noun_classes:
                for n_cls in paradigm.noun_classes:
                    matching = [w for w in living_nouns if GrammarEngine.classify_noun_dynamically(w, paradigm, civ.language) == n_cls]
                    sample_w = matching[0] if matching else (living_nouns[0] if living_nouns else list(civ.language.words.values())[0])
                    decl = GrammarEngine.get_noun_declension_table(sample_w, civ.language, paradigm)
                    noun_classes_data.append({
                        "id": n_cls.class_id,
                        "label": n_cls.label,
                        "marker": f"/{n_cls.marker.form}/" if n_cls.marker else "Ø",
                        "sample_form": sample_w.form,
                        "sample_gloss": SemanticTranslator.translate(sample_w, civ.language),
                        "declension": decl
                    })
            else:
                sample_w = living_nouns[0] if living_nouns else list(civ.language.words.values())[0]
                decl = GrammarEngine.get_noun_declension_table(sample_w, civ.language, paradigm)
                noun_classes_data.append({
                    "id": 1,
                    "label": "Universal Noun Class (Genderless)",
                    "marker": "Ø",
                    "sample_form": sample_w.form,
                    "sample_gloss": SemanticTranslator.translate(sample_w, civ.language),
                    "declension": decl
                })

            # 2. Full Verb Classes & Conjugation Tables
            verb_classes_data = []
            living_verbs = [w for w in civ.language.words.values() if GrammarEngine.get_part_of_speech(w) == "Verb"]
            
            if paradigm.verb_classes:
                for v_cls in paradigm.verb_classes:
                    matching = [w for w in living_verbs if GrammarEngine.classify_verb_dynamically(w, paradigm, civ.language).class_id == v_cls.class_id]
                    sample_v = matching[0] if matching else (living_verbs[0] if living_verbs else list(civ.language.words.values())[0])
                    conj = GrammarEngine.get_verb_conjugation_table(sample_v, civ.language, paradigm)
                    verb_classes_data.append({
                        "id": v_cls.class_id,
                        "label": v_cls.label,
                        "strategy": v_cls.morph_strategy,
                        "aspect": v_cls.aspect_label,
                        "thematic": f"/{v_cls.thematic_marker.form}/" if v_cls.thematic_marker else "Ø",
                        "sample_form": sample_v.form,
                        "sample_gloss": SemanticTranslator.translate(sample_v, civ.language),
                        "conjugation": conj
                    })
            else:
                sample_v = living_verbs[0] if living_verbs else list(civ.language.words.values())[0]
                conj = GrammarEngine.get_verb_conjugation_table(sample_v, civ.language, paradigm)
                verb_classes_data.append({
                    "id": 1,
                    "label": "Universal Verb Conjugation",
                    "strategy": "concatenative",
                    "aspect": "Aorist/Perf",
                    "thematic": "Ø",
                    "sample_form": sample_v.form,
                    "sample_gloss": SemanticTranslator.translate(sample_v, civ.language),
                    "conjugation": conj
                })

            # 3. Searchable Living Lexicon Sample
            sorted_words = sorted(civ.language.words.values(), key=lambda w: w.usage_frequency, reverse=True)[:60]
            lexicon_sample = [
                {
                    "id": w.id,
                    "form": w.form,
                    "plain": w.plain_form,
                    "pos": GrammarEngine.get_part_of_speech(w),
                    "gloss": SemanticTranslator.translate(w, civ.language),
                    "freq": round(w.usage_frequency, 2),
                    "age": self.world.current_epoch - w.generation_born
                }
                for w in sorted_words
            ]

            # 4. Tonal Status
            tone_tier = civ.language.profile.tone_tier
            if tone_tier == 0:
                tone_desc = "Non-Tonal (Stress-timed)"
            elif tone_tier <= 2:
                tone_desc = "Tone Tier 2: Pitch-Accent (High ˥ / Low ˩)"
            elif tone_tier <= 4:
                tone_desc = "Tone Tier 4: Register Contours (˥, ˧, ˩, ˧˥)"
            elif tone_tier <= 6:
                tone_desc = "Tone Tier 6: 6-Tone System (˥, ˧, ˩, ˧˥, ˥˩, ˨˩)"
            else:
                tone_desc = "Tone Tier 8: 8-Tone Contour System (˥, ˧, ˩, ˧˥, ˩˧, ˥˩, ˨˩, ˨˩˦)"

            civs_data.append({
                "name": name,
                "x": civ.x,
                "y": civ.y,
                "population": round(civ.population, 3),
                "alt": round(env.altitude, 2),
                "hum": round(env.humidity, 2),
                "temp": round(env.temperature, 2),
                "veg": round(env.vegetation, 2),
                "word_count": len(civ.language.words),
                "cv_ratio": stats.get("cv_ratio", 1.0),
                "entropy": stats.get("entropy_evenness", 1.0),
                "word_order": civ.language.word_order,
                "head_dir": round(civ.language.profile.head_directionality, 2),
                "synthesis_index": round(civ.language.profile.synthesis_index, 2),
                "alignment": alignment,
                "tone_tier": tone_tier,
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
                "noun_classes": noun_classes_data,
                "verb_classes": verb_classes_data,
                "lexicon": lexicon_sample,
            })

        return {
            "epoch": self.world.current_epoch,
            "civs": civs_data,
            "events": self.event_log[-15:],
            "terrain": self.terrain_grid,
        }


STATE = WorldSimulationState()


# =====================================================================
# HTTP Request Handler
# =====================================================================

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
            self.wfile.write(json.dumps(STATE.get_summary_state()).encode("utf-8"))

        elif parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(os.path.dirname(__file__), "index.html")
            if os.path.exists(html_path):
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"<h1>index.html not found. Place it in the same directory.</h1>")
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
            self.wfile.write(json.dumps(STATE.get_summary_state()).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")


def start_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, EvoLangHandler)
    print(f"\n==================================================================")
    print(f"  EVOLANG REAL-TIME VISUALIZER RUNNING")
    print(f"  Open your browser at: http://localhost:{port}")
    print(f"==================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualizer server...")
        httpd.server_close()


if __name__ == "__main__":
    start_server()