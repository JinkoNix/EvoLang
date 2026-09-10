"""
translator.py — POS-Constrained & Compositional Semantic Translator for EvoLang.
Guarantees strict grammatical role preservation: Verbs translate as verbs,
Nouns as nouns, Particles as particles, with syntax-appropriate adverbial/adjectival modifiers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from semantics import SemanticVector, DerivationType

if TYPE_CHECKING:
    from word import Word, LexicalCategory
    from language import Language


class SemanticTranslator:
    """Translates 7D coordinates into strictly POS-aligned, compositional natural language glosses."""

    # 1. NOUN ANCHORS (Physical & Abstract Entities)
    NOUN_VOCABULARY: list[tuple[SemanticVector, str]] = [
        (SemanticVector(1.0, 0.0, 0.7, 0.4, 0.4, 0.1, 0.3), "water"),
        (SemanticVector(1.0, 0.0, 0.7, 0.5, 0.8, 0.1, 0.8), "river"),
        (SemanticVector(1.0, 0.0, 0.7, 0.8, 0.6, 0.1, 1.0), "ocean"),
        (SemanticVector(0.8, 0.0, 0.6, 0.4, 0.9, 0.1, 0.6), "rain"),
        (SemanticVector(0.9, 0.0, 0.5, 0.2, 0.2, 0.1, 0.1), "dew"),
        (SemanticVector(0.9, 0.0, 0.4, 0.6, 0.2, 0.1, 0.4), "ice"),
        (SemanticVector(0.9, 0.1, 0.4, 0.7, 0.9, 0.1, 0.3), "fire"),
        (SemanticVector(0.9, 0.0, 0.3, 0.4, 0.6, 0.1, 0.4), "smoke"),
        (SemanticVector(0.9, 0.0, 0.3, 0.2, 0.1, 0.1, 0.2), "ash"),
        (SemanticVector(0.8, 0.0, 0.9, 1.0, 0.6, 0.1, 0.9), "sun"),
        (SemanticVector(0.8, 0.0, 0.6, 0.6, 0.3, 0.1, 0.7), "moon"),
        (SemanticVector(0.6, 0.0, 0.7, 0.3, 0.2, 0.1, 0.4), "star"),
        (SemanticVector(0.5, 0.0, 0.8, 0.4, 0.1, 0.1, 1.0), "sky"),
        (SemanticVector(0.7, 0.0, 0.5, 0.3, 0.5, 0.1, 0.6), "cloud"),
        (SemanticVector(0.6, 0.0, 0.6, 0.5, 0.8, 0.1, 0.8), "wind"),
        (SemanticVector(0.6, 0.0, 0.3, 1.0, 1.0, 0.1, 0.7), "thunder"),
        (SemanticVector(1.0, 0.0, 0.6, 0.7, 0.1, 0.1, 0.7), "earth"),
        (SemanticVector(1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.1), "stone"),
        (SemanticVector(1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.9), "mountain"),
        (SemanticVector(1.0, 0.0, 0.4, 0.8, 0.0, 0.0, 0.4), "cliff"),
        (SemanticVector(1.0, 0.0, 0.5, 0.2, 0.1, 0.0, 0.6), "sand"),
        (SemanticVector(1.0, 0.0, 0.4, 0.3, 0.3, 0.0, 0.3), "mud"),
        (SemanticVector(1.0, 0.0, 0.6, 0.8, 0.0, 0.0, 0.2), "iron"),
        (SemanticVector(0.9, 0.0, 0.9, 0.6, 0.0, 0.1, 0.2), "gold"),
        (SemanticVector(1.0, 0.4, 0.7, 0.6, 0.2, 0.1, 0.4), "tree"),
        (SemanticVector(1.0, 0.4, 0.6, 0.7, 0.2, 0.1, 0.9), "forest"),
        (SemanticVector(0.9, 0.4, 0.8, 0.2, 0.2, 0.2, 0.2), "flower"),
        (SemanticVector(1.0, 0.3, 0.6, 0.4, 0.1, 0.1, 0.2), "wood"),
        (SemanticVector(0.9, 0.3, 0.7, 0.3, 0.3, 0.2, 0.2), "leaf"),
        (SemanticVector(0.9, 0.3, 0.8, 0.4, 0.2, 0.1, 0.1), "fruit"),
        (SemanticVector(0.9, 0.3, 0.6, 0.3, 0.1, 0.1, 0.1), "seed"),

        # Kinship & People
        (SemanticVector(0.9, 1.0, 0.85, 0.40, 0.40, 0.95, 0.20), "mother"),
        (SemanticVector(0.9, 1.0, 0.75, 0.75, 0.60, 0.85, 0.30), "father"),
        (SemanticVector(0.9, 1.0, 0.85, 0.20, 0.50, 0.90, 0.10), "child"),
        (SemanticVector(0.9, 1.0, 0.70, 0.70, 0.70, 0.80, 0.30), "man"),
        (SemanticVector(0.9, 1.0, 0.80, 0.40, 0.50, 0.85, 0.20), "woman"),
        (SemanticVector(0.9, 1.0, 0.60, 0.50, 0.50, 0.70, 0.30), "person"),
        (SemanticVector(0.8, 1.0, 0.80, 0.40, 0.40, 0.90, 0.40), "friend"),
        (SemanticVector(0.8, 1.0, 0.20, 0.80, 0.70, 0.10, 0.40), "enemy"),
        (SemanticVector(0.8, 1.0, 0.60, 0.80, 0.50, 0.85, 0.60), "leader"),
        (SemanticVector(0.8, 1.0, 0.70, 0.50, 0.50, 0.90, 0.70), "tribe"),

        # Animals
        (SemanticVector(0.9, 0.9, 0.50, 0.60, 0.70, 0.20, 0.40), "animal"),
        (SemanticVector(0.9, 0.9, 0.30, 0.90, 0.90, 0.10, 0.50), "wolf"),
        (SemanticVector(0.9, 0.9, 0.70, 0.30, 0.80, 0.40, 0.30), "deer"),
        (SemanticVector(0.8, 0.8, 0.60, 0.30, 0.70, 0.30, 0.20), "bird"),
        (SemanticVector(0.8, 0.8, 0.40, 0.80, 0.90, 0.10, 0.40), "eagle"),
        (SemanticVector(0.9, 0.8, 0.60, 0.40, 0.70, 0.10, 0.30), "fish"),
        (SemanticVector(0.9, 0.8, 0.20, 0.60, 0.60, 0.10, 0.30), "snake"),

        # Body Anatomy & Tools
        (SemanticVector(0.9, 0.8, 0.60, 0.70, 0.40, 0.30, 0.20), "head"),
        (SemanticVector(1.0, 0.9, 0.60, 0.60, 0.50, 0.40, 0.50), "body"),
        (SemanticVector(0.9, 0.8, 0.60, 0.40, 0.50, 0.30, 0.20), "eye"),
        (SemanticVector(0.9, 0.8, 0.60, 0.40, 0.40, 0.40, 0.20), "ear"),
        (SemanticVector(0.9, 0.8, 0.60, 0.30, 0.30, 0.10, 0.10), "nose"),
        (SemanticVector(0.9, 0.8, 0.60, 0.50, 0.60, 0.60, 0.20), "mouth"),
        (SemanticVector(1.0, 0.8, 0.60, 0.60, 0.60, 0.40, 0.20), "hand"),
        (SemanticVector(1.0, 0.8, 0.60, 0.60, 0.70, 0.20, 0.30), "foot"),
        (SemanticVector(0.7, 0.9, 0.70, 0.60, 0.40, 0.60, 0.20), "heart"),
        (SemanticVector(0.9, 0.6, 0.40, 0.70, 0.50, 0.30, 0.20), "blood"),
        (SemanticVector(1.0, 0.5, 0.50, 0.80, 0.10, 0.10, 0.20), "bone"),
        (SemanticVector(0.9, 0.5, 0.50, 0.40, 0.20, 0.20, 0.40), "skin"),
        (SemanticVector(1.0, 0.0, 0.30, 0.90, 0.80, 0.10, 0.30), "sword"),
        (SemanticVector(1.0, 0.0, 0.50, 0.50, 0.50, 0.10, 0.20), "knife"),
        (SemanticVector(1.0, 0.0, 0.40, 0.80, 0.80, 0.10, 0.60), "spear"),
        (SemanticVector(1.0, 0.0, 0.50, 0.70, 0.70, 0.10, 0.50), "bow"),
        (SemanticVector(1.0, 0.0, 0.60, 0.80, 0.30, 0.20, 0.40), "shield"),
        (SemanticVector(1.0, 0.2, 0.80, 0.70, 0.10, 0.80, 0.60), "house"),
        (SemanticVector(1.0, 0.2, 0.70, 0.60, 0.20, 0.90, 0.80), "village"),

        # Nominalized Abstract & State Entities
        (SemanticVector(0.2, 0.9, 0.20, 0.10, 0.40, 0.70, 0.60), "ghost"),
        (SemanticVector(0.2, 0.9, 0.70, 0.20, 0.40, 0.80, 0.60), "ancestor"),
        (SemanticVector(0.5, 0.8, 0.10, 0.40, 0.40, 0.30, 0.50), "death"),
        (SemanticVector(0.4, 0.9, 0.60, 0.40, 0.40, 0.80, 0.30), "voice"),
        (SemanticVector(0.4, 0.9, 0.70, 0.50, 0.40, 0.90, 0.30), "song"),
        (SemanticVector(0.5, 0.7, 0.50, 0.40, 0.40, 0.30, 0.50), "journey"),
    ]

    # 2. VERB ANCHORS (Dynamic Actions & Cognitive Processes)
    VERB_VOCABULARY: list[tuple[SemanticVector, str]] = [
        (SemanticVector(0.5, 0.8, 0.60, 0.60, 0.80, 0.40, 0.30), "make"),
        (SemanticVector(0.6, 0.7, 0.50, 0.40, 0.70, 0.20, 0.50), "walk"),
        (SemanticVector(0.6, 0.7, 0.50, 0.80, 1.00, 0.20, 0.70), "run"),
        (SemanticVector(0.5, 0.7, 0.60, 0.70, 0.90, 0.10, 0.80), "fly"),
        (SemanticVector(0.5, 0.7, 0.60, 0.50, 0.70, 0.10, 0.50), "swim"),
        (SemanticVector(0.4, 0.8, 0.60, 0.40, 0.50, 0.30, 0.40), "see"),
        (SemanticVector(0.4, 0.8, 0.60, 0.40, 0.40, 0.40, 0.40), "hear"),
        (SemanticVector(0.4, 0.9, 0.60, 0.40, 0.70, 1.00, 0.30), "speak"),
        (SemanticVector(0.4, 0.9, 0.70, 0.50, 0.70, 0.90, 0.30), "sing"),
        # Nutrition & Sustenance Sub-Domain
        (SemanticVector(0.8, 0.8, 0.80, 0.50, 0.60, 0.30, 0.20), "eat"),
        (SemanticVector(0.7, 0.9, 0.75, 0.40, 0.70, 0.35, 0.10), "devour"),
        (SemanticVector(0.8, 0.5, 0.75, 0.35, 0.60, 0.60, 0.10), "feast"),
        (SemanticVector(0.7, 0.8, 0.85, 0.30, 0.50, 0.80, 0.20), "feed"),
        (SemanticVector(0.8, 0.8, 0.70, 0.40, 0.50, 0.20, 0.20), "drink"),
        
        (SemanticVector(0.4, 0.7, 0.60, 0.10, 0.10, 0.30, 0.20), "sleep"),
        (SemanticVector(0.5, 0.8, 0.10, 0.40, 0.40, 0.30, 0.50), "die"),
        (SemanticVector(0.6, 0.8, 0.10, 0.95, 0.95, 0.10, 0.40), "kill"),
        (SemanticVector(0.7, 0.8, 0.20, 0.85, 0.90, 0.20, 0.30), "strike"),
        (SemanticVector(0.6, 0.8, 0.30, 0.70, 0.80, 0.40, 0.40), "hunt"),
        (SemanticVector(0.4, 0.8, 0.85, 0.30, 0.20, 0.95, 0.30), "love"),
        (SemanticVector(0.5, 0.8, 0.70, 0.40, 0.60, 0.80, 0.40), "give"),
        (SemanticVector(0.7, 0.8, 0.80, 0.70, 0.70, 0.60, 0.50), "nourish"),
        (SemanticVector(0.6, 0.8, 0.85, 0.60, 0.60, 0.80, 0.40), "heal"),
        (SemanticVector(0.6, 0.8, 0.50, 0.80, 0.80, 0.85, 0.60), "lead"),
        (SemanticVector(0.8, 0.0, 0.80, 0.90, 0.70, 0.10, 0.80), "shine"),
        (SemanticVector(0.9, 0.0, 0.30, 0.90, 0.80, 0.10, 0.40), "cut"),
        (SemanticVector(0.9, 0.1, 0.40, 0.80, 0.90, 0.10, 0.40), "burn"),
    ]

    # 3. ADJECTIVE ANCHORS (Scalar Qualities & Attributes)
    ADJ_VOCABULARY: list[tuple[SemanticVector, str]] = [
        (SemanticVector(0.2, 0.3, 0.90, 0.60, 0.30, 0.70, 0.50), "good"),
        (SemanticVector(0.2, 0.3, 0.10, 0.60, 0.40, 0.30, 0.50), "bad"),
        (SemanticVector(0.4, 0.1, 0.60, 0.90, 0.30, 0.10, 0.90), "big"),
        (SemanticVector(0.4, 0.1, 0.50, 0.10, 0.30, 0.10, 0.10), "small"),
        (SemanticVector(0.3, 0.1, 0.60, 0.80, 0.10, 0.10, 0.80), "long"),
        (SemanticVector(0.3, 0.1, 0.50, 0.20, 0.10, 0.10, 0.20), "short"),
        (SemanticVector(0.6, 0.1, 0.50, 0.90, 0.80, 0.10, 0.20), "hard"),
        (SemanticVector(0.5, 0.1, 0.60, 0.20, 0.20, 0.30, 0.30), "soft"),
        (SemanticVector(0.6, 0.0, 0.70, 0.80, 0.60, 0.10, 0.30), "hot"),
        (SemanticVector(0.7, 0.0, 0.40, 0.60, 0.20, 0.10, 0.30), "cold"),
        (SemanticVector(0.3, 0.0, 0.90, 0.60, 0.40, 0.10, 0.80), "bright"),
        (SemanticVector(0.3, 0.0, 0.20, 0.60, 0.20, 0.10, 0.70), "dark"),
        (SemanticVector(0.4, 0.1, 0.85, 0.40, 0.30, 0.85, 0.50), "holy"),
        (SemanticVector(0.4, 0.1, 0.15, 0.80, 0.40, 0.15, 0.50), "cursed"),
        (SemanticVector(0.5, 0.3, 0.50, 0.80, 0.90, 0.20, 0.50), "swift"),
        (SemanticVector(0.7, 0.1, 0.60, 0.60, 0.10, 0.30, 0.70), "ancient"),
    ]

    # 4. PARTICLE ANCHORS (Closed-Class Functional Relators)
    PARTICLE_VOCABULARY: list[tuple[SemanticVector, str]] = [
        (SemanticVector(0.3, 0.0, 0.50, 0.30, 0.00, 0.00, 0.10), "this"),
        (SemanticVector(0.3, 0.0, 0.50, 0.30, 0.00, 0.00, 0.60), "that"),
        (SemanticVector(0.1, 0.0, 0.50, 0.20, 0.00, 0.00, 0.10), "in"),
        (SemanticVector(0.2, 0.0, 0.50, 0.60, 0.00, 0.00, 0.50), "on"),
        (SemanticVector(0.0, 0.0, 0.30, 0.70, 0.00, 0.00, 0.10), "not"),
        (SemanticVector(0.1, 0.0, 0.50, 0.20, 0.00, 0.00, 0.10), "one"),
        (SemanticVector(0.1, 0.0, 0.50, 0.60, 0.00, 0.20, 0.60), "many"),
        (SemanticVector(0.8, 1.0, 0.60, 0.50, 0.50, 0.30, 0.10), "I"),
        (SemanticVector(0.8, 1.0, 0.60, 0.50, 0.50, 0.80, 0.10), "you"),
    ]

    @classmethod
    def get_pos(cls, vector: SemanticVector) -> str:
        """Determines Part of Speech via continuous cognitive prototype attractor scores."""
        c, a, v, p, d, s, e = vector.coords

        score_verb = 1.70 * d + 0.50 * p - 0.40 * c
        score_noun = 1.40 * c + 1.20 * a - 0.50 * d

        qual_salience = max(abs(v - 0.50), abs(p - 0.50), abs(e - 0.50))
        score_adj = 1.60 * qual_salience + 0.30 * (1.0 - a) - 0.40 * d

        score_particle = 1.80 * vector.grammaticality_index

        scores = [
            (score_verb, "Verb"),
            (score_noun, "Noun"),
            (score_adj, "Adjective"),
            (score_particle, "Particle"),
        ]
        return max(scores, key=lambda item: item[0])[1]

    @classmethod
    def translate_vector_constrained(cls, vector: SemanticVector, target_pos: str | None = None) -> str:
        """Translates a vector with strict enforcement of its target grammatical category."""
        pos = target_pos or cls.get_pos(vector)

        if pos == "Verb":
            target_pool = cls.VERB_VOCABULARY
        elif pos == "Adjective":
            target_pool = cls.ADJ_VOCABULARY
        elif pos == "Particle":
            target_pool = cls.PARTICLE_VOCABULARY
        else:
            target_pool = cls.NOUN_VOCABULARY

        best_dist = 999.0
        best_word = "thing" if pos == "Noun" else ("act" if pos == "Verb" else "some")
        
        for anchor_vec, word_name in target_pool:
            d = vector.distance_to(anchor_vec)
            if d < best_dist:
                best_dist = d
                best_word = word_name

        c, a, v, p, d, s, e = vector.coords

        # Modifiers are strictly disabled for closed-class particles
        if pos == "Particle":
            return best_word

        # For content open classes (Nouns, Verbs, Adjectives), add syntax-appropriate modifiers
        if best_dist > 0.28:
            if pos == "Verb":
                # Adverbial modifiers for verbs
                if v >= 0.80: return f"sacredly {best_word}"
                if v <= 0.20: return f"darkly {best_word}"
                if p >= 0.80: return f"fiercely {best_word}"
                if e >= 0.80: return f"widely {best_word}"
                if d >= 0.85: return f"swiftly {best_word}"
            elif pos == "Noun":
                # Adjectival modifiers for nouns
                if v >= 0.80 and best_word not in ("gold", "sun"): return f"holy {best_word}"
                if v <= 0.20 and best_word not in ("enemy", "death", "blood"): return f"dark {best_word}"
                if p >= 0.80 and best_word not in ("mountain", "iron", "sword"): return f"mighty {best_word}"
                if e >= 0.80 and best_word not in ("ocean", "forest", "sky"): return f"great {best_word}"

        return best_word

    @classmethod
    def translate_word(cls, word: Word, lang: Language | None = None) -> str:
        """Translates a living Word, enforcing its grammatical category and resolving compounds."""
        from grammar import GrammarEngine
        word_pos = GrammarEngine.get_part_of_speech(word)
        base_gloss = cls.translate_vector_constrained(word.vector, target_pos=word_pos)
        deriv = word.derivation

        # 1. Compositional Compounds (e.g. fire + stone -> fire-stone (mountain))
        if deriv.derivation_type == DerivationType.COMPOUND and len(deriv.local_parent_ids) >= 2:
            parent_glosses = []
            if lang:
                for pid in deriv.local_parent_ids:
                    p_word = lang.words.get(pid)
                    if p_word and p_word.id != word.id:
                        p_pos = GrammarEngine.get_part_of_speech(p_word)
                        parent_glosses.append(cls.translate_vector_constrained(p_word.vector, target_pos=p_pos))

            if len(parent_glosses) >= 2 and parent_glosses[0] != parent_glosses[1]:
                compound_label = f"{parent_glosses[0]}-{parent_glosses[1]}"
                return f"{compound_label} ({base_gloss})" if base_gloss not in parent_glosses else compound_label

        # 2. Derivational Affixations (e.g. make + leader -> make+leader (lead))
        elif deriv.derivation_type == DerivationType.AFFIXATION and len(deriv.local_parent_ids) >= 2:
            if lang:
                stem_w = lang.words.get(deriv.local_parent_ids[0])
                affix_w = lang.words.get(deriv.local_parent_ids[1])
                if stem_w and affix_w:
                    s_pos = GrammarEngine.get_part_of_speech(stem_w)
                    a_pos = GrammarEngine.get_part_of_speech(affix_w)
                    stem_g = cls.translate_vector_constrained(stem_w.vector, target_pos=s_pos)
                    affix_g = cls.translate_vector_constrained(affix_w.vector, target_pos=a_pos)
                    return f"{stem_g}+{affix_g} ({base_gloss})"

        # 3. Direct Root / Semantic Mitosis / Loanwords
        return base_gloss

    @classmethod
    def translate(cls, target: SemanticVector | Word, lang: Language | None = None) -> str:
        """Universal polymorphic translation helper."""
        if hasattr(target, "derivation") and hasattr(target, "vector"):
            return cls.translate_word(target, lang)
        elif hasattr(target, "coords"):
            return cls.translate_vector_constrained(target)
        return str(target)

    @classmethod
    def identify_metaphor_pathway(cls, primary: SemanticVector, extended: SemanticVector) -> str:
        """Dynamically identifies the cognitive transfer pathway via vector divergence."""
        diffs = [extended.coords[i] - primary.coords[i] for i in range(7)]
        
        relational_gains = [(diffs[i], SemanticVector.DIMENSION_NAMES[i]) for i in range(2, 7)]
        relational_gains.sort(key=lambda item: item[0], reverse=True)
        max_gain, target_dim = relational_gains[0]

        if diffs[0] < -0.08 or diffs[1] < -0.08:
            if target_dim == "sociality" and max_gain > 0.05:
                return "Physical → Social / Hierarchical"
            elif target_dim == "extension" and max_gain > 0.05:
                return "Spatial → Temporal / Extensional"
            elif target_dim == "dynamism" and max_gain > 0.05:
                return "Stative → Process / Cognitive"
            elif target_dim in ("potency", "valence") and max_gain > 0.05:
                return "Physical → Evaluative / Intensive"
            return "Concrete → Abstract Projection"
        
        return "Semantic Drift"

    @classmethod
    def translate_polysemy(cls, word: Word, lang: Language | None = None) -> list[tuple[str, str, SemanticVector]]:
        """Returns fully glossed senses with their cognitive transfer pathways."""
        from grammar import GrammarEngine
        word_pos = GrammarEngine.get_part_of_speech(word)

        if not word.senses:
            return [(cls.translate_word(word, lang), "Primary Literal", word.vector)]

        primary_vec = word.senses[0]
        primary_gloss = cls.translate_word(word, lang)
        results = [(primary_gloss, "Primary Literal", primary_vec)]

        seen_meanings = {primary_gloss}

        for ext_vec in word.senses[1:]:
            ext_pos = cls.get_pos(ext_vec)
            ext_gloss = cls.translate_vector_constrained(ext_vec, target_pos=ext_pos)
            if ext_gloss not in seen_meanings:
                pathway_label = cls.identify_metaphor_pathway(primary_vec, ext_vec)
                results.append((ext_gloss, pathway_label, ext_vec))
                seen_meanings.add(ext_gloss)

        return results