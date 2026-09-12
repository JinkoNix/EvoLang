"""
grammar.py — Formal Generative Cognitive Morphosyntax with Continuous Strategy Dynamics.
Features Multi-Strategy Hybridity (Affixation, Ablaut, Umlaut, Transfixation, Reduplication, Infixation),
Demographic Morphological Leveling (T_strat), Case Syncretism, and Differential Object Marking.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from phoneme import Phoneme, PhonemeKind
from word import Word, Morpheme, LexicalCategory
from language import Language
from semantics import SemanticVector, SemanticSpace, DerivationRecord, DerivationType, ClineStage
from energy import ArticulatoryEnergyModel
from syllable import SyllableEngine, StressPattern, Tone


@dataclass
class EmergentClass:
    """An emergent gender/class category defined by its cognitive centroid in 7D space."""
    class_id: int
    label: str
    centroid: SemanticVector
    marker: Word | None = None


@dataclass
class EmergentVerbClass:
    """An emergent verbal conjugation track (Aktionsart) defined by Dynamism, Potency, and Sociality."""
    class_id: int
    label: str
    centroid: SemanticVector
    thematic_marker: Word | None = None
    aspect_label: str = "Aorist/Perf"
    morph_strategy: str = "concatenative"


@dataclass
class NumeralChain:
    """A hierarchical chain of somatic counting radices (Sub-Base -> Primary Base -> Super-Base)."""
    sub_base: int | None = None
    primary_base: int = 10
    super_base: int | None = None
    name: str = "Decimal (Base-10)"


@dataclass
class EmergentParadigm:
    """A collection of closed-class operators, pronouns, number markers, articles, and numeral chains."""
    temporal_operators: list[Word] = field(default_factory=list)
    spatial_operators: list[Word] = field(default_factory=list)
    determiners: list[Word] = field(default_factory=list)
    modal_operators: list[Word] = field(default_factory=list)
    noun_classes: list[EmergentClass] = field(default_factory=list)
    verb_classes: list[EmergentVerbClass] = field(default_factory=list)
    relativizers: list[Word] = field(default_factory=list)
    complementizers: list[Word] = field(default_factory=list)
    pronouns: dict[str, Word] = field(default_factory=dict)
    number_system: str = "Singular-Plural"
    number_operators: dict[str, Word] = field(default_factory=dict)
    article_system: str = "Transdefinite (No Articles)"
    articles: dict[str, Word] = field(default_factory=dict)
    numeral_chain: NumeralChain = field(default_factory=NumeralChain)
    digits: dict[int, Word] = field(default_factory=dict)
    radix_words: dict[str, Word] = field(default_factory=dict)
    formatted_cache: dict[int, str] = field(default_factory=dict)


class ClauseValency(Enum):
    INTRANSITIVE = "Intransitive (1-Arg: Subject)"
    TRANSITIVE = "Transitive (2-Arg: Agent + Patient)"
    DITRANSITIVE = "Ditransitive (3-Arg: Agent + Recipient + Theme)"


@dataclass
class NounPhrase:
    """A hierarchical Noun Phrase constituent with active Numeral Classifiers, Articles, and Concord."""
    head_noun: Word
    modifier: Word | None = None
    determiner: Word | None = None
    possessor: Word | None = None
    relative_clause: Clause | None = None
    case_operator: Word | None = None
    number_operator: Word | None = None
    numeral_value: int | None = None
    is_definite: bool | None = None
    case_label: str = "Nom"
    number_label: str = "Sg"
    assigned_class: EmergentClass | None = None
    is_subject: bool = False

    def render(self, lang: Language) -> tuple[str, str]:
        h = lang.profile.head_directionality
        s_idx = lang.profile.synthesis_index
        tokens: list[str] = []
        glosses: list[str] = []

        paradigm = GrammarEngine.discover_grammar_system(lang)

        # 1. Auto-Classification from 7D Semantics
        if self.assigned_class is None and paradigm.noun_classes:
            self.assigned_class = GrammarEngine.classify_noun_dynamically(self.head_noun, paradigm, lang)

        agr_op = self.assigned_class.marker if self.assigned_class else None
        cls_tag = f"Cls{self.assigned_class.class_id}" if self.assigned_class else ""

        # 2. Base Noun with Overt Class & Number Inflection
        base_noun = self.head_noun
        if agr_op and s_idx >= 0.35 and len(paradigm.noun_classes) >= 2:
            base_noun, is_class_bound = GrammarEngine.apply_spatial_relator(self.head_noun, agr_op, lang, category_type="class")
        else:
            is_class_bound = False

        if self.numeral_value and self.numeral_value > 1 and self.number_label == "Sg":
            if self.numeral_value == 2 and "du" in paradigm.number_operators:
                self.number_operator = paradigm.number_operators["du"]
                self.number_label = "Du"
            elif "pl" in paradigm.number_operators:
                self.number_operator = paradigm.number_operators["pl"]
                self.number_label = "Pl"

        if self.number_operator and self.number_label != "Sg" and s_idx >= 0.30:
            base_noun, is_num_bound = GrammarEngine.apply_spatial_relator(base_noun, self.number_operator, lang, category_type="number")
        else:
            is_num_bound = False

        # 3. Numeral & Classifier Processing
        num_tokens, num_gloss = [], []
        if self.numeral_value is not None:
            raw_num = GrammarEngine.format_numeral(self.numeral_value, lang, paradigm).strip("/")
            is_same_word = (agr_op and agr_op.id == self.head_noun.id)
            if s_idx < 0.35 and agr_op and not is_same_word and len(paradigm.noun_classes) >= 2:
                num_tokens.extend([raw_num, agr_op.form])
                num_gloss.extend([f"[Num:{self.numeral_value}]", f"[{cls_tag}:Classifier]"])
            else:
                num_tokens.append(raw_num)
                num_gloss.append(f"[Num:{self.numeral_value}]")

        # 4. Definite / Indefinite Articles with Concord
        art_op = None
        art_label = ""
        if self.is_definite is True and "def" in paradigm.articles:
            art_op = paradigm.articles["def"]
            art_label = "Def"
        elif self.is_definite is False and "indef" in paradigm.articles:
            art_op = paradigm.articles["indef"]
            art_label = "Indef"

        art_tokens, art_gloss = [], []
        is_art_bound = False
        if art_op:
            if s_idx < 0.40 or h <= 0.65:
                if agr_op and s_idx >= 0.25 and len(paradigm.noun_classes) >= 2:
                    art_concord, is_ab = GrammarEngine.apply_spatial_relator(art_op, agr_op, lang, category_type="class")
                    art_form = art_concord.form if is_ab else f"{art_op.form} {agr_op.form}"
                else:
                    art_form = art_op.form
                art_tokens = [art_form]
                art_gloss = [f"[{art_label}{'+' + cls_tag if cls_tag else ''}:{art_op.id}]"]
            else:
                base_noun, is_art_bound = GrammarEngine.apply_spatial_relator(base_noun, art_op, lang, category_type="case")
                if not is_art_bound:
                    art_tokens = [art_op.form]
                    art_gloss = [f"[{art_label}:{art_op.id}]"]

        # 5. Case Inflection
        prep_tokens, prep_gloss = [], []
        post_tokens, post_gloss = [], []
        
        if self.case_operator:
            inflected_noun, is_bound = GrammarEngine.apply_spatial_relator(base_noun, self.case_operator, lang, category_type="case")
            if is_bound:
                n_form = inflected_noun.form
                tags = [self.case_label]
                if is_art_bound: tags.append(art_label)
                if is_num_bound: tags.append(self.number_label)
                if is_class_bound and cls_tag: tags.append(cls_tag)
                n_gloss = f"{self.head_noun.id}[{'+'.join(tags)}]"
            else:
                n_form = base_noun.form
                tags = []
                if is_art_bound: tags.append(art_label)
                if is_num_bound: tags.append(self.number_label)
                if is_class_bound and cls_tag: tags.append(cls_tag)
                tag_str = f"[{'+'.join(tags)}]" if tags else ""
                n_gloss = f"{self.head_noun.id}{tag_str}"

                if h > 0.50:
                    prep_tokens = [self.case_operator.form]
                    prep_gloss = [f"[{self.case_label}:{self.case_operator.id}]"]
                else:
                    post_tokens = [self.case_operator.form]
                    post_gloss = [f"[{self.case_label}:{self.case_operator.id}]"]
        else:
            n_form = base_noun.form
            tags = [self.case_label] if self.case_label != "Nom" else []
            if is_art_bound: tags.append(art_label)
            if is_num_bound: tags.append(self.number_label)
            if is_class_bound and cls_tag: tags.append(cls_tag)
            tag_str = f"[{'+'.join(tags)}]" if tags else ""
            n_gloss = f"{self.head_noun.id}{tag_str}"

        # 6. Modifiers & Determiners Concord
        mod_tokens, mod_gloss = [], []
        if self.modifier:
            if agr_op and s_idx >= 0.30 and len(paradigm.noun_classes) >= 2:
                adj_inflected, is_bound = GrammarEngine.apply_spatial_relator(self.modifier, agr_op, lang, category_type="class")
                if is_bound:
                    mod_tokens.append(adj_inflected.form)
                    mod_gloss.append(f"{self.modifier.id}[Adj{'+' + cls_tag if cls_tag else ''}]")
                else:
                    mod_tokens.extend([agr_op.form, self.modifier.form])
                    mod_gloss.extend([f"[{cls_tag}:{agr_op.id}]", f"{self.modifier.id}[Adj]"])
            else:
                mod_tokens.append(self.modifier.form)
                mod_gloss.append(f"{self.modifier.id}[Adj]")

        det_tokens, det_gloss = [], []
        if self.determiner:
            det_tokens.append(self.determiner.form)
            det_gloss.append(f"[Det:{self.determiner.id}]")

        # 7. Possessor
        poss_tokens, poss_gloss = [], []
        if self.possessor:
            poss_tokens.append(self.possessor.form)
            poss_gloss.append(f"[Poss:{self.possessor.id}]")

        # 8. Word Order Assembly
        if h < 0.35:  # Head-Final / SOV
            tokens.extend(art_tokens + poss_tokens + num_tokens + det_tokens + mod_tokens)
            glosses.extend(art_gloss + poss_gloss + num_gloss + det_gloss + mod_gloss)
            if self.relative_clause:
                rc_ipa, rc_gl = self.relative_clause.render(lang)
                tokens.append(f"[{rc_ipa}]")
                glosses.append(f"[RC:{rc_gl}]")
            tokens.append(n_form)
            glosses.append(n_gloss)
            tokens.extend(post_tokens)
            glosses.extend(post_gloss)

        elif h > 0.65:  # Head-Initial / VSO
            tokens.extend(prep_tokens + art_tokens)
            glosses.extend(prep_gloss + art_gloss)
            tokens.append(n_form)
            glosses.append(n_gloss)
            tokens.extend(num_tokens + mod_tokens + det_tokens + poss_tokens)
            glosses.extend(num_gloss + mod_gloss + det_gloss + poss_gloss)
            if self.relative_clause:
                rc_ipa, rc_gl = self.relative_clause.render(lang)
                tokens.append(f"[{rc_ipa}]")
                glosses.append(f"[RC:{rc_gl}]")

        else:  # SVO
            tokens.extend(prep_tokens + art_tokens + num_tokens + det_tokens)
            glosses.extend(prep_gloss + art_gloss + num_gloss + det_gloss)
            tokens.append(n_form)
            glosses.append(n_gloss)
            tokens.extend(mod_tokens + poss_tokens + post_tokens)
            glosses.extend(mod_gloss + poss_gloss + post_gloss)
            if self.relative_clause:
                rc_ipa, rc_gl = self.relative_clause.render(lang)
                tokens.append(f"[{rc_ipa}]")
                glosses.append(f"[RC:{rc_gl}]")

        return " ".join(tokens), " ".join(glosses)


@dataclass
class Clause:
    """A full hierarchical clause: Predicate + Arguments + TAM + Person Agreement + Subordination."""
    predicate: Word
    valency: ClauseValency
    subject: NounPhrase
    direct_object: NounPhrase | None = None
    indirect_object: NounPhrase | None = None
    oblique_location: NounPhrase | None = None
    temporal_op: Word | None = None
    modal_op: Word | None = None
    agreement_op: Word | None = None
    person_marker: Word | None = None
    person_label: str = "3sg"
    complement_clause: Clause | None = None
    complementizer: Word | None = None
    relativizer: Word | None = None
    is_subordinate: bool = False
    is_past_or_perfective: bool = False

    def render(self, lang: Language) -> tuple[str, str]:
        h = lang.profile.head_directionality
        s_idx = lang.profile.synthesis_index

        inflected_verb, free_particles, is_suppletive, bound_labels = GrammarEngine.inflect_predicate(
            verb=self.predicate,
            temporal_op=self.temporal_op,
            modal_op=self.modal_op,
            agreement_op=self.agreement_op,
            person_op=self.person_marker,
            person_label=self.person_label,
            lang=lang,
            is_past_or_perfective=self.is_past_or_perfective,
        )

        v_tokens = [p.form for p in free_particles] + [inflected_verb.form]
        v_tag = f"{self.predicate.id}[V" + (f"+{'+'.join(bound_labels)}" if bound_labels else "") + "]"
        if is_suppletive:
            v_tag += "[Supp]"
        v_gloss = [f"[Part:{p.id}]" for p in free_particles] + [v_tag]

        subj_ipa, subj_gl = self.subject.render(lang)
        obj_ipa, obj_gl = self.direct_object.render(lang) if self.direct_object else ("", "")
        ind_ipa, ind_gl = self.indirect_object.render(lang) if self.indirect_object else ("", "")
        obl_ipa, obl_gl = self.oblique_location.render(lang) if self.oblique_location else ("", "")

        comp_ipa, comp_gl = ("", "")
        if self.complement_clause:
            c_ipa, c_gl = self.complement_clause.render(lang)
            comp_marker = f"{self.complementizer.form} " if self.complementizer else ""
            comp_m_gl = f"[Comp:{self.complementizer.id}] " if self.complementizer else ""
            comp_ipa = f"{comp_marker}[{c_ipa}]"
            comp_gl = f"{comp_m_gl}[CompP:{c_gl}]"

        clause_ipa, clause_gl = [], []

        if self.relativizer:
            clause_ipa.append(self.relativizer.form)
            clause_gl.append(f"[REL:{self.relativizer.id}]")

        # Pragmatic Scrambling in High-Synthesis Case Languages (Latin / Russian model)
        has_overt_case = (self.subject.case_operator is not None or s_idx >= 0.55)
        scramble_roll = random.random()

        if has_overt_case and s_idx >= 0.50 and scramble_roll < 0.25:
            if scramble_roll < 0.12 and obj_ipa:
                # O S V (Topicalization)
                clause_ipa.extend([obj_ipa, subj_ipa] + ([ind_ipa] if ind_ipa else []) + v_tokens)
                clause_gl.extend([obj_gl, subj_gl] + ([ind_gl] if ind_gl else []) + v_gloss)
            else:
                # V S O (Verb Focus)
                clause_ipa.extend(v_tokens + [subj_ipa] + ([obj_ipa] if obj_ipa else []))
                clause_gl.extend(v_gloss + [subj_gl] + ([obj_gl] if obj_gl else []))

        elif h < 0.35:  # SOV
            clause_ipa.append(subj_ipa); clause_gl.append(subj_gl)
            if ind_ipa: clause_ipa.append(ind_ipa); clause_gl.append(ind_gl)
            if obj_ipa: clause_ipa.append(obj_ipa); clause_gl.append(obj_gl)
            if obl_ipa: clause_ipa.append(obl_ipa); clause_gl.append(obl_gl)
            if comp_ipa: clause_ipa.append(comp_ipa); clause_gl.append(comp_gl)
            clause_ipa.extend(v_tokens); clause_gl.extend(v_gloss)

        elif h > 0.65:  # VSO
            clause_ipa.extend(v_tokens); clause_gl.extend(v_gloss)
            clause_ipa.append(subj_ipa); clause_gl.append(subj_gl)
            if obj_ipa: clause_ipa.append(obj_ipa); clause_gl.append(obj_gl)
            if ind_ipa: clause_ipa.append(ind_ipa); clause_gl.append(ind_gl)
            if obl_ipa: clause_ipa.append(obl_ipa); clause_gl.append(obl_gl)
            if comp_ipa: clause_ipa.append(comp_ipa); clause_gl.append(comp_gl)

        else:  # SVO
            clause_ipa.append(subj_ipa); clause_gl.append(subj_gl)
            clause_ipa.extend(v_tokens); clause_gl.extend(v_gloss)
            if obj_ipa: clause_ipa.append(obj_ipa); clause_gl.append(obj_gl)
            if ind_ipa: clause_ipa.append(ind_ipa); clause_gl.append(ind_gl)
            if obl_ipa: clause_ipa.append(obl_ipa); clause_gl.append(obl_gl)
            if comp_ipa: clause_ipa.append(comp_ipa); clause_gl.append(comp_gl)

        return " ".join(clause_ipa).strip(), " ".join(clause_gl).strip()


class GrammarEngine:
    """Formal morphosyntactic engine managing hybrid non-concatenative and concatenative grammar."""

    @staticmethod
    def calculate_semantic_specificity(word: Word) -> float:
        c, a, _, _, d, s, e = word.vector.coords
        return max(0.02, (c * 1.40 + a * 0.90) - (d + e + s) * 0.20)

    @classmethod
    def get_part_of_speech(cls, word: Word) -> str:
        """Determines Part of Speech via continuous cognitive prototype attractor scores."""
        if word.category == LexicalCategory.FUNCTIONAL_CLOSED:
            return "Particle"
        from translator import SemanticTranslator
        return SemanticTranslator.get_pos(word.vector)

    @classmethod
    def compress_bound_affix(cls, operator: Word) -> list[Phoneme]:
        """Compresses a functional operator into a light 1-2 segment bound affix."""
        p_list = list(operator.phonemes)
        if len(p_list) <= 2:
            return p_list

        if p_list[0].kind == PhonemeKind.CONSONANT and p_list[1].kind == PhonemeKind.VOWEL:
            return [p_list[0], p_list[1]]
        if p_list[0].kind == PhonemeKind.VOWEL and p_list[1].kind == PhonemeKind.CONSONANT:
            return [p_list[0], p_list[1]]

        return p_list[:2]

    @classmethod
    def should_fuse_morpheme(cls, host_word: Word, operator: Word, lang: Language, is_prefix: bool = False) -> bool:
        """Determines if a grammatical clitic fuses into a bound affix (Deterministic)."""
        s_idx = lang.profile.synthesis_index
        if s_idx < 0.30:
            return False

        if host_word.phonemes and operator.phonemes:
            p_host = host_word.phonemes[0] if is_prefix else host_word.phonemes[-1]
            p_op = operator.phonemes[-1] if is_prefix else operator.phonemes[0]
            boundary_friction = ArticulatoryEnergyModel.transition_cost(p_host, p_op)
        else:
            boundary_friction = 0.0

        return (s_idx * 3.50) >= (1.0 + boundary_friction * 0.15)

    @classmethod
    def apply_spatial_relator(
        cls,
        noun: Word,
        operator: Word,
        lang: Language,
        category_type: str = "case",
    ) -> tuple[Word, bool]:
        """Attaches grammatical relators with consistent directional morphology."""
        h = lang.profile.head_directionality
        is_prefix = (category_type == "class") or (h > 0.50)

        if not cls.should_fuse_morpheme(noun, operator, lang, is_prefix=is_prefix):
            return noun.copy(), False

        affix_phonemes = cls.compress_bound_affix(operator)

        m_stem_list = [m.copy() for m in noun.morphemes] if noun.morphemes else [Morpheme(noun.phonemes, root_family_id=noun.derivation.root_family_id)]
        m_case = Morpheme(affix_phonemes, root_family_id=operator.derivation.root_family_id, is_grammatical=True, stage=ClineStage.BOUND_AFFIX)

        morphemes = [m_case] + m_stem_list if is_prefix else m_stem_list + [m_case]
        raw_phonemes = affix_phonemes + noun.phonemes if is_prefix else noun.phonemes + affix_phonemes

        inflected_word = Word(
            vector=noun.vector,
            phonemes=raw_phonemes,
            morphemes=morphemes,
            derivation=DerivationRecord(root_family_id=noun.derivation.root_family_id, derivation_type=DerivationType.AFFIXATION, local_parent_ids=(noun.id,)),
            category=LexicalCategory.CONTENT_OPEN,
            is_ephemeral=True,
        )

        prof = lang.profile if lang else None
        inertia = getattr(lang, "cultural_inertia", 1.0) if lang else 1.0
        repaired = ArticulatoryEnergyModel.repair_word(inflected_word, prof, cultural_inertia=inertia) if prof else inflected_word
        final_word = SyllableEngine.apply_prosody(repaired, pattern=lang.profile.stress_pattern, tone_tier=lang.profile.tone_tier)
        return final_word, True

    @classmethod
    def apply_vocalic_ablaut(cls, word: Word, mode: str = "past", lang: Language | None = None) -> tuple[Word, bool]:
        """Applies non-concatenative vocalic apophony/ablaut (sing -> sang, foot -> feet)."""
        p_list = list(word.phonemes)
        v_indices = [i for i, p in enumerate(p_list) if p.kind == PhonemeKind.VOWEL]
        if not v_indices or not lang:
            return word.copy(), False

        v_idx = v_indices[0]
        orig_v = p_list[v_idx]
        h, b, r = orig_v.point

        if mode == "past":
            if h >= 3.0:
                new_v_point = (0.0, 1.0, 0.0) if b <= 1.0 else (4.0, 2.0, 1.0)
            else:
                new_v_point = (6.0, 2.0, 1.0)
        else:  # mode == 'plural' (Umlaut)
            if b >= 0.8 or h <= 1.5:
                new_v_point = (6.0, 0.0, 0.0) if h >= 3.0 else (4.0, 0.0, 0.0)
            else:
                return word.copy(), False

        p_list[v_idx] = orig_v.drift(point=new_v_point)
        cleaned = ArticulatoryEnergyModel.repair_phonemes(p_list, lang.profile)
        temp_w = Word(vector=word.vector, phonemes=cleaned, is_ephemeral=True)
        ablaut_word = SyllableEngine.apply_prosody(temp_w, pattern=lang.profile.stress_pattern, tone_tier=lang.profile.tone_tier)
        return ablaut_word, True

    @classmethod
    def apply_root_pattern_transfix(cls, word: Word, template: str = "a_a", lang: Language | None = None) -> tuple[Word, bool]:
        """Applies Semitic Root-and-Pattern Transfixation (k-t-b -> kataba / kutiba)."""
        if not lang:
            return word.copy(), False

        cons = [p for p in word.phonemes if p.kind == PhonemeKind.CONSONANT]
        if len(cons) != 3:
            return word.copy(), False

        c1, c2, c3 = cons[0], cons[1], cons[2]
        
        if template == "a_a":
            v1 = Phoneme(PhonemeKind.VOWEL, (0.0, 1.0, 0.0))
            v2 = Phoneme(PhonemeKind.VOWEL, (0.0, 1.0, 0.0))
            raw_p = [c1, v1, c2, v2, c3]
        elif template == "u_i":
            v1 = Phoneme(PhonemeKind.VOWEL, (6.0, 2.0, 1.0))
            v2 = Phoneme(PhonemeKind.VOWEL, (6.0, 0.0, 0.0))
            raw_p = [c1, v1, c2, v2, c3]
        else:
            return word.copy(), False

        cleaned = ArticulatoryEnergyModel.repair_phonemes(raw_p, lang.profile)
        temp_w = Word(vector=word.vector, phonemes=cleaned, is_ephemeral=True)
        transfix_word = SyllableEngine.apply_prosody(temp_w, pattern=lang.profile.stress_pattern, tone_tier=lang.profile.tone_tier)
        return transfix_word, True

    @classmethod
    def apply_reduplication(cls, word: Word, mode: str = "prefix", lang: Language | None = None) -> tuple[Word, bool]:
        """Applies iconic morphological reduplication (CV-copying or full root copying)."""
        if not word.phonemes or not lang:
            return word.copy(), False
        
        syls = SyllableEngine.syllabify(word)
        if not syls:
            return word.copy(), False

        if mode == "prefix":
            first_syl = syls[0]
            copy_p = [p.drift() for p in (first_syl.onset + first_syl.nucleus[:1])]
            raw_p = copy_p + list(word.phonemes)
        else:
            raw_p = list(word.phonemes) + [p.drift() for p in word.phonemes]

        cleaned = ArticulatoryEnergyModel.repair_phonemes(raw_p, lang.profile)
        temp_w = Word(vector=word.vector, phonemes=cleaned, is_ephemeral=True)
        return SyllableEngine.apply_prosody(temp_w, pattern=lang.profile.stress_pattern, tone_tier=lang.profile.tone_tier), True

    @classmethod
    def apply_infixation(cls, word: Word, infix_phonemes: Sequence[Phoneme], lang: Language | None = None) -> tuple[Word, bool]:
        """Inserts an infix bound morpheme inside the root immediately after the onset consonant."""
        if not word.phonemes or len(word.phonemes) < 2 or not lang:
            return word.copy(), False

        p_list = list(word.phonemes)
        onset_idx = 0
        while onset_idx < len(p_list) and p_list[onset_idx].kind == PhonemeKind.CONSONANT:
            onset_idx += 1

        if onset_idx == 0:
            onset_idx = 1

        raw_p = p_list[:onset_idx] + [p.drift() for p in infix_phonemes] + p_list[onset_idx:]
        cleaned = ArticulatoryEnergyModel.repair_phonemes(raw_p, lang.profile)
        temp_w = Word(vector=word.vector, phonemes=cleaned, is_ephemeral=True)
        return SyllableEngine.apply_prosody(temp_w, pattern=lang.profile.stress_pattern, tone_tier=lang.profile.tone_tier), True

    @classmethod
    def discover_grammar_system(cls, lang: Language) -> EmergentParadigm:
        cached = getattr(lang, "_cached_grammar_paradigm", None)
        if cached is not None:
            return cached

        candidates = list(lang.words.values())
        attn = lang.cultural_attention

        s_idx = getattr(lang.profile, "synthesis_index", 0.50)
        pop = getattr(lang, "population", 0.50)
        env = getattr(lang, "environment", None)
        alt = getattr(env, "altitude", 0.0) if env else 0.0
        hum = getattr(env, "humidity", 0.50) if env else 0.50
        veg = env.emergent_vegetation(population=pop, cultural_attention=lang.cultural_attention) if hasattr(env, "emergent_vegetation") else 0.50

        t_strat = 0.15 + (0.80 / (1.0 + math.exp(5.0 * (pop ** 1.2 - 0.45))))

        max_syncretism = max(1, min(3, int(round(1 + 2.5 * pop * s_idx))))
        assigned_word_ids: set[int] = set()
        assigned_family_ids: set[int] = set()
        form_frequencies: Counter = Counter()

        def get_best_candidates(
            domain_score_fn, 
            max_count: int, 
            allow_family_overlap: bool = False, 
            length_penalty_power: float = 1.80,
            ignore_assigned: bool = False
        ) -> list[Word]:
            scored = []
            for w in candidates:
                if not ignore_assigned and w.id in assigned_word_ids:
                    continue
                
                score = domain_score_fn(w)
                if score <= 0.01:
                    continue

                if length_penalty_power > 0.0:
                    score = score / (len(w.phonemes) ** length_penalty_power)
                
                score = score * (w.usage_frequency / cls.calculate_semantic_specificity(w))
                scored.append((score, w))

            if not scored:
                return []

            scored.sort(key=lambda item: item[0], reverse=True)

            selected = []
            for _, w in scored:
                fam = w.derivation.root_family_id
                form_key = w.plain_form

                if form_frequencies[form_key] >= max_syncretism and not ignore_assigned:
                    continue

                if allow_family_overlap or ignore_assigned or fam not in assigned_family_ids or len(selected) < 2:
                    selected.append(w)
                    assigned_word_ids.add(w.id)
                    assigned_family_ids.add(fam)
                    form_frequencies[form_key] += 1
                    if len(selected) >= max_count:
                        break
            return selected

        # 1. Operators & Functors Recruited with Strong Zipfian Length Penalty (Length^1.8)
        modals = get_best_candidates(
            lambda w: (w.vector.potency * 0.5 + (1.0 - w.vector.valence) * 0.5) * (1.0 - w.vector.concreteness * 0.6),
            max_count=6,
            length_penalty_power=1.80
        )
        temporals = get_best_candidates(
            lambda w: w.vector.dynamism * (1.0 - w.vector.concreteness * 0.6),
            max_count=8,
            length_penalty_power=1.80
        )
        spatials = get_best_candidates(
            lambda w: w.vector.extension * (1.0 - w.vector.dynamism * 0.5),
            max_count=8,
            length_penalty_power=1.80
        )
        determiners = get_best_candidates(
            lambda w: (w.vector.extension * 0.6 + (1.0 - w.vector.dynamism) * 0.4) * (1.0 - w.vector.animacy * 0.5),
            max_count=6,
            length_penalty_power=1.80
        )
        relativizers = get_best_candidates(
            lambda w: (w.vector.extension * 0.5 + (1.0 - w.vector.concreteness) * 0.5),
            max_count=4,
            length_penalty_power=1.80
        )
        complementizers = get_best_candidates(
            lambda w: (w.vector.sociality * 0.5 + w.vector.dynamism * 0.5) * (1.0 - w.vector.concreteness * 0.5),
            max_count=4,
            length_penalty_power=1.80
        )

        # 2. Personal Pronouns (Heavy Length Penalty Length^2.40 for 1-2 Segment Cores)
        pronouns: dict[str, Word] = {}
        p_1sg = get_best_candidates(lambda w: w.vector.animacy * 0.60 + (1.0 - w.vector.sociality) * 0.40, max_count=1, allow_family_overlap=True, length_penalty_power=2.40)
        p_2sg = get_best_candidates(lambda w: w.vector.animacy * 0.50 + w.vector.sociality * 0.50, max_count=1, allow_family_overlap=True, length_penalty_power=2.40)
        p_3sg = get_best_candidates(lambda w: w.vector.extension * 0.50 + (1.0 - w.vector.animacy) * 0.50, max_count=1, allow_family_overlap=True, length_penalty_power=2.40)

        if p_1sg: pronouns["1sg"] = p_1sg[0]
        if p_2sg: pronouns["2sg"] = p_2sg[0]
        if p_3sg: pronouns["3sg"] = p_3sg[0]

        p_pl = get_best_candidates(lambda w: w.vector.extension * 0.70 + w.vector.sociality * 0.30, max_count=3, allow_family_overlap=True, length_penalty_power=2.40)
        if len(p_pl) >= 3:
            pronouns["1pl"] = p_pl[0]
            pronouns["2pl"] = p_pl[1]
            pronouns["3pl"] = p_pl[2]
        elif p_pl:
            pronouns["1pl"] = p_pl[0]
            pronouns["2pl"] = p_2sg[0] if p_2sg else p_pl[0]
            pronouns["3pl"] = p_3sg[0] if p_3sg else p_pl[0]

        # 3. Definiteness & Articles
        ext_attn = attn[6]
        soc_attn = attn[5]
        con_attn = attn[0]

        p_def = get_best_candidates(lambda w: (w.vector.extension * 0.65 + (1.0 - w.vector.concreteness) * 0.35) * (1.0 - w.vector.animacy * 0.5), max_count=1, allow_family_overlap=True, length_penalty_power=2.0)
        p_indef = get_best_candidates(lambda w: (1.0 - w.vector.extension) * 0.60 + w.vector.concreteness * 0.40, max_count=1, allow_family_overlap=True, length_penalty_power=2.0)

        def_salience = (ext_attn * 0.45) + (pop * 0.40) - (s_idx * 0.25)
        articles: dict[str, Word] = {}

        if def_salience >= 0.40 and p_def:
            articles["def"] = p_def[0]
            if (pop >= 0.50 or s_idx < 0.40) and p_indef:
                articles["indef"] = p_indef[0]
                art_system = "Definite & Indefinite Articles"
            else:
                art_system = "Definite Article Only"
        else:
            art_system = "Transdefinite (No Articles)"

        # 4. Number System
        num_salience = s_idx * 2.0 + ext_attn * 0.50 - 0.40
        dual_salience = soc_attn * 0.65 + (1.0 - pop) * 0.55 + alt * 0.20 - 0.85
        number_operators: dict[str, Word] = {}

        if num_salience <= 0.30:
            num_system_name = "Transnumeral (No Grammatical Number)"
        elif dual_salience > 0.35:
            num_system_name = "Singular / Dual / Plural"
            ops = get_best_candidates(lambda w: w.vector.extension * 0.70 + w.vector.sociality * 0.30, max_count=2, length_penalty_power=2.0)
            if len(ops) >= 2:
                number_operators["du"], number_operators["pl"] = ops[0], ops[1]
            elif ops:
                number_operators["pl"] = ops[0]
        else:
            num_system_name = "Singular / Plural"
            ops = get_best_candidates(lambda w: w.vector.extension * 0.70 + w.vector.sociality * 0.30, max_count=1, length_penalty_power=2.0)
            if ops:
                number_operators["pl"] = ops[0]

        # 5. Somatic Counting Chain
        hand_load = sum(w.usage_frequency for w in candidates if w.vector.distance_to(SemanticSpace.HAND) < 0.25)
        body_load = sum(w.usage_frequency for w in candidates if w.vector.distance_to(SemanticSpace.BODY) < 0.25)
        cycle_load = sum(w.usage_frequency for w in candidates if w.vector.distance_to(SemanticSpace.MOON) < 0.25)

        sub_base = 5 if (hand_load > 2.2 or (pop < 0.30 and s_idx < 0.35)) else None
        primary_base = 12 if (cycle_load > 2.5 or (hum > 0.65 and ext_attn > 1.25)) else 10
        
        if body_load > 2.5 or (alt > 0.55 or veg > 0.65):
            super_base = 20
            chain_name = f"Hybrid [{'5+' if sub_base else ''}{primary_base}+20] Vigesimal"
        else:
            super_base = 100
            chain_name = f"Hybrid [{'5+' if sub_base else ''}{primary_base}+100] Decimal"

        num_chain = NumeralChain(sub_base=sub_base, primary_base=primary_base, super_base=super_base, name=chain_name)
        digits: dict[int, Word] = {}
        radix_words: dict[str, Word] = {}
        used_digit_ids: set[int] = set()

        d1 = get_best_candidates(
            lambda w: (1.0 - w.vector.extension) * 0.60 + w.vector.concreteness * 0.40, 
            max_count=1, 
            allow_family_overlap=True,
            length_penalty_power=2.0,
            ignore_assigned=True
        )
        if d1:
            digits[1] = d1[0]
            used_digit_ids.add(d1[0].id)

        max_digit_target = primary_base if sub_base is None else 5
        digit_cands = get_best_candidates(
            lambda w: (w.vector.concreteness * 0.50 + w.vector.potency * 0.50) if w.id not in used_digit_ids else 0.0, 
            max_count=max_digit_target, 
            allow_family_overlap=True,
            length_penalty_power=2.0,
            ignore_assigned=True
        )

        for d_idx in range(2, max_digit_target + 1):
            cand_idx = d_idx - 2
            chosen_w = digit_cands[cand_idx] if cand_idx < len(digit_cands) else (d1[0] if d1 else candidates[0])
            digits[d_idx] = chosen_w
            used_digit_ids.add(chosen_w.id)

        radix_cands = get_best_candidates(
            lambda w: (w.vector.extension * 0.80 + w.vector.potency * 0.20) if w.id not in used_digit_ids else 0.0,
            max_count=4,
            allow_family_overlap=True,
            length_penalty_power=1.8,
            ignore_assigned=True
        )

        if sub_base:
            radix_words["hand"] = radix_cands[0] if len(radix_cands) > 0 else (d1[0] if d1 else candidates[0])
        radix_words["base"] = radix_cands[1] if len(radix_cands) > 1 else (d1[0] if d1 else candidates[0])
        radix_words["super"] = radix_cands[2] if len(radix_cands) > 2 else (d1[0] if d1 else candidates[0])
        radix_words["hundred"] = radix_cands[3] if len(radix_cands) > 3 else (d1[0] if d1 else candidates[0])

        # 6. DYNAMIC EMERGENT NOUN CLASSES (Requires >= 2 for contrastive gender/class system)
        attn_weight = (ext_attn * 0.40 + soc_attn * 0.35 + con_attn * 0.25) / max(0.01, sum(attn) / 7.0)
        phi_noun = s_idx * (1.0 - 0.55 * min(1.5, pop)) * attn_weight
        num_classes_calc = int(round(10.0 * (phi_noun - 0.22)))
        num_classes = num_classes_calc if num_classes_calc >= 2 else 0

        emergent_classes: list[EmergentClass] = []
        if num_classes >= 2:
            living_nouns = [w for w in candidates if w.category == LexicalCategory.CONTENT_OPEN and w.vector.dynamism < 0.55]
            if not living_nouns:
                living_nouns = candidates[:num_classes]

            centroids = [living_nouns[0].vector]
            for _ in range(1, num_classes):
                farthest = max(living_nouns, key=lambda w: min(w.vector.weighted_distance_to(c, attn) for c in centroids))
                centroids.append(farthest.vector)

            for i, c_vec in enumerate(centroids):
                c, a, v, p, d, s, e = c_vec.coords
                features = []
                if a >= 0.50: features.append("Animate" if a < 0.75 else "Humanoid")
                else: features.append("Inanimate" if c >= 0.50 else "Abstract")
                if e >= 0.55: features.append("Extended/Shape")
                if p >= 0.60: features.append("Potent/Tool")
                if s >= 0.55: features.append("Social/Kin")

                label = "-".join(features) if features else "General"
                marker_cand = get_best_candidates(lambda w: 1.0 - w.vector.weighted_distance_to(c_vec, attn), max_count=1, allow_family_overlap=True, length_penalty_power=2.0)
                marker = marker_cand[0] if marker_cand else None

                emergent_classes.append(
                    EmergentClass(
                        class_id=i + 1,
                        label=f"Class {i+1} ({label})",
                        centroid=c_vec,
                        marker=marker,
                    )
                )

        # 7. DYNAMIC EMERGENT VERB CONJUGATION CLASSES WITH SYSTEMATIC STRATEGIES
        phi_verb = s_idx * (1.0 - 0.50 * min(1.5, pop))
        num_verb_classes = max(0, min(4, int(round(4.0 * (phi_verb - 0.18)))))

        emergent_verb_classes: list[EmergentVerbClass] = []
        if num_verb_classes > 0:
            living_verbs = [w for w in candidates if w.category == LexicalCategory.CONTENT_OPEN and w.vector.dynamism >= 0.50]
            if not living_verbs:
                living_verbs = candidates[:num_verb_classes]

            v_centroids = [living_verbs[0].vector]
            for _ in range(1, num_verb_classes):
                farthest_v = max(living_verbs, key=lambda w: min(w.vector.weighted_distance_to(c, attn) for c in v_centroids))
                v_centroids.append(farthest_v.vector)

            aspect_flavors = ["Aorist/Perf", "Durative/Imperf", "Directional/Telic", "Telic/Punctual"]

            for i, c_vec in enumerate(v_centroids):
                c, a, v, p, d, s, e = c_vec.coords
                features = []
                
                # Multi-Strategy Selection Modulated by Demographic Temperature (T_strat)
                if p >= 0.70 and d >= 0.70 and (random.random() < t_strat):
                    features.append("Dynamic/Action")
                    strategy = "ablaut" if (s_idx >= 0.40 and hum < 0.40) else "concatenative"
                elif s >= 0.60 and (random.random() < t_strat * 0.70):
                    features.append("Social/Communication")
                    strategy = "transfix" if (s_idx >= 0.50 and hum < 0.25) else "concatenative"
                elif d >= 0.80 and (random.random() < t_strat * 0.60):
                    features.append("Iterative/Continuous")
                    strategy = "reduplication"
                else:
                    features.append("General")
                    strategy = "concatenative"

                label = "-".join(features) if features else "General"
                theme_cand = get_best_candidates(
                    lambda w: 1.0 - w.vector.weighted_distance_to(c_vec, attn),
                    max_count=1,
                    allow_family_overlap=True,
                    length_penalty_power=2.0
                )

                emergent_verb_classes.append(
                    EmergentVerbClass(
                        class_id=i + 1,
                        label=f"Class {i+1} ({label})" if num_verb_classes > 1 else "Universal Regular Conjugation",
                        centroid=c_vec,
                        thematic_marker=theme_cand[0] if theme_cand else None,
                        aspect_label=aspect_flavors[i % len(aspect_flavors)],
                        morph_strategy=strategy,
                    )
                )

        paradigm = EmergentParadigm(
            temporal_operators=temporals,
            spatial_operators=spatials,
            determiners=determiners,
            modal_operators=modals,
            noun_classes=emergent_classes,
            verb_classes=emergent_verb_classes,
            relativizers=relativizers,
            complementizers=complementizers,
            pronouns=pronouns,
            number_system=num_system_name,
            number_operators=number_operators,
            article_system=art_system,
            articles=articles,
            numeral_chain=num_chain,
            digits=digits,
            radix_words=radix_words,
        )

        lang._cached_grammar_paradigm = paradigm
        return paradigm

    @classmethod
    def format_numeral(cls, value: int, lang: Language, paradigm: EmergentParadigm | None = None) -> str:
        para = paradigm or cls.discover_grammar_system(lang)
        if value in para.formatted_cache:
            return para.formatted_cache[value]

        if value in para.digits:
            result = f"/{para.digits[value].form}/"
            para.formatted_cache[value] = result
            return result

        chain = para.numeral_chain
        h = lang.profile.head_directionality
        s_idx = lang.profile.synthesis_index

        val_rem = value
        blocks: list[list[Phoneme]] = []

        if chain.super_base and val_rem >= chain.super_base:
            q_super, val_rem = divmod(val_rem, chain.super_base)
            super_word = para.radix_words.get("super") or para.radix_words.get("base")
            super_p = list(super_word.phonemes) if super_word else []

            if q_super > 1 and q_super in para.digits:
                mult_p = list(para.digits[q_super].phonemes)
                blocks.append(mult_p + super_p if h < 0.5 else super_p + mult_p)
            elif super_p:
                blocks.append(super_p)

        if val_rem >= chain.primary_base:
            q_prim, val_rem = divmod(val_rem, chain.primary_base)
            base_word = para.radix_words.get("base")
            base_p = list(base_word.phonemes) if base_word else []

            if q_prim > 1 and q_prim in para.digits:
                mult_p = list(para.digits[q_prim].phonemes)
                blocks.append(mult_p + base_p if h < 0.5 else base_p + mult_p)
            elif base_p:
                blocks.append(base_p)

        if chain.sub_base and val_rem >= chain.sub_base:
            hand_word = para.radix_words.get("hand", para.radix_words.get("base"))
            if hand_word:
                blocks.append(list(hand_word.phonemes))
            val_rem = val_rem - chain.sub_base

        if val_rem > 0:
            if val_rem in para.digits:
                blocks.append(list(para.digits[val_rem].phonemes))
            elif 1 in para.digits:
                unit_p = list(para.digits[1].phonemes)
                for _ in range(val_rem):
                    blocks.append(list(unit_p))

        ordered_blocks = blocks if h < 0.50 else list(reversed(blocks))

        mag_scale = min(1.0, math.log10(max(1, value) + 1.0) / 2.5)
        num_vec = SemanticVector(
            concreteness=0.10,
            animacy=0.0,
            valence=0.50,
            potency=0.20 + (mag_scale * 0.60),
            dynamism=0.0,
            sociality=0.10,
            extension=0.20 + (mag_scale * 0.75),
        )

        if s_idx >= 0.50:
            combined_p: list[Phoneme] = []
            for b in ordered_blocks:
                combined_p.extend(b)
            temp_word = Word(vector=num_vec, phonemes=combined_p)
            repaired = ArticulatoryEnergyModel.repair_word(temp_word, lang.profile)
            prosodified = SyllableEngine.apply_prosody(repaired, pattern=lang.profile.stress_pattern, tone_tier=lang.profile.tone_tier)
            result = f"/{prosodified.form}/"
        else:
            phrased_tokens: list[str] = []
            for b in ordered_blocks:
                w_tok = Word(vector=num_vec, phonemes=b)
                w_rep = ArticulatoryEnergyModel.repair_word(w_tok, lang.profile)
                w_pro = SyllableEngine.apply_prosody(w_rep, pattern=lang.profile.stress_pattern, tone_tier=lang.profile.tone_tier)
                phrased_tokens.append(w_pro.form)
            result = f"/{' '.join(phrased_tokens)}/"

        para.formatted_cache[value] = result
        return result

    @classmethod
    def classify_noun_dynamically(cls, word: Word, paradigm: EmergentParadigm, lang: Language) -> EmergentClass | None:
        if not paradigm.noun_classes or len(paradigm.noun_classes) < 2:
            return None
        return min(paradigm.noun_classes, key=lambda e_cls: word.vector.weighted_distance_to(e_cls.centroid, lang.cultural_attention))

    @classmethod
    def classify_verb_dynamically(cls, word: Word, paradigm: EmergentParadigm, lang: Language) -> EmergentVerbClass:
        if not paradigm.verb_classes:
            return EmergentVerbClass(class_id=1, label="Universal Verb", centroid=SemanticSpace.MAKE)
        return min(paradigm.verb_classes, key=lambda v_cls: word.vector.weighted_distance_to(v_cls.centroid, lang.cultural_attention))

    @classmethod
    def get_valency_frame(cls, verb: Word) -> ClauseValency:
        c, a, v, p, d, s, e = verb.vector.coords
        transitivity_score = (d * 0.50) + (p * 0.30) + (s * 0.20)
        if transitivity_score >= 0.65 and e >= 0.45:
            return ClauseValency.DITRANSITIVE
        if transitivity_score >= 0.45:
            return ClauseValency.TRANSITIVE
        return ClauseValency.INTRANSITIVE

    @classmethod
    def get_alignment_system(cls, lang: Language) -> str:
        return "Ergative-Absolutive" if lang.cultural_attention[3] > (lang.cultural_attention[5] * 1.15) else "Nominative-Accusative"

    @classmethod
    def resolve_suppletive_stem(cls, verb: Word, lang: Language, is_past_or_perfective: bool = False) -> tuple[Word, bool]:
        if not is_past_or_perfective or verb.usage_frequency < 4.0:
            return verb, False

        competitors = [
            w for w in lang.words.values()
            if w.id != verb.id
            and w.derivation.derivation_type == DerivationType.ROOT
            and w.vector.distance_to(verb.vector) < 0.18
            and w.usage_frequency >= 2.0
            and w.vector.dynamism >= 0.50
        ]
        if not competitors:
            return verb, False

        return max(competitors, key=lambda w: w.usage_frequency), True

    @classmethod
    def get_thematic_verb_stem(cls, verb: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> tuple[Word, EmergentVerbClass]:
        para = paradigm or cls.discover_grammar_system(lang)
        v_cls = cls.classify_verb_dynamically(verb, para, lang)
        s_idx = lang.profile.synthesis_index

        if v_cls.thematic_marker and s_idx >= 0.35:
            theme_stem, is_tb = cls.apply_spatial_relator(verb, v_cls.thematic_marker, lang, category_type="inf")
            return (theme_stem if is_tb else verb), v_cls
        return verb, v_cls

    @classmethod
    def inflect_predicate(
        cls,
        verb: Word,
        temporal_op: Word | None,
        modal_op: Word | None,
        agreement_op: Word | None,
        person_op: Word | None = None,
        person_label: str = "3sg",
        lang: Language | None = None,
        is_past_or_perfective: bool = False,
    ) -> tuple[Word, list[Word], bool, list[str]]:
        stem_word, was_suppletive = cls.resolve_suppletive_stem(verb, lang, is_past_or_perfective) if lang else (verb, False)
        bound_labels = []

        para = cls.discover_grammar_system(lang) if lang else None
        v_cls = cls.classify_verb_dynamically(stem_word, para, lang) if para and lang else None

        used_non_concatenative = False
        if lang and is_past_or_perfective and v_cls:
            if v_cls.morph_strategy == "transfix":
                trans_word, is_trans = cls.apply_root_pattern_transfix(stem_word, template="a_a", lang=lang)
                if is_trans:
                    stem_word = trans_word
                    bound_labels.append("TransfixPast")
                    used_non_concatenative = True
            elif v_cls.morph_strategy == "ablaut" or stem_word.usage_frequency >= 2.0:
                abl_word, is_abl = cls.apply_vocalic_ablaut(stem_word, mode="past", lang=lang)
                if is_abl:
                    stem_word = abl_word
                    bound_labels.append("AblautPast")
                    used_non_concatenative = True
            elif v_cls.morph_strategy == "reduplication":
                redup_word, is_redup = cls.apply_reduplication(stem_word, mode="prefix" if lang.profile.head_directionality > 0.5 else "full", lang=lang)
                if is_redup:
                    stem_word = redup_word
                    bound_labels.append("RedupAspect")
                    used_non_concatenative = True

        if lang and not used_non_concatenative:
            stem_word, _ = cls.get_thematic_verb_stem(stem_word, lang)

        current_host = stem_word
        prefix_stack = []
        suffix_stack = []
        free_particles = []

        active_temporal_op = None if used_non_concatenative else temporal_op

        ops_with_labels = [
            (active_temporal_op, "TAM"),
            (modal_op, "Mod"),
            (agreement_op, "Agr"),
            (person_op, person_label),
        ]

        h = lang.profile.head_directionality if lang else 0.50
        p_prefix = 1.0 / (1.0 + math.exp(-8.0 * (h - 0.50)))

        for op, label in ops_with_labels:
            if op is None:
                continue
            if lang and cls.should_fuse_morpheme(current_host, op, lang):
                affix_phonemes = cls.compress_bound_affix(op)
                m_op = Morpheme(affix_phonemes, root_family_id=op.derivation.root_family_id, is_grammatical=True, stage=ClineStage.BOUND_AFFIX)
                if random.random() < p_prefix:
                    prefix_stack.append(m_op)
                else:
                    suffix_stack.append(m_op)
                bound_labels.append(label)
                current_host = Word(vector=stem_word.vector, phonemes=stem_word.phonemes + affix_phonemes, is_ephemeral=True)
            else:
                free_particles.append(op)

        if not prefix_stack and not suffix_stack:
            return stem_word.copy(), free_particles, was_suppletive, bound_labels

        m_stem = Morpheme(stem_word.phonemes, root_family_id=stem_word.derivation.root_family_id)
        all_morphemes = prefix_stack + [m_stem] + suffix_stack

        raw_phonemes = []
        for m in all_morphemes:
            raw_phonemes.extend(m.phonemes)

        inflected_verb = Word(
            vector=stem_word.vector,
            phonemes=raw_phonemes,
            morphemes=all_morphemes,
            derivation=DerivationRecord(root_family_id=stem_word.derivation.root_family_id, derivation_type=DerivationType.AFFIXATION, local_parent_ids=(stem_word.id,)),
            category=LexicalCategory.CONTENT_OPEN,
            is_ephemeral=True,
        )

        prof = lang.profile if lang else None
        inertia = getattr(lang, "cultural_inertia", 1.0) if lang else 1.0
        repaired = ArticulatoryEnergyModel.repair_word(inflected_verb, prof, cultural_inertia=inertia) if prof else inflected_verb
        stress_p = prof.stress_pattern if prof else StressPattern.PENULTIMATE
        t_tier = prof.tone_tier if prof else 0
        final_verb = SyllableEngine.apply_prosody(repaired, pattern=stress_p, tone_tier=t_tier)
        return final_verb, free_particles, was_suppletive, bound_labels

    @classmethod
    def get_verb_conjugation_table(cls, verb: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> dict[str, str]:
        para = paradigm or cls.discover_grammar_system(lang)
        conj: dict[str, str] = {}
        s_idx = lang.profile.synthesis_index

        theme_stem, v_cls = cls.get_thematic_verb_stem(verb, lang, para)

        inf_op = None
        if len(para.spatial_operators) > 3:
            inf_op = para.spatial_operators[3]
        elif para.complementizers:
            inf_op = para.complementizers[0]
        elif para.spatial_operators:
            inf_op = para.spatial_operators[-1]

        if inf_op and s_idx >= 0.35:
            inf_word, is_b = cls.apply_spatial_relator(theme_stem, inf_op, lang, category_type="inf")
            conj["Infinitive"] = f"/{inf_word.form}/" if is_b else f"/{inf_op.form} {theme_stem.form}/"
        else:
            conj["Infinitive"] = f"/{theme_stem.form}/"

        ptcp_op = None
        if len(para.spatial_operators) > 4:
            ptcp_op = para.spatial_operators[4]
        elif len(para.relativizers) > 0:
            ptcp_op = para.relativizers[0]
        elif len(para.spatial_operators) > 1:
            ptcp_op = para.spatial_operators[1]

        if ptcp_op and s_idx >= 0.30:
            ptcp_word, is_b = cls.apply_spatial_relator(theme_stem, ptcp_op, lang, category_type="inf")
            conj["Participle"] = f"/{ptcp_word.form}/" if is_b else f"/{ptcp_op.form} {theme_stem.form}/"

        persons = ["1sg", "2sg", "3sg", "1pl", "2pl", "3pl"]
        for p_key in persons:
            p_op = para.pronouns.get(p_key)
            v_form, free_p, was_s, labels = cls.inflect_predicate(
                verb, temporal_op=None, modal_op=None, agreement_op=None,
                person_op=p_op, person_label=p_key, lang=lang, is_past_or_perfective=False
            )
            tokens = [p.form for p in free_p] + [v_form.form]
            conj[f"Pres.{p_key}"] = f"/{' '.join(tokens)}/"

        past_tag = "Past." + v_cls.aspect_label[:4]
        t_op = para.temporal_operators[0] if para.temporal_operators else None
        for p_key in ["1sg", "3sg", "1pl"]:
            p_op = para.pronouns.get(p_key)
            v_form, free_p, was_s, labels = cls.inflect_predicate(
                verb, temporal_op=t_op, modal_op=None, agreement_op=None,
                person_op=p_op, person_label=p_key, lang=lang, is_past_or_perfective=True
            )
            tokens = [p.form for p in free_p] + [v_form.form]
            conj[f"{past_tag}.{p_key}"] = f"/{' '.join(tokens)}/"

        return conj

    @classmethod
    def get_adjective_concord_table(cls, adj: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> dict[str, str]:
        para = paradigm or cls.discover_grammar_system(lang)
        concord_grid: dict[str, str] = {}
        s_idx = lang.profile.synthesis_index

        if not para.noun_classes or len(para.noun_classes) < 2:
            return {"Base": f"/{adj.form}/"}

        for n_cls in para.noun_classes:
            tag = f"Class {n_cls.class_id}"
            if n_cls.marker and s_idx >= 0.30:
                inflected, is_b = cls.apply_spatial_relator(adj, n_cls.marker, lang, category_type="class")
                concord_grid[tag] = f"/{inflected.form}/" if is_b else f"/{adj.form} {n_cls.marker.form}/"
            else:
                concord_grid[tag] = f"/{adj.form}/"

        return concord_grid

    @classmethod
    def render_lemma(cls, word: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> tuple[str, str]:
        para = paradigm or cls.discover_grammar_system(lang)
        alignment = cls.get_alignment_system(lang)
        s_idx = lang.profile.synthesis_index
        pos = cls.get_part_of_speech(word)

        if pos == "Particle":
            return word.form, "Particle / Closed"

        if pos == "Verb":
            theme_stem, v_cls = cls.get_thematic_verb_stem(word, lang, para)
            inf_op = None
            if len(para.spatial_operators) > 3:
                inf_op = para.spatial_operators[3]
            elif para.complementizers:
                inf_op = para.complementizers[0]
            elif para.spatial_operators:
                inf_op = para.spatial_operators[-1]

            if inf_op and s_idx >= 0.35:
                inf_word, is_b = cls.apply_spatial_relator(theme_stem, inf_op, lang, category_type="inf")
                if is_b:
                    return inf_word.form, f"Verb [Infinitive] | {v_cls.label}"
                return f"{inf_op.form} {theme_stem.form}", f"Verb [Infinitive] | {v_cls.label}"
            return theme_stem.form, f"Verb [Bare Infinitive] | {v_cls.label}"

        if pos == "Adjective":
            if para.noun_classes and len(para.noun_classes) >= 2 and para.noun_classes[0].marker:
                c1 = para.noun_classes[0]
                if s_idx >= 0.30:
                    inflected_adj, is_bound = cls.apply_spatial_relator(word, c1.marker, lang, category_type="class")
                    if is_bound:
                        return inflected_adj.form, f"Adjective [Class 1 Concord] | {c1.label}"
                    return f"{word.form} {c1.marker.form}", f"Adjective [Class 1 Concord] | {c1.label}"
                return word.form, f"Adjective [Class 1 Concord (Analytic)] | {c1.label}"
            return word.form, "Adjective [Base Descriptor]"

        assigned_cls = cls.classify_noun_dynamically(word, para, lang)
        class_label = assigned_cls.label if assigned_cls else "Neutral"
        citation_case = "Abs" if alignment == "Ergative-Absolutive" else "Nom"

        if assigned_cls and assigned_cls.marker and s_idx >= 0.35 and len(para.noun_classes) >= 2:
            lemma_word, is_bound = cls.apply_spatial_relator(word, assigned_cls.marker, lang, category_type="class")
            if is_bound:
                return lemma_word.form, f"Noun [{citation_case}.Sg] | {class_label}"

        return word.form, f"Noun [{citation_case}.Sg] | {class_label}"

    @classmethod
    def get_noun_declension_table(cls, noun: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> dict[str, str]:
        para = paradigm or cls.discover_grammar_system(lang)
        alignment = cls.get_alignment_system(lang)
        grid: dict[str, str] = {}
        s_idx = lang.profile.synthesis_index
        h = lang.profile.head_directionality
        is_prefix = (h > 0.50)

        assigned_cls = cls.classify_noun_dynamically(noun, para, lang)
        if assigned_cls and assigned_cls.marker and s_idx >= 0.35 and len(para.noun_classes) >= 2:
            base_lemma, _ = cls.apply_spatial_relator(noun, assigned_cls.marker, lang, category_type="class")
        else:
            base_lemma = noun

        number_stems: dict[str, Word] = {"Sg": base_lemma}
        
        for num_key in ("du", "pl"):
            num_label = num_key.capitalize()
            if num_key == "pl" and (noun.usage_frequency >= 2.0 or s_idx >= 0.40):
                umlaut_stem, is_uml = cls.apply_vocalic_ablaut(base_lemma, mode="plural", lang=lang)
                if is_uml:
                    number_stems[num_label] = umlaut_stem
                    continue

            if num_key in para.number_operators and s_idx >= 0.30:
                num_word, is_nb = cls.apply_spatial_relator(base_lemma, para.number_operators[num_key], lang, category_type="number")
                number_stems[num_label] = num_word if is_nb else base_lemma
            elif num_key in para.number_operators:
                number_stems[num_label] = base_lemma

        available_relators = para.spatial_operators
        def find_relator(vec: SemanticVector) -> Word | None:
            return min(available_relators, key=lambda w: w.vector.distance_to(vec)) if available_relators else None

        loc_op = find_relator(SemanticVector(0.20, 0.00, 0.50, 0.20, 0.00, 0.00, 0.10))
        dat_op = find_relator(SemanticVector(0.30, 0.70, 0.70, 0.30, 0.40, 0.80, 0.50))
        acc_op = find_relator(SemanticVector(0.20, 0.00, 0.50, 0.30, 0.00, 0.00, 0.50))

        for num_label, n_stem in number_stems.items():
            if alignment == "Ergative-Absolutive":
                grid[f"Abs.{num_label}"] = f"/{n_stem.form}/"
                if acc_op:
                    inflected, is_b = cls.apply_spatial_relator(n_stem, acc_op, lang, category_type="case")
                    grid[f"Erg.{num_label}"] = f"/{inflected.form}/" if is_b else (
                        f"/{acc_op.form} {n_stem.form}/" if is_prefix else f"/{n_stem.form} {acc_op.form}/"
                    )
            else:
                grid[f"Nom.{num_label}"] = f"/{n_stem.form}/"
                if acc_op and s_idx >= 0.45:
                    inflected, is_b = cls.apply_spatial_relator(n_stem, acc_op, lang, category_type="case")
                    grid[f"Acc.{num_label}"] = f"/{inflected.form}/" if is_b else (
                        f"/{acc_op.form} {n_stem.form}/" if is_prefix else f"/{n_stem.form} {acc_op.form}/"
                    )
                else:
                    grid[f"Acc.{num_label}"] = f"/{n_stem.form}/"

            if dat_op:
                inflected, is_b = cls.apply_spatial_relator(n_stem, dat_op, lang, category_type="case")
                grid[f"Dat.{num_label}"] = f"/{inflected.form}/" if is_b else (
                    f"/{dat_op.form} {n_stem.form}/" if is_prefix else f"/{n_stem.form} {dat_op.form}/"
                )

            if loc_op:
                inflected, is_b = cls.apply_spatial_relator(n_stem, loc_op, lang, category_type="case")
                grid[f"Loc.{num_label}"] = f"/{inflected.form}/" if is_b else (
                    f"/{loc_op.form} {n_stem.form}/" if is_prefix else f"/{n_stem.form} {loc_op.form}/"
                )

        return grid

    @classmethod
    def assemble_complex_clause(
        cls,
        lang: Language,
        subject: NounPhrase,
        predicate: Word,
        direct_object: NounPhrase | None = None,
        indirect_object: NounPhrase | None = None,
        oblique_location: NounPhrase | None = None,
        complement_clause: Clause | None = None,
        relativizer: Word | None = None,
        is_past_or_perfective: bool = False,
    ) -> Clause:
        paradigm = cls.discover_grammar_system(lang)
        valency = cls.get_valency_frame(predicate)
        alignment = cls.get_alignment_system(lang)
        s_idx = lang.profile.synthesis_index
        h = lang.profile.head_directionality

        temporal_op = paradigm.temporal_operators[0] if paradigm.temporal_operators else None
        modal_op = paradigm.modal_operators[0] if paradigm.modal_operators else None
        
        subj_class = cls.classify_noun_dynamically(subject.head_noun, paradigm, lang)
        subject.assigned_class = subj_class
        subject.is_subject = True
        agreement_op = subj_class.marker if (subj_class and len(paradigm.noun_classes) >= 2) else None

        person_op = paradigm.pronouns.get("3sg")
        person_label = "3sg"

        if direct_object:
            direct_object.assigned_class = cls.classify_noun_dynamically(direct_object.head_noun, paradigm, lang)

        available_relators = paradigm.spatial_operators
        def find_closest_relator(target_vec: SemanticVector) -> Word | None:
            if not available_relators:
                return None
            return min(available_relators, key=lambda w: w.vector.distance_to(target_vec))

        # 1. Subject & Direct Object Case Marking
        if alignment == "Ergative-Absolutive":
            if direct_object:
                erg_vec = SemanticVector(0.40, 0.80, 0.50, 0.85, 0.80, 0.30, 0.20)
                subject.case_operator = find_closest_relator(erg_vec)
                subject.case_label = "Erg" if subject.case_operator else "Abs"
                direct_object.case_operator = None
                direct_object.case_label = "Abs"
            else:
                subject.case_operator = None
                subject.case_label = "Abs"
        else:
            # Nominative-Accusative
            if (s_idx >= 0.50 or (h < 0.35 and available_relators)) and available_relators:
                nom_vec = SemanticVector(0.50, 0.80, 0.60, 0.50, 0.20, 0.50, 0.10)
                subject.case_operator = find_closest_relator(nom_vec)
                subject.case_label = "Nom" if subject.case_operator else "Nom(Ø)"
            else:
                subject.case_operator = None
                subject.case_label = "Nom(Ø)"
            
            if direct_object:
                is_animate = (direct_object.head_noun.vector.animacy >= 0.55)
                needs_accusative = (s_idx >= 0.45 or is_animate or h < 0.35)
                
                if needs_accusative and available_relators:
                    acc_vec = SemanticVector(0.20, 0.00, 0.50, 0.30, 0.00, 0.00, 0.50)
                    direct_object.case_operator = find_closest_relator(acc_vec)
                    direct_object.case_label = "Acc" if direct_object.case_operator else "Acc(Ø)"
                else:
                    direct_object.case_operator = None
                    direct_object.case_label = "Acc(Ø)"

        if indirect_object:
            dat_vec = SemanticVector(0.30, 0.70, 0.70, 0.30, 0.40, 0.80, 0.50)
            indirect_object.case_operator = find_closest_relator(dat_vec)
            indirect_object.case_label = "Dat" if indirect_object.case_operator else "Dat(Ø)"

        if oblique_location:
            loc_vec = SemanticVector(0.20, 0.00, 0.50, 0.20, 0.00, 0.00, 0.10)
            oblique_location.case_operator = find_closest_relator(loc_vec)
            oblique_location.case_label = "Loc" if oblique_location.case_operator else "Loc(Ø)"

        complementizer = paradigm.complementizers[0] if (complement_clause and paradigm.complementizers) else None

        return Clause(
            predicate=predicate,
            valency=valency,
            subject=subject,
            direct_object=direct_object,
            indirect_object=indirect_object,
            oblique_location=oblique_location,
            temporal_op=temporal_op,
            modal_op=modal_op,
            agreement_op=agreement_op,
            person_marker=person_op,
            person_label=person_label,
            complement_clause=complement_clause,
            complementizer=complementizer,
            relativizer=relativizer,
            is_subordinate=(relativizer is not None or complementizer is not None),
            is_past_or_perfective=is_past_or_perfective,
        )