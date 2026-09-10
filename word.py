"""
word.py — High-Performance Lexical Item with Lazy Form Caching and Atomic Synchronization.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Sequence

from phoneme import Phoneme, PhonemeKind
from semantics import SemanticVector, DerivationRecord, DerivationType, ClineStage


class LexicalCategory(Enum):
    CONTENT_OPEN = "content_open"
    FUNCTIONAL_CLOSED = "functional"


class Morpheme:
    """A sub-word unit carrying its phonetic payload and derivation lineage."""

    __slots__ = ('phonemes', 'root_family_id', 'is_grammatical', 'stage')

    def __init__(
        self,
        phonemes: Sequence[Phoneme],
        root_family_id: int,
        is_grammatical: bool = False,
        stage: ClineStage = ClineStage.FREE_ROOT,
    ):
        self.phonemes: list[Phoneme] = list(phonemes)
        self.root_family_id: int = root_family_id
        self.is_grammatical: bool = is_grammatical
        self.stage: ClineStage = stage

    @property
    def ipa(self) -> str:
        return "".join(p.ipa for p in self.phonemes)

    def copy(self) -> Morpheme:
        return Morpheme(
            phonemes=list(self.phonemes),
            root_family_id=self.root_family_id,
            is_grammatical=self.is_grammatical,
            stage=self.stage,
        )

    def __repr__(self) -> str:
        return f"Morpheme(/{self.ipa}/, root_family={self.root_family_id}, stage={self.stage.value})"


class Word:
    """Struct-packed living word with lazy form caching and capacity-relative memory decay."""

    __slots__ = (
        'id', 'vector', '_phonemes', '_syllables', 'morphemes',
        'derivation', 'usage_frequency', 'category', 'generation_born',
        'epochs_idle', 'senses', '_form', '_plain_form'
    )

    _ID_COUNTER = 1000

    def __init__(
        self,
        vector: SemanticVector,
        phonemes: Sequence[Phoneme] | None = None,
        syllables: Sequence | None = None,
        morphemes: Sequence[Morpheme] | None = None,
        derivation: DerivationRecord | None = None,
        word_id: int | None = None,
        usage_frequency: float = 1.0,
        category: LexicalCategory = LexicalCategory.CONTENT_OPEN,
        generation_born: int = 0,
        senses: Sequence[SemanticVector] | tuple[SemanticVector, ...] | None = None,
        is_ephemeral: bool = False,
    ):
        if word_id is not None:
            self.id = word_id
        elif is_ephemeral:
            self.id = -1
        else:
            Word._ID_COUNTER += 1
            self.id = Word._ID_COUNTER

        self.vector = vector
        self._syllables = list(syllables) if syllables else []
        self.morphemes = list(morphemes) if morphemes else []
        self._form = None
        self._plain_form = None

        if syllables:
            flat = []
            for s in self._syllables:
                flat.extend(s.to_phonemes())
            self._phonemes = flat
        elif phonemes is not None:
            self._phonemes = list(phonemes)
        elif self.morphemes:
            flat = []
            for m in self.morphemes:
                flat.extend(m.phonemes)
            self._phonemes = flat
        else:
            self._phonemes = []

        self.derivation = derivation or DerivationRecord(root_family_id=self.id)
        self.usage_frequency = max(0.01, float(usage_frequency))
        self.category = category
        self.generation_born = generation_born
        self.epochs_idle = 0

        if senses is not None:
            self.senses = senses if isinstance(senses, tuple) else tuple(senses)
        else:
            self.senses = (self.vector,)

        # Guarantee single root morpheme synchronization
        if not self.morphemes and self._phonemes:
            self.morphemes = [
                Morpheme(
                    phonemes=list(self._phonemes),
                    root_family_id=self.derivation.root_family_id,
                    is_grammatical=(self.category == LexicalCategory.FUNCTIONAL_CLOSED),
                )
            ]
        elif len(self.morphemes) == 1 and self._phonemes:
            self.morphemes[0].phonemes = list(self._phonemes)

    @property
    def phonemes(self) -> list[Phoneme]:
        return self._phonemes

    @phonemes.setter
    def phonemes(self, new_phonemes: Sequence[Phoneme]) -> None:
        self._phonemes = list(new_phonemes)
        self._form = None
        self._plain_form = None

    @property
    def syllables(self) -> list:
        return self._syllables

    @syllables.setter
    def syllables(self, new_syllables: Sequence) -> None:
        self._syllables = list(new_syllables)
        # Automatic Single-Source-of-Truth Resynchronization:
        if self._syllables:
            flat = []
            for s in self._syllables:
                flat.extend(s.to_phonemes())
            self._phonemes = flat
        self._form = None
        self._plain_form = None

    @property
    def form(self) -> str:
        if self._form is None:
            if self._syllables:
                self._form = "".join(getattr(s, "form", "") for s in self._syllables)
            elif self._phonemes:
                self._form = "".join(p.ipa for p in self._phonemes)
            else:
                self._form = ""
        return self._form

    @property
    def plain_form(self) -> str:
        if self._plain_form is None:
            raw = self.form
            self._plain_form = raw.replace("ˈ", "")
        return self._plain_form

    @property
    def is_proto_root(self) -> bool:
        return (
            self.derivation.derivation_type == DerivationType.ROOT
            and self.generation_born == 0
        )

    def add_sense(self, new_vec: SemanticVector) -> bool:
        max_senses = 1 + int(round(1.80 * math.log10(self.usage_frequency + 1.0)))
        max_senses = max(1, min(5, max_senses))

        if len(self.senses) < max_senses:
            if all(new_vec.distance_to(s) >= 0.15 for s in self.senses):
                self.senses = self.senses + (new_vec,)
                return True
        return False

    @property
    def phonetic_drift_multiplier(self) -> float:
        log_u = math.log(max(0.10, self.usage_frequency) + 1.0)
        return max(0.40, min(2.20, 0.50 + log_u * 0.60))

    @property
    def semantic_drift_multiplier(self) -> float:
        log_u = math.log(max(0.10, self.usage_frequency) + 1.0)
        return max(0.50, min(1.80, 0.60 + log_u * 0.40))

    def record_usage(self, boost: float = 1.0) -> None:
        self.usage_frequency += boost
        self.epochs_idle = 0

    def is_obsolete(self, max_idle_epochs: int = 4, cultural_inertia: float = 1.0) -> bool:
        """Evaluates memory decay: Higher cultural inertia retains idle words longer."""
        if self.category == LexicalCategory.FUNCTIONAL_CLOSED or self.is_proto_root:
            return False
        effective_max_idle = max(2, int(round(max_idle_epochs * cultural_inertia)))
        return (self.epochs_idle >= effective_max_idle) and (self.usage_frequency < 0.30)

    def copy(self) -> Word:
        w = Word(
            word_id=self.id,
            vector=self.vector,
            phonemes=list(self._phonemes),
            syllables=list(self._syllables),
            morphemes=[m.copy() for m in self.morphemes],
            derivation=self.derivation,
            usage_frequency=self.usage_frequency,
            category=self.category,
            generation_born=self.generation_born,
            senses=self.senses,
        )
        w._form = self._form
        w._plain_form = self._plain_form
        return w

    def __len__(self) -> int:
        return len(self._phonemes)

    def __getitem__(self, idx: int) -> Phoneme:
        return self._phonemes[idx]

    def __iter__(self):
        return iter(self._phonemes)

    def __repr__(self) -> str:
        cat = "FUNC" if self.category == LexicalCategory.FUNCTIONAL_CLOSED else "CONTENT"
        poly = f", {len(self.senses)} senses" if len(self.senses) > 1 else ""
        return f"Word(#{self.id}, /{self.form}/, U={self.usage_frequency:.2f}, {cat}{poly})"