# EvoLang

**Generative Linguistic Atlas & Biomechanical Evolution Observer**  
A continuous, first-principles computational linguistics and geopolitical simulation engine. Languages in EvoLang evolve under physical articulatory mechanics, 7D cognitive semantics, and emergent demographic dynamics.


## Key Features

### 1. Biomechanical & Thermodynamic Phonetics
* **Aerodynamic Constraints:** Models Phonation Threshold Pressure (PTP), Ohala’s supraglottal impedance walls, and respiratory moisture taxes across cold/arid biomes.
* **Continuous Articulatory Space:** Maps segments directly to continuous 3D coordinates (place/height, manner/backness, voicing/rounding) and authentic IPA.
* **Natural Prominence & Tonogenesis:** Weight-to-Stress emerges organically from acoustic mass. Tonal registers and contours evolve diachronically from onset voicing and laryngeal coda decay.

### 2. 7D Cognitive Semantics & Generative Grammar
* **Unified Cognitive Space:** Words occupy continuous vectors across 7 dimensions: *Concreteness, Animacy, Valence, Potency, Dynamism, Sociality, Extension*.
* **Emergent Typology:** Derives Morphosyntactic Alignment (Nominative-Accusative vs. Ergative-Absolutive), Differential Object Marking (DOM), Noun Classes, Verb Conjugations, and Numeral Radixes directly from cultural attention vectors.
* **Non-Concatenative Morphology:** Generates vocalic ablaut, Semitic-style root-and-pattern transfixation, reduplication, and affix erosion along grammaticalization clines.

### 3. Geopolitics & Urban Networks
* **Local City Demographics:** Settlements maintain independent populations, fortifications, and local carrying capacities—no artificial teleportation or global pooling.
* **Dynamic Transport Networks:** Pioneers and trade caravans carve physical roads on land and sea lanes across water that permanently lower transit friction.
* **Siege Warfare & Substrate Shockwaves:** Armies suffer logistical distance decay. Defeated capitals trigger demographic assimilation, transferring loanwords, phonemes, and grammar into the conqueror's language.
* **Dormant Classical Languages:** Extinct civilizations persist in a classical archive, allowing scholars and daughter colonies to revive archaic liturgical roots.

### 4. High-Performance Hardware Engine
* **Vectorized Spatial Fields:** Replaces cellular-automata loops with NumPy/PyTorch Euclidean Power Fields ($<1\text{ms}$ per tick).
* **Batch Semantic Filtering:** GPU-accelerated candidate distance validation using PyTorch CUDA with graceful CPU/NumPy fallbacks.

## Quick Start

### Prerequisites
* Python 3.9+
* NumPy (Required)
* PyTorch (Optional, enables CUDA acceleration)
* eSpeak-NG (Optional, enables live speech synthesis)

```bash
pip install numpy torch
```

### Running the Observer

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/evolang.git
   cd evolang
   ```

2. Start the simulation backend:
   ```bash
   python visualizer.py
   ```

3. Open your browser at **`http://localhost:8008`**.

## Project Structure

```text
├── articulatory_space.py  # 3D IPA resolver & secondary articulation mapping
├── energy.py              # Biomechanical aerodynamics, PTP, and transition costs
├── gpu_accelerator.py     # Hardware tensor backend (CUDA & NumPy SIMD)
├── grammar.py             # Emergent morphosyntax, DOM, and conjugation engine
├── index.html             # Client-side canvas visualizer & City Inspector
├── language.py            # Speech community, inventory clustering, and drift
├── lexicon.py             # Derivational morphology and vocabulary growth
├── phoneme.py             # Immutable phonetic profiles & physics invariants
├── pronouncer.py          # eSpeak-NG audio synthesis & IPA transpiler
├── semantics.py           # 7D cognitive vector space and prototype anchors
├── syllable.py            # Weight-to-stress, sonority syllabification & tones
├── translator.py          # POS-constrained compositional semantic translation
├── visualizer.py          # HTTP server and state serialization pipeline
├── word.py                # Structural word unit with lazy form & cost caching
└── world_map.py           # 2D physical terrain, cities, roads, and warfare
```

## Controls & Usage

* **Play / Pause / Step:** Control time evolution (ticks/epochs) and simulation pace directly from the control deck.
* **Map Layer Toggle:** Switch between Topography, Population Density, Vegetation, Humidity, and Temperature.
* **City Inspector:** Click on any settlement node on the map to inspect its living population, wall strength, original founder, and historical conquest chronicle.
* **7D Mindset & Grammar:** Inspect real-time cultural attention shifts, emergent vowel/consonant charts, active feature series, and live verb/noun paradigm tables.

## License

MIT License. Free for research, generative worldbuilding, and educational use.
