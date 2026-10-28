# EvoLang

**Generative Linguistic Atlas & Biomechanical Evolution Simulator**

EvoLang is a continuous computational linguistics engine that simulates language evolution from physical articulatory mechanics, environmental acoustics, 7D cognitive semantics, and territorial demographics.

Instead of procedural sound-change rules (`k -> tʃ`) or hardcoded grammar templates, languages emerge and diversify through thermodynamic trade-offs between motor effort, acoustic transmission, and communicative contrast.

## Architecture & Systems

### 1. Biomechanical Phonetics & Acoustic Ecology
* **Articulatory Mechanics:** Vowels and consonants map to continuous coordinates across independent articulatory tiers (lips, coronal tongue tip/blade, dorsal tongue body, and glottis).
* **Autosegmental Conservation:** Segment deletion preserves moraic mass, dorsal frontness, and laryngeal stiffness, producing compensatory vowel lengthening, umlaut, and tonogenesis (register and contour tones) organically.
* **Environmental Acoustics:** Models vocal fold hydration tax, air density ($\rho_{\text{air}}$), ambient noise, and canopy acoustic damping. Warm, humid lowlands penalize consonant clusters and favor open CV sonority; cold or high-altitude regions license ejectives and complex onsets.
* **Topological Mode-Seeking:** Emergent phonemic inventories are discovered as 3D energy-weighted density peaks ($U \times \text{duration} \times \text{salience}$) with local sharpness gating, cleanly separating contrastive phonemes from transitional allophones (e.g., unstressed schwa reduction, intervocalic spirantization).

### 2. 7D Cognitive Semantics & Lexicon Growth
* **7D Semantic Space:** Lexemes occupy continuous vectors across seven cognitive dimensions: *Concreteness, Animacy, Valence, Potency, Dynamism, Sociality, and Extension*.
* **Thermodynamic Neologisms:** Vocabulary expands through a competitive Boltzmann distribution between root creation, compounding, and derivational affixation, weighted by morphological synthesis, semantic density, and word-length cost.
* **Autonomous Grammaticalization:** High-frequency roots with low concreteness bleach into grammatical operators (case markers, TAM markers, applicatives) along continuous grammaticalization clines.
* **Communicative Contrast Maintenance:** Surfaces homophonic collisions to penalize ambiguous merges, driving functional differentiation.

### 3. Emergent Morphosyntax
* **Continuous Typology:** Head-directionality ($h \in [0, 1]$) continuously orders constituents (SOV $\leftrightarrow$ SVO $\leftrightarrow$ VSO) and adposition placement via Boltzmann potentials.
* **Morphological Synthesis:** Governs isolating, agglutinative, and fusional strategies based on morpheme-to-word ratios and boundary friction.
* **Syntactic Alignment:** Derives Ergative-Absolutive vs. Nominative-Accusative marking from the balance of cultural potency and sociality attention.
* **Phonotactic Repair:** Morpheme concatenation runs through sonority sequencing, hiatus resolution, and stress-timed prosody.

### 4. Geography & Demographics
* **Continuous 2D Geography:** Elevation, orographic rainfall, Hadley-cell subsidence, and emergent vegetation determine carrying capacity and resource profiles (grain, livestock, maritime, minerals).
* **Territory & Infrastructure:** Populations clear farm plots, expand borders via Euclidean power fields, and build roads that lower transit friction.
* **Conflict & Contact:** Deficits trigger border friction, conquest, lexical borrowing, and substrate influence.

## Project Structure

```text
├── articulatory_space.py  # 3D IPA mapping and suprasegmental projection
├── energy.py              # Biomechanical costs, transition friction, and cluster repair
├── gpu_accelerator.py     # Hardware tensor acceleration (CUDA / NumPy SIMD)
├── grammar.py             # Autonomous operator discovery, clausal assembly, and paradigms
├── index.html             # Web-based linguistic atlas, demographic map, and inspector
├── language.py            # Speech community state, continuous drift, and inventory discovery
├── lexicon.py             # Neologism dynamics, phonotactics, and compounding
├── phoneme.py             # Multi-tier articulatory physics and autosegmental invariants
├── pronouncer.py          # Universal IPA normalizer and eSpeak-NG audio synthesis
├── semantics.py           # 7D cognitive vector space and biological prototypes
├── syllable.py            # Sonority syllabification, stress assignment, and reduction
├── translator.py          # Compositional semantic translation and POS classification
├── visualizer.py          # HTTP server, state serialization, and simulation harness
├── word.py                # Lexical entry model with morpheme tracking and caching
└── world_map.py           # Orographic climate, economic production, and territory grid
```

## Quick Start

### Prerequisites
* Python 3.9+
* NumPy
* PyTorch *(optional, for CUDA acceleration)*
* eSpeak-NG *(optional, for audio synthesis)*

```bash
pip install numpy torch
```

### Running the Simulator

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/evolang.git
   cd evolang
   ```

2. Start the simulation server:
   ```bash
   python visualizer.py
   ```

3. Open your browser at:
   ```text
   http://localhost:8008
   ```

## Inspector Controls

* **Simulation Deck:** Play, pause, step (1, 15, 50 epochs), and adjust simulation speed.
* **Map Layers:** View Topography (Elevation), Population Density, Vegetation, Humidity, and Temperature.
* **Analytics Tabs:**
  * **Overview:** Articulatory effort (Joules), C:V ratio, phonemic entropy, and sociolinguistic tension (diglossia / standard canon divergence).
  * **Population:** Economic breakdown (grain, livestock, marine, minerals) and carrying capacity.
  * **7D Mindset:** Cultural attention distribution across the 7 cognitive axes.
  * **Phonology:** Active feature series (ejectives, fricatives, nasals), emergent vowel/consonant charts, and token frequency stats.
  * **Grammar:** Word order, synthesis index, noun declension, and verb conjugation tables.
  * **City Inspector:** Inspect settlement history, fortification levels, and local population.


## License

MIT License. Open for linguistic research, computational modeling, and generative worldbuilding.
