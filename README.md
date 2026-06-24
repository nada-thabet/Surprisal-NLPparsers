# Surprise syntaxique sur la parole naturelle (ds004408) — analyse en constituants

Pipeline qui calcule, pour chaque mot des stimuli du dataset **Broderick ds004408**, sa **surprise syntaxique** à l'aide d'un **analyseur en constituants** (grammaire *context-free* probabiliste / PCFG). Chaque mot conserve son **temps d'apparition**, ce qui permet d'aligner la mesure sur l'EEG et, à terme, de la comparer à une **surprise calculée par un modèle de langue (LLM)**.

> Note : une analyse en dépendances avait été testée mais s'est révélée non fiable (parseur NLTK trop faible). Elle est **mise de côté** ; on se concentre sur les constituants.

---

## 1. Données

**Dataset** : OpenNeuro `ds004408` — *EEG responses to continuous naturalistic speech* (Di Liberto, Broderick, Bialas, Lalor). Livre audio *The Old Man and the Sea*. Le dossier `stimuli` contient un fichier **TextGrid** (Praat) par segment, avec le minutage des mots (tier `words`) et des phonèmes.

On n'utilise que les 20 fichiers `audioNN.TextGrid` (pas l'EEG, pas les `.wav`).

### Télécharger uniquement les TextGrids
```bash
pip install openneuro-py
python -m openneuro download --dataset ds004408 --tag 1.0.8 --include "stimuli/*.TextGrid" --target-dir ds004408_stimuli
```

---

## 2. Installation
```bash
pip install nltk pandas numpy matplotlib wtpsplit onnxruntime
```
(NLTK télécharge ses corpus au premier lancement ; SaT télécharge son modèle une seule fois, ~855 Mo.)

---

## 3. Les scripts

| Script | Rôle | Sortie |
|---|---|---|
| `constituants_dataset.py` | Segmentation (wtpsplit/SaT) + analyse PCFG/Viterbi + **surprisal** par mot | `surprisal_constituants.csv`, `arbres_constituants.txt` |
| `kpi_constituants.py` | Indicateurs de la surprise syntaxique | `kpi_global.csv`, `kpi_par_POS.csv`, `fig_surprisal_par_POS.png` |

Scripts interactifs (test d'une phrase au clavier) : `nltk_parsers_interactive.py`, `surprisal_interactive_v2.py`.

---

## 4. Utilisation

> ⚠️ Lancer depuis le dossier des données, **dans le terminal** (pas le bouton « Run » de VS Code). Le `.` final = « traite le dossier courant ».

```bash
cd ds004408_stimuli/stimuli      # y placer les scripts .py
python constituants_dataset.py .
python kpi_constituants.py .
```

---

## 5. Fonctionnement

- **Lecture TextGrid** : extraction du tier `words` (mot + onset/offset), en ignorant silences et fichiers cachés `._`.
- **Segmentation en phrases** : avec **wtpsplit / SaT** (*Segment any Text*), un segmenteur neuronal adapté à l'anglais et au texte **sans ponctuation ni majuscules** (transcriptions de parole). Repli sur les pauses si SaT n'est pas installé.
- **Étiquetage** : les mots (en MAJUSCULES dans les TextGrids) sont mis en **minuscules** avant l'étiquetage, sinon ils sont tous étiquetés noms propres (NNP).
- **Surprisal** : PCFG induite du Penn Treebank, analyse Viterbi, `surprisal = -log2 P(mot | POS)` en bits. Arbres dé-binarisés (`un_chomsky_normal_form`) pour la lisibilité.

---

## 6. Fichiers produits

| Fichier | Contenu |
|---|---|
| `surprisal_constituants.csv` | mot, phrase, position, onset, offset, **POS, surprisal_bits, oov** |
| `arbres_constituants.txt` | arbre en constituants (lisible) de chaque phrase |
| `kpi_global.csv` | statistiques globales de surprisal |
| `kpi_par_POS.csv` | surprisal moyenne par catégorie grammaticale |
| `fig_surprisal_par_POS.png` | figure |

---

## 7. Résultats (20 segments)

- **Volume** : 11 122 mots, 1 216 phrases (segmentation SaT), 14,1 % de mots hors-vocabulaire (colonne `oov`).
- **Surprisal globale** : moyenne **5,14 bits**, médiane 4,77, P90 10,5, max 13,68.
- **Surprisal par POS** : mots de contenu élevés (noms NN **8,5**, adjectifs JJ 8,4, verbes 5-7), mots-outils bas (déterminants DT **1,6**, CC 1,1, TO/EX ≈ 0). Les mots porteurs de sens sont les plus « surprenants ».
- **Corrélation la plus forte** : surprisal × **durée du mot, r = +0,52** (les mots longs, donc les mots de contenu, sont les plus surprenants).

---

## 8. Corrections apportées

- **Bug NNP / « Mr. » corrigé** : mots mis en minuscules avant l'étiquetage (plus aucun NNP parasite).
- **Segmentation refaite** avec un outil adapté à l'**anglais** (wtpsplit/SaT), à la place du découpage par pauses, qui était syntaxiquement faux.
- **Dépendances mises de côté** (sortie du parseur NLTK non fiable) ; focus sur les constituants.
- Colonne **`oov`** ajoutée pour repérer les mots hors-vocabulaire.

---

## 9. Limites et suite

- **Vocabulaire WSJ** : les mots hors-vocabulaire (colonne `oov`) reçoivent une surprisal plafonnée (~13,68 bits). C'est la limite principale de la PCFG.
- **Suite** : comparer cette **surprise syntaxique** à la **surprise d'un modèle de langue (LLM)** calculée sur les mêmes mots, puis relier les deux au signal EEG.
