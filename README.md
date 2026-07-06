# Surprise syntaxique sur parole naturelle (ds004408)

Ce dépôt calcule, mot à mot, une **surprise syntaxique** sur les stimuli de parole
du dataset **Broderick ds004408** (livre audio *The Old Man and the Sea*). La chaîne
comprend trois étapes : (1) une **segmentation en phrases par indices prosodiques**
adaptée à la parole non ponctuée, corrigée par une contrainte morphosyntaxique ;
(2) une **analyse en constituants** (grammaire hors-contexte probabiliste) de chaque
phrase ; (3) le calcul d'une **surprise** par mot à partir de la grammaire, alignée
sur le temps d'apparition du mot (onset) en vue d'un couplage ultérieur avec l'EEG.

La surprise (*surprisal*, Hale 2001 ; Levy 2008) est définie comme le contenu
d'information d'un mot, `surprisal = -log2 P(mot | contexte)`. Ici le contexte est
syntaxique : `P(mot | catégorie)` issu d'une PCFG induite sur le Penn Treebank.

---

## 1. Données

Dataset : OpenNeuro `ds004408` (*EEG responses to continuous naturalistic speech*).
Le dossier `stimuli` fournit, par segment, un fichier **TextGrid** (Praat) contenant
le minutage des mots (tier `words`) et des phonèmes (tier `phones`). On n'utilise que
les 20 fichiers `audioNN.TextGrid`.

Téléchargement des seuls TextGrid :

```bash
pip install openneuro-py
python -m openneuro download --dataset ds004408 --tag 1.0.8 \
    --include "stimuli/*.TextGrid" --target-dir ./ds004408_stimuli
```

Le tier `phones` est nécessaire : l'allongement du dernier phonème avant une frontière
est l'un des deux indices de segmentation.

---

## 2. Installation

```bash
pip install nltk scikit-learn pandas numpy matplotlib
```

NLTK télécharge automatiquement ses ressources au premier lancement (`treebank`,
`punkt`, `averaged_perceptron_tagger`).

---

## 3. Scripts

| Script | Rôle | Entrée | Sortie |
|---|---|---|---|
| `segmentation_prosodique.py` | Segmentation en phrases (prosodie + POS) | TextGrid | `phrases_prosodiques.txt` |
| `constituants_dataset.py` | Analyse en constituants + surprise | TextGrid | `arbres_constituants.txt`, `surprisal_constituants.csv` |
| `kpi_constituants.py` | Statistiques de la surprise | `surprisal_constituants.csv` | `kpi_global.csv`, `kpi_par_POS.csv`, `fig_surprisal_par_POS.png` |

`segmentation_prosodique.py` est autonome (inspection de la segmentation) **et**
importé par `constituants_dataset.py`, qui l'utilise pour découper chaque segment
avant l'analyse syntaxique.

---

## 4. Utilisation

À lancer depuis le dossier des données, dans un terminal :

```bash
cd ds004408_stimuli/stimuli
python segmentation_prosodique.py .     # inspecter la segmentation seule
python constituants_dataset.py .        # arbres + surprise (20 fichiers)
python kpi_constituants.py .            # statistiques + figure
```

Pour tester un seul segment, remplacer `.` par un nom de fichier
(`python constituants_dataset.py audio01.TextGrid`).

---

## 5. Détail technique

### 5.1 Lecture des TextGrid

`read_words_and_phones()` extrait par expression régulière les intervalles des tiers
`words` et `phones` (`xmin`, `xmax`, `text`), en minuscules, en écartant les silences
(`sil`, `sp`, `spn`, …) et les fichiers cachés `._*`. Chaque mot est représenté par un
triplet `(mot, onset, offset)`.

### 5.2 Segmentation prosodique

La parole n'a ni ponctuation ni majuscules ; les frontières de phrases sont donc
estimées à partir du signal temporel, selon deux indices (Shriberg & Stolcke 2000 ;
l'ordre d'importance rapporté dans la littérature est durée > pause > pitch) :

1. **Pause** après chaque mot : `pause_i = onset(mot_{i+1}) − offset(mot_i)`.
2. **Allongement final** : durée du **dernier phonème** du mot. Cette durée est
   normalisée (z-score) **par phonème** — un `/t/` est comparé à la distribution des
   `/t/`, un `/a/` à celle des `/a/` — afin de neutraliser la durée intrinsèque des
   sons et de ne retenir que l'allongement pré-frontière.

La pause est également z-scorée (globalement). Le score de frontière est la somme
pondérée `W_PAUSE·z(pause) + W_LENG·z(allongement)`. Une position est candidate à une
frontière si :

- la pause absolue dépasse `PAUSE_MIN`, **ou**
- le score combiné dépasse `moyenne + THR_K · écart-type` (seuil relatif à la
  distribution du segment, sans quota de longueur imposé).

Aucune longueur de phrase n'est fixée a priori : une phrase longue sans rupture
prosodique reste entière ; `SENT_MAX` n'est qu'un garde-fou pour la complexité
cubique de l'analyseur de Viterbi.

### 5.3 Contrainte morphosyntaxique

La prosodie seule place parfois des frontières linguistiquement impossibles. Après
étiquetage en catégories (POS Penn Treebank via `nltk.pos_tag`), deux corrections
symétriques sont appliquées :

- **Fin de phrase (`NO_END`)** : une phrase ne peut pas se terminer sur un mot outil
  qui appelle un complément à droite (préposition `IN`, `TO`, déterminant `DT`,
  conjonction `CC`, possessif `PRP$`/`POS`, relatif `WDT`/`WP`, modal `MD`, particule
  `RP`…). Si une frontière tombe après un tel mot, elle est déplacée **avant** lui, qui
  rejoint la phrase suivante (corrige `… a fish in | the first…` → `… a fish | in the first…`).
- **Début de phrase (`NO_START`)** : une phrase ne peut pas commencer par un mot
  orphelin appartenant à la précédente — particule verbale (`RP` : *up*, *off*) ou
  préposition stranded (une `IN` immédiatement suivie d'une `CC`). La frontière est
  alors déplacée **après** ce mot (corrige `… hauled | up i could…` → `… hauled up | i could…`).

### 5.4 Analyse en constituants et surprise

`load_pcfg()` induit une grammaire hors-contexte probabiliste (PCFG) à partir des
arbres du Penn Treebank : `collapse_unary`, puis `chomsky_normal_form(horzMarkov=2)`
pour la binarisation, puis `induce_pcfg`. Chaque phrase est analysée par un
`ViterbiParser` (analyse la plus probable).

Pour chaque mot, la surprise syntaxique vaut `−log2 P(mot | catégorie)`, où la
probabilité lexicale est celle des règles `catégorie → mot` de la PCFG. L'arbre est
ensuite **débinarisé** (`un_chomsky_normal_form`) et ses feuilles restaurées en
minuscules pour une lecture naturelle.

Traitement des mots hors-vocabulaire (OOV) : les mots sont mis en minuscules avant
l'étiquetage (ce qui évite un étiquetage parasite en nom propre `NNP`). Un mot absent
du vocabulaire du Penn Treebank est remplacé, pour l'analyse, par le mot le plus
probable de sa catégorie prédite ; il est marqué `oov = 1` et reçoit une surprise
plafonnée (≈ 13,7 bits).

### 5.5 Statistiques

`kpi_constituants.py` calcule sur `surprisal_constituants.csv` : couverture (mots,
phrases, taux d'OOV), distribution de la surprise (moyenne, écart-type, quantiles),
surprise moyenne **par catégorie grammaticale**, corrélations avec la durée du mot
(`offset − onset`) et la position dans la phrase, top 15 des mots les plus surprenants,
et une figure de la surprise par POS.

---

## 6. Fichiers produits

| Fichier | Contenu |
|---|---|
| `phrases_prosodiques.txt` | phrases segmentées par fichier (contrôle qualité) |
| `arbres_constituants.txt` | arbre en constituants de chaque phrase |
| `surprisal_constituants.csv` | mot, phrase, position, onset, offset, **POS, surprisal_bits, oov** |
| `kpi_global.csv` | statistiques globales de la surprise |
| `kpi_par_POS.csv` | surprise moyenne par catégorie grammaticale |
| `fig_surprisal_par_POS.png` | figure correspondante |

---

## 7. Paramètres réglables

En tête de `segmentation_prosodique.py` :

| Paramètre | Défaut | Effet |
|---|---|---|
| `PAUSE_MIN` | `0.35` | pause absolue (s) déclenchant une frontière ; baisser → segmente plus |
| `THR_K` | `1.5` | seuil du score combiné (en écarts-types) ; baisser → segmente plus |
| `MIN_LEN` | `3` | longueur minimale avant d'autoriser une coupe |
| `SENT_MAX` | `45` | garde-fou de longueur maximale |
| `USE_POS` | `True` | active la correction morphosyntaxique |

---

## 8. Limites

- **Segmentation.** Sur de la parole lue, les pauses de virgule et de fin de phrase se
  recouvrent en durée ; un seuil unique sépare imparfaitement les deux et produit soit
  de la sur-segmentation (aux virgules lourdes), soit des phrases fusionnées. La
  contrainte morphosyntaxique corrige les frontières impossibles mais pas ce
  recouvrement. La segmentation obtenue relève donc de l'**unité intonative /
  propositionnelle** plus que de la phrase orthographique.
- **Vocabulaire.** La grammaire est apprise sur le Wall Street Journal ; les mots hors
  de ce domaine reçoivent une surprise plafonnée.

---

## 9. Perspectives

- 
- Comparer la surprise **syntaxique** à une surprise issue d'un **modèle de langue**
  (`−log2 P(mot | contexte)`), calculée sur le flux et donc moins dépendante de la
  segmentation.
- Relier la surprise, alignée sur l'onset, aux **réponses EEG**.
