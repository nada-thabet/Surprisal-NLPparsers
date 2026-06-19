# Analyse syntaxique des stimuli (ds004408) : constituants vs dépendances

Pipeline qui analyse la parole naturelle du dataset **Broderick ds004408** selon **deux modèles syntaxiques** et les compare :

1. **Constituants** (analyseur *context-free* / PCFG) → arbres + **surprisal** par mot.
2. **Dépendances** (TransitionParser NLTK) → structure tête→dépendant.

Les deux analyses sont alignées sur les mêmes mots (via leur **temps d'apparition**), ce qui permet de les comparer et, à terme, de les relier à l'EEG.

---

## 1. Données

**Dataset** : OpenNeuro `ds004408` — *EEG responses to continuous naturalistic speech* (Di Liberto, Broderick, Bialas, Lalor). Livre audio *The Old Man and the Sea*. Le dossier `stimuli` contient un fichier **TextGrid** (Praat) par segment, avec le minutage des mots (tier `words`) et des phonèmes.

On n'utilise **que** les 20 fichiers `audioNN.TextGrid` (pas l'EEG, pas les `.wav`).

### Télécharger uniquement les TextGrids
```bash
pip install openneuro-py
python -m openneuro download --dataset ds004408 --tag 1.0.8 --include "stimuli/*.TextGrid" --target-dir ds004408_stimuli
```

---

## 2. Installation des dépendances
```bash
pip install nltk scikit-learn pandas numpy matplotlib
```
(NLTK télécharge automatiquement ses corpus au premier lancement : `treebank`, `dependency_treebank`, `punkt`, taggers.)

---

## 3. Les scripts

| Script | Rôle | Sortie |
|---|---|---|
| `constituants_dataset.py` | Analyse context-free (PCFG/Viterbi) + **surprisal** | `surprisal_constituants.csv`, `arbres_constituants.txt` |
| `dependances_dataset.py` | Analyse en **dépendances** (TransitionParser) | `dependances.csv` |
| `comparer_kpi.py` | **Fusion + KPI** de comparaison | `comparaison.csv`, `kpi_*.csv`, figures `.png` |

Scripts interactifs (test d'une phrase au clavier) : `nltk_parsers_interactive.py` (Viterbi + Earley), `surprisal_interactive_v2.py` (PCFG + n-gramme).

---

## 4. Utilisation (dans l'ordre)

> ⚠️ **Lancer depuis le dossier des données, dans le terminal**. Le `.` final = « traite le dossier courant ».

```bash
cd ds004408_stimuli/stimuli      # y placer les 3 scripts .py
python constituants_dataset.py .
python dependances_dataset.py .
python comparer_kpi.py .
```
Au premier lancement, `dependances_dataset.py` entraîne le TransitionParser (~1-3 min).

---

## 5. Fonctionnement résumé

- **Lecture TextGrid** : extraction du tier `words` (mot + onset/offset), en ignorant silences et fichiers cachés `._`.
- **Découpage en phrases** : par les **pauses** (silences > `PAUSE_GAP = 0.35 s`), car les TextGrids n'ont pas de ponctuation.
- **Surprisal (constituants)** : PCFG induite du Penn Treebank, analyse Viterbi, `surprisal = -log2 P(mot | POS)` en bits. Les mots sont mis en **minuscules** avant l'étiquetage (voir Corrections).
- **Dépendances** : TransitionParser arc-eager entraîné sur `dependency_treebank` ; donne la tête de chaque mot (→ distance, profondeur). Bascule sur Stanford CoreNLP si un serveur tourne sur le port 9000.
- **Comparaison** : fusion des deux CSV **par le temps du mot (onset)**, puis KPI.

---

## 6. Corrections

- **Bug NNP / « Mr. » corrigé** : les mots des TextGrids étant en MAJUSCULES, l'étiqueteur les prenait pour des noms propres (NNP) et les remplaçait par « Mr. ». On met désormais les mots en **minuscules** avant l'étiquetage → POS correct (plus aucun NNP parasite : 2078 → 0) et beaucoup moins de mots hors-vocabulaire.
- **Arbres dé-binarisés** (`un_chomsky_normal_form`) → lisibles, sans étiquettes `S|<...>`.
- **Tokenisation des contractions** dans les scripts interactifs (`word_tokenize` : « it's » → `it` + `'s`).
- Colonne **`oov`** (0/1) ajoutée au CSV pour repérer les mots hors-vocabulaire.

---

## 7. Fichiers produits

| Fichier | Contenu |
|---|---|
| `surprisal_constituants.csv` | mot, phrase, position, onset, offset, **POS, surprisal_bits, oov** |
| `arbres_constituants.txt` | arbre en constituants (lisible) de chaque phrase |
| `dependances.csv` | mot, phrase, position, onset, offset, relation, **tête** |
| `comparaison.csv` | table fusionnée mot-par-mot (surprisal + structure dépendance + durée) |
| `kpi_global.csv`, `kpi_par_POS.csv`, `kpi_correlations.csv` | indicateurs |
| `fig_surprisal_par_POS.png`, `fig_surprisal_vs_distance.png` | figures |

---

## 8. Résultats (20 segments, après correction)

- **Volume** : 11 084 mots analysés en constituants, 11 419 en dépendances.
- **Surprisal globale** : moyenne **5,17 bits**, médiane 4,77, P90 10,5, max 13,68.
- **Surprisal par POS** : mots de contenu élevés (noms NN **8,6**, adjectifs JJ 8,5, verbes 7-8), mots-outils bas (déterminants DT **1,6**, CC 1,1, TO/EX ≈ 0). → les mots porteurs de sens sont les plus « surprenants ». Plus aucun NNP (bug corrigé).
- **Structure de dépendance** : distance tête moyenne 5,17 positions (1,22 s), profondeur moyenne 1,42.
- **Corrélation la plus forte** : surprisal × **durée du mot, r = +0,51** (n = 11 084) — les mots longs (mots de contenu) sont les plus surprenants. Effets faibles pour la distance de dépendance (r = +0,03 à +0,07) et la profondeur (r = −0,05).

---

## 9. Limites

- **Relations de dépendance non étiquetées** : le corpus `dependency_treebank` de NLTK donne la structure (têtes) mais **pas** les labels (`nsubj`, `dobj`, `det`…). Pour les obtenir → Stanford CoreNLP ou spaCy.
- **Vocabulaire WSJ** : les mots hors-vocabulaire (colonne `oov`) reçoivent une surprisal plafonnée (~13,68 bits).
- **Segmentation en phrases** approximée par les pauses (pas de ponctuation dans les TextGrids) — amélioration possible via le texte original ou MarsaTag.

