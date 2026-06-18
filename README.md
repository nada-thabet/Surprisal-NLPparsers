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
python -m openneuro download --dataset ds004408 --tag 1.0.8 --include "stimuli/*.TextGrid" --target-dir C:\Users\user\Downloads\ds004408_stimuli
```
Les fichiers arrivent dans `...\ds004408_stimuli\stimuli\audio01.TextGrid` … `audio20.TextGrid`.

---

## 2. Installation des dépendances
```bash
pip install nltk scikit-learn pandas numpy matplotlib
```
(NLTK télécharge automatiquement ses corpus au premier lancement : `treebank`, `dependency_treebank`, `punkt`, taggers.)

---

## 3. Les scripts

| Script | Rôle | Entrée | Sortie |
|---|---|---|---|
| `constituants_dataset.py` | Analyse context-free (PCFG/Viterbi) + **surprisal** | dossier de TextGrids | `surprisal_constituants.csv`, `arbres_constituants.txt` |
| `dependances_dataset.py` | Analyse en **dépendances** (TransitionParser) | dossier de TextGrids | `dependances.csv` |
| `comparer_kpi.py` | **Fusion + KPI** de comparaison | les 2 CSV ci-dessus | `comparaison.csv`, `kpi_*.csv`, figures `.png` |

Scripts annexes (interactifs, optionnels, pour tester une phrase au clavier) : `analyse_constituants.py`, `analyse_dependances.py`, `nltk_const_vs_dep.py`.

---

## 4. Utilisation (dans l'ordre)

> ⚠️ **Lancer depuis le dossier des données, dans le terminal** — pas avec le bouton « Run » de VS Code (qui exécute depuis le mauvais dossier). Le `.` final = « traite le dossier courant ».

```bash
cd C:\Users\user\Downloads\ds004408_stimuli\stimuli
# placer les 3 scripts .py dans ce dossier, puis :
python constituants_dataset.py .
python dependances_dataset.py .
python comparer_kpi.py .
```

Au premier lancement, `dependances_dataset.py` entraîne le TransitionParser (~1-3 min).

---

## 5. Fonctionnement résumé

- **Lecture TextGrid** : extraction du tier `words` (mot + onset/offset), en ignorant silences et fichiers cachés `._`.
- **Découpage en phrases** : par les **pauses** (silences > `PAUSE_GAP = 0.35 s`), car les TextGrids n'ont pas de ponctuation. Longueur max paramétrable (`MAX_WORDS`).
- **Surprisal (constituants)** : PCFG induite du Penn Treebank, analyse Viterbi, `surprisal = -log2 P(mot | POS)` en bits.
- **Dépendances** : TransitionParser arc-eager entraîné sur `dependency_treebank` ; donne la tête de chaque mot (→ distance, profondeur). *Bascule automatique sur Stanford CoreNLP si un serveur tourne sur le port 9000.*
- **Comparaison** : fusion des deux CSV **par le temps du mot (onset)**, puis KPI (voir §7).

---

## 6. Fichiers produits

| Fichier | Contenu |
|---|---|
| `surprisal_constituants.csv` | mot, phrase, position, onset, offset, **POS, surprisal_bits** |
| `arbres_constituants.txt` | arbre en constituants de chaque phrase |
| `dependances.csv` | mot, phrase, position, onset, offset, relation, **tête** |
| `comparaison.csv` | table fusionnée mot-par-mot (surprisal + structure dépendance + durée) |
| `kpi_global.csv` | statistiques globales de surprisal |
| `kpi_par_POS.csv` | surprisal moyenne par catégorie grammaticale |
| `kpi_correlations.csv` | corrélations surprisal × facteurs |
| `fig_surprisal_par_POS.png`, `fig_surprisal_vs_distance.png` | figures |
| `transitionparser_ptb.model` | modèle entraîné (cache) |

---

## 7. Résultats (run sur les 20 segments)

- **Volume** : 10 951 mots (constituants), 11 419 (dépendances), 10 951 fusionnés.
- **Surprisal globale** : moyenne **5,76 bits**, médiane 4,65, P90 12,1, max 13,68.
- **Surprisal par POS** : mots de contenu élevés (noms NN **11,0**, adjectifs JJ 9,2, verbes 7-8), mots-outils bas (déterminants DT **1,5**, TO/EX ≈ 0). → les mots porteurs de sens sont les plus « surprenants ».
- **Structure de dépendance** : distance tête moyenne 5,17 positions (1,22 s), profondeur moyenne 1,42.
- **Croisements** : surprisal légèrement croissante avec la distance de dépendance ; têtes (6,01) un peu plus surprenantes que dépendants (5,69) ; décroissante avec la profondeur.
- **Corrélation la plus forte** : surprisal × **durée du mot, r = +0,37** (les mots longs = mots de contenu = plus surprenants).

Synthèse détaillée : voir `rapport_constituants_vs_dependances.docx`.

---

## 8. Limites

- **Relations non étiquetées** : le corpus `dependency_treebank` de NLTK donne la structure (têtes) mais **pas** les labels (`nsubj`, `dobj`, `det`…). Pour les obtenir → **Stanford CoreNLP** ou **spaCy**.
- **Vocabulaire WSJ** : les mots hors-vocabulaire reçoivent une surprisal **plafonnée** (~13,68 bits).
- **Segmentation en phrases** approximée par les pauses (pas de ponctuation dans les TextGrids).

---

## 9. Dépannage (problèmes rencontrés et solutions)

| Symptôme | Cause | Solution |
|---|---|---|
| Fichiers `._audioNN.txt` vides | métadonnées macOS (AppleDouble) | utiliser les vrais `audioNN.TextGrid` ; ignorés automatiquement |
| `dependances.csv` → **0 mots** au 2e lancement | le modèle en cache ne contient pas le dictionnaire de features | utiliser la version de `load_transitionparser` qui **ré-entraîne** à chaque fois (ou supprimer `transitionparser_ptb.model` avant de relancer) |
| Erreur `int64 indices` (scikit-learn) | incompat. NLTK TransitionParser / sklearn récent | corrigé dans `dependances_dataset.py` (cast des index en **int32**) |
| Sections « relation » vides dans la comparaison | treebank NLTK sans labels (voir §8) | normal ; utiliser le `comparer_kpi.py` final (KPI sur la **structure**) |
