# Surprisal and word duration: syntactic (PCFG) vs GPT-2 surprisal

Code and data from my 2026 internship at the Laboratoire Parole et Langage (CNRS, Aix-Marseille), supervised by Philippe Blache. The question is whether a word's surprisal predicts how long it is spoken, beyond its frequency and length. It follows Demberg, Sayeed, Gorinski & Engonopoulos (2012) on two corpora:

- read speech: the audiobook of *The Old Man and the Sea* (OpenNeuro ds004408, 20 TextGrids in `stimuli_livre/`);
- spontaneous speech: the AMI meeting corpus.

Two measures are compared: incremental syntactic surprisal from a PCFG (Hale 2001, prefix probabilities computed with Jelinek & Lafferty 1991) and GPT-2 surprisal.

## Contents

| Folder | Content |
|---|---|
| `v2/` | Current version: both measures recomputed in nats (natural log) for both corpora, with the code, the data (`v2/donnees/`), every analysis of the report redone for syntactic and GPT-2 surprisal on the book and on AMI (plus the syntactic surprisal on all 171 AMI meetings and the DialoGPT test on AMI), the figures, the EEG files (`v2/indices_eeg/`) and two reports in French (`v2/comptes_rendus/`). |
| `version_rapport/` | The version used in the internship report: the scripts that generate the CSVs, the CSVs (in bits, -log2), the analysis scripts, the 14 figures and the surprisal-after-control files (`5_indices_eeg/`). AMI data: the 12 matched meetings for both measures, and the syntactic values for all 171 meetings. |
| `stimuli_livre/` | The 20 TextGrids of the audiobook, the book text (`oldmansea.pdf`, used to give GPT-2 the real text) and the prosodic sentence split used everywhere (`phrases_prosodiques.txt`). |

## Not included

- The audiobook audio (`.wav`). It is only needed to redo the prosodic segmentation from scratch; the split I used is in `stimuli_livre/phrases_prosodiques.txt` and in the CSVs.
- The AMI `*.words.xml` annotation files (available on the AMI corpus website). The CSVs contain every word used.
