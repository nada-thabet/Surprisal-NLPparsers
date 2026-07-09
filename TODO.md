# Feuille de route (to-do)

Objectif général : calculer plusieurs métriques de surprise (constituants, dépendances, LLM) sur la parole naturelle
(corpus ds004408, *The Old Man and the Sea*), les comparer entre elles, puis les relier à l'EEG.

## 1. Segmentation — comparaison automatique avec le texte original
Aligner automatiquement la segmentation prosodique sur le texte de *The Old Man and
the Sea* afin d'évaluer et de valider les frontières de phrases (mesure d'accord
entre les phrases prédites et les phrases réelles du livre).

## 2. Surprise syntaxique par constituants et fréquence
Calculer la surprise en constituants en contrôlant la fréquence lexicale et le nombre de phonèmes 
(protocole type Demberg et al. 2012).

## 3. Surprise — analyse syntaxique en dépendances
Calculer la surprise à partir d'une **analyse syntaxique en dépendances**
(structure tête → dépendant), en parallèle de la version en constituants.

## 4. Comparaison constituants vs dépendances
Comparer directement la surprise que donne l'analyse syntaxique en constituants avec la surprise que donne l'analyse syntaxique par dépendances.

## 5. Comparaison LLM vs constituants vs dépendances
Comparer les trois métriques entre elles : la surprise que donne le LLM, la surprise issue de l'analyse syntaxique en constituants, et la surprise issue de l'analyse syntaxique par dépendances.