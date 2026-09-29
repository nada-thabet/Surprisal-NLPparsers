# -*- coding: utf-8 -*-
"""
surprise_apres_controle.py

Cree deux fichiers pour The Old Man and the Sea avec la surprise apres
controle de la frequence :

    surprise_syntaxique_ctrlfreq.csv   a partir de surprisal_incrementale.csv
    surprise_llm_ctrlfreq.csv          a partir de surprisal_llm_textgrid.csv

La valeur est le residu de la regression lineaire de la surprise sur la
frequence (freq_zipf), c'est-a-dire la part de la surprise que la frequence ne
predit pas. Elle est en bits comme la surprise, et sa moyenne est 0 par
construction (donc environ la moitie des valeurs sont negatives).

Pas de dependance, seulement la bibliotheque standard.
A lancer depuis le dossier des deux CSV :

    python surprise_apres_controle.py
"""

import csv
import glob
import os

# --------------------------------------------------------------------
# Fichiers a traiter : (motifs de recherche, colonne source, colonne creee,
#                       nom du fichier de sortie)
# --------------------------------------------------------------------
TACHES = [
    (["surprisal_incrementale.csv", "surprisal_incrementale-*.csv"],
     "surprise_incrementale", "surprise_syn_ctrlfreq",
     "surprise_syntaxique_ctrlfreq.csv"),
    (["surprisal_llm_textgrid.csv", "surprisal_llm_textgrid-*.csv"],
     "surprise_llm", "surprise_llm_ctrlfreq",
     "surprise_llm_ctrlfreq.csv"),
]

DOSSIERS = [
    ".",
    os.path.expanduser(r"~\Downloads\ds004408_stimuli\stimuli"),
    os.path.expanduser(r"~\Downloads"),
]


def trouver(motifs):
    """Retourne le premier fichier correspondant, en excluant les fichiers AMI."""
    for dossier in DOSSIERS:
        for motif in motifs:
            trouves = sorted(glob.glob(os.path.join(dossier, motif)))
            trouves = [t for t in trouves
                       if "ami" not in os.path.basename(t).lower()]
            if trouves:
                return trouves[0]
    return None


def nombre(txt):
    """Convertit en float, retourne None si la valeur est absente ou invalide."""
    try:
        v = float(str(txt).strip())
    except (TypeError, ValueError):
        return None
    return v


def regression_simple(xs, ys):
    """Droite des moindres carres y = a + b*x. Retourne (a, b, r)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    b = sxy / sxx
    a = my - b * mx
    r = sxy / (sxx ** 0.5 * syy ** 0.5)
    return a, b, r


def traiter(motifs, col_surprise, col_residu, sortie):
    chemin = trouver(motifs)
    if chemin is None:
        print(f"  INTROUVABLE : {' / '.join(motifs)}")
        return

    with open(chemin, newline="", encoding="utf-8-sig") as f:
        lignes = list(csv.DictReader(f))
    if not lignes:
        print(f"  {chemin} : fichier vide")
        return

    entetes = list(lignes[0].keys())
    for requis in (col_surprise, "freq_zipf"):
        if requis not in entetes:
            print(f"  {chemin} : colonne '{requis}' absente")
            print(f"  colonnes presentes : {entetes}")
            return

    # couples valides pour estimer la droite
    valides = []
    for i, ligne in enumerate(lignes):
        y = nombre(ligne[col_surprise])
        x = nombre(ligne["freq_zipf"])
        if y is not None and x is not None:
            valides.append((i, x, y))

    xs = [x for _, x, _ in valides]
    ys = [y for _, _, y in valides]
    a, b, r = regression_simple(xs, ys)

    # residus
    residus = {}
    for i, x, y in valides:
        residus[i] = y - (a + b * x)

    # controles
    moyenne_res = sum(residus.values()) / len(residus)
    mr = moyenne_res
    mx = sum(xs) / len(xs)
    res_liste = [residus[i] for i, _, _ in valides]
    sxy = sum((x - mx) * (rr - mr) for (_, x, _), rr in zip(valides, res_liste))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((rr - mr) ** 2 for rr in res_liste)
    r_res = sxy / (sxx ** 0.5 * syy ** 0.5) if sxx and syy else 0.0

    zeros = sum(1 for x in xs if abs(x) < 1e-9)

    print(f"  source        : {chemin}")
    print(f"  lignes        : {len(lignes)}  (dont {len(valides)} exploitables)")
    if zeros:
        print(f"  freq_zipf = 0 : {zeros} mots absents de la base de frequence")
    print(f"  droite        : surprise = {a:.4f} + {b:.4f} x freq_zipf")
    print(f"  r(surprise, frequence)        = {r:+.4f}")
    print(f"  moyenne des residus           = {moyenne_res:+.2e}   (doit etre ~0)")
    print(f"  r(residus, frequence)         = {r_res:+.2e}   (doit etre ~0)")

    # ecriture
    with open(sortie, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=entetes + [col_residu])
        w.writeheader()
        for i, ligne in enumerate(lignes):
            ligne = dict(ligne)
            ligne[col_residu] = ("" if i not in residus
                                 else f"{residus[i]:.6f}")
            w.writerow(ligne)
    print(f"  ecrit         : {sortie}  ({len(lignes)} lignes)\n")


if __name__ == "__main__":
    print("Surprise apres controle de la frequence, The Old Man and the Sea\n")
    for motifs, col_src, col_res, sortie in TACHES:
        print(f"[{col_res}]")
        traiter(motifs, col_src, col_res, sortie)
    print("Termine.")
