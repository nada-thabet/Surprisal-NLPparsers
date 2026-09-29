# -*- coding: utf-8 -*-
"""
surprise_apres_controle_total.py

Cree deux fichiers pour The Old Man and the Sea avec la surprise apres
controle complet, c'est-a-dire en retirant toutes les variables de controle
du modele de duree, pas seulement la frequence :

    surprise_syntaxique_apres_controle.csv
    surprise_llm_apres_controle.csv

Controles retires (ceux du modele pour la parole lue) :
    longueur du mot            n_phones
    frequence lexicale         freq_zipf
    interaction longueur x frequence
    classe du mot              contenu (NN, VB, JJ, RB) ou fonction

La valeur est le residu de la regression lineaire multiple de la surprise sur
ces quatre variables : la part de la surprise qu'elles ne predisent pas. Elle
est en bits, et sa moyenne est 0 par construction (donc environ la moitie des
valeurs sont negatives).

Pas de dependance, seulement la bibliotheque standard.
A lancer depuis le dossier des deux CSV :

    python surprise_apres_controle_total.py
"""

import csv
import glob
import os

TACHES = [
    (["surprisal_incrementale.csv", "surprisal_incrementale-*.csv"],
     "surprise_incrementale", "surprise_syn_apres_controle",
     "surprise_syntaxique_apres_controle.csv"),
    (["surprisal_llm_textgrid.csv", "surprisal_llm_textgrid-*.csv"],
     "surprise_llm", "surprise_llm_apres_controle",
     "surprise_llm_apres_controle.csv"),
]

DOSSIERS = [
    ".",
    os.path.expanduser(r"~\Downloads\the old man and the sea"),
    os.path.expanduser(r"~\Downloads\ds004408_stimuli\stimuli"),
    os.path.expanduser(r"~\Downloads"),
]

CLASSES_CONTENU = ("NN", "VB", "JJ", "RB")


def trouver(motifs):
    for dossier in DOSSIERS:
        for motif in motifs:
            trouves = sorted(glob.glob(os.path.join(dossier, motif)))
            trouves = [t for t in trouves
                       if "ami" not in os.path.basename(t).lower()]
            if trouves:
                return trouves[0]
    return None


def nombre(txt):
    try:
        return float(str(txt).strip())
    except (TypeError, ValueError):
        return None


def resoudre(A, b):
    """Resout A x = b par elimination de Gauss avec pivot partiel."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("systeme singulier : une variable de controle "
                             "est constante ou redondante")
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / p
            if f:
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def moindres_carres(X, y):
    """Coefficients de la regression de y sur X (X contient deja la constante)."""
    p = len(X[0])
    XtX = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(p)]
           for i in range(p)]
    Xty = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(p)]
    return resoudre(XtX, Xty)


def correlation(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    if saa <= 0 or sbb <= 0:
        return 0.0
    return sab / (saa ** 0.5 * sbb ** 0.5)


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
    for requis in (col_surprise, "freq_zipf", "n_phones", "POS"):
        if requis not in entetes:
            print(f"  {chemin} : colonne '{requis}' absente")
            print(f"  colonnes presentes : {entetes}")
            return

    # lignes exploitables
    idx, L, F, C, Y = [], [], [], [], []
    for i, ligne in enumerate(lignes):
        y = nombre(ligne[col_surprise])
        f_ = nombre(ligne["freq_zipf"])
        l_ = nombre(ligne["n_phones"])
        if y is None or f_ is None or l_ is None:
            continue
        pos = str(ligne["POS"]).strip().upper()[:2]
        idx.append(i)
        L.append(l_)
        F.append(f_)
        C.append(1.0 if pos in CLASSES_CONTENU else 0.0)
        Y.append(y)

    n = len(Y)
    if n < 10:
        print(f"  {chemin} : trop peu de lignes exploitables ({n})")
        return

    # centrage des variables continues, pour le conditionnement
    mL = sum(L) / n
    mF = sum(F) / n
    Lc = [v - mL for v in L]
    Fc = [v - mF for v in F]
    INT = [a * b for a, b in zip(Lc, Fc)]

    X = [[1.0, Lc[k], Fc[k], INT[k], C[k]] for k in range(n)]
    beta = moindres_carres(X, Y)

    ajuste = [sum(b * x for b, x in zip(beta, X[k])) for k in range(n)]
    res = [Y[k] - ajuste[k] for k in range(n)]

    # diagnostics
    my = sum(Y) / n
    sst = sum((v - my) ** 2 for v in Y)
    sse = sum(v * v for v in res)
    r2 = 1 - sse / sst if sst else 0.0

    print(f"  source            : {chemin}")
    print(f"  lignes            : {len(lignes)}  (dont {n} exploitables)")
    print(f"  part de contenu   : {sum(C) / n:.1%}")
    print(f"  coefficients      : const {beta[0]:+.4f} | longueur {beta[1]:+.4f} | "
          f"frequence {beta[2]:+.4f} | interaction {beta[3]:+.4f} | contenu {beta[4]:+.4f}")
    print(f"  R2 du controle    : {r2:.4f}  "
          f"(part de la surprise expliquee par les controles)")
    print(f"  moyenne residus   : {sum(res) / n:+.2e}   (doit etre ~0)")
    for nom_v, v in (("frequence", F), ("longueur", L), ("contenu", C)):
        print(f"  r(residus, {nom_v:9s}) = {correlation(res, v):+.2e}   (doit etre ~0)")

    dico = {i: r for i, r in zip(idx, res)}
    with open(sortie, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=entetes + [col_residu])
        w.writeheader()
        for i, ligne in enumerate(lignes):
            ligne = dict(ligne)
            ligne[col_residu] = "" if i not in dico else f"{dico[i]:.6f}"
            w.writerow(ligne)
    print(f"  ecrit             : {sortie}  ({len(lignes)} lignes)\n")


if __name__ == "__main__":
    print("Surprise apres controle complet, The Old Man and the Sea")
    print("Controles : longueur, frequence, longueur x frequence, classe du mot\n")
    for motifs, col_src, col_res, sortie in TACHES:
        print(f"[{col_res}]")
        traiter(motifs, col_src, col_res, sortie)
    print("Termine.")
