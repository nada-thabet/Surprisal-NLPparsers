# -*- coding: utf-8 -*-
"""
Etape 2 : surprise syntaxique incrementale (Hale 2001), en nats.

    surprise_syn(w_k) = -ln [ P(w_1..w_k) / P(w_1..w_{k-1}) ]

P(prefixe) est la probabilite de prefixe d'une PCFG (Jelinek & Lafferty 1991).
La PCFG est apprise avec NLTK sur son echantillon du Penn Treebank (3 914 phrases
du WSJ). Le contexte est la phrase (prosodique pour le livre, du transcript pour
AMI) : on repart de S a chaque phrase.

Entree : mots_livre.csv ou mots_ami.csv (etape 1). Colonnes ajoutees :
    POS              etiquette NLTK (perceptron) du mot en minuscules
    hors_vocab       1 si le mot n'est pas dans la grammaire ; il est alors remplace
                     par le mot le plus probable de sa categorie (donc sa surprise
                     est celle du mot de remplacement)
    surprise_syn     en nats (>= 0)

Utilisation :
    python 2_surprise_syntaxique.py mots_livre.csv sortie.csv [--grammaire originale] [--proc 4]
--grammaire propre (par defaut) : treebank sans ponctuation, sans elements vides
    (-NONE-) et sans etiquettes fonctionnelles. Les mots en entree n'ont ni
    ponctuation ni traces, donc avec la grammaire d'origine toutes les analyses qui
    en ont besoin avaient une probabilite nulle.
--grammaire originale : la grammaire du rapport (valeurs du rapport x ln 2).
"""
import argparse, csv, os, sys, time
from multiprocessing import Pool

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import commun

G = P = None


def _init(propre):
    global G, P
    G = commun.induire_grammaire(propre=propre)
    P = commun.PrefixParser(G[0], G[1], G[2], G[5])


def _phrase(words):
    proc, tags, oov = commun.remplacer_hors_vocab(words, G[3], G[4])
    return P.surprisals(proc), tags, oov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entree"); ap.add_argument("sortie")
    ap.add_argument("--grammaire", choices=["propre", "originale"], default="propre")
    ap.add_argument("--proc", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()

    with open(a.entree, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f); cols = rd.fieldnames; rows = list(rd)
    cle = ["fichier", "phrase_id"] if "fichier" in cols else ["meeting", "locuteur", "phrase_id"]
    phrases, cur, prev = [], [], None
    for r in rows:
        k = tuple(r[c] for c in cle)
        if k != prev and cur:
            phrases.append(cur); cur = []
        cur.append(r); prev = k
    if cur:
        phrases.append(cur)
    print(f"  {len(rows)} mots, {len(phrases)} phrases (la plus longue : "
          f"{max(len(p) for p in phrases)} mots). Grammaire {a.grammaire}, {a.proc} processus.")
    # les phrases longues en premier, pour mieux repartir le travail entre processus
    ordre = sorted(range(len(phrases)), key=lambda i: -len(phrases[i]))
    t0 = time.time()
    with Pool(a.proc, initializer=_init, initargs=(a.grammaire == "propre",)) as pool:
        res = [None] * len(phrases)
        it = pool.imap(_phrase, [[r["mot"] for r in phrases[i]] for i in ordre], chunksize=1)
        for n, (i, r) in enumerate(zip(ordre, it), 1):
            res[i] = r
            if n % 500 == 0:
                print(f"    {n}/{len(phrases)} phrases ({time.time() - t0:.0f} s)", flush=True)
    with open(a.sortie, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(cols + ["POS", "hors_vocab", "surprise_syn"])
        for ph, (s, tags, oov) in zip(phrases, res):
            for r, si, t, o in zip(ph, s, tags, oov):
                wr.writerow([r[c] for c in cols] + [t, int(o), "" if si != si else f"{si:.4f}"])
    print(f"  Termine en {time.time() - t0:.0f} s -> {a.sortie}")


if __name__ == "__main__":
    main()
