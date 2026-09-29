# -*- coding: utf-8 -*-
"""
Verification du parseur incremental (commun.PrefixParser) : la probabilite de
prefixe qu'il calcule doit etre egale a la somme, faite a la main, des
probabilites de toutes les phrases qui commencent par ce prefixe (section 3.4 du
rapport : "les deux valeurs coincident").

Deux petites grammaires :
  1. une grammaire sans recursion, avec des regles unaires (S -> VP, NP -> N) et
     de l'ambiguite (rattachement du groupe prepositionnel au nom ou au verbe).
     Son langage est fini, donc on peut enumerer toutes les phrases et leurs
     probabilites (somme sur toutes les analyses) ;
  2. une grammaire recursive a gauche (S -> S X), qui fait travailler la cloture
     left-corner. Ici le langage est infini, mais on connait la valeur exacte :
     P(a x1 ... xk ...) = (0,4 x 0,5)^k.

Le test passe si l'ecart est inferieur a 1e-9 (en log naturel, donc en nats).
Il ne depend d'aucun fichier de donnees et tourne en une trentaine de secondes.

    pip install numpy scipy
    python test_prefixe.py
"""
import math, os, sys
from collections import defaultdict
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commun import PrefixParser

TOLERANCE = 1e-9

# ---------------------------------------------------------------------------
#  1. grammaire finie (regles binaires, unaires et lexicales)
# ---------------------------------------------------------------------------
BIN = [("S", "NP", "VP", 0.7),
       ("NP", "Det", "N", 0.5), ("NP", "NPb", "PP", 0.2), ("NPb", "Det", "N", 1.0),
       ("VP", "V", "NP", 0.5), ("VP", "VPb", "PP", 0.3), ("VPb", "V", "NP", 1.0),
       ("PP", "P", "NP2", 1.0), ("NP2", "Det", "N", 0.6)]
UN = [("S", "VP", 0.3), ("NP", "N", 0.3), ("VP", "V", 0.2), ("NP2", "N", 0.4)]
LEX = [("Det", "the", 0.6), ("Det", "a", 0.4),
       ("N", "man", 0.5), ("N", "fish", 0.5),
       ("V", "saw", 0.6), ("V", "fish", 0.4),
       ("P", "with", 1.0)]


def langage(bin_rules, un_rules, lex_rules, A, memo=None):
    """Toutes les suites de mots produites par A, avec leur probabilite
    (somme sur toutes les analyses). La grammaire ne doit pas etre recursive."""
    memo = {} if memo is None else memo
    if A in memo:
        return memo[A]
    dist = defaultdict(float)
    for X, w, p in lex_rules:
        if X == A:
            dist[(w,)] += p
    for X, B, p in un_rules:
        if X == A:
            for s, q in langage(bin_rules, un_rules, lex_rules, B, memo).items():
                dist[s] += p * q
    for X, B, C, p in bin_rules:
        if X == A:
            dB = langage(bin_rules, un_rules, lex_rules, B, memo)
            dC = langage(bin_rules, un_rules, lex_rules, C, memo)
            for (s1, q1), (s2, q2) in product(dB.items(), dC.items()):
                dist[s1 + s2] += p * q1 * q2
    memo[A] = dict(dist)
    return memo[A]


def prefixe_parseur(parser, mots):
    """P(w1..wk) pour chaque k, a partir des surprises en nats du parseur."""
    s = parser.surprisals(list(mots))
    return [math.exp(-sum(s[:k])) for k in range(1, len(mots) + 1)]


def test_grammaire_finie():
    phrases = langage(BIN, UN, LEX, "S")
    total = sum(phrases.values())
    print(f"1. Grammaire finie : {len(phrases)} phrases differentes, somme des probabilites = {total:.12f}")
    # probabilite de prefixe par enumeration
    pref = defaultdict(float)
    for s, q in phrases.items():
        for k in range(1, len(s) + 1):
            pref[s[:k]] += q
    parser = PrefixParser(BIN, UN, LEX, "S")
    ecart, n = 0.0, 0
    exemples = [("the", "man", "saw", "a", "fish", "with", "the", "man"),
                ("fish", "fish", "with", "fish"), ("saw", "the", "fish")]
    for s in list(phrases) + exemples:
        pp = prefixe_parseur(parser, s)
        for k in range(1, len(s) + 1):
            ecart = max(ecart, abs(math.log(pp[k - 1]) - math.log(pref[s[:k]])))
            n += 1
    for s in exemples:
        pp = prefixe_parseur(parser, s)
        print(f"   {' '.join(s):35s} P(phrase entiere comme prefixe) : parseur {pp[-1]:.10f}, "
              f"enumeration {pref[s]:.10f}")
    print(f"   {n} prefixes compares, ecart maximal |ln P_parseur - ln P_enumeration| = {ecart:.1e} nats")
    return ecart


def test_recursion_gauche():
    # S -> S X (0,4) | A (0,6) ; A -> a ; X -> b | c (0,5 chacun)
    parser = PrefixParser([("S", "S", "X", 0.4)], [("S", "A", 0.6)],
                          [("A", "a", 1.0), ("X", "b", 0.5), ("X", "c", 0.5)], "S")
    mots = ["a", "b", "c", "c", "b", "b", "c"]
    pp = prefixe_parseur(parser, mots)
    ecart = max(abs(math.log(pp[k]) - k * math.log(0.2)) for k in range(len(mots)))
    print(f"2. Grammaire recursive a gauche : P(a) = {pp[0]:.12f} (attendu 1), "
          f"P(a b c c b b c) = {pp[-1]:.4e} (attendu 0,2^6 = {0.2 ** 6:.4e})")
    print(f"   ecart maximal sur les {len(mots)} prefixes = {ecart:.1e} nats")
    return ecart


if __name__ == "__main__":
    e1 = test_grammaire_finie()
    e2 = test_recursion_gauche()
    ok = max(e1, e2) < TOLERANCE
    print(f"\n{'OK' if ok else 'ECHEC'} : ecart maximal {max(e1, e2):.1e} nats "
          f"({max(e1, e2) / math.log(2):.1e} bits), tolerance {TOLERANCE:.0e}")
    sys.exit(0 if ok else 1)
