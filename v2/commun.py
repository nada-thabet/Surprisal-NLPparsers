# -*- coding: utf-8 -*-
"""
Fonctions communes aux scripts de la version 2.

Ce qui change par rapport a la version du rapport (dossier version_rapport/) :
  - les surprises sont en nats (log naturel), comme au labo :
        surprise(mot) = -ln P(mot | contexte)
    (avant elles etaient en bits, -log2 ; bits = nats / ln 2) ;
  - la PCFG peut etre apprise sur un treebank nettoye (sans ponctuation, sans
    elements vides -NONE-, et NP-SBJ-1 devient NP), puisque les mots en entree
    n'ont ni ponctuation ni traces ;
  - plus aucune phrase n'est ignoree a cause de sa longueur ;
  - la frequence Zipf des sigles AMI est corrigee ("l_c_d_" devient "lcd").
"""

import math
import re
from collections import defaultdict

import numpy as np


# ===========================================================================
#  Probabilite de prefixe d'une PCFG en CNF (Jelinek & Lafferty 1991)
#  Meme algorithme que version_rapport/.../surprise_incrementale.py, mais en nats.
# ===========================================================================

class PrefixParser:
    """Surprise incrementale (Hale 2001) :
        surprise(w_k) = -ln [ P(w_1..w_k) / P(w_1..w_{k-1}) ]
    avec P(w_1..w_k) la probabilite de prefixe (Jelinek-Lafferty).
    La probabilite de prefixe ne peut que baisser quand on ajoute un mot,
    donc la surprise est toujours >= 0.
    """

    def __init__(self, bin_rules, un_rules, lex_rules, start):
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu
        self.start = start
        nts = {start}
        for A, B, C, p in bin_rules: nts |= {A, B, C}
        for A, B, p in un_rules:     nts |= {A, B}
        for A, w, p in lex_rules:    nts.add(A)
        self.NT = sorted(nts)
        self.idx = {A: i for i, A in enumerate(self.NT)}
        n = self.n = len(self.NT)
        # regles binaires rangees en tableaux numpy pour vectoriser le calcul
        self.bA = np.array([self.idx[A] for A, B, C, p in bin_rules], dtype=np.int64)
        self.bB = np.array([self.idx[B] for A, B, C, p in bin_rules], dtype=np.int64)
        self.bC = np.array([self.idx[C] for A, B, C, p in bin_rules], dtype=np.int64)
        self.bP = np.array([p for A, B, C, p in bin_rules])
        self.lex_by_word = defaultdict(list)
        for A, w, p in lex_rules:
            self.lex_by_word[w].append((self.idx[A], p))
        # cloture left-corner (I - PL)^-1 et cloture unaire (I - PU)^-1
        rows, cols, vals = [], [], []
        for A, B, C, p in bin_rules:
            rows.append(self.idx[A]); cols.append(self.idx[B]); vals.append(p)
        for A, B, p in un_rules:
            rows.append(self.idx[A]); cols.append(self.idx[B]); vals.append(p)
        PL = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        self._luL = splu((sp.eye(n, format="csc") - PL).tocsc())
        rows, cols, vals = [], [], []
        for A, B, p in un_rules:
            rows.append(self.idx[A]); cols.append(self.idx[B]); vals.append(p)
        PU = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        self._luU = splu((sp.eye(n, format="csc") - PU).tocsc())

    def _lex(self, w):
        v = np.zeros(self.n)
        for a, p in self.lex_by_word.get(w, []):
            v[a] += p
        return v

    def _binary(self, left_vecs, right_vecs):
        """Somme sur m et sur les regles A -> B C de p * left_m[B] * right_m[C], par A."""
        out = np.zeros(self.n)
        if not left_vecs:
            return out
        Lm = np.array(left_vecs)[:, self.bB]       # (m, R)
        Rm = np.array(right_vecs)[:, self.bC]      # (m, R)
        s = (Lm * Rm).sum(axis=0) * self.bP
        np.add.at(out, self.bA, s)
        return out

    def surprisals(self, words):
        """Surprise (nats) de chaque mot ; les mots doivent etre dans le vocabulaire."""
        L = len(words)
        e = [[None] * (L + 2) for _ in range(L + 2)]   # inside e[i][j]
        prev = 1.0
        surps = []
        for end in range(1, L + 1):
            w = words[end - 1]
            lex = self._lex(w)
            e[end][end] = self._luU.solve(lex) if lex.any() else lex
            for i in range(end - 1, 0, -1):
                ebin = self._binary([e[i][m] for m in range(i, end)],
                                    [e[m + 1][end] for m in range(i, end)])
                e[i][end] = self._luU.solve(ebin) if ebin.any() else ebin
            pi = [None] * (end + 2)
            for i in range(end, 0, -1):
                c = lex.copy() if i == end else np.zeros(self.n)
                if i < end:
                    c += self._binary([e[i][j] for j in range(i, end)],
                                      [pi[j + 1] for j in range(i, end)])
                pi[i] = self._luL.solve(c)
            pk = pi[1][self.idx[self.start]]
            surps.append(-math.log(pk / prev) if pk > 0 and prev > 0 else float("nan"))
            prev = pk
        return surps


# ===========================================================================
#  Grammaire : PCFG en CNF apprise sur le Penn Treebank de NLTK
# ===========================================================================

PONCT = {',', '.', ':', '``', "''", '-LRB-', '-RRB-', '-NONE-', '#', '$'}


def _nettoyer(t):
    """Enleve la ponctuation et les elements vides, et simplifie NP-SBJ-1 en NP."""
    from nltk import Tree
    if isinstance(t, str):
        return t
    if t.label() in PONCT:
        return None
    kids = [k for k in (_nettoyer(k) for k in t) if k is not None]
    if not kids:
        return None
    lab = t.label()
    if not lab.startswith('-'):
        lab = re.split(r'[-=]', lab)[0] or lab
    return Tree(lab, kids)


def induire_grammaire(propre=True):
    """Renvoie (bin, un, lex, vocab, best_pos, start).
    Avec propre=False on retrouve exactement la grammaire du rapport."""
    import nltk
    for r in ('treebank', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'):
        nltk.download(r, quiet=True)
    from nltk.corpus import treebank
    from nltk import induce_pcfg, Nonterminal
    prods = []
    for t in treebank.parsed_sents():
        if propre:
            t = _nettoyer(t)
            if t is None:
                continue
        t.collapse_unary(collapsePOS=False)
        t.chomsky_normal_form(horzMarkov=0)
        prods += t.productions()
    g = induce_pcfg(Nonterminal('S'), prods)
    b, u, lx, vocab = [], [], [], set()
    for p in g.productions():
        A, rhs, pr = str(p.lhs()), p.rhs(), p.prob()
        if len(rhs) == 1 and isinstance(rhs[0], str):
            lx.append((A, rhs[0].lower(), pr)); vocab.add(rhs[0].lower())
        elif len(rhs) == 1:
            u.append((A, str(rhs[0]), pr))
        else:
            b.append((A, str(rhs[0]), str(rhs[1]), pr))
    best = {}
    for A, w, pr in lx:
        if A not in best or pr > best[A][1]:
            best[A] = (w, pr)
    return b, u, lx, vocab, {k: v[0] for k, v in best.items()}, 'S'


def remplacer_hors_vocab(words, vocab, best_pos):
    """Meme regle que fix_oov (constituants_dataset.py) : un mot hors vocabulaire
    est remplace par le mot le plus probable de sa categorie POS.
    Renvoie (terminaux, POS, hors_vocab)."""
    import nltk
    low = [w.lower() for w in words]
    tags = [t for _, t in nltk.pos_tag(low)]
    proc, oov = [], []
    for wl, pos in zip(low, tags):
        if wl in vocab:
            proc.append(wl); oov.append(False)
        elif pos in best_pos:
            proc.append(best_pos[pos]); oov.append(True)
        else:
            proc.append(wl); oov.append(True)
    return proc, tags, oov


# ===========================================================================
#  Modele de langue (GPT-2 ou autre)
# ===========================================================================

def charger_modele(nom_modele):
    """Modele de langue causal Hugging Face (gpt2, gpt2-xl, ...) ou dossier local."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(nom_modele)
    model = AutoModelForCausalLM.from_pretrained(nom_modele).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    print(f"  Modele {nom_modele} sur {dev}")
    return tok, model, dev


def surprises_tokens(texte, tok, model, dev, verbeux=True, fenetre=1024):
    """Surprise (nats) de chaque token de `texte`, lu en continu depuis le debut
    avec une fenetre glissante de `fenetre` tokens (1024 au plus) ; chaque token a
    au moins une demi-fenetre de contexte. Renvoie (surprises, offsets)."""
    import torch
    enc = tok(texte, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    fen = min(getattr(model.config, "n_positions", None)
              or getattr(model.config, "max_position_embeddings", 1024), 1024, fenetre)
    pas = fen // 2
    surp = [None] * len(ids)
    debut = 0                                  # premier token encore sans surprise
    if verbeux:
        print(f"  {len(ids)} tokens, fenetre {fen}")
    while debut < len(ids):
        ctx0 = max(0, debut - (fen - 1 - pas))          # >= fen-1-pas tokens de contexte
        fin = min(len(ids), ctx0 + fen - 1)
        x = [bos] + ids[ctx0:fin]
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([x], device=dev)).logits[0].float(), -1)
        for t in range(debut, fin):                      # token t est en position t-ctx0+1
            surp[t] = -lp[t - ctx0, ids[t]].item()       # predit par la position precedente
        debut = fin
        if verbeux:
            print(f"    {debut}/{len(ids)} tokens", flush=True)
    return surp, offs



# ===========================================================================
#  Frequence
# ===========================================================================

def est_sigle(mot):
    """Sigles AMI : lettres suivies de '_' ("l_c_d_", "t_v_s")."""
    return "_" in mot and re.fullmatch(r"([a-z]_)+[a-z]?s?", mot) is not None


def forme_ecrite(mot):
    """Forme ecrite usuelle : 'l_c_d_' -> 'LCD', 't_v_s' -> 'TVs', 'i' -> 'I'."""
    if est_sigle(mot):
        lettres = mot.split("_")
        fin = lettres[-1]
        return "".join(x.upper() for x in lettres[:-1]) + (fin.upper() if len(fin) == 1 and fin != "s" else fin)
    if mot == "i" or mot.startswith("i'"):
        return "I" + mot[1:]
    return mot


def freq_zipf(mot):
    from wordfreq import zipf_frequency
    return round(zipf_frequency(forme_ecrite(mot).lower(), "en"), 3)
