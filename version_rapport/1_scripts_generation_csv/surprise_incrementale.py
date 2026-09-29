# -*- coding: utf-8 -*-
"""
Surprise syntaxique incrementale (Hale 2001), calculee sur les TextGrid de
The Old Man and the Sea.

J'avais d'abord essaye une surprise "structurale" (trigramme de POS), mais elle
mesure surtout la previsibilite de la categorie du mot, et elle est tres liee a la
longueur (les categories les plus surprenantes sont souvent des mots-outils courts).
Ici on mesure plutot le cout de construction de l'arbre au fur et a mesure, avec les
probabilites de prefixe d'une PCFG, comme dans Demberg et al. (2012) :

    surprise(mot_k) = -log2 [ P(w_1..w_k) / P(w_1..w_{k-1}) ]

avec P(w_1..w_k) = probabilite que la grammaire produise une phrase qui commence par
w_1..w_k (somme sur toutes les analyses possibles du debut de phrase).

Methode : algorithme de Jelinek & Lafferty (1991) sur une PCFG en forme normale
de Chomsky.
  - e[i][j][A] : probabilite que A produise exactement w_i..w_j (inside, type CKY)
  - pi[i][A]   : probabilite que A produise une chaine qui commence par w_i..w_L
        pi[i] = RL . c[i]   avec RL = (I - PL)^-1  (cloture left-corner)
  - P(prefixe) = pi[1][S]
Les deux clotures (left-corner et unaire) sont calculees avec une factorisation LU
creuse (scipy), sinon c'est trop lent avec plusieurs milliers de regles.

J'ai verifie l'algorithme contre une enumeration exhaustive sur une petite grammaire
(_test_prefix.py).

Utilisation :
    pip install nltk scipy numpy wordfreq
    python surprise_incrementale.py .                # dossier de .TextGrid (ds004408)
    python surprise_incrementale.py mon_audio.TextGrid

Sortie : surprisal_incrementale.csv
    fichier, phrase_id, position, mot, onset, offset, POS,
    surprise_incrementale, freq_zipf, n_phones
"""

import os, re, sys, csv, glob, math
import numpy as np


# ===========================================================================
#  Probabilite de prefixe (Jelinek-Lafferty, PCFG en CNF)
# ===========================================================================

class PrefixParser:
    """Surprise incrementale de chaque mot d'une phrase, a partir d'une PCFG en CNF.
    Les regles sont passees en trois listes : binaires (A, B, C, p), unaires (A, B, p)
    et lexicales (A, mot, p).
    """

    def __init__(self, bin_rules, un_rules, lex_rules, start):
        self.start = start
        nts = {start}
        for A, B, C, p in bin_rules: nts |= {A, B, C}
        for A, B, p in un_rules:     nts |= {A, B}
        for A, w, p in lex_rules:    nts.add(A)
        self.NT = sorted(nts)
        self.idx = {A: i for i, A in enumerate(self.NT)}
        n = len(self.NT)

        self.bin = bin_rules
        # regles binaires groupees par partie gauche
        from collections import defaultdict
        self.bin_by_lhs = defaultdict(list)
        for A, B, C, p in bin_rules:
            self.bin_by_lhs[A].append((B, C, p))
        # regles lexicales groupees par mot
        self.lex_by_word = defaultdict(list)
        for A, w, p in lex_rules:
            self.lex_by_word[w].append((A, p))

        # matrices de cloture (creuses)
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu
        # PL : coin gauche = 1er symbole des regles binaires, plus les regles unaires
        rows, cols, vals = [], [], []
        for A, B, C, p in bin_rules:
            rows.append(self.idx[A]); cols.append(self.idx[B]); vals.append(p)
        for A, B, p in un_rules:
            rows.append(self.idx[A]); cols.append(self.idx[B]); vals.append(p)
        PL = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        self._luL = splu((sp.eye(n, format="csc") - PL).tocsc())   # (I-PL) x = c
        # PU : regles unaires seules, pour l'inside
        rows, cols, vals = [], [], []
        for A, B, p in un_rules:
            rows.append(self.idx[A]); cols.append(self.idx[B]); vals.append(p)
        PU = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        self._luU = splu((sp.eye(n, format="csc") - PU).tocsc())   # (I-PU) x = ebin
        self.n = n

    def _solveL(self, c):     # RL . c  = (I-PL)^-1 c
        return self._luL.solve(c)

    def _solveU(self, c):     # RU . c  = (I-PU)^-1 c
        return self._luU.solve(c)

    def surprisals(self, words):
        """Surprise (en bits) de chaque mot. Les mots doivent etre dans le vocabulaire."""
        n = self.n
        L = len(words)
        # e[i][j] : vecteur inside de w_i..w_j (indices a partir de 1)
        e = [[None] * (L + 2) for _ in range(L + 2)]
        prefix = [1.0]                     # P(prefixe vide) = 1
        surps = []

        for end in range(1, L + 1):        # on ajoute les mots un par un
            w = words[end - 1]
            # inside : case e[end][end] (lexicale) puis colonne end
            ebin = np.zeros(n)
            for A, p in self.lex_by_word.get(w, []):
                ebin[self.idx[A]] += p
            e[end][end] = self._apply_unary(ebin)
            for i in range(end - 1, 0, -1):
                ebin = np.zeros(n)
                for A, B, C, p in self.bin:
                    bi = self.idx[B]; ci = self.idx[C]
                    s = 0.0
                    for m in range(i, end):
                        eb = e[i][m]; ec = e[m + 1][end]
                        if eb is None or ec is None:
                            continue
                        vb = eb[bi]
                        if vb:
                            s += vb * ec[ci]
                    if s:
                        ebin[self.idx[A]] += p * s
                e[i][end] = self._apply_unary(ebin) if ebin.any() else np.zeros(n)

            # pi[i] pour le prefixe courant, de i = end jusqu'a 1
            pi = [None] * (end + 2)
            for i in range(end, 0, -1):
                c = np.zeros(n)
                if i == end:               # le mot en derniere position
                    for A, p in self.lex_by_word.get(w, []):
                        c[self.idx[A]] += p
                for A, B, C, p in self.bin:      # B couvre i..j (j < end), C continue apres
                    bi = self.idx[B]; ci = self.idx[C]
                    acc = 0.0
                    for j in range(i, end):
                        eb = e[i][j]
                        if eb is None:
                            continue
                        vb = eb[bi]
                        if vb:
                            acc += vb * pi[j + 1][ci]
                    if acc:
                        c[self.idx[A]] += p * acc
                pi[i] = self._solveL(c)
            pk = pi[1][self.idx[self.start]]
            prev = prefix[-1]
            if pk > 0 and prev > 0:
                surps.append(-math.log2(pk / prev))
            else:
                surps.append(float("nan"))
            prefix.append(pk)
        return surps

    def _apply_unary(self, ebin):
        if not ebin.any():
            return ebin
        return self._solveU(ebin)


# ===========================================================================
#  Grammaire : PCFG apprise sur le Penn Treebank de NLTK
# ===========================================================================

def induce_grammar(horzMarkov=0):
    """PCFG en CNF apprise sur l'echantillon du Penn Treebank de NLTK.
    Avec horzMarkov=0 la grammaire reste petite, donc les matrices de cloture aussi.
    Renvoie (bin, un, lex, vocab, best_pos, start)."""
    import nltk
    for r in ['treebank', 'punkt', 'punkt_tab',
              'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
        nltk.download(r, quiet=True)
    from nltk.corpus import treebank
    from nltk import induce_pcfg, Nonterminal
    prods = []
    for t in treebank.parsed_sents():
        t.collapse_unary(collapsePOS=False)
        t.chomsky_normal_form(horzMarkov=horzMarkov)
        prods += t.productions()
    g = induce_pcfg(Nonterminal('S'), prods)

    bin_rules, un_rules, lex_rules = [], [], []
    vocab = set()
    for p in g.productions():
        A = str(p.lhs()); rhs = p.rhs(); pr = p.prob()
        if len(rhs) == 1 and isinstance(rhs[0], str):
            lex_rules.append((A, rhs[0].lower(), pr)); vocab.add(rhs[0].lower())
        elif len(rhs) == 1:
            un_rules.append((A, str(rhs[0]), pr))
        elif len(rhs) == 2:
            bin_rules.append((A, str(rhs[0]), str(rhs[1]), pr))
    # mot le plus probable de chaque POS (sert a remplacer les mots hors vocabulaire)
    best_pos = {}
    for A, w, pr in lex_rules:
        if A not in best_pos or pr > best_pos[A][1]:
            best_pos[A] = (w, pr)
    best_pos = {k: v[0] for k, v in best_pos.items()}
    return bin_rules, un_rules, lex_rules, vocab, best_pos, 'S'


# ===========================================================================
#  Programme principal : segmentation prosodique, surprise par mot, CSV
# ===========================================================================

def main():
    # segmentation prosodique et variables de controle (constituants_dataset.py)
    from constituants_dataset import (load_sentences, fix_oov, word_freq, count_phones)
    arg = sys.argv[1] if len(sys.argv) > 1 else "."
    if os.path.isdir(arg):
        folder = arg
        files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                       if re.search(r'\.textgrid$', f, re.I)
                       and not os.path.basename(f).startswith("._"))
    else:
        folder = os.path.dirname(arg) or "."
        files = [arg] if os.path.exists(arg) else []
    if not files:
        print(f"  Aucun .TextGrid trouve : {os.path.abspath(arg)}"); return

    print(f"  {len(files)} fichier(s). Induction de la PCFG (CNF, horzMarkov=0)...")
    bin_r, un_r, lex_r, vocab, best_pos, start = induce_grammar(horzMarkov=0)
    print(f"  Grammaire : {len(bin_r)} binaires, {len(un_r)} unaires, "
          f"{len(lex_r)} lexicales, {len(vocab)} mots.")
    print("  Construction des clotures (factorisation LU creuse)...")
    parser = PrefixParser(bin_r, un_r, lex_r, start)
    print(f"  {parser.n} non-terminaux. Pret.")

    SENT_MAX = 40                          # phrases plus longues ignorees (cout en O(n^3))
    out_csv = os.path.join(folder, "surprisal_incrementale.csv")
    n_words = n_sent = n_skip = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["fichier", "phrase_id", "position", "mot", "onset", "offset",
                     "POS", "surprise_incrementale", "freq_zipf", "n_phones"])
        for path in files:
            name = os.path.basename(path)
            sents, phones = load_sentences(path)
            print(f"  - {name}: {len(sents)} phrases")
            for sid, sent in enumerate(sents, 1):
                plain = [w for (w, on, off) in sent]
                if len(plain) > SENT_MAX:
                    n_skip += 1; continue
                # remplacement des mots hors vocabulaire (meme regle que fix_oov)
                proc, info = fix_oov(plain, vocab, best_pos)
                try:
                    surps = parser.surprisals(proc)
                except Exception as ex:
                    print(f"      (phrase {sid} ignoree : {ex})"); n_skip += 1; continue
                n_sent += 1
                for i, (w, on, off) in enumerate(sent):
                    s = surps[i] if i < len(surps) else float("nan")
                    pos = info[i][1] if i < len(info) else ""
                    wr.writerow([name, sid, i + 1, w.lower(),
                                 "" if on is None else f"{on:.3f}",
                                 "" if off is None else f"{off:.3f}",
                                 pos, "" if s != s else f"{s:.4f}",
                                 word_freq(w), count_phones(phones, on, off)])
                    n_words += 1
    print(f"\n  Termine. {n_words} mots, {n_sent} phrases ({n_skip} ignorees).")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
