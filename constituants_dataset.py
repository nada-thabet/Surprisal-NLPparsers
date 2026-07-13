# -*- coding: utf-8 -*-
"""
ANALYSE EN CONSTITUANTS (context-free) + SURPRISAL  --  DATASET Broderick ds004408

SEGMENTATION (retour Philippe -- refs Shriberg & Stolcke) :
  Le decoupage en phrases se fait par la PROSODIE (module segmentation_prosodique.py) :
  pause reelle entre mots + allongement du dernier phone avant la frontiere. C'est la
  methode adaptee a la PAROLE (ni texte ecrit, ni LLM). Les deux tiers du TextGrid
  ("words" et "phones") suffisent.


CORRECTIONS precedentes conservees :
  - mots mis en MINUSCULES avant l'etiquetage (fini NNP/'Mr.').
  - arbres de-binarises (un_chomsky_normal_form), colonne 'oov'.

Usage :  python constituants_dataset.py .
"""

import os, re, sys, csv, glob, math, nltk

OUTPUTS = {"arbres_constituants.txt", "surprisal_constituants.csv",
           "phrases_prosodiques.txt", "dependances.csv", "comparaison.csv"}

for r in ['treebank', 'punkt', 'punkt_tab',
          'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(r, quiet=True)

from nltk.corpus import treebank
from nltk import induce_pcfg, Nonterminal
from nltk.parse import ViterbiParser

# --- segmenteur prosodique (fichier segmentation_prosodique.py, meme dossier) ---
from segmentation_prosodique import (read_words_and_phones, segment_prosodic,
                                     load_pitch_reset)


# ---------------------------------------------------------------------------
#  LECTURE DES STIMULI  (TextGrid Praat ; repli texte brut)
# ---------------------------------------------------------------------------

def load_sentences(path):
    """Renvoie (phrases, phones). phrases = liste de [(mot, onset, offset)]."""
    words, phones = read_words_and_phones(path)
    if words is not None:
        reset = load_pitch_reset(path, words)      # pitch si un .wav est a cote
        return segment_prosodic(words, phones, reset=reset), phones
    # repli : fichier texte sans tier 'words' -> une phrase par ligne tokenisee
    txt = open(path, encoding='utf-8', errors='ignore').read()
    sents = []
    for line in txt.splitlines():
        toks = nltk.word_tokenize(line)
        if toks:
            sents.append([(t, None, None) for t in toks])
    return sents, None


# --- facteurs de controle pour la regression (cf. Demberg et al. 2012) --------

def count_phones(phones, on, off):
    """Nombre de phonemes du mot [on, off] (duree canonique de reference ~ MARY)."""
    if not phones or on is None or off is None:
        return ""
    n = sum(1 for (p, po, pf) in phones if pf > on + 1e-6 and po < off - 1e-6)
    return n


def word_freq(w):
    """Frequence Zipf du mot (0-8) via wordfreq ; '' si la librairie manque."""
    try:
        from wordfreq import zipf_frequency
        return round(zipf_frequency(w.lower(), 'en'), 3)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
#  PCFG  (context-free + surprisal)
# ---------------------------------------------------------------------------

def load_pcfg():
    print("  Induction de la PCFG (Penn Treebank)...")
    prods = []
    for t in treebank.parsed_sents():
        t.collapse_unary(collapsePOS=False); t.chomsky_normal_form(horzMarkov=2)
        prods += t.productions()
    grammar = induce_pcfg(Nonterminal('S'), prods)
    vocab, best_pos = set(), {}
    for p in grammar.productions():
        if len(p.rhs()) == 1 and isinstance(p.rhs()[0], str):
            w, pos = p.rhs()[0], str(p.lhs()); vocab.add(w)
            if pos not in best_pos or p.prob() > best_pos[pos][1]:
                best_pos[pos] = (w, p.prob())
    best_pos = {k: v[0] for k, v in best_pos.items()}
    lex_probs = {}
    for p in grammar.productions():
        if len(p.rhs()) == 1 and isinstance(p.rhs()[0], str):
            key = (str(p.lhs()), p.rhs()[0].lower())
            lex_probs[key] = lex_probs.get(key, 0.0) + p.prob()
    return ViterbiParser(grammar, trace=0), vocab, best_pos, lex_probs


# ---------------------------------------------------------------------------
#  SURPRISE STRUCTURALE  (independante du mot : -log2 P(categorie | contexte))
#  Corrige le biais WSJ : "boy" compte comme un NN attendu, pas comme un mot rare.
# ---------------------------------------------------------------------------

def load_pos_lm():
    """Modele trigramme des categories (POS) appris sur le Penn Treebank."""
    from collections import Counter
    uni, bi, tri, V = Counter(), Counter(), Counter(), set()
    for sent in treebank.tagged_sents():
        tags = ['<s>', '<s>'] + [t for (_, t) in sent] + ['</s>']
        for t in tags:
            V.add(t)
        for i in range(2, len(tags)):
            uni[tags[i]] += 1
            bi[(tags[i - 1], tags[i])] += 1
            tri[(tags[i - 2], tags[i - 1], tags[i])] += 1
    return {'uni': uni, 'bi': bi, 'tri': tri,
            'V': len(V), 'N': sum(uni.values())}


def pos_surprisal(lm, t2, t1, t):
    """-log2 P(t | t2, t1) par interpolation tri/bi/uni (lissage add-1)."""
    uni, bi, tri, V, N = lm['uni'], lm['bi'], lm['tri'], lm['V'], lm['N']
    p_uni = (uni.get(t, 0) + 1) / (N + V)
    c1 = uni.get(t1, 0)
    p_bi = (bi.get((t1, t), 0) + 1) / (c1 + V) if c1 else p_uni
    c2 = bi.get((t2, t1), 0)
    p_tri = (tri.get((t2, t1, t), 0) + 1) / (c2 + V) if c2 else p_bi
    p = 0.6 * p_tri + 0.3 * p_bi + 0.1 * p_uni
    return -math.log2(p)


def fix_oov(words, vocab, best_pos):
    low = [w.lower() for w in words]
    tagged = nltk.pos_tag(low)             # minuscules -> POS correct
    proc, info = [], []
    for (orig, (wl, pos)) in zip(words, tagged):
        disp = orig.lower()
        if wl in vocab:
            proc.append(wl);            info.append((disp, pos, False))
        elif orig in vocab:
            proc.append(orig);          info.append((disp, pos, False))
        elif pos in best_pos:
            proc.append(best_pos[pos]); info.append((disp, pos, True))
        else:
            proc.append(wl);            info.append((disp, pos, True))
    return proc, info


def restore_leaves(tree, words):
    for i in range(len(tree.leaves())):
        if i < len(words):
            tree[tree.leaf_treeposition(i)] = words[i]
    return tree


def parse_sentence(viterbi, vocab, best_pos, lex_probs, pos_lm, plain_words):
    proc, info = fix_oov(plain_words, vocab, best_pos)
    try:
        parses = list(viterbi.parse(proc + ['.']))
    except Exception:
        parses = []
    if not parses:
        return None, None
    tree = parses[0]
    leaves = [s for s in tree.subtrees(lambda t: t.height() == 2)]
    tags = [st.label() for st in leaves][:len(plain_words)]     # sequence de POS du parse
    rows = []
    for i, st in enumerate(leaves):
        if i >= len(plain_words):
            break
        tree_pos = st.label(); pw = st.leaves()[0]
        ow, opos, is_oov = info[i]
        prob = lex_probs.get((tree_pos, pw.lower()), 1e-10)          # surprise lexicale (WSJ)
        show_pos = opos if is_oov else tree_pos
        t2 = tags[i - 2] if i - 2 >= 0 else '<s>'                    # surprise structurale
        t1 = tags[i - 1] if i - 1 >= 0 else '<s>'
        struct = pos_surprisal(pos_lm, t2, t1, tree_pos)
        rows.append((ow, show_pos, -math.log2(prob), struct, is_oov))
    try:
        tree.un_chomsky_normal_form()
    except Exception:
        pass
    restore_leaves(tree, [w.lower() for w in plain_words])
    return tree, rows


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "."
    if os.path.isdir(arg):
        folder = arg
        allf = [f for f in glob.glob(os.path.join(folder, "*"))
                if not os.path.basename(f).startswith("._")]
        tg = [f for f in allf if re.search(r'\.textgrid$', f, re.I)]
        if tg:
            files = sorted(tg)                    # priorite aux TextGrid
        else:                                     # repli : .txt hors sorties
            files = sorted(f for f in allf if re.search(r'\.txt$', f, re.I)
                           and os.path.basename(f) not in OUTPUTS)
    else:
        folder = os.path.dirname(arg) or "."
        files = [arg] if os.path.exists(arg) else []
    if not files:
        print(f"  Aucun fichier .txt/.TextGrid trouve : {os.path.abspath(arg)}"); return

    print(f"  {len(files)} fichier(s) a traiter.")
    print("  [segmentation] prosodique (pause + allongement final).")
    viterbi, vocab, best_pos, lex_probs = load_pcfg()
    print("  Modele de categories (surprise structurale)...")
    pos_lm = load_pos_lm()

    out_csv = os.path.join(folder, "surprisal_constituants.csv")
    out_tree = os.path.join(folder, "arbres_constituants.txt")
    n_words = n_sent = n_fail = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv, \
         open(out_tree, "w", encoding="utf-8") as ftree:
        wr = csv.writer(fcsv)
        wr.writerow(["fichier", "phrase_id", "position", "mot",
                     "onset", "offset", "POS",
                     "surprisal_bits", "surprise_structurale", "oov",
                     "freq_zipf", "n_phones"])
        for path in files:
            name = os.path.basename(path)
            sents, phones = load_sentences(path)
            print(f"  - {name}: {sum(len(s) for s in sents)} mots -> {len(sents)} phrases")
            for sid, sent in enumerate(sents, 1):
                plain = [w for (w, on, off) in sent]
                tree, rows = parse_sentence(viterbi, vocab, best_pos, lex_probs, pos_lm, plain)
                n_sent += 1
                if rows is None:
                    n_fail += 1; continue
                ftree.write(f"### {name} | phrase {sid} : {' '.join(w.lower() for w in plain)}\n")
                ftree.write(tree.pformat(margin=70) + "\n\n")
                for pos_i, (w, pos, surp, struct, is_oov) in enumerate(rows):
                    on, off = sent[pos_i][1], sent[pos_i][2]
                    wr.writerow([name, sid, pos_i + 1, w,
                                 "" if on is None else f"{on:.3f}",
                                 "" if off is None else f"{off:.3f}",
                                 pos, f"{surp:.4f}", f"{struct:.4f}", int(is_oov),
                                 word_freq(w), count_phones(phones, on, off)])
                    n_words += 1

    print(f"\n  Termine. {n_words} mots, {n_sent} phrases ({n_fail} non analysees).")
    print(f"  -> {out_csv}\n  -> {out_tree}")


if __name__ == "__main__":
    main()
