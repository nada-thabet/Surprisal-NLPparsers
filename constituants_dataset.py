# -*- coding: utf-8 -*-
"""
ANALYSE EN CONSTITUANTS (context-free) + SURPRISAL  --  DATASET Broderick ds004408

SEGMENTATION (refs Shriberg & Stolcke) :
  Le decoupage en phrases se fait par la PROSODIE (module segmentation_prosodique.py) :
  pause reelle entre mots + allongement du dernier phone avant la frontiere. C'est la
  methode adaptee a la PAROLE. Les deux tiers du TextGrid
  ("words" et "phones") suffisent.
  -> placez segmentation_prosodique.py dans le MEME dossier que ce script.

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
from segmentation_prosodique import read_words_and_phones, segment_prosodic


# ---------------------------------------------------------------------------
#  LECTURE DES STIMULI  (TextGrid Praat ; repli texte brut)
# ---------------------------------------------------------------------------

def load_sentences(path):
    """Renvoie une liste de phrases, chaque phrase = [(mot, onset, offset)]."""
    words, phones = read_words_and_phones(path)
    if words is not None:
        return segment_prosodic(words, phones)
    # repli : fichier texte sans tier 'words' -> une phrase par ligne tokenisee
    txt = open(path, encoding='utf-8', errors='ignore').read()
    sents = []
    for line in txt.splitlines():
        toks = nltk.word_tokenize(line)
        if toks:
            sents.append([(t, None, None) for t in toks])
    return sents


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


def parse_sentence(viterbi, vocab, best_pos, lex_probs, plain_words):
    proc, info = fix_oov(plain_words, vocab, best_pos)
    try:
        parses = list(viterbi.parse(proc + ['.']))
    except Exception:
        parses = []
    if not parses:
        return None, None
    tree = parses[0]
    leaves = [s for s in tree.subtrees(lambda t: t.height() == 2)]
    rows = []
    for i, st in enumerate(leaves):
        if i >= len(plain_words):
            break
        tree_pos = st.label(); pw = st.leaves()[0]
        ow, opos, is_oov = info[i]
        prob = lex_probs.get((tree_pos, pw.lower()), 1e-10)
        show_pos = opos if is_oov else tree_pos
        rows.append((ow, show_pos, -math.log2(prob), is_oov))
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
        files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                       if re.search(r'\.(txt|textgrid)$', f, re.I)
                       and os.path.basename(f) not in OUTPUTS
                       and not os.path.basename(f).startswith("._"))
    else:                                   # un seul fichier passe en argument
        folder = os.path.dirname(arg) or "."
        files = [arg] if os.path.exists(arg) else []
    if not files:
        print(f"  Aucun fichier .txt/.TextGrid trouve : {os.path.abspath(arg)}"); return

    print(f"  {len(files)} fichier(s) a traiter.")
    print("  [segmentation] prosodique (pause + allongement final).")
    viterbi, vocab, best_pos, lex_probs = load_pcfg()

    out_csv = os.path.join(folder, "surprisal_constituants.csv")
    out_tree = os.path.join(folder, "arbres_constituants.txt")
    n_words = n_sent = n_fail = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv, \
         open(out_tree, "w", encoding="utf-8") as ftree:
        wr = csv.writer(fcsv)
        wr.writerow(["fichier", "phrase_id", "position", "mot",
                     "onset", "offset", "POS", "surprisal_bits", "oov"])
        for path in files:
            name = os.path.basename(path)
            sents = load_sentences(path)
            print(f"  - {name}: {sum(len(s) for s in sents)} mots -> {len(sents)} phrases")
            for sid, sent in enumerate(sents, 1):
                plain = [w for (w, on, off) in sent]
                tree, rows = parse_sentence(viterbi, vocab, best_pos, lex_probs, plain)
                n_sent += 1
                if rows is None:
                    n_fail += 1; continue
                ftree.write(f"### {name} | phrase {sid} : {' '.join(w.lower() for w in plain)}\n")
                ftree.write(tree.pformat(margin=70) + "\n\n")
                for pos_i, (w, pos, surp, is_oov) in enumerate(rows):
                    on, off = sent[pos_i][1], sent[pos_i][2]
                    wr.writerow([name, sid, pos_i + 1, w,
                                 "" if on is None else f"{on:.3f}",
                                 "" if off is None else f"{off:.3f}",
                                 pos, f"{surp:.4f}", int(is_oov)])
                    n_words += 1

    print(f"\n  Termine. {n_words} mots, {n_sent} phrases ({n_fail} non analysees).")
    print(f"  -> {out_csv}\n  -> {out_tree}")


if __name__ == "__main__":
    main()