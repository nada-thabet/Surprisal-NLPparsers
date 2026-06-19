
"""
ANALYSE EN CONSTITUANTS (context-free) + SURPRISAL -- DATASET Broderick ds004408
CORRECTIONS :
  - mots des TextGrids en MAJUSCULES -> etaient tous tagues NNP et remplaces par
    "Mr.". On met desormais en MINUSCULES avant l'etiquetage : POS correct, moins d'OOV.
  - arbres de-binarises (un_chomsky_normal_form) -> lisibles, sans 'S|<...>'.
  - vrai mot + POS predit affiches pour les mots hors-vocabulaire (colonne 'oov').

"""
import os, re, sys, csv, glob, math, nltk

MAX_WORDS = 15
PAUSE_GAP = 0.35
OUTPUTS = {"arbres_constituants.txt", "surprisal_constituants.csv",
           "dependances.csv", "comparaison.csv"}

for r in ['treebank', 'punkt', 'punkt_tab',
          'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(r, quiet=True)

from nltk.corpus import treebank
from nltk import induce_pcfg, Nonterminal
from nltk.parse import ViterbiParser


def read_textgrid_words(txt):
    items = re.split(r'item\s*\[\d+\]\s*:', txt); block = None
    for it in items:
        m = re.search(r'name\s*=\s*"([^"]*)"', it)
        if m and m.group(1).strip().lower() in ('words', 'word'): block = it; break
    if block is None: return None
    words = []
    for iv in re.finditer(r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"', block):
        on, off, w = float(iv.group(1)), float(iv.group(2)), iv.group(3).strip()
        if w and w.lower() not in ('sil', 'sp', '<p>', 'sps', 'br'):
            words.append((w, on, off))
    return words


def read_words(path):
    txt = open(path, encoding='utf-8', errors='ignore').read()
    if 'ooTextFile' in txt or 'IntervalTier' in txt or 'item [' in txt:
        w = read_textgrid_words(txt)
        if w: return w
    return [(tok, None, None) for tok in nltk.word_tokenize(txt)]


def segment_sentences(words, gap=PAUSE_GAP, max_words=MAX_WORDS):
    sents, cur = [], []
    for i, (w, on, off) in enumerate(words):
        cur.append((w, on, off)); boundary = False
        if re.search(r'[.!?]$', w): boundary = True
        elif on is not None and i + 1 < len(words):
            nxt = words[i + 1][1]
            if nxt is not None and nxt - off > gap: boundary = True
        if boundary or len(cur) >= max_words: sents.append(cur); cur = []
    if cur: sents.append(cur)
    return sents


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
    # CORRECTION : etiqueter en MINUSCULES (sinon tout est NNP -> 'Mr.')
    low = [w.lower() for w in words]
    tagged = nltk.pos_tag(low)
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
        if i >= len(plain_words): break
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


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                   if re.search(r'\.(txt|textgrid)$', f, re.I)
                   and os.path.basename(f) not in OUTPUTS
                   and not os.path.basename(f).startswith("._"))
    if not files:
        print(f"  Aucun fichier .txt/.TextGrid dans : {os.path.abspath(folder)}"); return
    print(f"  {len(files)} fichier(s) a traiter.")
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
            words = read_words(path); sents = segment_sentences(words)
            print(f"  - {name}: {len(words)} mots -> {len(sents)} phrases")
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