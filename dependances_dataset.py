# -*- coding: utf-8 -*-
"""
ANALYSE EN DEPENDANCES (arbres etiquetes) -- DATASET Broderick ds004408
  Principal: Stanford CoreNLP ; Repli: NLTK TransitionParser (corrige int32).
  pip install nltk scikit-learn
  python dependances_dataset.py [DOSSIER]
Sortie: dependances.csv  (fichier, phrase, position, mot, onset, offset, relation, tete)
"""
import os, re, sys, csv, glob, pickle, tempfile
import numpy as np
import nltk

MAX_WORDS = 30
PAUSE_GAP = 0.35
CORENLP_URL = "http://localhost:9000"
HERE = os.path.dirname(os.path.abspath(__file__))
TP_MODEL = os.path.join(HERE, "transitionparser_ptb.model")
TP_TRAIN_SENTS = 200
OUTPUTS = {"arbres_constituants.txt", "surprisal_constituants.csv", "dependances.csv"}

for r in ['dependency_treebank','punkt','punkt_tab',
          'averaged_perceptron_tagger','averaged_perceptron_tagger_eng']:
    nltk.download(r, quiet=True)

from nltk.parse.dependencygraph import DependencyGraph


# ---------- lecture TextGrid ----------
def read_textgrid_words(txt):
    items = re.split(r'item\s*\[\d+\]\s*:', txt); block = None
    for it in items:
        m = re.search(r'name\s*=\s*"([^"]*)"', it)
        if m and m.group(1).strip().lower() in ('words','word'): block = it; break
    if block is None: return None
    words = []
    for iv in re.finditer(r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"', block):
        on, off, w = float(iv.group(1)), float(iv.group(2)), iv.group(3).strip()
        if w and w.lower() not in ('sil','sp','<p>','sps','br'):
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
        elif on is not None and i+1 < len(words):
            nxt = words[i+1][1]
            if nxt is not None and nxt - off > gap: boundary = True
        if boundary or len(cur) >= max_words: sents.append(cur); cur = []
    if cur: sents.append(cur)
    return sents


# ---------- TransitionParser corrige (int32 pour scikit-learn recent) ----------
def _i32(M):
    M = M.tocsr()
    M.indices = M.indices.astype(np.int32)
    M.indptr  = M.indptr.astype(np.int32)
    return M


def make_tp32():
    from nltk.parse.transitionparser import TransitionParser, Transition, Configuration
    from sklearn.datasets import load_svmlight_file
    from sklearn import svm
    from operator import itemgetter
    from copy import deepcopy
    from numpy import array
    from scipy import sparse

    class TP32(TransitionParser):
        def train(self, depgraphs, modelfile, verbose=False):
            f = tempfile.NamedTemporaryFile(prefix="tp.train", dir=tempfile.gettempdir(), delete=False)
            try:
                if self._algorithm == self.ARC_STANDARD:
                    self._create_training_examples_arc_std(depgraphs, f)
                else:
                    self._create_training_examples_arc_eager(depgraphs, f)
                f.close()
                x, y = load_svmlight_file(f.name)
                x = _i32(x)
                model = svm.SVC(kernel="poly", degree=2, coef0=0, gamma=0.2, C=0.5,
                                verbose=False, probability=True)
                model.fit(x, y)
                pickle.dump(model, open(modelfile, "wb"))
            finally:
                os.remove(f.name)

        def parse(self, depgraphs, modelFile):
            with open(modelFile, "rb") as fh:
                model = pickle.load(fh)
            op = Transition(self._algorithm)
            result = []
            for dg in depgraphs:
                conf = Configuration(dg)
                while len(conf.buffer) > 0:
                    col=[]; row=[]; data=[]
                    for ft in conf.extract_features():
                        if ft in self._dictionary:
                            col.append(self._dictionary[ft]); row.append(0); data.append(1.0)
                    x = sparse.csr_matrix((array(data),(array(row),array(sorted(col)))),
                                          shape=(1,len(self._dictionary)))
                    x = _i32(x)
                    probs = model.predict_proba(x)[0]
                    order = sorted({i:probs[i] for i in range(len(probs))}.items(),
                                   key=itemgetter(1), reverse=True)
                    for idx,_ in order:
                        yp = model.classes_[idx]
                        if yp in self._match_transition:
                            st = self._match_transition[yp]; base = st.split(":")[0]
                            if base==Transition.LEFT_ARC:
                                if op.left_arc(conf, st.split(":")[1])!=-1: break
                            elif base==Transition.RIGHT_ARC:
                                if op.right_arc(conf, st.split(":")[1])!=-1: break
                            elif base==Transition.REDUCE:
                                if op.reduce(conf)!=-1: break
                            elif base==Transition.SHIFT:
                                if op.shift(conf)!=-1: break
                        else:
                            raise ValueError("transition inconnue")
                nd = deepcopy(dg)
                for k in nd.nodes:
                    nd.nodes[k]["rel"]=""; nd.nodes[k]["head"]=0
                for h,r,c in conf.arcs:
                    nd.nodes[c]["head"]=h; nd.nodes[c]["rel"]=r
                result.append(nd)
            return result
    return TP32


# ---------- parseurs ----------
def connect_corenlp():
    try:
        from nltk.parse.corenlp import CoreNLPDependencyParser
        dep = CoreNLPDependencyParser(url=CORENLP_URL)
        list(dep.raw_parse("This is a test ."))
        return dep
    except Exception:
        return None


def load_transitionparser():
    try:
        import sklearn  # noqa
    except Exception:
        print("  X scikit-learn manquant : pip install scikit-learn"); return None
    from nltk.corpus import dependency_treebank
    TP32 = make_tp32()
    tp = TP32('arc-eager')
    sents = list(dependency_treebank.parsed_sents())[:TP_TRAIN_SENTS]
    print(f"  Entrainement du TransitionParser ({TP_TRAIN_SENTS} phrases, ~1-3 min)...")
    try:
        tp.train(sents, TP_MODEL, verbose=False)
    except Exception as e:
        print("  X", e); return None
    return tp


def words_to_input_depgraph(plain):
    lines = []
    for i, (w, t) in enumerate(nltk.pos_tag(plain), 1):
        lines.append("\t".join([str(i), w, w, t, t, "_", "0", "ROOT", "_", "_"]))
    return DependencyGraph("\n".join(lines))


def parse_dep(plain, corenlp, tp):
    if corenlp is not None:
        try: return next(corenlp.parse(plain))
        except Exception:
            try: return next(corenlp.raw_parse(" ".join(plain)))
            except Exception: return None
    if tp is not None:
        try: return tp.parse([words_to_input_depgraph(plain)], TP_MODEL)[0]
        except Exception: return None
    return None


# ---------- main ----------
def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                   if re.search(r'\.(txt|textgrid)$', f, re.I)
                   and os.path.basename(f) not in OUTPUTS
                   and not os.path.basename(f).startswith("._"))
    if not files:
        print(f"  Aucun .TextGrid dans {os.path.abspath(folder)}"); return
    print(f"  {len(files)} fichier(s).")
    corenlp = connect_corenlp(); tp = None
    if corenlp is not None: print("  [CoreNLP] serveur detecte.")
    else:
        print("  [CoreNLP] non detecte -> repli TransitionParser.")
        tp = load_transitionparser()
        if tp is None: print("  X Aucun parseur."); return
    out_csv = os.path.join(folder, "dependances.csv")
    n_words = n_sent = n_fail = 0
    with open(out_csv,"w",newline="",encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["fichier","phrase_id","position","mot","onset","offset","relation","tete"])
        for path in files:
            name = os.path.basename(path)
            words = read_words(path); sents = segment_sentences(words)
            print(f"  - {name}: {len(words)} mots -> {len(sents)} phrases")
            for sid, sent in enumerate(sents, 1):
                plain = [w for (w, on, off) in sent]
                dg = parse_dep(plain, corenlp, tp); n_sent += 1
                if dg is None: n_fail += 1; continue
                for ni in range(1, len(dg.nodes)):
                    node = dg.nodes[ni]
                    if not node.get('word'): continue
                    hi = node.get('head')
                    head_w = "ROOT" if hi in (0, None) else dg.nodes[hi].get('word','?')
                    pos_i = ni - 1
                    on = sent[pos_i][1] if pos_i < len(sent) else None
                    off = sent[pos_i][2] if pos_i < len(sent) else None
                    wr.writerow([name, sid, ni, node['word'],
                                 "" if on is None else f"{on:.3f}",
                                 "" if off is None else f"{off:.3f}",
                                 node.get('rel','_'), head_w])
                    n_words += 1
    print(f"\n  Termine. {n_words} mots, {n_sent} phrases ({n_fail} non analysees).")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()