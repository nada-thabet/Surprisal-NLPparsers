# -*- coding: utf-8 -*-
"""
Analyse en constituants + surprise sur AMI (parole spontanee, plusieurs locuteurs).

Premiere version du test sur AMI (analyse en constituants avec Viterbi). Elle a ete
remplacee par surprise_incrementale_ami.py, mais les deux scripts AMI finaux utilisent
toujours read_ami_words() de ce fichier.

Entree : les annotations NXT d'AMI, dossier words/ (fichiers *.words.xml).
Chaque <w starttime=".." endtime="..">mot</w> est un mot aligne. La ponctuation du
transcript ( <w punc="true">.</w> ) donne les fins de phrase, donc pour AMI je n'utilise
pas la segmentation prosodique (et pas besoin de l'audio).

La longueur du mot (n_phones pour ds004408) est remplacee par le nombre de lettres.

  pip install nltk wordfreq
  python ami_dataset.py  chemin/vers/words

Sortie : surprisal_ami.csv
  (meeting, locuteur, phrase_id, position, mot, onset, offset, POS,
   surprisal_bits, surprise_structurale, oov, freq_zipf, n_letters)
"""

import os, re, sys, csv, glob
import xml.etree.ElementTree as ET

# PCFG, modele de categories et analyse en constituants de constituants_dataset.py
from constituants_dataset import (load_pcfg, load_pos_lm, parse_sentence, word_freq)

SENT_MAX = 40          # longueur max d'une phrase (le parsing est en O(n^3))
END_PUNC = {'.', '?', '!'}


def read_ami_words(path):
    """Lit un *.words.xml et renvoie la liste des phrases, chaque phrase = [(mot, onset, offset)].
    Les fins de phrase viennent de la ponctuation du transcript."""
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        print(f"  [!] lecture impossible {os.path.basename(path)} : {e}"); return []
    words, bound = [], []                       # mots, et fin de phrase apres le mot ou non
    for el in root.iter():
        tag = el.tag.split('}')[-1]             # on enleve le namespace nite:
        if tag != 'w':
            continue
        txt = (el.text or '').strip()
        st, en = el.get('starttime'), el.get('endtime')
        is_punc = (el.get('punc') == 'true') or (st is None) or (en is None)
        if is_punc:                             # la ponctuation marque une fin de phrase
            if txt in END_PUNC and bound:
                bound[-1] = True
            continue
        if not txt:
            continue
        words.append((txt.lower(), float(st), float(en)))
        bound.append(False)
    # decoupage en phrases : ponctuation, et coupure a SENT_MAX mots
    sents, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        if bound[i] or len(cur) >= SENT_MAX:
            sents.append(cur); cur = []
    if cur:
        sents.append(cur)
    return sents


def n_letters(w):
    return len(re.sub(r'[^a-z0-9]', '', w.lower()))


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "words"
    files = sorted(glob.glob(os.path.join(folder, "*.words.xml")))
    if not files:
        print(f"  Aucun *.words.xml dans : {os.path.abspath(folder)}")
        print("  Indique le dossier 'words' des annotations AMI."); return
    print(f"  {len(files)} fichiers AMI (locuteur x reunion).")
    viterbi, vocab, best_pos, lex_probs = load_pcfg()
    print("  Modele de categories (surprise structurale)...")
    pos_lm = load_pos_lm()

    out_csv = os.path.join(folder, "surprisal_ami.csv")
    n_words = n_sent = n_fail = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["meeting", "locuteur", "phrase_id", "position", "mot",
                     "onset", "offset", "POS", "surprisal_bits", "surprise_structurale",
                     "oov", "freq_zipf", "n_letters"])
        for path in files:
            base = os.path.basename(path).replace(".words.xml", "")   # ex: EN2001a.A
            meeting, _, spk = base.partition(".")
            sents = read_ami_words(path)
            print(f"  - {base}: {sum(len(s) for s in sents)} mots -> {len(sents)} phrases")
            for sid, sent in enumerate(sents, 1):
                plain = [w for (w, on, off) in sent]
                tree, rows = parse_sentence(viterbi, vocab, best_pos, lex_probs, pos_lm, plain)
                n_sent += 1
                if rows is None:
                    n_fail += 1; continue
                for pos_i, (w, pos, surp, struct, is_oov) in enumerate(rows):
                    on, off = sent[pos_i][1], sent[pos_i][2]
                    wr.writerow([meeting, spk, sid, pos_i + 1, w,
                                 f"{on:.3f}", f"{off:.3f}", pos,
                                 f"{surp:.4f}", f"{struct:.4f}", int(is_oov),
                                 word_freq(w), n_letters(w)])
                    n_words += 1

    print(f"\n  Termine. {n_words} mots, {n_sent} phrases ({n_fail} non analysees).")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
