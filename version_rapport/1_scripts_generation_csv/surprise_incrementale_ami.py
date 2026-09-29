# -*- coding: utf-8 -*-
"""
Surprise syntaxique incrementale (Hale 2001) sur AMI (parole spontanee).

Meme mesure que surprise_incrementale.py :
    surprise(mot) = -log2 [ P(w_1..w_k) / P(w_1..w_{k-1}) ]
avec les probabilites de prefixe de la PCFG (Jelinek-Lafferty), mais sur les
fichiers *.words.xml d'AMI. Ici pas de segmentation prosodique : les phrases
viennent de la ponctuation du transcript (lecture dans ami_dataset.py), et les
phrases de plus de 40 mots sont coupees en morceaux de 40.

A mettre dans le meme dossier que surprise_incrementale.py, ami_dataset.py,
constituants_dataset.py et segmentation_prosodique.py.

    pip install nltk scipy numpy wordfreq
    python surprise_incrementale_ami.py <dossier contenant les .words.xml>

Sur tout AMI (171 reunions) le calcul est long : pour tester, mieux vaut
commencer par un dossier avec quelques fichiers .words.xml.

Sortie : surprisal_incrementale_ami.csv
    meeting, locuteur, phrase_id, position, mot, onset, offset, POS,
    surprise_incrementale, freq_zipf, n_letters
"""

import os, sys, csv, glob

from surprise_incrementale import PrefixParser, induce_grammar


def main():
    from ami_dataset import read_ami_words, n_letters, SENT_MAX
    from constituants_dataset import fix_oov, word_freq

    folder = sys.argv[1] if len(sys.argv) > 1 else "words"
    files = sorted(glob.glob(os.path.join(folder, "*.words.xml")))
    if not files:
        print(f"  Aucun *.words.xml dans : {os.path.abspath(folder)}"); return

    print(f"  {len(files)} fichiers AMI. Induction de la PCFG (CNF, horzMarkov=0)...")
    bin_r, un_r, lex_r, vocab, best_pos, start = induce_grammar(horzMarkov=0)
    print(f"  Grammaire : {len(bin_r)} bin, {len(un_r)} un, {len(lex_r)} lex. "
          "Construction des clotures...")
    parser = PrefixParser(bin_r, un_r, lex_r, start)
    print(f"  {parser.n} non-terminaux. Pret.")

    out_csv = os.path.join(folder, "surprisal_incrementale_ami.csv")
    n_words = n_sent = n_skip = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["meeting", "locuteur", "phrase_id", "position", "mot",
                     "onset", "offset", "POS", "surprise_incrementale",
                     "freq_zipf", "n_letters"])
        for path in files:
            base = os.path.basename(path).replace(".words.xml", "")   # ex: ES2008b.B
            meeting, _, spk = base.partition(".")
            sents = read_ami_words(path)
            print(f"  - {base}: {len(sents)} phrases")
            for sid, sent in enumerate(sents, 1):
                plain = [w for (w, on, off) in sent]
                if len(plain) > SENT_MAX:
                    n_skip += 1; continue
                proc, info = fix_oov(plain, vocab, best_pos)
                try:
                    surps = parser.surprisals(proc)
                except Exception as ex:
                    print(f"      (phrase {sid} ignoree : {ex})"); n_skip += 1; continue
                n_sent += 1
                for i, (w, on, off) in enumerate(sent):
                    s = surps[i] if i < len(surps) else float("nan")
                    pos = info[i][1] if i < len(info) else ""
                    wr.writerow([meeting, spk, sid, i + 1, w.lower(),
                                 f"{on:.3f}", f"{off:.3f}", pos,
                                 "" if s != s else f"{s:.4f}",
                                 word_freq(w), n_letters(w)])
                    n_words += 1
            fcsv.flush()          # on garde les resultats partiels si ca plante
    print(f"\n  Termine. {n_words} mots, {n_sent} phrases ({n_skip} ignorees).")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
