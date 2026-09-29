# -*- coding: utf-8 -*-
"""
Etape 1 (livre) : table des mots de The Old Man and the Sea, une ligne par mot.

Sortie : mots_livre.csv
    fichier, phrase_id, position, mot, onset, offset, freq_zipf, n_phones

Deux modes :
  python 1_mots_livre.py textgrid <dossier des audioNN.TextGrid (+ .wav)>
      refait la segmentation prosodique (segmentation_prosodique.py de version_rapport).
      Sans les .wav il n'y a pas de pitch, et environ 3 frontieres sur 800 changent.
  python 1_mots_livre.py csv ../version_rapport/2_csv_du_rapport/surprisal_llm_textgrid.csv
      reprend exactement les phrases du rapport (segmentation faite avec les .wav,
      la meme que stimuli_livre/phrases_prosodiques.txt). C'est ce mode que j'ai utilise.

Aucune phrase n'est enlevee (dans la version du rapport, la surprise syntaxique
sautait les 16 phrases de plus de 40 mots, soit 741 mots).
"""
import csv, glob, os, re, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", "version_rapport", "1_scripts_generation_csv"))
COLS = ["fichier", "phrase_id", "position", "mot", "onset", "offset", "freq_zipf", "n_phones"]


def depuis_textgrid(dossier):
    from constituants_dataset import load_sentences, count_phones, word_freq
    files = sorted(f for f in glob.glob(os.path.join(dossier, "*"))
                   if re.search(r"\.textgrid$", f, re.I) and not os.path.basename(f).startswith("._"))
    for path in files:
        name = os.path.basename(path)
        sents, phones = load_sentences(path)
        print(f"  - {name}: {len(sents)} phrases")
        for sid, sent in enumerate(sents, 1):
            for i, (w, on, off) in enumerate(sent, 1):
                yield [name, sid, i, w.lower(), f"{on:.3f}", f"{off:.3f}",
                       word_freq(w), count_phones(phones, on, off)]


def depuis_csv(chemin):
    with open(chemin, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            yield [r[c] for c in COLS]


def main():
    mode, source = sys.argv[1], sys.argv[2]
    rows = depuis_textgrid(source) if mode == "textgrid" else depuis_csv(source)
    out = sys.argv[3] if len(sys.argv) > 3 else "mots_livre.csv"
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(COLS)
        for r in rows:
            wr.writerow(r); n += 1
    print(f"  {n} mots -> {out}")


if __name__ == "__main__":
    main()
