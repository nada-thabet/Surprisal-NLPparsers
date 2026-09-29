# -*- coding: utf-8 -*-
"""
Etape 1 (AMI) : table des mots des reunions AMI, une ligne par mot.

Sortie : mots_ami.csv
    meeting, locuteur, phrase_id, position, mot, onset, offset, freq_zipf, n_letters

  python 1_mots_ami.py xml <dossier des *.words.xml>
      meme lecture que la version du rapport (ami_dataset.py) : phrases decoupees
      par la ponctuation du transcript, et coupees en morceaux de 40 mots au plus.
  python 1_mots_ami.py csv ../version_rapport/2_csv_du_rapport/surprisal_llm_ami.csv
      reprend exactement les mots et les phrases du rapport (12 reunions ES2002-ES2004).

Changement : la frequence Zipf des sigles ("l_c_d_", "t_v_") valait 0 dans la version
du rapport (341 occurrences) ; elle est maintenant calculee sur la forme ecrite ("lcd", "tv").
"""
import csv, glob, os, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, "..", "version_rapport", "1_scripts_generation_csv"))
from commun import freq_zipf

COLS = ["meeting", "locuteur", "phrase_id", "position", "mot", "onset", "offset",
        "freq_zipf", "n_letters"]


def depuis_xml(dossier):
    from ami_dataset import read_ami_words, n_letters
    for path in sorted(glob.glob(os.path.join(dossier, "*.words.xml"))):
        meeting, _, spk = os.path.basename(path).replace(".words.xml", "").partition(".")
        for sid, sent in enumerate(read_ami_words(path), 1):
            for i, (w, on, off) in enumerate(sent, 1):
                yield [meeting, spk, sid, i, w.lower(), f"{on:.3f}", f"{off:.3f}",
                       freq_zipf(w.lower()), n_letters(w)]


def depuis_csv(chemin):
    with open(chemin, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            yield [r["meeting"], r["locuteur"], r["phrase_id"], r["position"], r["mot"],
                   r["onset"], r["offset"], freq_zipf(r["mot"]), r["n_letters"]]


def main():
    mode, source = sys.argv[1], sys.argv[2]
    rows = depuis_xml(source) if mode == "xml" else depuis_csv(source)
    out = sys.argv[3] if len(sys.argv) > 3 else "mots_ami.csv"
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(COLS)
        for r in rows:
            wr.writerow(r); n += 1
    print(f"  {n} mots -> {out}")


if __name__ == "__main__":
    main()
