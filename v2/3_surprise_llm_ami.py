# -*- coding: utf-8 -*-
"""
Etape 3 (AMI) : surprise GPT-2 (par defaut), en nats, avec le contexte de la reunion.

    surprise_llm(mot) = -ln P(mot | tout ce qui a ete dit avant dans la reunion)

Dans la version du rapport (surprise_llm_ami.py), GPT-2 repartait de zero a chaque
enonce et ne voyait jamais ce qui avait ete dit avant. Ici je remets les enonces de
tous les locuteurs d'une reunion dans l'ordre chronologique (debut du premier mot),
un par ligne, et GPT-2 lit la reunion en continu (fenetre glissante, voir
commun.surprises_tokens).

Mise en forme du texte donne au modele (le transcript AMI est en minuscules et sans
ponctuation apres lecture) :
  - "i", "i'm", ... deviennent "I", "I'm" ; les sigles "l_c_d_" deviennent "LCD" ;
  - majuscule au premier mot de chaque enonce ;
  - point a la fin de l'enonce quand il finissait par une ponctuation dans le
    transcript (c'est-a-dire tous sauf les morceaux de 40 mots de ami_dataset.py).
La surprise d'un mot est la somme des surprises de ses tokens ; la ponctuation
ajoutee n'est comptee pour aucun mot.

Options (pour DialoGPT, voir 10_dialogpt.py) :
  --tours      DialoGPT a ete entraine sur des conversations Reddit ou chaque tour de
               parole se termine par le jeton de fin de texte (<|endoftext|>). Avec
               --tours, un changement de locuteur est marque par ce jeton, et les
               enonces qui se suivent chez le meme locuteur sont mis bout a bout.
               Le jeton n'est compte pour aucun mot.
  --fenetre N  taille de la fenetre glissante en tokens (1024 par defaut). DialoGPT
               a ete entraine sur des dialogues courts et devient beaucoup moins bon
               avec un long contexte.
  --enonce     chaque enonce est lu seul, sans ce qui precede (comme dans la version
               du rapport).
  --sans-tiret les mots composes sont ecrits avec une espace ("flip top", "mm hmm").
               Les textes d'entrainement de DialoGPT n'ont presque pas de tirets :
               avec le tiret, DialoGPT-small donne 56 a 227 nats a ces mots.

Entree : mots_ami.csv (etape 1) ou la sortie de l'etape 2.
Colonne ajoutee : surprise_llm (nats).

Utilisation :
    pip install torch transformers
    python 3_surprise_llm_ami.py mots_ami_syn.csv sortie.csv [gpt2] [--tours] [--fenetre 1024] [--enonce] [--sans-tiret]
"""
import argparse, csv, os, sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commun import charger_modele, forme_ecrite, surprises_tokens

MORCEAU = 40      # taille des morceaux coupes par ami_dataset.read_ami_words


def forme_sans_tiret(mot):
    """'flip-top' -> 'flip top', 'anti-r_s_i_' -> 'anti RSI', "t_v_'s" -> "TV's"."""
    parties = []
    for x in mot.split("-"):
        base, apos, fin = x.partition("'")
        parties.append(forme_ecrite(base) + apos + fin)
    return " ".join(x for x in parties if x) or mot


def texte_reunion(enonces, fin_tour=None, forme=forme_ecrite):
    """enonces : liste de (indices de lignes, mots, locuteur), dans l'ordre chronologique.
    Sans fin_tour, un enonce par ligne. Avec fin_tour (jeton de fin de texte), ce jeton
    separe deux locuteurs differents et les enonces d'un meme locuteur sont mis a la suite.
    Renvoie (texte, spans), avec spans[indice de ligne] = (debut, fin) du mot dans le texte."""
    morceaux, spans, pos, avant = [], {}, 0, None
    for idx, mots, loc in enonces:
        if morceaux:
            sep = "\n" if fin_tour is None else (fin_tour if loc != avant else " ")
            morceaux.append(sep); pos += len(sep)
        avant = loc
        for j, (i, m) in enumerate(zip(idx, mots)):
            w = forme(m)
            if j == 0:
                w = w[:1].upper() + w[1:]
            else:
                morceaux.append(" "); pos += 1
            morceaux.append(w); spans[i] = (pos, pos + len(w)); pos += len(w)
        if len(mots) != MORCEAU:
            morceaux.append("."); pos += 1
    return "".join(morceaux), spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entree"); ap.add_argument("sortie")
    ap.add_argument("modele", nargs="?", default="gpt2")
    ap.add_argument("--tours", action="store_true")
    ap.add_argument("--fenetre", type=int, default=1024)
    ap.add_argument("--enonce", action="store_true")
    ap.add_argument("--sans-tiret", action="store_true")
    args = ap.parse_args()
    entree, sortie, nom_modele, tours = args.entree, args.sortie, args.modele, args.tours
    with open(entree, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f); cols = rd.fieldnames; rows = list(rd)
    cols = [c for c in cols if c not in ("surprise_llm",)]   # si on relance, la colonne est remplacee
    reunions = OrderedDict()
    for i, r in enumerate(rows):
        reunions.setdefault(r["meeting"], OrderedDict()).setdefault(
            (r["locuteur"], r["phrase_id"]), []).append(i)
    tok, model, dev = charger_modele(nom_modele)
    fin_tour = tok.eos_token if tours else None
    forme = forme_sans_tiret if args.sans_tiret else forme_ecrite
    val = [None] * len(rows)
    for k, (meeting, enon) in enumerate(reunions.items(), 1):
        ordre = sorted(enon.values(), key=lambda idx: float(rows[idx[0]]["onset"]))
        liste = [(idx, [rows[i]["mot"] for i in idx], rows[idx[0]]["locuteur"]) for idx in ordre]
        # toute la reunion d'un coup, ou un enonce a la fois avec --enonce
        blocs = [[e] for e in liste] if args.enonce else [liste]
        ntok = 0
        for bloc in blocs:
            texte, spans = texte_reunion(bloc, fin_tour, forme)
            surp, offs = surprises_tokens(texte, tok, model, dev, verbeux=False, fenetre=args.fenetre)
            par_char = {c: i for i, (a, b) in spans.items() for c in range(a, b)}
            for (a, b), sv in zip(offs, surp):
                if b > a and (b - 1) in par_char:
                    i = par_char[b - 1]
                    val[i] = sv if val[i] is None else val[i] + sv
            ntok += len(surp)
        print(f"  [{k}/{len(reunions)}] {meeting} : {len(enon)} enonces, {ntok} tokens", flush=True)
    manque = sum(v is None for v in val)
    if manque > 0.01 * len(rows):
        sys.exit(f"  ERREUR : {manque} mots sans token attribue ; verifier le tokenizer.")
    with open(sortie, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(cols + ["surprise_llm"])
        for r, v in zip(rows, val):
            wr.writerow([r[c] for c in cols] + ["" if v is None else f"{v:.4f}"])
    print(f"  {len(rows) - manque}/{len(rows)} mots -> {sortie}")


if __name__ == "__main__":
    main()
