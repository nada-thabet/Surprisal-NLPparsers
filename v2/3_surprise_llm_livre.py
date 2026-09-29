# -*- coding: utf-8 -*-
"""
Etape 3 (livre) : surprise GPT-2 (par defaut), en nats, sur le vrai texte du livre.

    surprise_llm(mot) = -ln P(mot | tout le texte qui precede)
                      = somme des -ln P(token | contexte) sur les tokens du mot

Differences avec la version du rapport (surprise_llm_textgrid.py), qui :
  1. donnait a GPT-2 les mots des TextGrid tels quels, donc en majuscules
     ("HE WAS AN OLD MAN WHO FISHED ALONE"), ce qui est tres improbable pour
     GPT-2 et gonfle les surprises ;
  2. repartait de zero a chaque phrase prosodique, sans le texte d'avant ;
  3. donnait un texte sans ponctuation ("the boy s parents").
Ici chaque mot des TextGrid est aligne sur le texte du livre (oldmansea.pdf, avec
la casse, la ponctuation et les apostrophes) et GPT-2 lit le livre en continu avec
une fenetre glissante (1024 tokens pour GPT-2, chaque token a au moins 512 tokens
de contexte). Un token de ponctuation seul n'est compte pour aucun mot ; un token
va au mot qui contient son dernier caractere.

Entree : mots_livre.csv (etape 1) ou la sortie de l'etape 2.
Colonnes ajoutees : surprise_llm (nats), aligne_texte (1 si le mot a ete retrouve
dans le texte du livre ; sinon la surprise est vide).

Utilisation :
    pip install torch transformers pymupdf
    python 3_surprise_llm_livre.py mots_livre_syn.csv ../stimuli_livre/oldmansea.pdf sortie.csv [gpt2]
Le 4e argument est le nom du modele Hugging Face (gpt2, gpt2-xl, ...) ou un dossier local.
"""
import csv, difflib, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commun import charger_modele, surprises_tokens


# ---------------------------------------------------------------------------
#  Texte du livre
# ---------------------------------------------------------------------------

def texte_du_livre(pdf):
    import fitz  # pymupdf
    lignes = []
    for page in fitz.open(pdf):
        for l in page.get_text().split("\n"):
            s = l.strip()
            if re.fullmatch(r"-\s*\d+\s*-", s):                          # "- 12 -"
                continue
            if s.startswith("The Old Man and the Sea") and "Asiaing" in s:  # en-tete
                continue
            lignes.append(l)
    t = "\n".join(lignes)
    t = re.sub(r"\[\d+\]", " ", t)                    # reperes de page "[9]"
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"-\n", "-", t)                        # tiret en fin de ligne
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    # le recit commence a "He was an old man"
    return t[t.index("He was an old man"):].strip()


def mots_du_texte(t):
    """Unites alphanumeriques du texte avec leur position ("boy's" donne boy et s)."""
    return [(m.group().lower(), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9]+", t)]


def aligner(mots_tg, texte):
    """Pour chaque mot des TextGrid, sa position (debut, fin) dans le texte, ou None."""
    unites = mots_du_texte(texte)
    # un mot du TextGrid avec apostrophe ("didn't") donne plusieurs unites
    tg_unites, proprio = [], []
    for k, w in enumerate(mots_tg):
        for u in re.findall(r"[a-z0-9]+", w.lower()):
            tg_unites.append(u); proprio.append(k)
    sm = difflib.SequenceMatcher(None, tg_unites, [u[0] for u in unites], autojunk=False)
    spans = [None] * len(mots_tg)
    for a, b, n in sm.get_matching_blocks():
        for d in range(n):
            k = proprio[a + d]; s0, s1 = unites[b + d][1], unites[b + d][2]
            spans[k] = (s0, s1) if spans[k] is None else (min(spans[k][0], s0), max(spans[k][1], s1))
    return spans


def main():
    entree, pdf, sortie = sys.argv[1], sys.argv[2], sys.argv[3]
    nom_modele = sys.argv[4] if len(sys.argv) > 4 else "gpt2"
    with open(entree, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f); cols = rd.fieldnames; rows = list(rd)
    cols = [c for c in cols if c not in ("surprise_llm", "aligne_texte")]   # si on relance, colonnes remplacees
    texte = texte_du_livre(pdf)
    spans = aligner([r["mot"] for r in rows], texte)
    n_ok = sum(s is not None for s in spans)
    print(f"  {n_ok}/{len(rows)} mots des TextGrid retrouves dans le texte du livre")
    # le modele lit le texte du debut du recit jusqu'au dernier mot des audios
    fin_txt = max(s[1] for s in spans if s)
    texte = texte[:fin_txt]
    surp, offs = surprises_tokens(texte, *charger_modele(nom_modele))
    # chaque token va au mot qui contient son dernier caractere
    par_char = {}
    for k, s in enumerate(spans):
        if s:
            for c in range(s[0], s[1]):
                par_char[c] = k
    val = [None] * len(spans)
    for (a, b), sv in zip(offs, surp):
        if b > a and (b - 1) in par_char:
            k = par_char[b - 1]
            val[k] = sv if val[k] is None else val[k] + sv
    n_val = sum(v is not None for v in val)
    print(f"  {n_val} mots ont une surprise")
    if n_val < 0.95 * len(rows):
        sys.exit("  ERREUR : trop de mots sans token attribue ; verifier le tokenizer.")
    with open(sortie, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(cols + ["surprise_llm", "aligne_texte"])
        for r, v in zip(rows, val):
            wr.writerow([r[c] for c in cols] + ["" if v is None else f"{v:.4f}", int(v is not None)])
    print(f"  -> {sortie}")


if __name__ == "__main__":
    main()
