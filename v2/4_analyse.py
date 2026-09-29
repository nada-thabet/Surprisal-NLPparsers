# -*- coding: utf-8 -*-
"""
Etape 4 : est-ce que la surprise predit la duree du mot, au-dela de la frequence et
de la longueur ? Memes modeles que dans le rapport (resultats_comparaison.py), sur
les CSV v2 (surprises en nats).

  parole lue : duree ~ n_phones * freq_zipf + contenu + surprise          (MCO)
  AMI        : duree ~ n_letters * freq_zipf + contenu + debit + surprise
               MCO, puis modele mixte avec intercept aleatoire par locuteur,
               ajuste en ML avec l'optimiseur powell (lbfgs ne convergeait pas
               sur ces donnees)
Nettoyage : 0 < duree <= 2 s ; pour AMI, sans les mots d'hesitation et avec des
enonces d'au moins 4 mots.

Les coefficients sont en secondes par nat (multiplier par ln 2 pour avoir des
secondes par bit comme dans le rapport). Les correlations et les p-valeurs ne
dependent pas de la base du log.

    pip install pandas numpy statsmodels
    python 4_analyse.py surprise_livre_v2.csv surprise_ami_v2.csv
"""
import sys, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
HESIT = {"uh", "um", "mm", "hmm", "er", "erm", "ah", "oh", "huh", "eh", "mmm", "hm", "mm-hmm",
         "uh-huh", "yeah", "yep", "okay", "mmhmm"}


def preparer(path):
    d = pd.read_csv(path)
    ami = "meeting" in d.columns
    d["duree"] = d["offset"] - d["onset"]
    d["contenu"] = d["POS"].astype(str).str.upper().str[:2].isin(["NN", "VB", "JJ", "RB"]).astype(int)
    if ami:
        d["w"] = d["mot"].astype(str).str.lower()
        u = d.groupby(["meeting", "locuteur", "phrase_id"]).agg(
            n=("w", "size"), t0=("onset", "min"), t1=("offset", "max")).reset_index()
        u["debit"] = u["n"] / (u["t1"] - u["t0"]).clip(lower=1e-3)
        d = d.merge(u[["meeting", "locuteur", "phrase_id", "n", "debit"]],
                    on=["meeting", "locuteur", "phrase_id"], how="left")
        d = d[(d["n"] >= 4) & (~d["w"].isin(HESIT))]
        d["spk"] = d["meeting"].astype(str) + "." + d["locuteur"].astype(str)
    d = d[(d["duree"] > 0) & (d["duree"] <= 2)]
    return d.copy(), ami


def fmt(m, nom="s"):
    c, p = m.params[nom], m.pvalues[nom]
    lo, hi = m.conf_int().loc[nom]
    return f"{c:+.5f} s/nat (p = {p:.2g}, IC95 [{lo:+.5f} ; {hi:+.5f}])"


def analyser(d, ami, col, titre):
    L = "n_letters" if ami else "n_phones"
    d = d.dropna(subset=["duree", col, "freq_zipf", L] + (["debit"] if ami else [])).copy()
    if d.empty:
        print(f"== {titre} : colonne {col} vide, rien a analyser\n"); return
    d["s"] = d[col]
    ctrl = f"{L} * freq_zipf + contenu" + (" + debit" if ami else "")
    rds, rdf, rsf = (np.corrcoef(d[a], d[b])[0, 1] for a, b in
                     (("duree", "s"), ("duree", "freq_zipf"), ("s", "freq_zipf")))
    print(f"== {titre}  (n = {len(d)} mots" +
          (f", {d['meeting'].nunique()} reunions, {d['spk'].nunique()} locuteurs)" if ami else ")"))
    print(f"   r(surprise, duree) = {rds:+.3f}   r(surprise, freq) = {rsf:+.3f}   "
          f"EXTRA = {rds - rsf * rdf:+.3f}")
    print(f"   MCO           : {fmt(smf.ols(f'duree ~ {ctrl} + s', d).fit())}")
    if ami:
        mm = smf.mixedlm(f"duree ~ {ctrl} + s", d, groups=d["spk"]).fit(reml=False, method="powell")
        print(f"   mixte (1|loc) : {fmt(mm)}   converge = {mm.converged}")
    if "hors_vocab" in d:
        dv = d[d["hors_vocab"] == 0]
        print(f"   MCO, mots du vocabulaire de la grammaire seulement (n = {len(dv)}) : "
              f"{fmt(smf.ols(f'duree ~ {ctrl} + s', dv).fit())}")
    print()


def main():
    for path in sys.argv[1:]:
        d, ami = preparer(path)
        corpus = "AMI" if ami else "PAROLE LUE"
        for col, nom in (("surprise_syn", "syntaxique (PCFG)"), ("surprise_llm", "LLM")):
            if col in d.columns:
                analyser(d, ami, col, f"{corpus} - {nom}")


if __name__ == "__main__":
    main()
