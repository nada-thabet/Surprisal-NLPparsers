# -*- coding: utf-8 -*-
"""
Etape 5 : tous les chiffres de la comparaison syntaxique / GPT-2 sur les CSV v2,
comme resultats_comparaison.py pour la version du rapport.

Pour chaque cas (syntaxique ou GPT-2, parole lue ou AMI) :
  - le nombre de mots ;
  - les correlations brutes surprise/duree, frequence/duree, surprise/frequence (r2,
    et 1 - r2 = la part de la surprise que la frequence n'explique pas) ;
  - le coefficient de la surprise sans puis avec controle de la frequence, et
    controle par controle (frequence seule, + longueur, modele complet) ;
  - pour AMI : le modele mixte avec intercept aleatoire par locuteur (powell), avec
    et sans la frequence ;
  - le modele complet sur les seuls mots connus de la grammaire (hors_vocab = 0) ;
  - l'EXTRA (observe moins ce qu'on attendrait si la surprise n'agissait que via la
    frequence) et la correlation partielle.

Il y a un cinquieme cas, la surprise syntaxique sur les 171 reunions d'AMI (la
ligne "AMI complet" du rapport ; GPT-2 n'a ete calcule que sur les 12 reunions),
puis la comparaison 171 reunions / 12 reunions appariees de la section 4.3.

Les surprises sont en nats, donc les coefficients en secondes par nat
(multiplier par ln 2 = 0,693 pour avoir des secondes par bit).

    pip install pandas numpy statsmodels
    python 5_resultats_complets.py donnees/surprise_livre_v2.csv donnees/surprise_ami_v2.csv donnees/surprise_syn_ami_171reunions_v2.csv.gz
Sortie : resultats_complets_v2.txt (et le meme texte dans le terminal)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ICI = os.path.dirname(os.path.abspath(__file__))
HESIT = {"uh", "um", "mm", "hmm", "er", "erm", "ah", "oh", "huh", "eh", "mmm", "hm", "mm-hmm",
         "uh-huh", "yeah", "yep", "okay", "mmhmm"}


def prep(path, col):
    """Charge un CSV v2 et garde la colonne de surprise `col` dans s.
    Meme nettoyage que le rapport : 0 < duree <= 2 s ; pour AMI, sans hesitations
    et avec des enonces d'au moins 4 mots."""
    d = pd.read_csv(path)
    ami = "meeting" in d.columns
    L = "n_letters" if ami else "n_phones"
    d["duree"] = d["offset"] - d["onset"]
    d["content"] = d["POS"].astype(str).str.upper().str[:2].isin(["NN", "VB", "JJ", "RB"]).astype(int)
    if ami:
        d["w"] = d["mot"].astype(str).str.lower()
        u = d.groupby(["meeting", "locuteur", "phrase_id"]).agg(
            n=("w", "size"), t0=("onset", "min"), t1=("offset", "max")).reset_index()
        u["rate"] = u["n"] / (u["t1"] - u["t0"]).clip(lower=1e-3)
        d = d.merge(u[["meeting", "locuteur", "phrase_id", "n", "rate"]],
                    on=["meeting", "locuteur", "phrase_id"], how="left")
        d = d[(d["n"] >= 4) & (~d["w"].isin(HESIT))]
        d["spk"] = d["meeting"].astype(str) + "." + d["locuteur"].astype(str)
    d = d[(d["duree"] > 0) & (d["duree"] <= 2)]
    d = d.dropna(subset=["duree", col, "freq_zipf", L] + (["rate"] if ami else [])).copy()
    d["s"] = d[col]
    return d, ami, L


def r(a, b):
    return np.corrcoef(a, b)[0, 1]


def fmt(m, nom="s"):
    c, p = m.params[nom], m.pvalues[nom]
    lo, hi = m.conf_int().loc[nom]
    return f"{c:+.5f}  (p = {p:.2g}, IC95 [{lo:+.5f} ; {hi:+.5f}])"


def analyse(path, col, nom, out):
    d, ami, L = prep(path, col)
    rate = " + rate" if ami else ""
    ctrl = f"{L} * freq_zipf + content{rate}"
    rds, rdf, rsf = r(d["duree"], d["s"]), r(d["duree"], d["freq_zipf"]), r(d["s"], d["freq_zipf"])
    attendu = rsf * rdf
    partiel = (rds - rdf * rsf) / np.sqrt((1 - rdf ** 2) * (1 - rsf ** 2))
    p = lambda t="": out.append(t)
    p("=" * 70)
    p(f"  {nom}   (n = {len(d)} mots" +
      (f", {d['meeting'].nunique()} reunions, {d['spk'].nunique()} locuteurs)" if ami else ")"))
    p("-" * 70)
    p("  Correlations brutes :")
    p(f"    surprise ~ duree     : r = {rds:+.3f}")
    p(f"    frequence ~ duree    : r = {rdf:+.3f}")
    p(f"    surprise ~ frequence : r = {rsf:+.3f}   (r2 = {rsf * rsf:.2f}, "
      f"part non expliquee par la frequence = {100 * (1 - rsf * rsf):.0f} %)")
    p("  Effet de la surprise sur la duree (s/nat), controle par controle :")
    p(f"    sans frequence        : {fmt(smf.ols(f'duree ~ {L} + content{rate} + s', d).fit())}")
    p(f"    frequence seule       : {fmt(smf.ols('duree ~ freq_zipf + s', d).fit())}")
    p(f"    frequence + longueur  : {fmt(smf.ols(f'duree ~ freq_zipf + {L} + s', d).fit())}")
    p(f"    modele complet (MCO)  : {fmt(smf.ols(f'duree ~ {ctrl} + s', d).fit())}")
    if ami:
        mm = smf.mixedlm(f"duree ~ {ctrl} + s", d, groups=d["spk"]).fit(reml=False, method="powell")
        p(f"    modele mixte (1|loc)  : {fmt(mm)}   converge = {mm.converged}")
        mnf = smf.mixedlm(f"duree ~ {L} + content{rate} + s", d, groups=d["spk"]).fit(reml=False, method="powell")
        p(f"    mixte sans frequence  : {fmt(mnf)}   converge = {mnf.converged}")
    dv = d[d["hors_vocab"] == 0]
    p(f"    mots connus de la grammaire seulement (n = {len(dv)}) : "
      f"{fmt(smf.ols(f'duree ~ {ctrl} + s', dv).fit())}")
    p("  Analyse EXTRA (a frequence controlee) :")
    p(f"    observe {rds:+.3f}  vs  attendu via frequence {attendu:+.3f}  ->  EXTRA = {rds - attendu:+.3f}")
    p(f"    correlation partielle = {partiel:+.4f}")
    p()


def complet_vs_apparie(path171, ami12, out):
    """Section 4.3 du rapport : l'effet syntaxique sur les 171 reunions, puis sur les
    seules 12 reunions ou GPT-2 a aussi ete calcule (memes controles)."""
    d, _, L = prep(path171, "surprise_syn")
    reunions = set(pd.read_csv(ami12, usecols=["meeting"])["meeting"])
    ctrl = f"{L} * freq_zipf + content + rate"
    p = lambda t="": out.append(t)
    p("=" * 70)
    p("  AMI SYNTAXIQUE : 171 reunions contre les 12 reunions appariees avec GPT-2")
    p("-" * 70)
    for nom, dd in (("171 reunions", d), ("12 reunions", d[d["meeting"].isin(reunions)])):
        mco = smf.ols(f"duree ~ {ctrl} + s", dd).fit()
        mm = smf.mixedlm(f"duree ~ {ctrl} + s", dd, groups=dd["spk"]).fit(reml=False, method="powell")
        p(f"  {nom:13s}(n = {len(dd)}, {dd['spk'].nunique()} locuteurs)")
        p(f"    MCO          : {fmt(mco)}")
        p(f"    mixte (1|loc): {fmt(mm)}   converge = {mm.converged}")
    p()


def main():
    livre = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ICI, "donnees", "surprise_livre_v2.csv")
    ami = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ICI, "donnees", "surprise_ami_v2.csv")
    ami171 = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ICI, "donnees", "surprise_syn_ami_171reunions_v2.csv.gz")
    out = ["", "RESULTATS COMPLETS V2 (surprise en nats) : syntaxique et GPT-2, parole lue et AMI", ""]
    analyse(livre, "surprise_syn", "PAROLE LUE  -  SYNTAXIQUE (PCFG)", out)
    analyse(livre, "surprise_llm", "PAROLE LUE  -  GPT-2", out)
    analyse(ami, "surprise_syn", "AMI         -  SYNTAXIQUE (PCFG)", out)
    analyse(ami, "surprise_llm", "AMI         -  GPT-2", out)
    analyse(ami171, "surprise_syn", "AMI 171 REUNIONS  -  SYNTAXIQUE (PCFG)", out)
    complet_vs_apparie(ami171, ami, out)
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(ICI, "resultats_complets_v2.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
