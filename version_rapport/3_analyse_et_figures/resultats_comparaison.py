# -*- coding: utf-8 -*-
"""
Tous les chiffres de la comparaison syntaxique / GPT-2 et de l'analyse de frequence.

Pour chaque cas (syntaxique ou GPT-2, parole lue ou AMI) :
  - le nombre de mots ;
  - les correlations brutes surprise/duree, frequence/duree, surprise/frequence (et r2) ;
  - le coefficient de la surprise dans le modele complet, avec puis sans la frequence ;
  - la correlation partielle (a frequence controlee) et l'EXTRA
    (observe moins ce qu'on attendrait si la surprise n'agissait que via la frequence) ;
  - pour AMI : le coefficient du modele mixte (intercept aleatoire par locuteur =
    reunion + lettre), ajuste en ML (reml=False) et en REML avec lbfgs.
    Les deux lignes AMI du rapport viennent de ces ajustements : ML pour la syntaxe
    (+0,00077 ; p = 0,028), REML pour GPT-2 (-0,00179 ; p = 1,0e-13). Attention, ces
    ajustements lbfgs ne convergent pas bien (variance locuteur a 0). Avec powell ou
    nm, qui convergent, on trouve +0,00076 (p = 0,030) et -0,00180 : les conclusions
    ne changent pas.

AMI apparie : la surprise syntaxique est limitee aux reunions du CSV GPT-2
(12 reunions ES2002a..ES2004d), pour que les deux mesures portent sur les memes
60 285 mots et 48 locuteurs (rapport, section 4.3).

  pip install pandas numpy statsmodels
  python resultats_comparaison.py
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf

# ===================== CONFIG : chemins des CSV =====================
CSV_SYN_LU  = "surprisal_incrementale.csv"
CSV_LLM_LU  = "surprisal_llm_textgrid.csv"
CSV_SYN_AMI = "surprisal_incrementale_ami.csv"
CSV_LLM_AMI = "surprisal_llm_ami.csv"
# ===================================================================
FILLED={"uh","um","mm","hmm","er","erm","ah","oh","huh","eh","mmm","hm","mm-hmm","uh-huh","yeah","yep","okay","mmhmm"}

def prep(path, lenc, ami=False):
    d = pd.read_csv(path)
    sc = "surprise_incrementale" if "surprise_incrementale" in d.columns else "surprise_llm"
    for c in (sc, "onset", "offset", "freq_zipf", lenc):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["duree"] = d["offset"] - d["onset"]; d["s"] = d[sc]
    d["content"] = (d["POS"].astype(str).str.upper().str[:2].isin(["NN","VB","JJ","RB"]).astype(int))
    if ami:
        d["w"] = d["mot"].astype(str).str.lower()
        g = d.groupby(["meeting","locuteur","phrase_id"])
        u = g.agg(n=("w","size"), t0=("onset","min"), t1=("offset","max")).reset_index()
        u["rate"] = u["n"] / (u["t1"]-u["t0"]).clip(lower=1e-3)
        d = d.merge(u[["meeting","locuteur","phrase_id","n","rate"]], on=["meeting","locuteur","phrase_id"], how="left")
        d = d[(d["n"] >= 4) & (~d["w"].isin(FILLED))].dropna(subset=["duree","s","freq_zipf",lenc,"rate"])
    else:
        d = d.dropna(subset=["duree","s","freq_zipf",lenc])
    return d[(d["duree"] > 0) & (d["duree"] <= 2)].copy()

def r(a, b): return np.corrcoef(a, b)[0, 1]

def reunions_communes():
    """Reunions presentes dans les deux CSV AMI (AMI apparie)."""
    a = set(pd.read_csv(CSV_SYN_AMI, usecols=["meeting"])["meeting"])
    b = set(pd.read_csv(CSV_LLM_AMI, usecols=["meeting"])["meeting"])
    return a & b

def fmt(m, nom="s"):
    c, p = m.params[nom], m.pvalues[nom]; lo, hi = m.conf_int().loc[nom]
    return f"{c:+.5f}  (p = {p:.2g}, IC95 [{lo:+.5f} ; {hi:+.5f}])"

def analyse(path, lenc, nom, ami=False):
    d = prep(path, lenc, ami)
    if ami:
        d = d[d["meeting"].isin(reunions_communes())].copy()
        d["spk"] = d["meeting"].astype(str) + "." + d["locuteur"].astype(str)
    rds = r(d["duree"], d["s"]); rdf = r(d["duree"], d["freq_zipf"]); rsf = r(d["s"], d["freq_zipf"])
    attendu = rsf * rdf                                   # correlation attendue via la seule frequence
    extra = rds - attendu
    partiel = (rds - rdf*rsf) / np.sqrt((1-rdf**2)*(1-rsf**2))
    L = "n_letters" if ami else "n_phones"
    ctrl = f"{L} * freq_zipf + content" + (" + rate" if ami else "")
    m_freq      = smf.ols(f"duree ~ {ctrl} + s", d).fit()                             # avec frequence
    coef_freq   = m_freq.params["s"]
    coef_nofreq = smf.ols(f"duree ~ {L} + content{' + rate' if ami else ''} + s", d).fit().params["s"]  # sans frequence
    print("="*70)
    print(f"  {nom}   (n = {len(d)} mots" + (f", {d['meeting'].nunique()} reunions, {d['spk'].nunique()} locuteurs)" if ami else ")"))
    print("-"*70)
    print(f"  Correlations brutes :")
    print(f"    surprise ~ duree     : r = {rds:+.3f}")
    print(f"    frequence ~ duree    : r = {rdf:+.3f}")
    print(f"    surprise ~ frequence : r = {rsf:+.3f}   (r2 = {rsf*rsf:.2f})")
    print(f"  Effet de la surprise sur la duree (modele joint) :")
    print(f"    coef AVEC controle de la frequence = {coef_freq:+.5f}")
    print(f"    coef SANS controle de la frequence = {coef_nofreq:+.5f}")
    print(f"    MCO, avec frequence : {fmt(m_freq)}")
    if ami:
        for reml in (False, True):
            mm = smf.mixedlm(f"duree ~ {ctrl} + s", d, groups=d["spk"]).fit(reml=reml, method="lbfgs")
            print(f"    MODELE MIXTE (1|locuteur), {'REML' if reml else 'ML  '} : {fmt(mm)}")
    print(f"  Analyse EXTRA (a frequence controlee) :")
    print(f"    observe {rds:+.3f}  vs  attendu via frequence {attendu:+.3f}  ->  EXTRA = {extra:+.3f}")
    print(f"    correlation partielle = {partiel:+.4f}")
    print()

if __name__ == "__main__":
    print("\nRESULTATS DE COMPARAISON (surprise -> duree) et FREQUENCE\n")
    analyse(CSV_SYN_LU,  "n_phones",  "PAROLE LUE  -  SYNTAXIQUE (constituants)")
    analyse(CSV_LLM_LU,  "n_phones",  "PAROLE LUE  -  LLM (GPT-2)")
    analyse(CSV_SYN_AMI, "n_letters", "AMI         -  SYNTAXIQUE (constituants)", ami=True)
    analyse(CSV_LLM_AMI, "n_letters", "AMI         -  LLM (GPT-2)", ami=True)
