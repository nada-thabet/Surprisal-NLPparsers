# -*- coding: utf-8 -*-
"""
Figures de comparaison duree / surprise, a partir des 4 CSV.

Il faut les 4 CSV (surprise syntaxique et GPT-2, parole lue et AMI), avec leurs
chemins dans la partie CONFIG ci-dessous, puis : python figures_comparaisons.py

Chaque figure a deux panneaux qui comparent deux cas :
  - bleu  : duree brute par decile de surprise (rien n'est retire) ;
  - rouge : duree residuelle apres les controles, avec la droite de tendance.

  pip install pandas numpy statsmodels matplotlib
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ===================== CONFIG : chemins des CSV =====================
CSV_SYN_LU  = "surprisal_incrementale.csv"       # parole lue, surprise syntaxique (incrementale)
CSV_LLM_LU  = "surprisal_llm_textgrid.csv"       # parole lue, surprise LLM (GPT-2)
CSV_SYN_AMI = "surprisal_incrementale_ami.csv"   # AMI, surprise syntaxique
CSV_LLM_AMI = "surprisal_llm_ami.csv"            # AMI, surprise LLM (GPT-2)
# ===================================================================

BLUE="#2C4A7C"; RED="#9E2A2B"; DRED="#7A1F1F"
FILLED={"uh","um","mm","hmm","er","erm","ah","oh","huh","eh","mmm","hm","mm-hmm","uh-huh","yeah","yep","okay","mmhmm"}

def prep(path, lenc, ami=False):
    """Charge un CSV, calcule la duree et le debit (AMI), applique le nettoyage."""
    d = pd.read_csv(path)
    scol = "surprise_incrementale" if "surprise_incrementale" in d.columns else "surprise_llm"
    for c in (scol, "onset", "offset", "freq_zipf", lenc):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["duree"] = d["offset"] - d["onset"]; d["s"] = d[scol]
    d["content"] = (d["POS"].astype(str).str.upper().str[:2].isin(["NN","VB","JJ","RB"]).astype(int))
    if ami:
        d["w"] = d["mot"].astype(str).str.lower()
        g = d.groupby(["meeting","locuteur","phrase_id"])
        u = g.agg(n=("w","size"), t0=("onset","min"), t1=("offset","max")).reset_index()
        u["rate"] = u["n"] / (u["t1"]-u["t0"]).clip(lower=1e-3)
        d = d.merge(u[["meeting","locuteur","phrase_id","n","rate"]], on=["meeting","locuteur","phrase_id"], how="left")
        d = d[(d["n"] >= 4) & (~d["w"].isin(FILLED))]          # sans disfluences, enonces d'au moins 4 mots
        d = d.dropna(subset=["duree","s","freq_zipf",lenc,"rate"])
    else:
        d = d.dropna(subset=["duree","s","freq_zipf",lenc])
    return d[(d["duree"] > 0) & (d["duree"] <= 2)].copy()

def dec(d, y):
    dd = d.dropna(subset=["s", y]).copy(); dd["bin"] = pd.qcut(dd["s"], 10, duplicates="drop")
    g = dd.groupby("bin", observed=True)[y].agg(["mean","sem"])
    return [iv.mid for iv in g.index], g["mean"].values, g["sem"].values

def res(d, formula):
    d = d.copy(); d["r"] = smf.ols(formula, d).fit().resid; return d

def two_panels(specs, y, color, out, titre, ylim, trend=False):
    """specs = [(df, titre_panneau), (df, titre_panneau)]."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, (d, t) in enumerate(specs):
        x, m, e = dec(d, y)
        if trend:
            b = np.polyfit(d["s"], d[y], 1)[0]; a = np.mean(d[y]) - b*np.mean(d["s"]); xr = np.linspace(min(x), max(x), 50)
            ax[k].errorbar(x, m, yerr=e, fmt="o", color=color, alpha=.55, capsize=3)
            ax[k].plot(xr, a + b*np.array(xr), "-", color=DRED, lw=2.4, label=f"pente {b:+.4f}")
            ax[k].axhline(0, color="grey", lw=.8, ls=":"); ax[k].legend(fontsize=8)
        else:
            ax[k].errorbar(x, m, yerr=e, fmt="o-", color=color, capsize=3)
        ax[k].set_title(t, color=color); ax[k].set_xlabel("surprise (bits)")
        ax[k].set_ylabel("duree residuelle (s)" if trend else "duree moyenne (s)"); ax[k].set_ylim(ylim)
    fig.suptitle(titre, fontsize=13, fontweight="bold"); fig.tight_layout(); fig.savefig(out, dpi=125); plt.close()
    print("  ->", out)

def main():
    print("Chargement des CSV...")
    syn  = prep(CSV_SYN_LU,  "n_phones")
    llm  = prep(CSV_LLM_LU,  "n_phones")
    syA  = prep(CSV_SYN_AMI, "n_letters", ami=True)
    llA  = prep(CSV_LLM_AMI, "n_letters", ami=True)

    print("Figures : syntaxique vs LLM (parole lue)")
    two_panels([(syn,"Syntaxique (constituants)"),(llm,"LLM (GPT-2)")], "duree", BLUE,
               "cmp_lu_brut.png", "PAROLE LUE : duree BRUTE selon la surprise (panneaux bleus)", (0.10, 0.46))
    two_panels([(res(syn,"duree ~ n_phones*freq_zipf + content"),"Syntaxique (avec frequence)"),
                (res(llm,"duree ~ n_phones*freq_zipf + content"),"LLM (avec frequence)")], "r", RED,
               "cmp_lu_rouge.png", "PAROLE LUE : duree APRES CONTROLE (panneaux rouges)", (-0.016, 0.016), trend=True)
    two_panels([(res(llm,"duree ~ n_phones + content"),"LLM SANS controle de frequence"),
                (res(llm,"duree ~ n_phones*freq_zipf + content"),"LLM AVEC controle de frequence")], "r", RED,
               "cmp_lu_llm_freq.png", "PAROLE LUE, LLM : effet du controle de la FREQUENCE", (-0.025, 0.02), trend=True)

    print("Figures : syntaxique vs LLM (AMI)")
    two_panels([(syA,"Syntaxique (constituants)"),(llA,"LLM (GPT-2)")], "duree", BLUE,
               "cmp_ami_brut.png", "PAROLE SPONTANEE (AMI) : duree BRUTE selon la surprise (bleus)", (0.18, 0.46))
    two_panels([(res(syA,"duree ~ n_letters*freq_zipf + content + rate"),"Syntaxique (avec frequence)"),
                (res(llA,"duree ~ n_letters*freq_zipf + content + rate"),"LLM (avec frequence)")], "r", RED,
               "cmp_ami_rouge.png", "PAROLE SPONTANEE (AMI) : duree APRES CONTROLE (rouges)", (-0.02, 0.02), trend=True)
    two_panels([(res(llA,"duree ~ n_letters + content + rate"),"LLM SANS controle de frequence"),
                (res(llA,"duree ~ n_letters*freq_zipf + content + rate"),"LLM AVEC controle de frequence")], "r", RED,
               "cmp_ami_llm_freq.png", "PAROLE SPONTANEE (AMI), LLM : effet du controle de la FREQUENCE", (-0.02, 0.02), trend=True)

    print("Figures : comparaison des corpus (parole lue vs AMI)")
    two_panels([(syn,"Parole lue"),(syA,"AMI (spontané)")], "duree", BLUE,
               "cmp_syn_corpus_brut.png", "SYNTAXIQUE : duree BRUTE, parole lue vs AMI (bleus)", (0.10, 0.46))
    two_panels([(res(syn,"duree ~ n_phones*freq_zipf + content"),"Parole lue"),
                (res(syA,"duree ~ n_letters*freq_zipf + content + rate"),"AMI (spontané)")], "r", RED,
               "cmp_syn_corpus_rouge.png", "SYNTAXIQUE : duree APRES CONTROLE, parole lue vs AMI (rouges)", (-0.02, 0.02), trend=True)
    two_panels([(llm,"Parole lue"),(llA,"AMI (spontané)")], "duree", BLUE,
               "cmp_llm_corpus_brut.png", "LLM (GPT-2) : duree BRUTE, parole lue vs AMI (bleus)", (0.10, 0.46))
    two_panels([(res(llm,"duree ~ n_phones*freq_zipf + content"),"Parole lue"),
                (res(llA,"duree ~ n_letters*freq_zipf + content + rate"),"AMI (spontané)")], "r", RED,
               "cmp_llm_corpus_rouge.png", "LLM (GPT-2) : duree APRES CONTROLE, parole lue vs AMI (rouges)", (-0.02, 0.02), trend=True)
    print("Termine : 10 figures generees dans le dossier courant.")

if __name__ == "__main__":
    main()
