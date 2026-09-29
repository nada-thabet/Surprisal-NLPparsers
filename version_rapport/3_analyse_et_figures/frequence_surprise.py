# -*- coding: utf-8 -*-
"""
Lien entre frequence et surprise : correlations et figures.

Le but est de voir a quel point chaque surprise (syntaxique et GPT-2) est liee a la
frequence du mot. C'est ce lien qui explique pourquoi l'effet de GPT-2 disparait
quand on controle la frequence.

Sorties :
  - freq_corr_lu.png / freq_corr_ami.png : nuage de densite surprise vs frequence
    (syntaxique a gauche, GPT-2 a droite), avec r et r2 ;
  - freq_binned_lu.png / freq_binned_ami.png : surprise moyenne par decile de frequence ;
  - freq_corr_barres.png : |r| pour les 4 cas.
Les correlations sont aussi affichees dans le terminal.

  pip install pandas numpy matplotlib
  python frequence_surprise.py
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ===================== CONFIG : chemins des CSV =====================
CSV_SYN_LU  = "surprisal_incrementale.csv"
CSV_LLM_LU  = "surprisal_llm_textgrid.csv"
CSV_SYN_AMI = "surprisal_incrementale_ami.csv"
CSV_LLM_AMI = "surprisal_llm_ami.csv"
# ===================================================================
BLUE="#2C4A7C"; RED="#9E2A2B"; DRED="#7A1F1F"; rng=np.random.default_rng(0)

from resultats_comparaison import prep   # meme nettoyage que les chiffres du rapport

def load(path):
    """Surprise et frequence, avec le meme nettoyage que pour les modeles : 0 < duree <= 2 s,
    et pour AMI sans disfluences et avec des enonces d'au moins 4 mots (les CSV AMI
    contiennent deja seulement les 12 reunions appariees). Sans ce nettoyage, on obtient
    r = -0,42 / -0,66 sur AMI au lieu de -0,48 / -0,63."""
    ami = "meeting" in pd.read_csv(path, nrows=1).columns
    d = prep(path, "n_letters" if ami else "n_phones", ami)
    return d["s"].values, d["freq_zipf"].values

def corr(s, f):
    return np.corrcoef(f, s)[0, 1]

def sample(x, y, n=80000):
    if len(x) > n:
        i = rng.choice(len(x), n, replace=False); return x[i], y[i]
    return x, y

def hexbin_fig(pS, pL, out, titre):
    """Nuage de densite surprise vs frequence : syntaxique a gauche, GPT-2 a droite."""
    sS, fS = load(pS); sL, fL = load(pL)
    fS2, sS2 = sample(fS, sS); fL2, sL2 = sample(fL, sL)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for a_, (f, s, t, yl) in zip(ax, [(fS2, sS2, "Surprise SYNTAXIQUE vs fréquence", "surprise syntaxique (bits)"),
                                      (fL2, sL2, "Surprise LLM (GPT-2) vs fréquence", "surprise LLM (bits)")]):
        r = corr(s, f); a_.hexbin(f, s, gridsize=35, cmap="Blues", mincnt=1)
        b, a0 = np.polyfit(f, s, 1); xr = np.linspace(f.min(), f.max(), 50)
        a_.plot(xr, a0 + b*xr, "-", color=RED, lw=2.5)
        a_.set_title(f"{t}\n r = {r:+.2f}   (r² = {r*r:.2f})", fontsize=11)
        a_.set_xlabel("fréquence (Zipf)"); a_.set_ylabel(yl)
    fig.suptitle(titre, fontsize=13, fontweight="bold"); fig.tight_layout(); fig.savefig(out, dpi=125); plt.close()
    print("  ->", out)
    return corr(*load(pS)[::-1]) if False else None

def binned_fig(pS, pL, out, titre):
    """Surprise moyenne par decile de frequence (syntaxique et GPT-2)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a_, (p, t, c, yl) in zip(ax, [(pS, "Syntaxique", BLUE, "surprise syntaxique (bits)"),
                                      (pL, "LLM (GPT-2)", RED, "surprise LLM (bits)")]):
        s, f = load(p); d = pd.DataFrame({"s": s, "f": f}); d["bin"] = pd.qcut(d["f"], 10, duplicates="drop")
        g = d.groupby("bin", observed=True).agg(fm=("f","mean"), sm=("s","mean"), se=("s","sem"))
        a_.errorbar(g["fm"], g["sm"], yerr=g["se"], fmt="o-", color=c, capsize=3)
        a_.set_title(f"{t} : surprise moyenne par décile de fréquence", fontsize=11)
        a_.set_xlabel("fréquence (Zipf)"); a_.set_ylabel(yl)
    fig.suptitle(titre, fontsize=13, fontweight="bold"); fig.tight_layout(); fig.savefig(out, dpi=125); plt.close()
    print("  ->", out)

def main():
    rS_lu = corr(*load(CSV_SYN_LU)); rL_lu = corr(*load(CSV_LLM_LU))
    rS_am = corr(*load(CSV_SYN_AMI)); rL_am = corr(*load(CSV_LLM_AMI))
    # corr(f, s) == corr(s, f), donc l'ordre renvoye par load ne change rien
    print("CORRELATIONS surprise ~ frequence :")
    print(f"  Parole lue : syntaxique {rS_lu:+.3f}  |  LLM {rL_lu:+.3f}")
    print(f"  AMI        : syntaxique {rS_am:+.3f}  |  LLM {rL_am:+.3f}")

    hexbin_fig(CSV_SYN_LU, CSV_LLM_LU, "freq_corr_lu.png",  "PAROLE LUE : lien surprise / fréquence (syntaxique vs LLM)")
    hexbin_fig(CSV_SYN_AMI, CSV_LLM_AMI, "freq_corr_ami.png", "AMI (12 réunions appariées) : lien surprise / fréquence (syntaxique vs LLM)")
    binned_fig(CSV_SYN_LU, CSV_LLM_LU, "freq_binned_lu.png",  "PAROLE LUE : surprise moyenne selon la fréquence")
    binned_fig(CSV_SYN_AMI, CSV_LLM_AMI, "freq_binned_ami.png", "AMI : surprise moyenne selon la fréquence")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["Syntaxique\nlue","LLM\nlue","Syntaxique\nAMI","LLM\nAMI"]
    vals = [abs(rS_lu), abs(rL_lu), abs(rS_am), abs(rL_am)]; cols = [BLUE, RED, BLUE, RED]
    ax.bar(labels, vals, color=cols)
    for i, v in enumerate(vals): ax.text(i, v+0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("|corrélation| avec la fréquence  (|r|)"); ax.set_ylim(0, 0.9)
    ax.set_title("Force du lien surprise / fréquence selon la mesure", fontsize=12)
    fig.tight_layout(); fig.savefig("freq_corr_barres.png", dpi=130); plt.close(); print("  -> freq_corr_barres.png")

if __name__ == "__main__":
    main()
