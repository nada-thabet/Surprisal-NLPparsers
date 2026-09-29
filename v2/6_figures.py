# -*- coding: utf-8 -*-
"""
Etape 6 : toutes les figures de la comparaison, refaites avec les surprises v2 (en nats).
Ce sont les 16 figures de version_rapport/4_figures, plus 3 figures AMI :

  - 10 figures en deciles (duree brute en bleu, duree apres controle en rouge) :
      cmp_lu_*   : parole lue, syntaxique vs GPT-2
      cmp_ami_*  : AMI, syntaxique vs GPT-2
      cmp_syn_corpus_*, cmp_llm_corpus_* : parole lue vs AMI pour chaque mesure
      *_llm_freq : GPT-2 sans puis avec controle de la frequence
  - 5 figures surprise / frequence (freq_corr_*, freq_binned_*, freq_corr_barres) ;
  - la figure de l'EXTRA (fig_extra) ;
  - 3 figures AMI en plus : la surprise syntaxique sur les 171 reunions contre les
    12 reunions appariees (cmp_ami171_*), et la figure rouge AMI ou le residu
    vient d'un modele mixte, donc sans l'ecart propre a chaque locuteur
    (cmp_ami_rouge_locuteur).

Sur chaque figure en deciles, un point = un decile de surprise, place au milieu de
l'intervalle, avec l'erreur standard. La pente de la droite rouge est descriptive ;
les coefficients testes sont dans resultats_complets_v2.txt.

    pip install pandas numpy statsmodels matplotlib
    python 6_figures.py
Sortie : dossier figures/
"""
import importlib.util, os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ICI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ICI, "figures")
LIVRE = os.path.join(ICI, "donnees", "surprise_livre_v2.csv")
AMI = os.path.join(ICI, "donnees", "surprise_ami_v2.csv")
AMI171 = os.path.join(ICI, "donnees", "surprise_syn_ami_171reunions_v2.csv.gz")

# meme preparation que les chiffres (5_resultats_complets.py)
_spec = importlib.util.spec_from_file_location("res", os.path.join(ICI, "5_resultats_complets.py"))
_res = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_res)
prep = _res.prep

BLUE = "#2C4A7C"; RED = "#9E2A2B"; DRED = "#7A1F1F"; LIGHT = "#B8C4D9"
rng = np.random.default_rng(0)


def charger(path, col):
    return prep(path, col)[0]


def dec(d, y):
    dd = d.copy(); dd["bin"] = pd.qcut(dd["s"], 10, duplicates="drop")
    g = dd.groupby("bin", observed=True)[y].agg(["mean", "sem"])
    return [iv.mid for iv in g.index], g["mean"].values, g["sem"].values


def res(d, formule):
    d = d.copy(); d["r"] = smf.ols(formule, d).fit().resid; return d


def deux_panneaux(specs, y, color, nom, titre, ylim, trend=False):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, (d, t) in enumerate(specs):
        x, m, e = dec(d, y)
        if trend:
            b = np.polyfit(d["s"], d[y], 1)[0]; a = np.mean(d[y]) - b * np.mean(d["s"])
            xr = np.linspace(min(x), max(x), 50)
            ax[k].errorbar(x, m, yerr=e, fmt="o", color=color, alpha=.55, capsize=3)
            ax[k].plot(xr, a + b * xr, "-", color=DRED, lw=2.4, label=f"pente {b:+.4f}")
            ax[k].axhline(0, color="grey", lw=.8, ls=":"); ax[k].legend(fontsize=8)
        else:
            ax[k].errorbar(x, m, yerr=e, fmt="o-", color=color, capsize=3)
        ax[k].set_title(t, color=color); ax[k].set_xlabel("surprise (nats)")
        ax[k].set_ylabel("durée résiduelle (s)" if trend else "durée moyenne (s)"); ax[k].set_ylim(ylim)
    fig.suptitle(titre, fontsize=13, fontweight="bold"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, nom), dpi=125); plt.close(); print("  ->", nom)


def figures_deciles(syn, llm, syA, llA):
    cl = "duree ~ n_phones*freq_zipf + content"
    ca = "duree ~ n_letters*freq_zipf + content + rate"
    deux_panneaux([(syn, "Syntaxique (PCFG)"), (llm, "GPT-2")], "duree", BLUE,
                  "cmp_lu_brut.png", "PAROLE LUE : durée BRUTE selon la surprise", (0.05, 0.46))
    deux_panneaux([(res(syn, cl), "Syntaxique (avec fréquence)"), (res(llm, cl), "GPT-2 (avec fréquence)")], "r", RED,
                  "cmp_lu_rouge.png", "PAROLE LUE : durée APRÈS CONTRÔLE", (-0.02, 0.02), True)
    deux_panneaux([(res(llm, "duree ~ n_phones + content"), "GPT-2 SANS contrôle de fréquence"),
                   (res(llm, cl), "GPT-2 AVEC contrôle de fréquence")], "r", RED,
                  "cmp_lu_llm_freq.png", "PAROLE LUE, GPT-2 : effet du contrôle de la fréquence", (-0.03, 0.03), True)
    deux_panneaux([(syA, "Syntaxique (PCFG)"), (llA, "GPT-2")], "duree", BLUE,
                  "cmp_ami_brut.png", "PAROLE SPONTANÉE (AMI) : durée BRUTE selon la surprise", (0.12, 0.47))
    deux_panneaux([(res(syA, ca), "Syntaxique (avec fréquence)"), (res(llA, ca), "GPT-2 (avec fréquence)")], "r", RED,
                  "cmp_ami_rouge.png", "PAROLE SPONTANÉE (AMI) : durée APRÈS CONTRÔLE", (-0.035, 0.035), True)
    deux_panneaux([(res(llA, "duree ~ n_letters + content + rate"), "GPT-2 SANS contrôle de fréquence"),
                   (res(llA, ca), "GPT-2 AVEC contrôle de fréquence")], "r", RED,
                  "cmp_ami_llm_freq.png", "AMI, GPT-2 : effet du contrôle de la fréquence", (-0.035, 0.04), True)
    deux_panneaux([(syn, "Parole lue"), (syA, "AMI (spontané)")], "duree", BLUE,
                  "cmp_syn_corpus_brut.png", "SYNTAXIQUE : durée BRUTE, parole lue vs AMI", (0.05, 0.47))
    deux_panneaux([(res(syn, cl), "Parole lue"), (res(syA, ca), "AMI (spontané)")], "r", RED,
                  "cmp_syn_corpus_rouge.png", "SYNTAXIQUE : durée APRÈS CONTRÔLE, parole lue vs AMI", (-0.03, 0.03), True)
    deux_panneaux([(llm, "Parole lue"), (llA, "AMI (spontané)")], "duree", BLUE,
                  "cmp_llm_corpus_brut.png", "GPT-2 : durée BRUTE, parole lue vs AMI", (0.10, 0.46))
    deux_panneaux([(res(llm, cl), "Parole lue"), (res(llA, ca), "AMI (spontané)")], "r", RED,
                  "cmp_llm_corpus_rouge.png", "GPT-2 : durée APRÈS CONTRÔLE, parole lue vs AMI", (-0.035, 0.035), True)


def figures_frequence(syn, llm, syA, llA):
    def nuage(dS, dL, nom, titre):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
        for a_, (d, t, yl) in zip(ax, [(dS, "Surprise SYNTAXIQUE vs fréquence", "surprise syntaxique (nats)"),
                                       (dL, "Surprise GPT-2 vs fréquence", "surprise GPT-2 (nats)")]):
            f, s = d["freq_zipf"].values, d["s"].values
            if len(f) > 80000:
                i = rng.choice(len(f), 80000, replace=False); f, s = f[i], s[i]
            rr = np.corrcoef(d["freq_zipf"], d["s"])[0, 1]
            a_.hexbin(f, s, gridsize=35, cmap="Blues", mincnt=1)
            b, a0 = np.polyfit(f, s, 1); xr = np.linspace(f.min(), f.max(), 50)
            a_.plot(xr, a0 + b * xr, "-", color=RED, lw=2.5)
            a_.set_title(f"{t}\n r = {rr:+.2f}   (r² = {rr * rr:.2f})", fontsize=11)
            a_.set_xlabel("fréquence (Zipf)"); a_.set_ylabel(yl)
        fig.suptitle(titre, fontsize=13, fontweight="bold"); fig.tight_layout()
        fig.savefig(os.path.join(OUT, nom), dpi=125); plt.close(); print("  ->", nom)

    def deciles_freq(dS, dL, nom, titre):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        for a_, (d, t, c) in zip(ax, [(dS, "Syntaxique", BLUE), (dL, "GPT-2", RED)]):
            dd = d[["s", "freq_zipf"]].copy(); dd["bin"] = pd.qcut(dd["freq_zipf"], 10, duplicates="drop")
            g = dd.groupby("bin", observed=True).agg(fm=("freq_zipf", "mean"), sm=("s", "mean"), se=("s", "sem"))
            a_.errorbar(g["fm"], g["sm"], yerr=g["se"], fmt="o-", color=c, capsize=3)
            a_.set_title(f"{t} : surprise moyenne par décile de fréquence", fontsize=11)
            a_.set_xlabel("fréquence (Zipf)"); a_.set_ylabel("surprise (nats)")
        fig.suptitle(titre, fontsize=13, fontweight="bold"); fig.tight_layout()
        fig.savefig(os.path.join(OUT, nom), dpi=125); plt.close(); print("  ->", nom)

    nuage(syn, llm, "freq_corr_lu.png", "PAROLE LUE : lien surprise / fréquence")
    nuage(syA, llA, "freq_corr_ami.png", "AMI (12 réunions appariées) : lien surprise / fréquence")
    deciles_freq(syn, llm, "freq_binned_lu.png", "PAROLE LUE : surprise moyenne selon la fréquence")
    deciles_freq(syA, llA, "freq_binned_ami.png", "AMI : surprise moyenne selon la fréquence")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    vals = [abs(np.corrcoef(d["s"], d["freq_zipf"])[0, 1]) for d in (syn, llm, syA, llA)]
    ax.bar(["Syntaxique\nlue", "GPT-2\nlue", "Syntaxique\nAMI", "GPT-2\nAMI"], vals, color=[BLUE, RED, BLUE, RED])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("|corrélation| avec la fréquence  (|r|)"); ax.set_ylim(0, 0.9)
    ax.set_title("Force du lien surprise / fréquence selon la mesure", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "freq_corr_barres.png"), dpi=130); plt.close()
    print("  -> freq_corr_barres.png")


def res_locuteur(d, formule):
    """Residu d'un modele mixte (1|locuteur) : on enleve aussi l'ecart propre a chaque locuteur."""
    d = d.copy()
    d["r"] = smf.mixedlm(formule, d, groups=d["spk"]).fit(reml=False, method="powell").resid
    return d


def figures_ami_complement(syA, llA, s171):
    """AMI : les 171 reunions contre les 12 reunions appariees (surprise syntaxique),
    et la figure rouge AMI avec le locuteur controle (modele mixte)."""
    ca = "duree ~ n_letters*freq_zipf + content + rate"
    deux_panneaux([(s171, "171 réunions"), (syA, "12 réunions appariées avec GPT-2")], "duree", BLUE,
                  "cmp_ami171_brut.png", "AMI, SYNTAXIQUE : durée BRUTE, 171 réunions vs 12 réunions", (0.12, 0.47))
    deux_panneaux([(res(s171, ca), "171 réunions"), (res(syA, ca), "12 réunions appariées avec GPT-2")], "r", RED,
                  "cmp_ami171_rouge.png", "AMI, SYNTAXIQUE : durée APRÈS CONTRÔLE, 171 réunions vs 12 réunions",
                  (-0.03, 0.03), True)
    deux_panneaux([(res_locuteur(syA, ca), "Syntaxique (avec fréquence et locuteur)"),
                   (res_locuteur(llA, ca), "GPT-2 (avec fréquence et locuteur)")], "r", RED,
                  "cmp_ami_rouge_locuteur.png", "AMI : durée APRÈS CONTRÔLE, locuteur compris (modèle mixte)",
                  (-0.035, 0.035), True)


def figure_extra(syn, llm, syA, llA):
    cas = []
    for lab, d, c in (("Syntaxique\nlue", syn, BLUE), ("GPT-2\nlue", llm, RED),
                      ("Syntaxique\nAMI", syA, BLUE), ("GPT-2\nAMI", llA, RED)):
        rds = np.corrcoef(d["duree"], d["s"])[0, 1]; rdf = np.corrcoef(d["duree"], d["freq_zipf"])[0, 1]
        rsf = np.corrcoef(d["s"], d["freq_zipf"])[0, 1]
        cas.append((lab, rds, rsf * rdf, rds - rsf * rdf, c))
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    fig.suptitle("L'EXTRA : la surprise apporte-t-elle quelque chose au-delà de la fréquence ?",
                 fontsize=14, fontweight="bold")
    ax[0].bar(x - w / 2, [c[2] for c in cas], w, color=LIGHT, label="attendu via la fréquence (r_sf × r_df)")
    ax[0].bar(x + w / 2, [c[1] for c in cas], w, color=[c[4] for c in cas], label="observé r(durée, surprise)")
    for i, (_, o, a, e, _) in enumerate(cas):
        ax[0].annotate("", xy=(i + w / 2, o), xytext=(i + w / 2, a),
                       arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
        ax[0].text(i + w / 2 + 0.07, max(o, a) + 0.01, f"EXTRA\n{e:+.3f}", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([c[0] for c in cas]); ax[0].set_ylim(0, 0.62)
    ax[0].set_ylabel("corrélation avec la durée"); ax[0].set_title("Observé vs attendu via la fréquence")
    ax[0].legend(fontsize=8, loc="upper right")
    ext = [c[3] for c in cas]
    ax[1].bar(x, ext, 0.55, color=[c[4] for c in cas])
    for i, e in enumerate(ext):
        ax[1].text(i, e + (0.003 if e >= 0 else -0.003), f"{e:+.3f}", ha="center",
                   va="bottom" if e >= 0 else "top", fontsize=12)
    ax[1].axhline(0, color="grey", lw=1); ax[1].set_ylim(-0.035, 0.085)
    ax[1].set_xticks(x); ax[1].set_xticklabels([c[0] for c in cas])
    ax[1].set_ylabel("EXTRA (effet propre à fréquence contrôlée)")
    ax[1].set_title("L'EXTRA : ce qui survit au contrôle de la fréquence")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(OUT, "fig_extra.png"), dpi=125); plt.close()
    print("  -> fig_extra.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Chargement des CSV v2...")
    syn, llm = charger(LIVRE, "surprise_syn"), charger(LIVRE, "surprise_llm")
    syA, llA = charger(AMI, "surprise_syn"), charger(AMI, "surprise_llm")
    figures_deciles(syn, llm, syA, llA)
    figures_frequence(syn, llm, syA, llA)
    figure_extra(syn, llm, syA, llA)
    figures_ami_complement(syA, llA, charger(AMI171, "surprise_syn"))
    print("Termine : 19 figures dans figures/")


if __name__ == "__main__":
    main()
