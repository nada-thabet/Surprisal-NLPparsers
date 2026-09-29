# -*- coding: utf-8 -*-
"""
Etape 9 : le dernier decile de surprise (les 10 % de mots les plus surprenants),
refait avec les surprises v2 (en nats) pour les quatre cas.

Comme analyse_dernier_decile.py et figures_avec_sans_dernier_decile.py du stage :
  1. taille et etendue du 10e decile ;
  2. profil de ces mots compare aux autres (frequence, longueur, duree, residu,
     part de mots de contenu), formes les plus frequentes, categories ;
  3. residu moyen dans le 10e decile selon la frequence ;
  4. coefficient de la surprise avec la frequence en lineaire, avec un terme
     freq_zipf^2, et sans le 10e decile (MCO ; le modele mixte AMI est dans
     resultats_robustesse_v2.txt) ;
  5. les 20 mots du 10e decile au residu le plus negatif.
Et les figures en deciles en deux versions : 10 deciles (en haut) et 9 deciles,
dernier retire (en bas).

Enlever le dernier decile de la figure ne change que le dessin. Pour savoir s'il
change le resultat, il faut regarder le coefficient du point 4.

    pip install pandas numpy statsmodels matplotlib
    python 9_dernier_decile.py
Sorties : resultats_dernier_decile_v2.txt et figures/dec_{lu,ami}_{brut,rouge}.png
"""
import importlib.util, os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
ICI = os.path.dirname(os.path.abspath(__file__))
LIVRE = os.path.join(ICI, "donnees", "surprise_livre_v2.csv")
AMI = os.path.join(ICI, "donnees", "surprise_ami_v2.csv")
FIG = os.path.join(ICI, "figures")
BLUE, RED, DRED = "#2C4A7C", "#9E2A2B", "#7A1F1F"

# meme preparation que les chiffres principaux (5_resultats_complets.py)
_spec = importlib.util.spec_from_file_location("res", os.path.join(ICI, "5_resultats_complets.py"))
_res = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_res)
prep, fmt = _res.prep, _res.fmt

out = []
p = lambda t="": out.append(str(t))


def charger(path, col):
    d, ami, L = prep(path, col)
    d["w"] = d["mot"].astype(str).str.lower()
    formule = f"duree ~ {L}*freq_zipf + content" + (" + rate" if ami else "")
    d["r"] = smf.ols(formule, d).fit().resid
    d["dec"] = pd.qcut(d["s"], 10, labels=False, duplicates="drop")
    return d, L, formule


def analyser(d, L, formule, nom):
    p("=" * 78); p(nom); p("=" * 78)
    top, reste = d[d["dec"] == d["dec"].max()], d[d["dec"] < d["dec"].max()]

    p("\n1. TAILLE ET ETENDUE DU DECILE 10")
    p(f"   mots dans le decile 10 : {len(top)}  ({100 * len(top) / len(d):.1f} % des mots)")
    p(f"   surprise : min {top['s'].min():.2f}, mediane {top['s'].median():.2f}, "
      f"moyenne {top['s'].mean():.2f}, max {top['s'].max():.2f} nats")
    p(f"   les 9 autres deciles vont de {d['s'].min():.2f} a {top['s'].min():.2f} nats")

    p("\n2. PROFIL DES MOTS DU DECILE 10, COMPARE AU RESTE")
    lignes = [{"groupe": g, "n": len(x), "freq_zipf_moy": round(x["freq_zipf"].mean(), 2),
               "longueur_moy": round(x[L].mean(), 2), "duree_brute_moy": round(x["duree"].mean(), 4),
               "residu_moy": round(x["r"].mean(), 5), "part_contenu": round(x["content"].mean(), 3)}
              for g, x in (("decile 10", top), ("deciles 1-9", reste))]
    p(pd.DataFrame(lignes).to_string(index=False))
    p("\n   30 formes les plus frequentes dans le decile 10 :")
    p("   " + ", ".join(top["w"].value_counts().head(30).index.tolist()))
    p("\n   Categories grammaticales, part dans le decile 10 et ailleurs :")
    a = top["POS"].astype(str).str.upper().str[:2].value_counts(normalize=True)
    b = reste["POS"].astype(str).str.upper().str[:2].value_counts(normalize=True)
    comp = pd.DataFrame({"decile 10": a, "deciles 1-9": b}).fillna(0)
    comp["ecart"] = comp["decile 10"] - comp["deciles 1-9"]
    p(comp.sort_values("ecart", ascending=False).head(10).round(3).to_string())

    p("\n3. RESIDU MOYEN DANS LE DECILE 10, PAR NIVEAU DE FREQUENCE")
    t = top.copy()
    t["qfreq"] = pd.qcut(t["freq_zipf"], 4, duplicates="drop")
    p(t.groupby("qfreq", observed=True)[["r", "duree", "freq_zipf", L]].mean().round(4).to_string())

    p("\n4. COEFFICIENT DE LA SURPRISE SELON LA FORME DU CONTROLE DE FREQUENCE (MCO, s/nat)")
    p(f"   frequence lineaire          : {fmt(smf.ols(formule + ' + s', d).fit())}")
    p(f"   + terme freq_zipf^2         : {fmt(smf.ols(formule + ' + I(freq_zipf**2) + s', d).fit())}")
    p(f"   lineaire, sans le decile 10 : {fmt(smf.ols(formule + ' + s', reste).fit())}")

    p("\n5. LES 20 MOTS DU DECILE 10 AU RESIDU LE PLUS NEGATIF")
    p(top.nsmallest(20, "r")[["w", "POS", "s", "freq_zipf", L, "duree", "r"]].round(3).to_string(index=False))
    p()


def panneau(ax, d, y, couleur, titre, sans_dernier, tendance):
    dd = d[d["dec"] < d["dec"].max()] if sans_dernier else d
    g = dd.groupby("dec")[[y, "s"]].agg(["mean", "sem"])
    x, m, e = g[("s", "mean")].values, g[(y, "mean")].values, g[(y, "sem")].values
    ax.errorbar(x, m, yerr=e, fmt="o" if tendance else "o-", color=couleur,
                alpha=.6 if tendance else 1, capsize=3)
    if tendance:
        # pente calculee sur les mots gardes, pas sur les points
        b = np.polyfit(dd["s"], dd[y], 1)[0]; a = dd[y].mean() - b * dd["s"].mean()
        xr = np.linspace(min(x), max(x), 50)
        ax.plot(xr, a + b * xr, "-", color=DRED, lw=2.4, label=f"pente {b:+.4f}")
        ax.axhline(0, color="grey", lw=.8, ls=":"); ax.legend(fontsize=8)
    ax.set_title(titre, color=couleur, fontsize=10); ax.set_xlabel("surprise (nats)")
    ax.set_ylabel("durée résiduelle (s)" if tendance else "durée moyenne (s)")


def figure(specs, nom, titre, couleur, tendance, ylim):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    for ligne, sans in enumerate((False, True)):
        for k, (d, st) in enumerate(specs):
            suffixe = "9 déciles (dernier retiré)" if sans else "10 déciles"
            panneau(axes[ligne][k], d, "r" if tendance else "duree", couleur, f"{st}, {suffixe}", sans, tendance)
            axes[ligne][k].set_ylim(ylim)
    fig.suptitle(titre, fontsize=13, fontweight="bold"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, nom), dpi=125); plt.close(); print("  ->", nom)


def main():
    os.makedirs(FIG, exist_ok=True)
    cas = {}
    for path, col, nom in ((LIVRE, "surprise_syn", "PAROLE LUE, SURPRISE SYNTAXIQUE"),
                           (LIVRE, "surprise_llm", "PAROLE LUE, SURPRISE GPT-2"),
                           (AMI, "surprise_syn", "AMI (12 REUNIONS), SURPRISE SYNTAXIQUE"),
                           (AMI, "surprise_llm", "AMI (12 REUNIONS), SURPRISE GPT-2")):
        d, L, formule = charger(path, col)
        analyser(d, L, formule, nom)
        cas[nom] = d
    txt = "\n".join(["", "LE DERNIER DECILE DE SURPRISE, V2 (surprise en nats)", ""] + out)
    with open(os.path.join(ICI, "resultats_dernier_decile_v2.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    syn, llm, syA, llA = cas.values()
    figure([(syn, "Syntaxique"), (llm, "GPT-2")], "dec_lu_brut.png",
           "PAROLE LUE : durée brute", BLUE, False, (0.05, 0.45))
    figure([(syn, "Syntaxique"), (llm, "GPT-2")], "dec_lu_rouge.png",
           "PAROLE LUE : durée après contrôle", RED, True, (-0.02, 0.02))
    figure([(syA, "Syntaxique"), (llA, "GPT-2")], "dec_ami_brut.png",
           "AMI : durée brute", BLUE, False, (0.12, 0.47))
    figure([(syA, "Syntaxique"), (llA, "GPT-2")], "dec_ami_rouge.png",
           "AMI : durée après contrôle", RED, True, (-0.035, 0.035))


if __name__ == "__main__":
    main()
