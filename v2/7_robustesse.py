# -*- coding: utf-8 -*-
"""
Etape 7 : les analyses complementaires et les tests de robustesse, sur les CSV v2.

Les cinq cas : parole lue syntaxique et GPT-2, AMI 12 reunions syntaxique et GPT-2,
AMI 171 reunions syntaxique.
Une partie reprend des analyses faites pendant le stage a cote des resultats
principaux : le modele de base et l'AIC, la methode en deux temps
(regression_duree*.py), le terme de frequence au carre et le dernier decile
(analyse_dernier_decile.py, figures_avec_sans_dernier_decile.py), AMI nettoye contre
non nettoye (regression_duree_ami.py et regression_duree_ami_propre.py), le modele
mixte en ML et en REML (resultats_comparaison.py). Le reste est ajoute en v2 :
l'ICC, le test du rapport de vraisemblance, les optimiseurs powell et nm, les deux
mesures sur les memes mots, et les variantes du nettoyage une par une.

Pour chaque cas :
  A. le nettoyage etape par etape, les mots de contenu et les mots-outils, les
     correlations brutes de la duree avec la longueur (et avec le debit pour AMI) ;
  B. le modele complet en entier : tous les coefficients, le R2, et pour AMI la
     variance entre locuteurs ;
  C. modele de base (controles seuls) contre modele avec surprise : AIC et test
     du rapport de vraisemblance ;
  D. la methode en deux temps de Demberg : on enleve d'abord l'effet des controles
     (residu), puis on regarde le lien entre ce residu et la surprise. C'est la
     methode du debut du stage, a comparer au modele joint (section 4.1 du rapport) ;
  E. la forme du controle de frequence et le dernier decile : on ajoute
     freq_zipf^2, puis on refait le modele sans le 10e decile de surprise.
Pour AMI en plus :
  F. le modele mixte avec les trois optimiseurs (lbfgs, powell, nm), en ML et en
     REML, avec la variance par locuteur (12 reunions) ;
  G. syntaxe et GPT-2 sur exactement les memes mots (12 reunions) ;
  H. la sensibilite au nettoyage : sans retirer les hesitations, sans le seuil de
     4 mots par enonce, sans le debit, puis sans ces trois elements a la fois
     (le filtre 0 < duree <= 2 s est garde partout).

Nettoyage de base et modeles : les memes que 5_resultats_complets.py.
Coefficients de la surprise en s/nat (x ln 2 = 0,693 pour des s/bit).

    pip install pandas numpy scipy statsmodels
    python 7_robustesse.py
Sortie : resultats_robustesse_v2.txt (environ 2 minutes, surtout pour les 171 reunions)
"""
import importlib.util, os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

warnings.filterwarnings("ignore")
ICI = os.path.dirname(os.path.abspath(__file__))
LIVRE = os.path.join(ICI, "donnees", "surprise_livre_v2.csv")
AMI = os.path.join(ICI, "donnees", "surprise_ami_v2.csv")
AMI171 = os.path.join(ICI, "donnees", "surprise_syn_ami_171reunions_v2.csv.gz")

# meme preparation que les chiffres principaux (5_resultats_complets.py)
_spec = importlib.util.spec_from_file_location("res", os.path.join(ICI, "5_resultats_complets.py"))
_res = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_res)
prep, fmt, HESIT = _res.prep, _res.fmt, _res.HESIT

out = []
p = lambda t="": out.append(t)


def mixte(formule, d, reml=False, method="powell"):
    return smf.mixedlm(formule, d, groups=d["spk"]).fit(reml=reml, method=method)


def ctrl_de(L, ami, rate=True):
    return f"{L} * freq_zipf + content" + (" + rate" if ami and rate else "")


# ---------------------------------------------------------------------------
#  A. nettoyage et correlations brutes
# ---------------------------------------------------------------------------
def nettoyage(path, col, d, ami, L):
    brut = pd.read_csv(path)
    brut["duree"] = brut["offset"] - brut["onset"]
    p("  A. Nettoyage et correlations brutes")
    if ami:
        w = brut["mot"].astype(str).str.lower()
        n = brut.groupby(["meeting", "locuteur", "phrase_id"])["mot"].transform("size")
        e1 = brut[n >= 4]
        e2 = e1[~w[n >= 4].isin(HESIT)]
        e3 = e2[(e2["duree"] > 0) & (e2["duree"] <= 2)]
        p(f"    mots au depart                       : {len(brut)}")
        p(f"    enonces d'au moins 4 mots            : {len(e1)}")
        p(f"    sans les hesitations                 : {len(e2)}")
        p(f"    0 < duree <= 2 s                     : {len(e3)}")
    else:
        e3 = brut[(brut["duree"] > 0) & (brut["duree"] <= 2)]
        p(f"    mots au depart                       : {len(brut)}")
        p(f"    0 < duree <= 2 s                     : {len(e3)}")
    p(f"    avec une valeur de surprise          : {len(d)}")
    p(f"    mots de contenu / mots-outils        : {int(d['content'].sum())} / {int((1 - d['content']).sum())}")
    p(f"    r(duree, longueur {L:9s})         = {np.corrcoef(d['duree'], d[L])[0, 1]:+.3f}")
    if ami:
        p(f"    r(duree, debit)                      = {np.corrcoef(d['duree'], d['rate'])[0, 1]:+.3f}")


# ---------------------------------------------------------------------------
#  B. modele complet en entier ; C. AIC et rapport de vraisemblance
# ---------------------------------------------------------------------------
def tableau(m, noms):
    ic = m.conf_int()
    for t in noms:
        if t in m.params:
            p(f"      {t:20s} {m.params[t]:+.5f}   p = {m.pvalues[t]:.2g}   "
              f"IC95 [{ic.loc[t, 0]:+.5f} ; {ic.loc[t, 1]:+.5f}]")


def modele_complet(d, ami, L):
    ctrl = ctrl_de(L, ami)
    noms = ["Intercept", L, "freq_zipf", f"{L}:freq_zipf", "content"] + (["rate"] if ami else []) + ["s"]
    base = smf.ols(f"duree ~ {ctrl}", d).fit()
    comp = smf.ols(f"duree ~ {ctrl} + s", d).fit()
    p("  B. Modele complet (MCO) : duree ~ " + ctrl + " + surprise")
    tableau(comp, noms)
    p(f"      R2 = {comp.rsquared:.4f}  (R2 ajuste {comp.rsquared_adj:.4f}),  "
      f"R2 sans la surprise = {base.rsquared:.4f}")
    res = {"base": base, "comp": comp}
    if ami:
        mb = mixte(f"duree ~ {ctrl}", d)
        mm = mixte(f"duree ~ {ctrl} + s", d)
        vl, ve = float(mm.cov_re.iloc[0, 0]), float(mm.scale)
        p("     Modele mixte (1|locuteur), ML, powell :")
        tableau(mm, noms)
        p(f"      variance entre locuteurs = {vl:.6f}, variance residuelle = {ve:.6f}, "
          f"part due au locuteur (ICC) = {vl / (vl + ve):.3f}, converge = {mm.converged}")
        res.update(mbase=mb, mcomp=mm)
    # C
    p("  C. Controles seuls contre controles + surprise")
    lrt = 2 * (comp.llf - base.llf)
    p(f"    MCO   : AIC {base.aic:.1f} -> {comp.aic:.1f}  (gain {base.aic - comp.aic:+.1f}),  "
      f"rapport de vraisemblance chi2(1) = {lrt:.1f}, p = {chi2.sf(lrt, 1):.2g}")
    if ami:
        mb, mm = res["mbase"], res["mcomp"]
        lrt = 2 * (mm.llf - mb.llf)
        p(f"    mixte : AIC {mb.aic:.1f} -> {mm.aic:.1f}  (gain {mb.aic - mm.aic:+.1f}),  "
          f"rapport de vraisemblance chi2(1) = {lrt:.1f}, p = {chi2.sf(lrt, 1):.2g}")
    return res


# ---------------------------------------------------------------------------
#  D. methode en deux temps (Demberg) contre modele joint
# ---------------------------------------------------------------------------
def deux_temps(d, ami, res):
    p("  D. Methode en deux temps (residu des controles, puis residu ~ surprise)")
    dd = d.copy()
    base = res["mbase"] if ami else res["base"]
    dd["residu"] = base.resid
    m2 = smf.ols("residu ~ s", dd).fit()
    p(f"    etape 1 : {'modele mixte (1|locuteur)' if ami else 'MCO'} sur les controles seuls")
    p(f"    etape 2 : residu ~ surprise : {fmt(m2)}   R2 = {m2.rsquared:.5f}")
    p(f"    a comparer au modele joint   : {fmt(res['mcomp'] if ami else res['comp'])}")


# ---------------------------------------------------------------------------
#  E. forme du controle de frequence et dernier decile
# ---------------------------------------------------------------------------
def dernier_decile(d, ami, L):
    ctrl = ctrl_de(L, ami)
    dec = pd.qcut(d["s"], 10, labels=False, duplicates="drop")
    d9 = d[dec < dec.max()]
    seuil = d.loc[dec == dec.max(), "s"].min()
    p("  E. Forme du controle de frequence et dernier decile de surprise")
    p(f"    modele complet, tous les mots              : {fmt(smf.ols(f'duree ~ {ctrl} + s', d).fit())}")
    p(f"    + terme quadratique I(freq_zipf**2)        : "
      f"{fmt(smf.ols(f'duree ~ {ctrl} + I(freq_zipf**2) + s', d).fit())}")
    p(f"    sans le 10e decile (s < {seuil:.2f} nats, n = {len(d9)}) : {fmt(smf.ols(f'duree ~ {ctrl} + s', d9).fit())}")
    if ami:
        p(f"    mixte, tous les mots                       : {fmt(mixte(f'duree ~ {ctrl} + s', d))}")
        p(f"    mixte + I(freq_zipf**2)                    : "
          f"{fmt(mixte(f'duree ~ {ctrl} + I(freq_zipf**2) + s', d))}")
        p(f"    mixte sans le 10e decile                   : {fmt(mixte(f'duree ~ {ctrl} + s', d9))}")


# ---------------------------------------------------------------------------
#  F. modele mixte : optimiseurs et ML / REML
# ---------------------------------------------------------------------------
def optimiseurs(d, L):
    f = f"duree ~ {ctrl_de(L, True)} + s"
    p("  F. Modele mixte selon l'optimiseur (lbfgs etait celui du rapport)")
    for reml in (False, True):
        for meth in ("lbfgs", "powell", "nm"):
            mm = mixte(f, d, reml=reml, method=meth)
            vl = float(mm.cov_re.iloc[0, 0])
            degenere = vl < 1e-8 or not np.isfinite(mm.llf)
            p(f"    {'REML' if reml else 'ML  '} {meth:6s}: {fmt(mm)}   var. locuteurs = {vl:.6f}"
              + ("   <- variance nulle, vraisemblance infinie : ajustement a rejeter" if degenere else ""))


# ---------------------------------------------------------------------------
#  H. sensibilite au nettoyage AMI
# ---------------------------------------------------------------------------
def prep_variante(path, col, hesit=True, min4=True):
    """Comme prep() mais on peut garder les hesitations ou les enonces courts."""
    d = pd.read_csv(path)
    d["duree"] = d["offset"] - d["onset"]
    d["content"] = d["POS"].astype(str).str.upper().str[:2].isin(["NN", "VB", "JJ", "RB"]).astype(int)
    d["w"] = d["mot"].astype(str).str.lower()
    u = d.groupby(["meeting", "locuteur", "phrase_id"]).agg(
        n=("w", "size"), t0=("onset", "min"), t1=("offset", "max")).reset_index()
    u["rate"] = u["n"] / (u["t1"] - u["t0"]).clip(lower=1e-3)
    d = d.merge(u[["meeting", "locuteur", "phrase_id", "n", "rate"]],
                on=["meeting", "locuteur", "phrase_id"], how="left")
    if min4:
        d = d[d["n"] >= 4]
    if hesit:
        d = d[~d["w"].isin(HESIT)]
    d["spk"] = d["meeting"].astype(str) + "." + d["locuteur"].astype(str)
    d = d[(d["duree"] > 0) & (d["duree"] <= 2)]
    d = d.dropna(subset=["duree", col, "freq_zipf", "n_letters", "rate"]).copy()
    d["s"] = d[col]
    return d


def sensibilite(path, col, d):
    p("  H. Sensibilite au nettoyage (MCO / modele mixte)")
    variantes = [
        ("nettoyage du rapport", d, True),
        ("sans le debit dans le modele", d, False),
        ("en gardant les hesitations", prep_variante(path, col, hesit=False), True),
        ("sans le seuil de 4 mots", prep_variante(path, col, min4=False), True),
        ("ni hesitations, ni seuil, ni debit", prep_variante(path, col, hesit=False, min4=False), False),
    ]
    for nom, dd, avec_debit in variantes:
        f = f"duree ~ {ctrl_de('n_letters', True, avec_debit)} + s"
        mco, mm = smf.ols(f, dd).fit(), mixte(f, dd)
        p(f"    {nom:34s} (n = {len(dd)})")
        p(f"      MCO   : {fmt(mco)}")
        p(f"      mixte : {fmt(mm)}")


# ---------------------------------------------------------------------------
#  G. memes mots pour les deux mesures (AMI 12 reunions)
# ---------------------------------------------------------------------------
def memes_mots():
    p("=" * 78)
    p("  AMI 12 REUNIONS : SYNTAXE ET GPT-2 SUR EXACTEMENT LES MEMES MOTS")
    p("-" * 78)
    brut = pd.read_csv(AMI)
    manque = brut["surprise_syn"].isna()
    p(f"  {int(manque.sum())} mots du CSV n'ont pas de surprise syntaxique (la grammaire donne une "
      f"probabilite nulle au prefixe), dont : {', '.join(brut.loc[manque, 'mot'].value_counts().index[:4])}")
    cle = ["meeting", "locuteur", "phrase_id", "position"]
    syn = prep(AMI, "surprise_syn")[0][cle]
    for col, nom in (("surprise_syn", "syntaxique"), ("surprise_llm", "GPT-2")):
        d, _, L = prep(AMI, col)
        d = d.merge(syn, on=cle)
        f = f"duree ~ {ctrl_de(L, True)} + s"
        p(f"  {nom:10s} (n = {len(d)}, {d['spk'].nunique()} locuteurs)")
        p(f"    MCO   : {fmt(smf.ols(f, d).fit())}")
        p(f"    mixte : {fmt(mixte(f, d))}")
    p()


def cas(path, col, titre):
    d, ami, L = prep(path, col)
    p("=" * 78)
    p(f"  {titre}   (n = {len(d)} mots" +
      (f", {d['meeting'].nunique()} reunions, {d['spk'].nunique()} locuteurs)" if ami else ")"))
    p("-" * 78)
    nettoyage(path, col, d, ami, L)
    res = modele_complet(d, ami, L)
    deux_temps(d, ami, res)
    dernier_decile(d, ami, L)
    if ami and "171" not in titre:
        optimiseurs(d, L)
    if ami:
        sensibilite(path, col, d)
    p()
    print(f"  {titre} : fait", flush=True)


def main():
    p("")
    p("ANALYSES COMPLEMENTAIRES ET ROBUSTESSE, V2 (surprise en nats, coefficients en s/nat)")
    p("")
    cas(LIVRE, "surprise_syn", "PAROLE LUE  -  SYNTAXIQUE (PCFG)")
    cas(LIVRE, "surprise_llm", "PAROLE LUE  -  GPT-2")
    cas(AMI, "surprise_syn", "AMI 12 REUNIONS  -  SYNTAXIQUE (PCFG)")
    cas(AMI, "surprise_llm", "AMI 12 REUNIONS  -  GPT-2")
    memes_mots()
    cas(AMI171, "surprise_syn", "AMI 171 REUNIONS  -  SYNTAXIQUE (PCFG)")
    txt = "\n".join(out)
    with open(os.path.join(ICI, "resultats_robustesse_v2.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
