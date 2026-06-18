# -*- coding: utf-8 -*-
"""
COMPARAISON MULTI-FACTEURS : constituants (surprisal) vs dependances (structure)
Le corpus dependency_treebank de NLTK ne fournit PAS de labels de relation ;
on exploite donc la STRUCTURE (tete, distance, profondeur).
  pip install pandas numpy matplotlib
  python comparer_kpi.py .
"""
import os, sys
import numpy as np
import pandas as pd

folder = sys.argv[1] if len(sys.argv) > 1 else "."
f_con = os.path.join(folder, "surprisal_constituants.csv")
f_dep = os.path.join(folder, "dependances.csv")
for f in (f_con, f_dep):
    if not os.path.exists(f):
        print(f"  Fichier introuvable : {f}"); sys.exit(1)

con = pd.read_csv(f_con); dep = pd.read_csv(f_dep)
con["onset"] = con["onset"].round(3); dep["onset"] = dep["onset"].round(3)

dep["tete_position"] = np.nan; dep["tete_onset"] = np.nan; dep["profondeur"] = np.nan
for (fic, pid), g in dep.groupby(["fichier", "phrase_id"]):
    mot2pos = {}
    for _, r in g.iterrows():
        mot2pos.setdefault(str(r["mot"]).lower(), r["position"])
    pos2onset = dict(zip(g["position"], g["onset"]))
    pos2head = dict(zip(g["position"], g["tete"]))
    for idx, r in g.iterrows():
        h = str(r["tete"]).lower()
        hp = 0 if h == "root" else mot2pos.get(h, np.nan)
        dep.at[idx, "tete_position"] = hp
        if hp and not np.isnan(hp):
            dep.at[idx, "tete_onset"] = pos2onset.get(hp, np.nan)
        d, cur, seen = 0, r["position"], set()
        while True:
            hh = str(pos2head.get(cur, "root")).lower()
            if hh == "root" or cur in seen: break
            seen.add(cur); cur = mot2pos.get(hh, None)
            if cur is None: break
            d += 1
        dep.at[idx, "profondeur"] = d

dep["dist_pos"] = (dep["position"] - dep["tete_position"]).abs()
dep.loc[dep["tete_position"] == 0, "dist_pos"] = np.nan
dep["dist_temps"] = (dep["onset"] - dep["tete_onset"]).abs()
heads = dep.groupby(["fichier", "phrase_id"])["tete"].apply(
    lambda s: set(str(x).lower() for x in s))
dep["est_tete"] = dep.apply(
    lambda r: str(r["mot"]).lower() in heads.get((r["fichier"], r["phrase_id"]), set()), axis=1)

m = pd.merge(con, dep, on=["fichier", "onset"], how="outer", suffixes=("_con", "_dep"))
m["mot"] = m["mot_con"].fillna(m["mot_dep"]); m["offset"] = m["offset_con"].fillna(m["offset_dep"])
m["duree"] = m["offset"] - m["onset"]
m["surprisal_bits"] = pd.to_numeric(m["surprisal_bits"], errors="coerce")
m.to_csv(os.path.join(folder, "comparaison.csv"), index=False)

both = m.dropna(subset=["surprisal_bits", "tete_position"]).copy()
has_rel = m["relation"].notna().sum() > 0

def section(t): print("\n" + "=" * 60 + f"\n  {t}\n" + "=" * 60)

section("1. COUVERTURE DES DEUX MODELES")
print(f"  Fichiers traites            : {con['fichier'].nunique()}")
print(f"  Mots (constituants)         : {len(con)}")
print(f"  Mots (dependances)          : {len(dep)}")
print(f"  Mots fusionnes (les deux)   : {len(both)}")
print(f"  Relations etiquetees ?      : {'oui' if has_rel else 'non (treebank NLTK sans labels)'}")

section("2. KPI SURPRISAL  (modele constituants)")
s = pd.to_numeric(con["surprisal_bits"], errors="coerce").dropna()
stats = {"n": len(s), "moyenne": s.mean(), "ecart_type": s.std(), "min": s.min(),
         "P10": s.quantile(.10), "P25": s.quantile(.25), "mediane": s.median(),
         "P75": s.quantile(.75), "P90": s.quantile(.90), "P95": s.quantile(.95), "max": s.max()}
for k, v in stats.items():
    print(f"  {k:<12}: {v:.3f}" if isinstance(v, float) else f"  {k:<12}: {v}")
pd.DataFrame([stats]).to_csv(os.path.join(folder, "kpi_global.csv"), index=False)

section("3. SURPRISAL PAR CATEGORIE GRAMMATICALE (POS)")
con["surp"] = pd.to_numeric(con["surprisal_bits"], errors="coerce")
gpos = con.groupby("POS")["surp"].agg(["count", "mean", "std"]).sort_values("mean", ascending=False)
gpos = gpos[gpos["count"] >= 10]; gpos.to_csv(os.path.join(folder, "kpi_par_POS.csv"))
print(gpos.round(2).to_string())

section("4. KPI STRUCTURE DE DEPENDANCE")
print(f"  Distance tete (positions) : moy {dep['dist_pos'].mean():.2f} | med {dep['dist_pos'].median():.2f} | max {dep['dist_pos'].max():.0f}")
print(f"  Distance tete (secondes)  : moy {dep['dist_temps'].mean():.2f}s | med {dep['dist_temps'].median():.2f}s")
print(f"  Profondeur dans l'arbre   : moy {dep['profondeur'].mean():.2f} | max {dep['profondeur'].max():.0f}")
print(f"  Mots qui sont tete d'au moins un autre : {100*dep['est_tete'].mean():.1f} %")
if has_rel:
    rc = dep["relation"].value_counts()
    print("\n  Distribution des relations :")
    print(pd.DataFrame({"n": rc, "pct": (100*rc/len(dep)).round(1)}).head(20).to_string())
else:
    print("  (relations non etiquetees -> pour nsubj/dobj/det..., voir Stanford CoreNLP / spaCy)")

section("5. CROISEMENT  surprisal x distance de dependance")
both["dist_bin"] = pd.cut(both["dist_pos"], [0,1,2,3,5,8,100], labels=["1","2","3","4-5","6-8","9+"])
gb = both.groupby("dist_bin", observed=True)["surprisal_bits"].agg(["count","mean","std"])
print(gb.round(2).to_string())

section("6. CROISEMENT  surprisal : tetes vs dependants")
g = both.groupby("est_tete")["surprisal_bits"].agg(["count","mean","std"])
g.index = ["dependant (feuille)" if not i else "tete (noeud interne)" for i in g.index]
print(g.round(2).to_string())

section("7. CROISEMENT  surprisal x profondeur dans l'arbre")
gp = both.groupby("profondeur")["surprisal_bits"].agg(["count","mean"])
print(gp[gp["count"] >= 10].round(2).to_string())

section("8. CORRELATIONS (Pearson) avec la surprisal")
def corr(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 3: return np.nan, 0
    return np.corrcoef(d.iloc[:,0], d.iloc[:,1])[0,1], len(d)
facteurs = {"distance de dependance (positions)": both["dist_pos"],
            "distance de dependance (temps, s)": both["dist_temps"],
            "duree du mot (s)": both["duree"],
            "profondeur dans l'arbre": both["profondeur"],
            "position dans la phrase": both.get("position_con")}
cor_rows=[]
for nom, fac in facteurs.items():
    if fac is None: continue
    r, n = corr(both["surprisal_bits"], pd.to_numeric(fac, errors="coerce"))
    cor_rows.append({"facteur": nom, "r": round(r,3), "n": n})
    print(f"  surprisal ~ {nom:<36}: r = {r:+.3f}  (n={n})")
pd.DataFrame(cor_rows).to_csv(os.path.join(folder, "kpi_correlations.csv"), index=False)

section("9. TOP 15 mots les plus surprenants (avec tete + distance)")
cols = ["fichier","mot","POS","surprisal_bits","tete","dist_pos"]
top = both.sort_values("surprisal_bits", ascending=False).head(15)
print(top[[c for c in cols if c in top.columns]].to_string(index=False))

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5)); gpos["mean"].plot(kind="bar", color="#4C72B0")
    plt.title("Surprisal moyenne par POS"); plt.ylabel("bits")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(folder,"fig_surprisal_par_POS.png"), dpi=120); plt.close()
    plt.figure(figsize=(8,5)); gb["mean"].plot(kind="bar", color="#4C72B0")
    plt.title("Surprisal vs distance de dependance"); plt.ylabel("surprisal (bits)")
    plt.xlabel("distance mot <-> tete"); plt.tight_layout()
    plt.savefig(os.path.join(folder,"fig_surprisal_vs_distance.png"), dpi=120); plt.close()
    print("\n  Figures .png generees.")
except Exception as e:
    print(f"\n  (figures non generees : {e})")

print("\n  -> comparaison.csv, kpi_global.csv, kpi_par_POS.csv, kpi_correlations.csv")