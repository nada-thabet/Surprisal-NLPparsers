# -*- coding: utf-8 -*-
"""
KPI CONSTITUANTS (surprisal seule) -- remplace comparer_kpi.py.
Ne lit QUE surprisal_constituants.csv (plus de dependances).
  pip install pandas numpy matplotlib
  python kpi_constituants.py .
"""
import os, sys
import numpy as np
import pandas as pd

folder = sys.argv[1] if len(sys.argv) > 1 else "."
f_con = os.path.join(folder, "surprisal_constituants.csv")
if not os.path.exists(f_con):
    print(f"  Fichier introuvable : {f_con}"); sys.exit(1)

con = pd.read_csv(f_con)
con["surprisal_bits"] = pd.to_numeric(con["surprisal_bits"], errors="coerce")
con["onset"] = pd.to_numeric(con["onset"], errors="coerce")
con["offset"] = pd.to_numeric(con["offset"], errors="coerce")
con["duree"] = con["offset"] - con["onset"]

def section(t): print("\n" + "=" * 56 + f"\n  {t}\n" + "=" * 56)

section("1. COUVERTURE")
print(f"  Fichiers : {con['fichier'].nunique()}")
print(f"  Phrases  : {con.groupby('fichier')['phrase_id'].nunique().sum()}")
print(f"  Mots     : {len(con)}")
if "oov" in con.columns:
    n_oov = int(pd.to_numeric(con["oov"], errors="coerce").fillna(0).sum())
    print(f"  Mots hors-vocabulaire (oov) : {n_oov} ({100*n_oov/len(con):.1f} %)")

section("2. SURPRISAL (statistiques globales)")
s = con["surprisal_bits"].dropna()
stats = {"n": len(s), "moyenne": s.mean(), "ecart_type": s.std(), "min": s.min(),
         "P10": s.quantile(.10), "P25": s.quantile(.25), "mediane": s.median(),
         "P75": s.quantile(.75), "P90": s.quantile(.90), "P95": s.quantile(.95),
         "max": s.max()}
for k, v in stats.items():
    print(f"  {k:<12}: {v:.3f}" if isinstance(v, float) else f"  {k:<12}: {v}")
pd.DataFrame([stats]).to_csv(os.path.join(folder, "kpi_global.csv"), index=False)

section("3. SURPRISAL PAR CATEGORIE GRAMMATICALE (POS)")
gpos = con.groupby("POS")["surprisal_bits"].agg(["count", "mean", "std"]).sort_values("mean", ascending=False)
gpos = gpos[gpos["count"] >= 10]
gpos.to_csv(os.path.join(folder, "kpi_par_POS.csv"))
print(gpos.round(2).to_string())

section("4. CORRELATIONS (Pearson) avec la surprisal")
def corr(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 3: return np.nan, 0
    return np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1], len(d)
for nom, fac in {"duree du mot (s)": con["duree"],
                 "position dans la phrase": con["position"]}.items():
    r, n = corr(con["surprisal_bits"], pd.to_numeric(fac, errors="coerce"))
    print(f"  surprisal ~ {nom:<26}: r = {r:+.3f}  (n={n})")

section("5. TOP 15 mots les plus surprenants")
cols = [c for c in ["fichier", "mot", "POS", "surprisal_bits", "oov"] if c in con.columns]
print(con.sort_values("surprisal_bits", ascending=False).head(15)[cols].to_string(index=False))

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5)); gpos["mean"].plot(kind="bar", color="#4C72B0")
    plt.title("Surprisal moyenne par POS"); plt.ylabel("bits")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(folder, "fig_surprisal_par_POS.png"), dpi=120); plt.close()
    print("\n  Figure fig_surprisal_par_POS.png generee.")
except Exception as e:
    print(f"\n  (figure non generee : {e})")

print("\n  -> kpi_global.csv, kpi_par_POS.csv")