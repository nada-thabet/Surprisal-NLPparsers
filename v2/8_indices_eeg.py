# -*- coding: utf-8 -*-
"""
Etape 8 : les indices de surprise pour l'equipe EEG (section 5.4 du rapport),
recalcules avec les surprises v2 (en nats), pour The Old Man and the Sea.

Meme calcul que version_rapport/5_indices_eeg/ :

  *_ctrlfreq         residu de la regression surprise ~ frequence (freq_zipf) :
                     la part de la surprise que la frequence ne predit pas
                     (surprise_apres_controle.py)
  *_apres_controle   residu de la regression
                     surprise ~ longueur + frequence + longueur x frequence + classe
                     (longueur et frequence centrees, classe = mot de contenu ou
                     mot-outil) (surprise_apres_controle_total.py)

Les quatre indices du rapport sont la surprise syntaxique et GPT-2, avec et sans
controle de la frequence : ce sont les colonnes surprise_syn, surprise_llm et les
deux fichiers *_ctrlfreq. Les fichiers *_apres_controle sont la version avec tous
les controles.

Un residu a une moyenne nulle par construction, donc environ la moitie des valeurs
sont negatives. Les surprises elles-memes ne sont jamais negatives.
Tous les mots sont gardes (pas de filtre sur la duree) ; un mot sans surprise
garde une case vide.

    pip install pandas numpy
    python 8_indices_eeg.py
Sortie : dossier indices_eeg/ (quatre CSV)
"""
import os
import numpy as np
import pandas as pd

ICI = os.path.dirname(os.path.abspath(__file__))
LIVRE = os.path.join(ICI, "donnees", "surprise_livre_v2.csv")
OUT = os.path.join(ICI, "indices_eeg")
COLS = ["fichier", "phrase_id", "position", "mot", "onset", "offset", "POS"]


def residu(d, col, controles):
    """Residu de la regression de d[col] sur les colonnes `controles` (avec constante)."""
    ok = d[[col] + controles].notna().all(axis=1)
    X = np.column_stack([np.ones(ok.sum())] + [d.loc[ok, c].to_numpy(float) for c in controles])
    y = d.loc[ok, col].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = pd.Series(np.nan, index=d.index)
    res[ok] = y - X @ beta
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return res, beta, r2, ok


def ecrire(texte, col, nom_col, res, fichier):
    """Ecrit les colonnes d'origine telles quelles (texte) et le residu a 6 decimales."""
    t = texte[COLS + [col, "freq_zipf", "n_phones"]].copy()
    t[nom_col] = [f"{v:.6f}" if pd.notna(v) else "" for v in res]
    t.to_csv(os.path.join(OUT, fichier), index=False)
    print(f"  ecrit : indices_eeg/{fichier}  ({len(t)} lignes, {int(res.notna().sum())} valeurs)")


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(LIVRE)
    texte = pd.read_csv(LIVRE, dtype=str, keep_default_na=False)
    d["contenu"] = d["POS"].astype(str).str.upper().str[:2].isin(["NN", "VB", "JJ", "RB"]).astype(float)
    for col, court in (("surprise_syn", "syntaxique"), ("surprise_llm", "llm")):
        abr = "syn" if col == "surprise_syn" else "llm"
        print(f"[{col}]")
        # controle de la frequence seule
        res, beta, r2, ok = residu(d, col, ["freq_zipf"])
        print(f"  frequence seule : {col} = {beta[0]:+.4f} {beta[1]:+.4f} x freq_zipf   R2 = {r2:.4f}")
        print(f"    moyenne des residus {res.mean():+.1e}, r(residu, frequence) = "
              f"{np.corrcoef(res[ok], d.loc[ok, 'freq_zipf'])[0, 1]:+.1e}")
        ecrire(texte, col, f"surprise_{abr}_ctrlfreq", res, f"surprise_{court}_ctrlfreq_v2.csv")
        # controle complet : longueur x frequence + classe, variables continues centrees
        ok = d[[col, "freq_zipf", "n_phones"]].notna().all(axis=1)
        d["Lc"] = d["n_phones"] - d.loc[ok, "n_phones"].mean()
        d["Fc"] = d["freq_zipf"] - d.loc[ok, "freq_zipf"].mean()
        d["LxF"] = d["Lc"] * d["Fc"]
        res, beta, r2, ok = residu(d, col, ["Lc", "Fc", "LxF", "contenu"])
        print(f"  controle complet : constante {beta[0]:+.4f} | longueur {beta[1]:+.4f} | frequence "
              f"{beta[2]:+.4f} | interaction {beta[3]:+.4f} | contenu {beta[4]:+.4f}   R2 = {r2:.4f}")
        print(f"    moyenne des residus {res.mean():+.1e}, r(residu, frequence / longueur / contenu) = " +
              " / ".join(f"{np.corrcoef(res[ok], d.loc[ok, c])[0, 1]:+.1e}"
                         for c in ("freq_zipf", "n_phones", "contenu")))
        ecrire(texte, col, f"surprise_{abr}_apres_controle", res, f"surprise_{court}_apres_controle_v2.csv")
        print()


if __name__ == "__main__":
    main()
