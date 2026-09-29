# -*- coding: utf-8 -*-
"""
Figure de l'EXTRA (figure 5 du rapport).

Pour chaque cas (syntaxique / LLM, parole lue / AMI apparie) :
    observe  = r(duree, surprise)
    attendu  = r(surprise, frequence) x r(duree, frequence)   (lien qui passe seulement par la frequence)
    EXTRA    = observe - attendu
Panneau de gauche : attendu (gris) et observe (couleur) cote a cote, l'ecart est l'EXTRA.
Panneau de droite : l'EXTRA seul.

Les valeurs sont recalculees a partir des CSV, avec le meme nettoyage que
resultats_comparaison.py.

  pip install pandas numpy matplotlib statsmodels
  python figure_extra.py
Sortie : fig_extra.png (et le tableau des valeurs dans le terminal)
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from resultats_comparaison import prep, r, reunions_communes, \
    CSV_SYN_LU, CSV_LLM_LU, CSV_SYN_AMI, CSV_LLM_AMI

BLUE = "#2C4A7C"; RED = "#9E2A2B"; LIGHT = "#B8C4D9"

def valeurs(path, lenc, ami=False):
    d = prep(path, lenc, ami)
    if ami:
        d = d[d["meeting"].isin(reunions_communes())]
    rds = r(d["duree"], d["s"]); rdf = r(d["duree"], d["freq_zipf"]); rsf = r(d["s"], d["freq_zipf"])
    attendu = rsf * rdf
    return rds, attendu, rds - attendu

def main():
    cas = [("Syntaxique\nlue",  CSV_SYN_LU,  "n_phones",  False, BLUE),
           ("LLM\nlue",         CSV_LLM_LU,  "n_phones",  False, RED),
           ("Syntaxique\nAMI",  CSV_SYN_AMI, "n_letters", True,  BLUE),
           ("LLM\nAMI",         CSV_LLM_AMI, "n_letters", True,  RED)]
    res = [(lab, *valeurs(p, l, a), c) for lab, p, l, a, c in cas]

    print(f"{'cas':18s} {'observe':>9s} {'attendu':>9s} {'EXTRA':>8s}")
    for lab, obs, att, ext, _ in res:
        print(f"{lab.replace(chr(10), ' '):18s} {obs:+9.3f} {att:+9.3f} {ext:+8.3f}")

    x = np.arange(len(res)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    fig.suptitle("L'EXTRA : la surprise apporte-t-elle quelque chose au-delà de la fréquence ?",
                 fontsize=14, fontweight="bold")
    att = [a for _, _, a, _, _ in res]; obs = [o for _, o, _, _, _ in res]
    ax[0].bar(x - w/2, att, w, color=LIGHT, label="attendu via la fréquence (r_sf × r_df)")
    ax[0].bar(x + w/2, obs, w, color=[c for *_, c in res], label="observé r(durée, surprise)")
    for i, (_, o, a, e, _) in enumerate(res):
        ax[0].annotate("", xy=(i + w/2, o), xytext=(i + w/2, a),
                       arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
        ax[0].text(i + w/2 + 0.07, max(o, a) + 0.01, f"EXTRA\n{e:+.2f}", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([lab for lab, *_ in res])
    ax[0].set_ylabel("corrélation avec la durée")
    ax[0].set_title("Observé vs attendu via la fréquence"); ax[0].legend(fontsize=8, loc="upper right")

    ext = [e for _, _, _, e, _ in res]
    ax[1].bar(x, ext, 0.55, color=[c for *_, c in res])
    for i, e in enumerate(ext):
        ax[1].text(i, e + (0.005 if e >= 0 else -0.005), f"{e:+.2f}", ha="center",
                   va="bottom" if e >= 0 else "top", fontsize=12)
    ax[1].axhline(0, color="grey", lw=1)
    ax[1].set_ylim(min(ext + [0]) - 0.022, max(ext) + 0.024)
    ax[1].set_xticks(x); ax[1].set_xticklabels([lab for lab, *_ in res])
    ax[1].set_ylabel("EXTRA (effet propre à fréquence contrôlée)")
    ax[1].set_title("L'EXTRA : ce qui survit au contrôle de la fréquence")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig("fig_extra.png", dpi=125); plt.close()
    print("  -> fig_extra.png")

if __name__ == "__main__":
    main()
