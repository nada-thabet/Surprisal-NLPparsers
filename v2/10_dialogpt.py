# -*- coding: utf-8 -*-
"""
Etape 10 : le test avec un modele conversationnel (DialoGPT) sur AMI.

Dans le rapport (sections 3.5 et 5.2), le coefficient negatif de GPT-2 sur AMI est
attribue au domaine d'entrainement de GPT-2 (du texte ecrit, pas de la parole
spontanee), et un test avec DialoGPT (GPT-2 reentraine sur des conversations
Reddit) allait dans ce sens. Ici je reprends ce test avec la methode v2 : meme
nettoyage, memes modeles de duree.

DialoGPT a ete entraine sur des echanges courts, et il devient beaucoup moins bon
quand on lui donne toute la reunion : avec une fenetre de 1024 tokens sa surprise
moyenne monte a plus de 9 nats (4,2 pour GPT-2), et meme des suites tres
previsibles ("would be") deviennent surprenantes. J'ai donc essaye plusieurs tailles
de fenetre (option --fenetre) sur deux reunions (ES2002a et ES2003b, mots composes
sans tiret) et garde pour chaque modele celle qui donne la surprise moyenne la plus
basse, c'est-a-dire le modele qui predit le mieux le texte :
  DialoGPT-small  : un enonce par ligne, fenetre de 64 tokens (5,26 nats ; 5,46 avec
                    32, 5,60 avec 128, 9,79 avec 256) ;
  DialoGPT-medium : tours separes par le jeton de fin de texte, fenetre de 32 tokens
                    (5,41 nats ; 6,06 avec 64, 6,33 avec 128 ; 5,57 sans les tours).
GPT-2 est meilleur avec la fenetre la plus longue (4,28 nats avec 1024, 4,83 avec 64).

Les mots composes : les textes d'entrainement de DialoGPT n'ont presque pas de
tirets, et il donne des surprises enormes aux mots ecrits avec un tiret (flip-top,
jog-dial) : 73 nats en moyenne (de 56 a 227) pour DialoGPT-small, 44 pour
DialoGPT-medium, contre 14 pour GPT-2. Ces 138 mots analyses sont longs, et a eux
seuls ils aplatissent la pente de DialoGPT. Tous les fichiers de la comparaison
sont donc calcules avec l'option --sans-tiret ("flip top"), pour GPT-2 comme pour
DialoGPT. Un seul fichier garde les tirets, DialoGPT-small enonce par enonce, pour
retrouver les conditions du test fait pendant le stage.

Comme la fenetre change avec le modele, GPT-2 est aussi recalcule avec 64 tokens :
la comparaison GPT-2 / DialoGPT-small a 64 tokens ne change alors que les donnees
d'entrainement (meme taille de modele, 124 millions de parametres, meme contexte,
meme texte). J'ajoute aussi la lecture enonce par enonce, sans ce qui precede, comme
dans la version du rapport, et DialoGPT-small avec 1024 tokens pour montrer ce que
donne un contexte trop long.

Les surprises sont calculees par 3_surprise_llm_ami.py (dossier donnees/dialogpt/),
depuis v2/ avec E=donnees/surprise_ami_v2.csv et O=donnees/dialogpt :
    python 3_surprise_llm_ami.py $E $O/gpt2_fenetre64.csv gpt2 --fenetre 64 --sans-tiret
    python 3_surprise_llm_ami.py $E $O/DialoGPT-small_fenetre64.csv microsoft/DialoGPT-small --fenetre 64 --sans-tiret
    python 3_surprise_llm_ami.py $E $O/DialoGPT-medium_tours_fenetre32.csv microsoft/DialoGPT-medium --tours --fenetre 32 --sans-tiret
    python 3_surprise_llm_ami.py $E $O/gpt2_enonce.csv gpt2 --enonce --sans-tiret
    python 3_surprise_llm_ami.py $E $O/DialoGPT-small_enonce.csv microsoft/DialoGPT-small --enonce --sans-tiret
    python 3_surprise_llm_ami.py $E $O/DialoGPT-small_enonce_tirets.csv microsoft/DialoGPT-small --enonce
    python 3_surprise_llm_ami.py $E $O/DialoGPT-small_tours_fenetre1024.csv microsoft/DialoGPT-small --tours --sans-tiret

Ce script reprend, pour chaque fichier, toutes les lignes de 5_resultats_complets.py,
puis un tableau recapitulatif, GPT-2 et DialoGPT-small mis ensemble dans le meme
modele (pour voir ce que DialoGPT apporte en plus de GPT-2) et deux figures en deciles.

    pip install pandas numpy statsmodels matplotlib
    python 10_dialogpt.py
Sorties : resultats_dialogpt_v2.txt, figures/cmp_ami_dialogpt.png (GPT-2 contre
DialoGPT-small, fenetre de 64 tokens) et figures/cmp_ami_dialogpt_enonce.png (les
memes, enonce par enonce)
"""
import importlib.util, os, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ICI = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ICI, "donnees")

# (modele, lecture, fichier dans donnees/) ; tous sans tiret sauf le fichier *_tirets
MODELES = [
    ("GPT-2", "reunion en continu, fenetre 1024 (valeurs v2)", "surprise_ami_v2.csv"),
    ("GPT-2", "reunion en continu, fenetre 64", "dialogpt/gpt2_fenetre64.csv"),
    ("DialoGPT-small", "reunion en continu, fenetre 64", "dialogpt/DialoGPT-small_fenetre64.csv"),
    ("DialoGPT-medium", "tours separes, fenetre 32", "dialogpt/DialoGPT-medium_tours_fenetre32.csv"),
    ("GPT-2", "enonce par enonce", "dialogpt/gpt2_enonce.csv"),
    ("DialoGPT-small", "enonce par enonce", "dialogpt/DialoGPT-small_enonce.csv"),
    ("DialoGPT-small", "enonce par enonce, avec les tirets (comme le rapport)",
     "dialogpt/DialoGPT-small_enonce_tirets.csv"),
    ("DialoGPT-small", "tours separes, fenetre 1024", "dialogpt/DialoGPT-small_tours_fenetre1024.csv"),
]

def charger_module(nom, fichier):
    spec = importlib.util.spec_from_file_location(nom, os.path.join(ICI, fichier))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


res5 = charger_module("res", "5_resultats_complets.py")
fig6 = charger_module("fig", "6_figures.py")
prep, fmt, analyse = res5.prep, res5.fmt, res5.analyse


def main():
    out = ["", "AMI (12 REUNIONS) : GPT-2 CONTRE DIALOGPT, V2 (surprise en nats, coefficients en s/nat)", ""]
    ref = pd.read_csv(os.path.join(D, "surprise_ami_v2.csv"))["surprise_llm"]
    recap, fichiers = [], {}
    for nom, lecture, f in MODELES:
        chemin = os.path.join(D, f)
        if not os.path.exists(chemin):
            print(f"  {f} absent, saute"); continue
        fichiers[(nom, lecture)] = chemin
        analyse(chemin, "surprise_llm", f"AMI  -  {nom}, {lecture}", out)
        d, _, L = prep(chemin, "surprise_llm")
        ctrl = f"{L} * freq_zipf + content + rate"
        mm = smf.mixedlm(f"duree ~ {ctrl} + s", d, groups=d["spk"]).fit(reml=False, method="powell")
        rds, rdf, rsf = (np.corrcoef(d[a], d[b])[0, 1] for a, b in
                         (("duree", "s"), ("duree", "freq_zipf"), ("s", "freq_zipf")))
        brut = pd.read_csv(chemin)["surprise_llm"]
        recap.append((nom, lecture, len(d), d["s"].mean(), brut.corr(ref), rsf, rds - rsf * rdf, mm))
        print(f"  {nom}, {lecture} : fait", flush=True)
    out.append("=" * 78)
    out.append("  RECAPITULATIF (modele mixte (1|locuteur), controles complets)")
    out.append("  surprise moyenne : plus elle est basse, mieux le modele predit les mots d'AMI")
    out.append("-" * 78)
    for nom, lecture, n, moy, rg, rsf, extra, mm in recap:
        out.append(f"  {nom}, {lecture}")
        out.append(f"      n = {n}   surprise moyenne = {moy:.2f} nats   r avec GPT-2 v2 = {rg:.3f}   "
                   f"r(surprise, freq) = {rsf:+.3f}   EXTRA = {extra:+.3f}")
        out.append(f"      coefficient : {fmt(mm)}")
    # les deux surprises dans le meme modele : le coefficient de DialoGPT est alors
    # celui de la partie de sa surprise qui n'est pas deja dans GPT-2
    out.append("")
    out.append("=" * 78)
    out.append("  GPT-2 ET DIALOGPT-SMALL DANS LE MEME MODELE (mixte, controles complets)")
    out.append("-" * 78)
    for lecture in ("reunion en continu, fenetre 64", "enonce par enonce"):
        cles = [("GPT-2", lecture), ("DialoGPT-small", lecture)]
        if not all(c in fichiers for c in cles):
            continue
        d, _, L = prep(fichiers[cles[0]], "surprise_llm")
        # prep garde le numero de ligne du CSV, les deux fichiers ont les memes lignes
        d["s_dialo"] = pd.read_csv(fichiers[cles[1]])["surprise_llm"].loc[d.index].values
        d = d.dropna(subset=["s_dialo"])
        mm = smf.mixedlm(f"duree ~ {L} * freq_zipf + content + rate + s + s_dialo", d,
                         groups=d["spk"]).fit(reml=False, method="powell")
        out.append(f"  {lecture}   (n = {len(d)}, r entre les deux surprises = {d['s'].corr(d['s_dialo']):.3f})")
        out.append(f"      GPT-2          : {fmt(mm, 's')}")
        out.append(f"      DialoGPT-small : {fmt(mm, 's_dialo')}   converge = {mm.converged}")
    txt = "\n".join(out)
    with open(os.path.join(ICI, "resultats_dialogpt_v2.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)

    # figures : duree apres controle par decile de surprise, GPT-2 contre DialoGPT-small
    ca = "duree ~ n_letters*freq_zipf + content + rate"
    for lecture, nom_fig, titre in (
            ("reunion en continu, fenetre 64", "cmp_ami_dialogpt.png",
             "AMI : durée APRÈS CONTRÔLE, même contexte (64 tokens)"),
            ("enonce par enonce", "cmp_ami_dialogpt_enonce.png",
             "AMI : durée APRÈS CONTRÔLE, énoncé par énoncé")):
        cles = [("GPT-2", lecture), ("DialoGPT-small", lecture)]
        if all(c in fichiers for c in cles):
            specs = [(fig6.res(fig6.charger(fichiers[c], "surprise_llm"), ca), c[0]) for c in cles]
            fig6.deux_panneaux(specs, "r", fig6.RED, nom_fig, titre, (-0.035, 0.035), True)


if __name__ == "__main__":
    main()
