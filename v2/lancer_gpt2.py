# -*- coding: utf-8 -*-
"""
Calcule la surprise GPT-2 v2 pour les deux corpus, puis les resultats.

    pip install torch transformers pymupdf nltk scipy numpy wordfreq pandas statsmodels
    python lancer_gpt2.py            # GPT-2 (telecharge au premier lancement)
    python lancer_gpt2.py gpt2-xl    # ou un autre modele Hugging Face / dossier local

Ajoute la colonne surprise_llm (nats) a donnees/surprise_livre_v2.csv et
donnees/surprise_ami_v2.csv, puis ecrit resultats_v2.txt. Avec un autre modele que
gpt2, les fichiers s'appellent ..._v2_<modele>.csv.
Sans GPU, GPT-2 (petit modele) prend quelques minutes.
"""
import os, subprocess, sys

ICI = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ICI, "donnees")
modele = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
suffixe = "" if modele == "gpt2" else "_" + os.path.basename(modele.rstrip("/\\")).replace("/", "-")
livre = os.path.join(D, "surprise_livre_v2.csv")
ami = os.path.join(D, "surprise_ami_v2.csv")
livre_out = livre.replace(".csv", f"{suffixe}.csv")
ami_out = ami.replace(".csv", f"{suffixe}.csv")
py = sys.executable
subprocess.run([py, os.path.join(ICI, "3_surprise_llm_livre.py"), livre,
                os.path.join(ICI, "..", "stimuli_livre", "oldmansea.pdf"), livre_out, modele], check=True)
subprocess.run([py, os.path.join(ICI, "3_surprise_llm_ami.py"), ami, ami_out, modele], check=True)
res = subprocess.run([py, os.path.join(ICI, "4_analyse.py"), livre_out, ami_out],
                     check=True, capture_output=True, text=True).stdout
with open(os.path.join(ICI, f"resultats_v2{suffixe}.txt"), "w", encoding="utf-8") as f:
    f.write(f"Modele de langue : {modele}\n\n" + res)
print(res)
print("Termine. Fichiers a renvoyer :", livre_out, ami_out, sep="\n  ")
