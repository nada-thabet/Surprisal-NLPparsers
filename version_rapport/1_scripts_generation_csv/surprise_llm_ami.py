# -*- coding: utf-8 -*-
"""
Surprise GPT-2 sur AMI (parole spontanee).

Meme principe que la surprise incrementale, mais la probabilite du mot vient d'un
modele de langue neuronal (GPT-2) et pas d'une PCFG :

    surprise_llm(mot) = -log2 P(mot | mots precedents de l'enonce)   selon GPT-2

GPT-2 predit chaque mot a partir du debut de l'enonce. Sa surprise melange donc
lexique, semantique et syntaxe, alors que la PCFG ne voit que la syntaxe : c'est la
mesure que je compare a la surprise syntaxique.

Sortie : surprisal_llm_ami.csv, memes colonnes que surprisal_incrementale_ami.csv
    meeting, locuteur, phrase_id, position, mot, onset, offset, POS,
    surprise_llm, freq_zipf, n_letters

A installer une fois (le modele se telecharge au premier lancement) :
    pip install torch transformers wordfreq nltk

A mettre dans le meme dossier que ami_dataset.py.

Utilisation :
    python surprise_llm_ami.py <dossier des .words.xml>
    python surprise_llm_ami.py <dossier> gpt2-medium        # modele plus gros, plus lent

Sur CPU c'est lent sur tout AMI (~1 M de mots), donc mieux vaut commencer par
5 a 10 fichiers. S'il y a un GPU (CUDA), il est utilise automatiquement.
"""

import os, sys, csv, glob, math


def main():
    # imports lourds, faits seulement ici
    from ami_dataset import read_ami_words, n_letters, SENT_MAX
    import nltk
    for r in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        nltk.download(r, quiet=True)
    try:
        from wordfreq import zipf_frequency
    except Exception:
        zipf_frequency = None
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    folder = sys.argv[1] if len(sys.argv) > 1 else "words"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
    files = sorted(glob.glob(os.path.join(folder, "*.words.xml")))
    if not files:
        print(f"  Aucun *.words.xml dans : {os.path.abspath(folder)}"); return

    print(f"  {len(files)} fichiers AMI. Chargement du LLM '{model_name}'...")
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    bos = tok.eos_token_id                     # sert de contexte pour le 1er mot
    print(f"  Modele pret sur {device}.")

    def word_surprisals(words):
        """Surprise GPT-2 (bits) de chaque mot d'un enonce = somme sur ses tokens."""
        text = " ".join(words)
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)                 # tokens BPE de l'enonce
        offs = enc["offset_mapping"][0].tolist()          # span de caracteres de chaque token
        inp = torch.cat([torch.tensor([[bos]], device=device), ids], dim=1)   # BOS au debut
        with torch.no_grad():
            logits = model(inp).logits
        logp = torch.log_softmax(logits, dim=-1)
        n_tok = ids.shape[1]
        # surprise de chaque token = -log2 P(token | contexte precedent)
        surp = [-(logp[0, t, ids[0, t]].item()) / math.log(2) for t in range(n_tok)]
        # position (en caracteres) de chaque mot dans text
        spans, pos = [], 0
        for w in words:
            st = text.index(w, pos); spans.append((st, st + len(w))); pos = st + len(w)
        ws = [0.0] * len(words)
        # chaque token est attribue au mot qui contient son dernier caractere
        for t, (a, b) in enumerate(offs):
            if b <= a:
                continue                                  # token special (span vide)
            c = b - 1
            for wi, (s0, s1) in enumerate(spans):
                if s0 <= c < s1:
                    ws[wi] += surp[t]; break
        return ws

    out_csv = os.path.join(folder, "surprisal_llm_ami.csv")
    n_words = n_sent = n_skip = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["meeting", "locuteur", "phrase_id", "position", "mot",
                     "onset", "offset", "POS", "surprise_llm", "freq_zipf", "n_letters"])
        for path in files:
            base = os.path.basename(path).replace(".words.xml", "")   # ex: ES2008b.B
            meeting, _, spk = base.partition(".")
            sents = read_ami_words(path)
            print(f"  - {base}: {len(sents)} phrases")
            for sid, sent in enumerate(sents, 1):
                plain = [w for (w, on, off) in sent]
                if not plain or len(plain) > SENT_MAX:
                    n_skip += 1; continue
                try:
                    surps = word_surprisals(plain)
                    tags = [t for (_, t) in nltk.pos_tag([w.lower() for w in plain])]
                except Exception as ex:
                    print(f"      (phrase {sid} ignoree : {ex})"); n_skip += 1; continue
                n_sent += 1
                for i, (w, on, off) in enumerate(sent):
                    fq = round(zipf_frequency(w.lower(), "en"), 3) if zipf_frequency else ""
                    wr.writerow([meeting, spk, sid, i + 1, w.lower(),
                                 f"{on:.3f}", f"{off:.3f}", tags[i], f"{surps[i]:.4f}",
                                 fq, n_letters(w)])
                    n_words += 1
            fcsv.flush()          # sauvegarde apres chaque fichier
    print(f"\n  Termine. {n_words} mots, {n_sent} phrases ({n_skip} ignorees).")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
