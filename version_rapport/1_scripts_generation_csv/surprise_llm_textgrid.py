# -*- coding: utf-8 -*-
"""
Surprise GPT-2 sur la parole lue (TextGrid de The Old Man and the Sea).

Meme calcul que pour AMI (surprise_llm_ami.py), mais sur les .TextGrid, avec les
phrases de ma segmentation prosodique (constituants_dataset.load_sentences).
Pour chaque mot :

    surprise_llm(mot) = -log2 P(mot | mots precedents de la phrase)   selon GPT-2

Sortie : surprisal_llm_textgrid.csv, memes colonnes que surprisal_incrementale.csv
    fichier, phrase_id, position, mot, onset, offset, POS,
    surprise_llm, freq_zipf, n_phones

A mettre dans le meme dossier que constituants_dataset.py et segmentation_prosodique.py.

    pip install torch transformers wordfreq nltk
    python surprise_llm_textgrid.py .                       # dossier des .TextGrid
    python surprise_llm_textgrid.py . C:\\Users\\user\\gpt2_local   # modele en local (probleme SSL)
"""

import os, re, sys, csv, glob, math


def main():
    # segmentation prosodique et variables de controle
    from constituants_dataset import load_sentences, count_phones, word_freq
    import nltk
    for r in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        nltk.download(r, quiet=True)
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    arg = sys.argv[1] if len(sys.argv) > 1 else "."
    model_name = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
    if os.path.isdir(arg):
        folder = arg
        files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                       if re.search(r'\.textgrid$', f, re.I)
                       and not os.path.basename(f).startswith("._"))
    else:
        folder = os.path.dirname(arg) or "."
        files = [arg] if os.path.exists(arg) else []
    if not files:
        print(f"  Aucun .TextGrid trouve : {os.path.abspath(arg)}"); return

    print(f"  {len(files)} fichier(s). Chargement du LLM '{model_name}'...")
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    bos = tok.eos_token_id
    print(f"  Modele pret sur {device}.")

    def word_surprisals(words):
        """Surprise GPT-2 (bits) de chaque mot = somme sur ses tokens."""
        text = " ".join(words)
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        offs = enc["offset_mapping"][0].tolist()
        inp = torch.cat([torch.tensor([[bos]], device=device), ids], dim=1)
        with torch.no_grad():
            logits = model(inp).logits
        logp = torch.log_softmax(logits, dim=-1)
        n_tok = ids.shape[1]
        surp = [-(logp[0, t, ids[0, t]].item()) / math.log(2) for t in range(n_tok)]
        spans, pos = [], 0
        for w in words:
            st = text.index(w, pos); spans.append((st, st + len(w))); pos = st + len(w)
        ws = [0.0] * len(words)
        for t, (a, b) in enumerate(offs):
            if b <= a:
                continue
            c = b - 1
            for wi, (s0, s1) in enumerate(spans):
                if s0 <= c < s1:
                    ws[wi] += surp[t]; break
        return ws

    out_csv = os.path.join(folder, "surprisal_llm_textgrid.csv")
    n_words = n_sent = n_skip = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["fichier", "phrase_id", "position", "mot", "onset", "offset",
                     "POS", "surprise_llm", "freq_zipf", "n_phones"])
        for path in files:
            name = os.path.basename(path)
            sents, phones = load_sentences(path)      # segmentation prosodique
            print(f"  - {name}: {len(sents)} phrases")
            for sid, sent in enumerate(sents, 1):
                plain = [w for (w, on, off) in sent]
                if not plain:
                    n_skip += 1; continue
                try:
                    surps = word_surprisals(plain)
                    tags = [t for (_, t) in nltk.pos_tag([w.lower() for w in plain])]
                except Exception as ex:
                    print(f"      (phrase {sid} ignoree : {ex})"); n_skip += 1; continue
                n_sent += 1
                for i, (w, on, off) in enumerate(sent):
                    wr.writerow([name, sid, i + 1, w.lower(),
                                 "" if on is None else f"{on:.3f}",
                                 "" if off is None else f"{off:.3f}",
                                 tags[i], f"{surps[i]:.4f}",
                                 word_freq(w), count_phones(phones, on, off)])
                    n_words += 1
            fcsv.flush()
    print(f"\n  Termine. {n_words} mots, {n_sent} phrases ({n_skip} ignorees).")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
