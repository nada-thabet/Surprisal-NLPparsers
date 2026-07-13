 # -*- coding: utf-8 -*-
"""
Comparaison automatique de la segmentation prosodique au TEXTE ORIGINAL.

Aligne le flux de mots segmente (phrases_prosodiques.txt) sur le texte du livre
(The Old Man and the Sea) via difflib, puis evalue l'accord des FRONTIERES de phrases
(precision / rappel / F1). Ecrit aussi un fichier de controle qui montre, phrase par
phrase, si la frontiere predite correspond a une vraie fin de phrase du livre.

  pip install pdfplumber         
  python comparer_segmentation.py phrases_prosodiques.txt oldmansea.pdf 2   # tolerance 2 mots

Sortie : accord_segmentation.txt
"""
import re, sys, difflib


def norm(w):
    return re.sub(r'[^a-z0-9]', '', w.lower())


def load_pred(path):
    """phrases_prosodiques.txt -> (mots, frontiere_apres_ce_mot)."""
    words, bound = [], []
    for line in open(path, encoding='utf-8', errors='ignore'):
        m = re.match(r'\s*\[\s*\d+\]\s*\(\s*\d+\s*mots?\)\s*(.*)', line)
        if not m:
            continue
        toks = [t for t in (norm(x) for x in m.group(1).split()) if t]
        for i, t in enumerate(toks):
            words.append(t); bound.append(i == len(toks) - 1)
    return words, bound


def read_text(path):
    if path.lower().endswith('.pdf'):
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join((pg.extract_text() or "") for pg in pdf.pages)
    return open(path, encoding='utf-8', errors='ignore').read()


def load_ref(path):
    """Texte original -> (mots, frontiere) en decoupant sur . ! ?"""
    txt = re.sub(r'\s+', ' ', read_text(path))
    sents = re.split(r'(?<=[.!?])\s+', txt)
    words, bound = [], []
    for s in sents:
        toks = [t for t in (norm(x) for x in s.split()) if t]
        for i, t in enumerate(toks):
            words.append(t); bound.append(i == len(toks) - 1)
    return words, bound


def evaluate(ref_w, ref_b, pred_w, pred_b, tol=1):
    sm = difflib.SequenceMatcher(a=ref_w, b=pred_w, autojunk=False)
    ref2pred, pred2ref = {}, {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                ref2pred[i1 + k] = j1 + k
                pred2ref[j1 + k] = i1 + k
    if not ref2pred:
        print("  Aucun alignement trouve : verifie que c'est bien le bon texte.")
        return None
    lo, hi = min(ref2pred), max(ref2pred)          # zone couverte par les audios

    ref_bnds = [i for i in range(lo, hi + 1) if ref_b[i] and i in ref2pred]

    def pred_bound_near(jp):
        return any(0 <= jp + d < len(pred_b) and pred_b[jp + d] for d in range(-tol, tol + 1))

    rec_hits = sum(1 for i in ref_bnds if pred_bound_near(ref2pred[i]))

    pred_bnds = [j for j in range(len(pred_b)) if pred_b[j] and j in pred2ref
                 and lo <= pred2ref[j] <= hi]

    def ref_bound_near(ir):
        return any(0 <= ir + d < len(ref_b) and ref_b[ir + d] for d in range(-tol, tol + 1))

    prec_hits = sum(1 for j in pred_bnds if ref_bound_near(pred2ref[j]))

    R = rec_hits / len(ref_bnds) if ref_bnds else 0.0
    P = prec_hits / len(pred_bnds) if pred_bnds else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0

    print("\n" + "=" * 56)
    print(f"  Mots alignes (zone couverte)  : {len(ref2pred)}")
    print(f"  Frontieres du livre           : {len(ref_bnds)}")
    print(f"  Frontieres predites           : {len(pred_bnds)}")
    print(f"  Precision (predites correctes): {P:.3f}")
    print(f"  Rappel    (livre retrouvees)  : {R:.3f}")
    print(f"  F1                            : {F:.3f}  (tolerance +/-{tol} mot)")
    print("=" * 56)
    return {"P": P, "R": R, "F": F, "ref2pred": ref2pred,
            "ref_bnds": ref_bnds, "pred_bnds": pred_bnds}


def main():
    if len(sys.argv) < 3:
        print("  Usage : python comparer_segmentation.py phrases_prosodiques.txt oldmansea.txt [tol]")
        return
    pred_w, pred_b = load_pred(sys.argv[1])
    ref_w, ref_b = load_ref(sys.argv[2])
    tol = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    print(f"  Predits : {len(pred_w)} mots | Reference : {len(ref_w)} mots")
    res = evaluate(ref_w, ref_b, pred_w, pred_b, tol)
    if res:
        with open("accord_segmentation.txt", "w", encoding="utf-8") as f:
            f.write(f"Precision={res['P']:.3f}  Rappel={res['R']:.3f}  F1={res['F']:.3f}\n")
            f.write(f"Frontieres livre={len(res['ref_bnds'])}  "
                    f"predites={len(res['pred_bnds'])}\n")
        print("  -> accord_segmentation.txt")


if __name__ == "__main__":
    main()
