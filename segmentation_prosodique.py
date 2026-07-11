# -*- coding: utf-8 -*-
"""
SEGMENTEUR PROSODIQUE POUR L'ORAL  --  dataset Broderick ds004408 (TextGrid)

"""

import os, re, sys, glob, math, statistics

# ---- parametres reglables --------------------------------------------------
PAUSE_MIN  = 0.45    # pause absolue (s) >= ce seuil -> frontiere (baisse->coupe plus)
THR_K      = 2.0     # OU score combine >= moyenne + THR_K * ecart-type (baisse->coupe plus)
MIN_LEN    = 3       # on ne coupe pas avant d'avoir ce nb de mots (evite les miettes)
SENT_MAX   = 45      # garde-fou : coupe si trop long (Viterbi est en O(n^3))
W_PAUSE    = 1.0     # poids de la pause dans le score combine
W_LENG     = 1.0     # poids de l'allongement final
USE_POS    = True    # contrainte morphosyntaxique (retour Philippe)
SIL        = {'sil', 'sp', 'spn', '', '<p>', 'br', 'noise', 'sps'}

# POS (Penn Treebank) sur lesquels une phrase ne peut PAS SE TERMINER : mots-outils
# qui appellent un complement a droite. Frontiere apres -> deplacee AVANT le mot.
# Corrige "... a fish in" ou "... flour sacks and".
NO_END = {'IN',    # preposition / subordonnant (in, of, that...)
          'TO',    # to
          'DT',    # determinant (the, a...)
          'PDT',   # predeterminant (all, both...)
          'CC',    # conjonction de coordination (and, or, but)
          'PRP$',  # possessif (his, their...)
          'POS',   # 's possessif
          'WDT',   # which, that (relatif)
          'WP', 'WP$',  # who, whose
          'MD',    # modal (can, will...)
          'RP',    # particule (up, off... appartient au verbe)
          'EX'}    # there existentiel

# Symetrique : une phrase ne peut PAS COMMENCER par un mot orphelin qui appartient
# a la phrase precedente. Frontiere ouvrant sur ce mot -> deplacee APRES lui.
# Corrige "at | and the steady weather" et "up | i could go".
NO_START = {'RP'}    # particule verbale (up, off, out...)


# ---------------------------------------------------------------------------
#  LECTURE DES DEUX TIERS DU TEXTGRID
# ---------------------------------------------------------------------------

def _read_tier(txt, names):
    """Renvoie [(label, xmin, xmax)] du premier tier dont le nom est dans `names`."""
    items = re.split(r'item\s*\[\d+\]\s*:', txt)
    block = None
    for it in items:
        m = re.search(r'name\s*=\s*"([^"]*)"', it)
        if m and m.group(1).strip().lower() in names:
            block = it
            break
    if block is None:
        return None
    out = []
    for iv in re.finditer(
            r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"', block):
        on, off, lab = float(iv.group(1)), float(iv.group(2)), iv.group(3).strip()
        out.append((lab, on, off))
    return out


def read_words_and_phones(path):
    txt = open(path, encoding='utf-8', errors='ignore').read()
    raw_w = _read_tier(txt, {'words', 'word'})
    raw_p = _read_tier(txt, {'phones', 'phone', 'phon'})
    if raw_w is None:
        return None, None
    words = [(w, on, off) for (w, on, off) in raw_w
             if w and w.lower() not in SIL]
    phones = None
    if raw_p is not None:
        phones = [(p, on, off) for (p, on, off) in raw_p
                  if p and p.lower() not in SIL]
    return words, phones


# ---------------------------------------------------------------------------
#  CALCUL DES INDICES PROSODIQUES
# ---------------------------------------------------------------------------

def _zstats(values):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0, 1.0
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals) or 1.0
    return mu, sd


def _last_phone_of_word(phones, on, off):
    """Duree du dernier phone (non silence) contenu dans [on, off], et son label."""
    inside = [(p, po, pf) for (p, po, pf) in phones
              if pf > on + 1e-6 and po < off - 1e-6]      # chevauche l'intervalle mot
    if not inside:
        return None, None
    p, po, pf = inside[-1]
    return p.lower(), pf - po


def compute_cues(words, phones):
    """Renvoie, pour chaque mot i, (pause_after_i, dur_last_phone_i, label_last_phone_i)."""
    n = len(words)
    pause, leng, labels = [], [], []
    for i, (w, on, off) in enumerate(words):
        if i + 1 < n:
            pa = max(0.0, words[i + 1][1] - off)
        else:
            pa = None
        pause.append(pa)
        if phones:
            lab, d = _last_phone_of_word(phones, on, off)
        else:
            lab, d = None, (off - on)      # sans phones : duree du mot entier
        labels.append(lab)
        leng.append(d)
    return pause, leng, labels


def boundary_scores(words, phones):
    """Score de frontiere apres chaque mot (plus c'est grand, plus c'est une fin de phrase)."""
    pause, leng, labels = compute_cues(words, phones)
    mu_p, sd_p = _zstats(pause)                       # z-score pause (global)
    by_lab = {}                                       # z-score allongement PAR phone
    for lab, d in zip(labels, leng):
        if d is not None:
            by_lab.setdefault(lab, []).append(d)
    lab_stats = {lab: _zstats(v) for lab, v in by_lab.items()}
    scores = []
    for pa, d, lab in zip(pause, leng, labels):
        z_p = 0.0 if pa is None else (pa - mu_p) / sd_p
        if d is None or lab not in lab_stats:
            z_l = 0.0
        else:
            mu_l, sd_l = lab_stats[lab]
            z_l = (d - mu_l) / sd_l
        scores.append((W_PAUSE * z_p + W_LENG * z_l, pa))
    return scores      # liste de (score, pause_en_secondes)


# ---------------------------------------------------------------------------
#  INFORMATION MORPHOSYNTAXIQUE  (POS, retour Philippe)
# ---------------------------------------------------------------------------

def pos_tags(words):
    """Etiquette POS (Penn Treebank) via NLTK. None si NLTK indisponible."""
    try:
        import nltk
        for r in ('averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'):
            nltk.download(r, quiet=True)
        return [t for (_, t) in nltk.pos_tag([w.lower() for w in words])]
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  SEGMENTATION  (prosodie + contrainte morphosyntaxique)
# ---------------------------------------------------------------------------

def segment_prosodic(words, phones,
                     pause_min=PAUSE_MIN, thr_k=THR_K, min_len=MIN_LEN,
                     sent_max=SENT_MAX, use_pos=USE_POS):
    """words = [(mot, on, off)] -> liste de phrases [(mot, on, off)].

    1) PROSODIE : frontiere candidate si pause >= pause_min OU score combine eleve.
    2) MORPHOSYNTAXE : (a) pas de fin sur un mot-outil (NO_END) -> deplace a gauche ;
                       (b) pas de debut sur un mot orphelin (NO_START) -> deplace a droite.
    3) garde-fous de longueur (min_len, sent_max). Aucune longueur imposee.
    """
    n = len(words)
    if n == 0:
        return []
    scores = boundary_scores(words, phones)
    vals = [s for s, _ in scores[:-1]]
    mu, sd = _zstats(vals)
    thr = mu + thr_k * sd

    # 1) frontieres candidates (prosodie)
    cut = [False] * n
    for i in range(n - 1):
        score, pause = scores[i]
        if (pause is not None and pause >= pause_min) or (score >= thr):
            cut[i] = True

    # 2) contrainte morphosyntaxique
    if use_pos:
        tags = pos_tags([w for (w, on, off) in words])
        if tags and len(tags) == n:
            # 2a) pas de frontiere APRES un mot-outil -> deplace a gauche
            for i in range(n - 1):
                if cut[i] and tags[i] in NO_END:
                    cut[i] = False
                    if i - 1 >= 0:
                        cut[i - 1] = True
            # 2b) pas de phrase qui COMMENCE par un mot orphelin -> deplace a droite
            for i in range(n - 1):
                if cut[i]:
                    j = i + 1
                    stranded = (tags[j] in NO_START or
                                (tags[j] == 'IN' and j + 1 < n and tags[j + 1] == 'CC'))
                    if stranded:
                        cut[i] = False
                        cut[j] = True

    # 3) construction avec garde-fous de longueur
    sents, cur = [], []
    for i, (w, on, off) in enumerate(words):
        cur.append((w, on, off))
        if i == n - 1:
            sents.append(cur); cur = []
        elif len(cur) >= sent_max:
            sents.append(cur); cur = []
        elif cut[i] and len(cur) >= min_len:
            sents.append(cur); cur = []
    if cur:
        sents.append(cur)
    return sents


def segment_textgrid_file(path, **kw):
    words, phones = read_words_and_phones(path)
    if words is None:
        return None, None, None
    sents = segment_prosodic(words, phones, **kw)
    return sents, words, phones


# ---------------------------------------------------------------------------
#  MAIN autonome : inspecter la qualite de la segmentation
# ---------------------------------------------------------------------------

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "."
    if os.path.isdir(arg):
        files = sorted(f for f in glob.glob(os.path.join(arg, "*"))
                       if re.search(r'\.textgrid$', f, re.I)
                       and not os.path.basename(f).startswith("._"))
    else:
        files = [arg]
    if not files:
        print(f"  Aucun .TextGrid dans : {os.path.abspath(arg)}"); return

    out = os.path.join(arg if os.path.isdir(arg) else ".", "phrases_prosodiques.txt")
    with open(out, "w", encoding="utf-8") as fout:
        for path in files:
            name = os.path.basename(path)
            sents, words, phones = segment_textgrid_file(path)
            if sents is None:
                print(f"  - {name}: pas de tier 'words'"); continue
            has_ph = "avec phones" if phones else "SANS phones (pause seule)"
            lens = [len(s) for s in sents]
            moy = sum(lens) / len(lens) if lens else 0
            mx = max(lens) if lens else 0
            n_cap = sum(1 for L in lens if L >= SENT_MAX)
            cap = f"  [!! {n_cap} phrase(s) coupees par SENT_MAX={SENT_MAX}]" if n_cap else ""
            print(f"  - {name}: {len(words)} mots -> {len(sents)} phrases "
                  f"(moy {moy:.1f}, max {mx} mots) [{has_ph}]{cap}")
            fout.write(f"===== {name}  ({has_ph}) =====\n")
            for sid, s in enumerate(sents, 1):
                phrase = " ".join(w.lower() for (w, on, off) in s)
                fout.write(f"[{sid:>3}] ({len(s):>2} mots) {phrase}\n")
            fout.write("\n")
    print(f"\n  Phrases ecrites dans : {out}")
    print("  -> ouvre ce fichier pour juger la qualite de la segmentation.")


if __name__ == "__main__":
    main()