# -*- coding: utf-8 -*-
"""
Segmentation en phrases de la parole lue (dataset Broderick ds004408, TextGrid).

Les TextGrid n'ont ni ponctuation ni majuscules. Je decoupe le flux de mots en
phrases avec des indices prosodiques, puis je corrige avec des regles sur les POS.

Score prosodique apres chaque mot :
    score = W_PAUSE * z(pause) + W_LENG * z(allongement) + W_PITCH * z(reset_pitch)
On coupe si  pause >= PAUSE_MIN  ou  score >= moyenne + THR_K * ecart-type.

Regles grammaticales (POS), activables une par une avec le parametre `rules` :
    no_end        : pas de fin de phrase sur un mot-outil (in, and, the...)
    no_end_words  : pas de fin sur un pronom sujet (i, we, he...)
    no_start      : pas de debut sur une particule seule
    no_start_words: pas de debut sur "of"/"nor" ou un pronom objet (him, them...)
    subject_verb  : pas de coupure entre un sujet (nom) et son verbe conjugue
    adv_no_end    : pas de fin sur un adverbe de degre (almost...) ou un auxiliaire (was, had...)
    no_verb       : une phrase sans verbe est collee a sa voisine

Utilisation :
    python segmentation_prosodique.py .
    python segmentation_prosodique.py audio01.TextGrid
"""

import os, re, sys, glob, math, statistics

# ---- parametres (regles avec regler_parametres.py) ----
PAUSE_MIN  = 0.40    # pause (s) au-dessus de laquelle on coupe directement ; plus bas = plus de coupures
THR_K      = 3.0     # on coupe aussi si score >= moyenne + THR_K*ecart-type ; plus bas = plus de coupures
MIN_LEN    = 4       # nombre de mots minimum avant d'autoriser une coupure
SENT_MAX   = 60      # longueur maximale d'une phrase (garde-fou pour l'analyseur)
W_PAUSE    = 1.0     # poids de la pause dans le score
W_LENG     = 1.0     # poids de l'allongement final dans le score
W_PITCH    = 0.5     # poids du reset de pitch (si un .wav est present)
USE_POS    = True    # True = on applique les regles grammaticales ; False = prosodie seule
SIL        = {'sil', 'sp', 'spn', '', '<p>', 'br', 'noise', 'sps'}   # silences a ignorer

# ---- listes pour les regles grammaticales ----------------------------------
# POS sur lesquels une phrase ne peut pas finir (mots-outils qui attendent un complement).
NO_END = {'IN', 'TO', 'DT', 'PDT', 'CC', 'PRP$', 'POS', 'WDT', 'WP', 'WP$', 'MD', 'RP', 'EX'}
# Mots (pronoms sujets) sur lesquels une phrase ne peut pas finir.
NO_END_WORDS = {'i', 'we', 'he', 'she', 'they'}
# POS sur lesquels une phrase ne peut pas commencer (particule seule).
NO_START = {'RP'}
# Mots sur lesquels une phrase ne peut pas commencer :
#   - prepositions non frontales (of, nor)  - pronoms objets (him, them, us)
# (trouves en regardant les debuts de phrase du texte original).
NO_START_WORDS = {'of', 'nor', 'him', 'them', 'us'}
# Mots sur lesquels une phrase ne peut pas finir (on garde le complement) :
#   - adverbes de degre (almost, very...)   - auxiliaires (was, had, is...)
# (trouves en regardant les fins de phrase du texte original).
ADV_NO_END = {'almost', 'very', 'quite', 'too', 'rather', 'extremely', 'fairly',
              'is', 'are', 'was', 'were', 'am', 'be', 'been',
              'had', 'has', 'have', 'do', 'does', 'did'}
# Regle sujet-verbe : on ne coupe pas entre un nom (sujet) et un verbe conjugue.
SUBJECT_NOUNS = {'NN', 'NNS', 'NNP', 'NNPS'}
PRED_VERBS    = {'VBD', 'VBZ'}
# Etiquettes de verbe (pour "toute phrase a un verbe").
VERBS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD'}

# Regles appliquees par defaut (on peut en enlever pour l'ablation).
DEFAULT_RULES = {'no_end', 'no_end_words', 'no_start', 'no_start_words',
                 'subject_verb', 'adv_no_end', 'no_verb'}


# ---------------------------------------------------------------------------
#  Lecture des tiers words et phones du TextGrid
# ---------------------------------------------------------------------------

def _read_tier(txt, names):
    """Extrait les intervalles (label, debut, fin) du premier tier dont le nom est dans `names`."""
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
    """Lit un TextGrid et renvoie (words, phones), chacun = liste de (label, debut, fin)."""
    txt = open(path, encoding='utf-8', errors='ignore').read()
    raw_w = _read_tier(txt, {'words', 'word'})
    raw_p = _read_tier(txt, {'phones', 'phone', 'phon'})
    if raw_w is None:
        return None, None
    words = [(w, on, off) for (w, on, off) in raw_w if w and w.lower() not in SIL]
    phones = None
    if raw_p is not None:
        phones = [(p, on, off) for (p, on, off) in raw_p if p and p.lower() not in SIL]
    return words, phones


# ---------------------------------------------------------------------------
#  Pitch (F0) : reset de pitch a la frontiere (il faut le .wav)
# ---------------------------------------------------------------------------

def find_wav(textgrid_path):
    """Cherche un .wav de meme nom a cote du TextGrid ; renvoie son chemin ou None."""
    base = os.path.splitext(textgrid_path)[0]
    for cand in (base + ".wav", base + ".WAV"):
        if os.path.exists(cand):
            return cand
    return None


def load_pitch_reset(textgrid_path, words, win=0.10):
    """Reset de pitch (demi-tons) apres chaque mot : F0(debut mot suivant) - F0(fin mot).
    None si pas de .wav ou parselmouth absent."""
    wav = find_wav(textgrid_path)
    if wav is None:
        return None
    try:
        import numpy as np
        import parselmouth
    except Exception:
        print("  [pitch] parselmouth absent (pip install praat-parselmouth) -> pitch ignore.")
        return None
    try:
        snd = parselmouth.Sound(wav)
        pit = snd.to_pitch(time_step=0.01)
        times = pit.xs()
        f0 = pit.selected_array['frequency']
    except Exception as e:
        print(f"  [pitch] lecture {os.path.basename(wav)} impossible ({e}) -> pitch ignore.")
        return None

    def f0win(a, b):
        m = (times >= a) & (times < b) & (f0 > 0)
        return float(np.median(f0[m])) if m.any() else None

    n = len(words)
    reset = []
    for i, (w, on, off) in enumerate(words):
        if i + 1 < n:
            fe = f0win(off - win, off) or f0win(on, off)
            nxt_on = words[i + 1][1]
            fs = f0win(nxt_on, nxt_on + win) or f0win(nxt_on, words[i + 1][2])
            reset.append(12 * math.log2(fs / fe) if (fe and fs and fe > 0 and fs > 0) else None)
        else:
            reset.append(None)
    return reset


# ---------------------------------------------------------------------------
#  Indices prosodiques et score
# ---------------------------------------------------------------------------

def _zstats(values):
    """(moyenne, ecart-type) en ignorant les None ; (0,1) si trop peu de valeurs."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0, 1.0
    return statistics.mean(vals), (statistics.pstdev(vals) or 1.0)


def _last_phone_of_word(phones, on, off):
    """(label, duree) du dernier phoneme du mot [on, off]."""
    inside = [(p, po, pf) for (p, po, pf) in phones if pf > on + 1e-6 and po < off - 1e-6]
    if not inside:
        return None, None
    p, po, pf = inside[-1]
    return p.lower(), pf - po


def compute_cues(words, phones):
    """Renvoie (pause_apres, duree_dernier_phoneme, label_dernier_phoneme) par mot."""
    n = len(words)
    pause, leng, labels = [], [], []
    for i, (w, on, off) in enumerate(words):
        pause.append(max(0.0, words[i + 1][1] - off) if i + 1 < n else None)
        if phones:
            lab, d = _last_phone_of_word(phones, on, off)
        else:
            lab, d = None, (off - on)
        labels.append(lab)
        leng.append(d)
    return pause, leng, labels


def boundary_scores(words, phones, reset=None):
    """Score de frontiere par mot = W_PAUSE*z(pause)+W_LENG*z(allongement)+W_PITCH*z(reset).
    Renvoie une liste de (score, pause_en_secondes)."""
    pause, leng, labels = compute_cues(words, phones)
    mu_p, sd_p = _zstats(pause)
    by_lab = {}
    for lab, d in zip(labels, leng):
        if d is not None:
            by_lab.setdefault(lab, []).append(d)
    lab_stats = {lab: _zstats(v) for lab, v in by_lab.items()}
    if reset is not None:
        mu_r, sd_r = _zstats(reset)
    else:
        reset = [None] * len(words)
    scores = []
    for pa, d, lab, r in zip(pause, leng, labels, reset):
        z_p = 0.0 if pa is None else (pa - mu_p) / sd_p
        z_l = 0.0 if (d is None or lab not in lab_stats) else (d - lab_stats[lab][0]) / lab_stats[lab][1]
        z_r = 0.0 if r is None else W_PITCH * (r - mu_r) / sd_r
        scores.append((W_PAUSE * z_p + W_LENG * z_l + z_r, pa))
    return scores


def pos_tags(words):
    """Etiquettes POS (Penn Treebank) via NLTK ; None si NLTK indisponible."""
    try:
        import nltk
        for r in ('averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'):
            nltk.download(r, quiet=True)
        return [t for (_, t) in nltk.pos_tag([w.lower() for w in words])]
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Segmentation
# ---------------------------------------------------------------------------

def segment_prosodic(words, phones, reset=None,
                     pause_min=PAUSE_MIN, thr_k=THR_K, min_len=MIN_LEN,
                     sent_max=SENT_MAX, use_pos=USE_POS, rules=None):
    """Decoupe `words` en phrases. `rules` = ensemble des regles grammaticales a appliquer
    (None = DEFAULT_RULES). Sert a l'ablation (activer/desactiver chaque regle)."""
    if rules is None:
        rules = DEFAULT_RULES
    n = len(words)
    if n == 0:
        return []
    scores = boundary_scores(words, phones, reset)
    vals = [s for s, _ in scores[:-1]]
    mu, sd = _zstats(vals)
    thr = mu + thr_k * sd

    # 1) frontieres candidates d'apres la prosodie
    cut = [False] * n
    for i in range(n - 1):
        score, pause = scores[i]
        if (pause is not None and pause >= pause_min) or (score >= thr):
            cut[i] = True

    # 2) corrections grammaticales
    tags = None
    if use_pos:
        tags = pos_tags([w for (w, on, off) in words])
        low_words = [w.lower() for (w, on, off) in words]
        if tags and len(tags) == n:
            # 2a) pas de fin sur un mot-outil ou un pronom sujet : on recule la frontiere d'un mot
            for i in range(n - 1):
                if not cut[i]:
                    continue
                bad_end = (('no_end' in rules and tags[i] in NO_END) or
                           ('no_end_words' in rules and low_words[i] in NO_END_WORDS))
                if bad_end:
                    cut[i] = False
                    if i - 1 >= 0:
                        cut[i - 1] = True
            # 2b) pas de debut sur un mot orphelin : on avance la frontiere d'un mot
            if 'no_start' in rules:
                for i in range(n - 1):
                    if cut[i]:
                        j = i + 1
                        stranded = (tags[j] in NO_START or
                                    (tags[j] == 'IN' and j + 1 < n and tags[j + 1] == 'CC'))
                        if stranded:
                            cut[i] = False
                            cut[j] = True
            # 2c) pas de coupure entre un sujet (nom) et son verbe conjugue
            if 'subject_verb' in rules:
                for i in range(n - 1):
                    if cut[i] and tags[i] in SUBJECT_NOUNS and tags[i + 1] in PRED_VERBS:
                        cut[i] = False
            # 2d) pas de debut par "of", "nor" ou un pronom objet : on enleve la frontiere
            if 'no_start_words' in rules:
                for i in range(n - 1):
                    if cut[i] and low_words[i + 1] in NO_START_WORDS:
                        cut[i] = False
            # 2e) pas de fin sur un adverbe de degre ou un auxiliaire : on garde le complement
            if 'adv_no_end' in rules:
                for i in range(n - 1):
                    if cut[i] and low_words[i] in ADV_NO_END:
                        cut[i] = False

    # 3) construction des phrases (avec leurs etiquettes)
    raw, cur, cur_t = [], [], []
    for i, (w, on, off) in enumerate(words):
        cur.append((w, on, off))
        cur_t.append(tags[i] if (tags and len(tags) == n) else None)
        if i == n - 1:
            raw.append((cur, cur_t)); cur, cur_t = [], []
        elif len(cur) >= sent_max:
            raw.append((cur, cur_t)); cur, cur_t = [], []
        elif cut[i] and len(cur) >= min_len:
            raw.append((cur, cur_t)); cur, cur_t = [], []
    if cur:
        raw.append((cur, cur_t))

    # 4) une phrase sans verbe est fusionnee avec la precedente
    if 'no_verb' in rules and tags and len(tags) == n:
        items = [(p, t, any(x in VERBS for x in t)) for p, t in raw]
        out = []
        for p, t, has_verb in items:
            if not has_verb and out:
                out[-1] = (out[-1][0] + p, out[-1][1] + t, out[-1][2])
            else:
                out.append((p, t, has_verb))
        if len(out) >= 2 and not out[0][2]:
            out[1] = (out[0][0] + out[1][0], out[0][1] + out[1][1], out[1][2])
            out = out[1:]
        raw = [(p, t) for (p, t, h) in out]

    return [p for (p, t) in raw]


def segment_textgrid_file(path, **kw):
    """Lit un TextGrid, calcule le pitch si un .wav est a cote, renvoie (phrases, mots, phonemes)."""
    words, phones = read_words_and_phones(path)
    if words is None:
        return None, None, None
    reset = load_pitch_reset(path, words)
    sents = segment_prosodic(words, phones, reset=reset, **kw)
    return sents, words, phones


# ---------------------------------------------------------------------------
#  Programme principal : ecrit phrases_prosodiques.txt
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
            has_ph = "avec phones" if phones else "SANS phones"
            has_ph += " + pitch" if find_wav(path) else " (pas de .wav)"
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


if __name__ == "__main__":
    main()
