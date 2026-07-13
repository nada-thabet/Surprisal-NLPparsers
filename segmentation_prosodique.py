# -*- coding: utf-8 -*-
"""
SEGMENTEUR PROSODIQUE POUR L'ORAL  --  dataset Broderick ds004408 (TextGrid)

Idee (references de Ph. Blache : Shriberg & Stolcke ; semi-supervise Isik Univ.) :
  Pour la PAROLE, on ne segmente ni par le texte ecrit ni par un LLM, mais par les
  INDICES PROSODIQUES du signal. L'ordre d'importance mesure dans la litterature est :
      duree (allongement pre-frontiere)  >  pause  >  pitch
  Ces indices sont DEJA dans le TextGrid :
      - tier "words"  -> pauses entre mots + duree des mots
      - tier "phones" -> allongement du DERNIER phone avant une frontiere
  (Le pitch necessiterait le .wav ; il pourra etre ajoute ensuite via parselmouth.)

Ce module combine deux features, chacune normalisee (z-score) :
      1. pause_after  : silence entre le mot et le suivant
      2. allongement  : duree du dernier phone du mot, z-scoree PAR phone
                        (on compare un /t/ a la moyenne des /t/, etc. -> controle la
                         duree intrinseque des phonemes)
  Score de frontiere = W_PAUSE * z(pause) + W_LENG * z(allongement).
  On pose une frontiere quand le score est eleve, avec garde-fous de longueur.

Usage autonome (pour inspecter la qualite des phrases) :
      python segmentation_prosodique.py .            # traite tous les TextGrid
      python segmentation_prosodique.py audio01.TextGrid

Integration dans constituants_dataset.py :
    from segmentation_prosodique import read_words_and_phones, segment_prosodic
    words, phones = read_words_and_phones(path)
    sents = segment_prosodic(words, phones)   # meme format [(mot, on, off)] qu'avant
"""

import os, re, sys, glob, math, statistics

# ---- parametres reglables --------------------------------------------------
# On NE FIXE PLUS de longueur de phrase : une frontiere n'est posee que si la
# PROSODIE la justifie. Les phrases longues restent donc longues.
PAUSE_MIN  = 0.45    # pause absolue (s) >= ce seuil -> frontiere (baisse->coupe plus)
THR_K      = 2.0     # OU score combine >= moyenne + THR_K * ecart-type (baisse->coupe plus)
MIN_LEN    = 3       # on ne coupe pas avant d'avoir ce nb de mots (evite les miettes)
SENT_MAX   = 60      # garde-fou : coupe si trop long (Viterbi est en O(n^3))
W_PAUSE    = 1.0     # poids de la pause dans le score combine
W_LENG     = 1.0     # poids de l'allongement final
W_PITCH    = 1.0     # poids du RESET de pitch (3e indice, si le .wav est present)
USE_POS    = True    # contrainte morphosyntaxique (retour Philippe) : voir NO_END
SIL        = {'sil', 'sp', 'spn', '', '<p>', 'br', 'noise', 'sps'}

# Categories morphosyntaxiques (POS Penn Treebank) sur lesquelles une phrase ne peut
# PAS se terminer : mots-outils qui appellent un complement a droite. Si la prosodie
# propose une frontiere apres un tel mot, on la deplace AVANT lui (le mot rejoint la
# phrase suivante). Corrige les cas "... a fish in" ou "... flour sacks and".
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

# Symetrique : une phrase ne peut PAS COMMENCER par ces mots orphelins qui
# appartiennent a la phrase precedente. Si la prosodie ouvre une phrase sur un tel
# mot, on deplace la frontiere APRES lui (il rejoint la phrase precedente).
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
#  PITCH (F0)  --  3e indice : reset de pitch a la frontiere
#  A une vraie fin de phrase la voix descend puis REMONTE au mot suivant (reset).
#  A une virgule elle reste suspendue -> peu de reset. Besoin du fichier .wav.
# ---------------------------------------------------------------------------

def find_wav(textgrid_path):
    """Cherche un .wav de meme nom a cote du TextGrid ; None si absent."""
    base = os.path.splitext(textgrid_path)[0]
    for cand in (base + ".wav", base + ".WAV"):
        if os.path.exists(cand):
            return cand
    return None


def load_pitch_reset(textgrid_path, words, win=0.10):
    """Reset de pitch (demi-tons) apres chaque mot : F0(debut mot suivant) - F0(fin mot).
    Renvoie une liste alignee sur `words` (None si pas de .wav / parselmouth absent)."""
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
        # pause apres le mot i
        if i + 1 < n:
            pa = max(0.0, words[i + 1][1] - off)
        else:
            pa = None
        pause.append(pa)
        # allongement du dernier phone
        if phones:
            lab, d = _last_phone_of_word(phones, on, off)
        else:
            lab, d = None, (off - on)      # sans phones : duree du mot entier
        labels.append(lab)
        leng.append(d)
    return pause, leng, labels


def boundary_scores(words, phones, reset=None):
    """Score de frontiere apres chaque mot (plus c'est grand, plus c'est une fin de phrase).
    Si `reset` (pitch) est fourni, il est ajoute comme 3e indice (poids W_PITCH)."""
    pause, leng, labels = compute_cues(words, phones)

    # z-score de la pause (global)
    mu_p, sd_p = _zstats(pause)

    # z-score de l'allongement, PAR phone (compare /t/ aux /t/, /a/ aux /a/...)
    by_lab = {}
    for lab, d in zip(labels, leng):
        if d is not None:
            by_lab.setdefault(lab, []).append(d)
    lab_stats = {lab: _zstats(v) for lab, v in by_lab.items()}

    # z-score du reset de pitch (global), si disponible
    if reset is not None:
        mu_r, sd_r = _zstats(reset)
    else:
        reset = [None] * len(words)

    scores = []
    for pa, d, lab, r in zip(pause, leng, labels, reset):
        z_p = 0.0 if pa is None else (pa - mu_p) / sd_p
        if d is None or lab not in lab_stats:
            z_l = 0.0
        else:
            mu_l, sd_l = lab_stats[lab]
            z_l = (d - mu_l) / sd_l
        z_r = 0.0 if r is None else W_PITCH * (r - mu_r) / sd_r
        scores.append((W_PAUSE * z_p + W_LENG * z_l + z_r, pa))
    return scores      # liste de (score, pause_en_secondes)


# ---------------------------------------------------------------------------
#  INFORMATION MORPHOSYNTAXIQUE  (POS, retour Philippe)
# ---------------------------------------------------------------------------

def pos_tags(words):
    """Etiquette POS (Penn Treebank) via NLTK. Renvoie None si NLTK indisponible."""
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

def segment_prosodic(words, phones, reset=None,
                     pause_min=PAUSE_MIN, thr_k=THR_K, min_len=MIN_LEN,
                     sent_max=SENT_MAX, use_pos=USE_POS):
    """words = [(mot, on, off)] -> liste de phrases (chaque phrase = liste de (mot,on,off)).

    0) PITCH (reset) : si `reset` est fourni (depuis le .wav), il devient un 3e indice.
    1) PROSODIE : frontiere candidate si pause >= pause_min OU score combine eleve.
    2) MORPHOSYNTAXE (use_pos) : une frontiere qui tomberait apres un mot-outil
       (POS dans NO_END : preposition, determinant, conjonction, to, possessif...)
       est illegale -> on la deplace AVANT ce mot, qui rejoint la phrase suivante.
       Corrige "... a fish in" et "... flour sacks and".
    3) garde-fous de longueur (min_len, sent_max).
    Aucune longueur de phrase n'est imposee.
    """
    n = len(words)
    if n == 0:
        return []
    scores = boundary_scores(words, phones, reset)
    vals = [s for s, _ in scores[:-1]]           # scores hors dernier mot
    mu, sd = _zstats(vals)
    thr = mu + thr_k * sd                         # seuil RELATIF a la moyenne (pas un quota)

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
                        cut[i - 1] = True         # le mot-outil finit la phrase precedente
            # 2b) pas de phrase qui COMMENCE par un mot orphelin -> deplace a droite
            for i in range(n - 1):
                if cut[i]:
                    j = i + 1                     # mot qui ouvrirait la phrase suivante
                    stranded = (tags[j] in NO_START or
                                (tags[j] == 'IN' and j + 1 < n and tags[j + 1] == 'CC'))
                    if stranded:
                        cut[i] = False
                        cut[j] = True             # le mot orphelin rejoint la phrase precedente

    # 3) construction avec garde-fous de longueur
    sents, cur = [], []
    for i, (w, on, off) in enumerate(words):
        cur.append((w, on, off))
        if i == n - 1:
            sents.append(cur); cur = []
        elif len(cur) >= sent_max:                # garde-fou Viterbi
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
    reset = load_pitch_reset(path, words)     # None si pas de .wav a cote
    sents = segment_prosodic(words, phones, reset=reset, **kw)
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
    print("  -> ouvre ce fichier pour juger la qualite de la segmentation.")


if __name__ == "__main__":
    main()
