import math
import nltk

for r in ['treebank', 'brown', 'punkt', 'punkt_tab',
          'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(r, quiet=True)

from nltk.corpus import treebank, brown
from nltk import induce_pcfg, Nonterminal
from nltk.parse import ViterbiParser
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline


# =============================================================================
# CHARGEMENT DES MODÈLES
# =============================================================================

def load_models():
    print("━"*60)
    print("  Chargement des modèles (une seule fois)...")
    print("━"*60)

    # N-gramme
    print("  [1/2] N-gramme trigramme (corpus Brown)...")
    sents = [list(s) for s in brown.sents()]
    train, vocab = padded_everygram_pipeline(3, sents)
    ng = Laplace(3)
    ng.fit(train, vocab)
    print(f"        Vocabulaire : {len(ng.vocab)} tokens  ✓")

    # PCFG
    print("  [2/2] PCFG depuis Penn Treebank (~30 secondes)...")
    prods = []
    for t in treebank.parsed_sents():
        t.collapse_unary(collapsePOS=False)
        t.chomsky_normal_form(horzMarkov=2)
        prods += t.productions()
    grammar = induce_pcfg(Nonterminal('S'), prods)
    parser  = ViterbiParser(grammar, trace=0)

    # Vocabulaire PCFG + meilleur proxy par POS
    vocab_pcfg, best_pos = set(), {}
    for p in grammar.productions():
        if len(p.rhs()) == 1 and isinstance(p.rhs()[0], str):
            w, pos = p.rhs()[0], str(p.lhs())
            vocab_pcfg.add(w)
            if pos not in best_pos or p.prob() > best_pos[pos][1]:
                best_pos[pos] = (w, p.prob())
    best_pos = {k: v[0] for k, v in best_pos.items()}

    # ── CORRECTION BUG ──────────────────────────────────────────────────────
    # Problème : "The" (majuscule) et "the" (minuscule) sont deux règles
    # distinctes dans la grammaire. En faisant .lower(), les deux clés
    # deviennent ("DT", "the") et le dict Python écrase la première valeur
    # par la seconde → probabilité faussement basse → surprisal trop élevé.
    #
    # Fix : on ADDITIONNE les probabilités de toutes les variantes de casse.
    # P_corrigée("DT", "the") = P(DT→"The") + P(DT→"the")
    # ────────────────────────────────────────────────────────────────────────
    lex_probs = {}
    for p in grammar.productions():
        if len(p.rhs()) == 1 and isinstance(p.rhs()[0], str):
            key = (str(p.lhs()), p.rhs()[0].lower())
            lex_probs[key] = lex_probs.get(key, 0.0) + p.prob()  # ← somme, pas écrasement

    print(f"        {len(grammar.productions())} productions  ✓")
    print("━"*60)
    print("  Modèles prêts !\n")

    return ng, parser, grammar, vocab_pcfg, best_pos, lex_probs


# =============================================================================
# TOKENISATION
# =============================================================================

def tokenize(sentence):
    import re
    sentence = sentence.strip()
    sentence = re.sub(r"([.,!?;:])", r" \1", sentence)
    if not sentence.rstrip().endswith(('.', '!', '?')):
        sentence = sentence.rstrip() + " ."
    return sentence.split()


# =============================================================================
# N-GRAMME
# =============================================================================

def calc_ngram(words, model, n=3):
    padded = ['<s>'] * (n - 1) + words
    results = []
    for i in range(n - 1, len(padded)):
        w   = padded[i]
        ctx = tuple(padded[i - (n-1):i])
        p   = model.score(w, ctx)
        results.append((w, -math.log2(p) if p > 0 else float('inf')))
    return results


# =============================================================================
# PCFG
# =============================================================================

def fix_oov(words, vocab, best):
    tagged = nltk.pos_tag(words)
    out, rep = [], []
    for w, pos in tagged:
        if w in vocab:            out.append(w);          rep.append(None)
        elif w.lower() in vocab:  out.append(w.lower());  rep.append(None)
        elif pos in best:         out.append(best[pos]);  rep.append(f"{w}→{best[pos]}")
        else:                     out.append(w);          rep.append(f"{w}=OOV")
    return out, rep

def calc_pcfg(words, parser, vocab, best, lex_probs):
    proc, rep = fix_oov(words, vocab, best)
    oovs = [r for r in rep if r]

    try:
        parses = list(parser.parse(proc))
    except Exception as e:
        return None, None, oovs, str(e)

    if not parses:
        return None, None, oovs, "Aucun parse trouvé (phrase hors-couverture grammaticale)"

    tree = parses[0]
    results = []
    for i, st in enumerate([s for s in tree.subtrees(lambda t: t.height() == 2)]):
        pos, pw = st.label(), st.leaves()[0]
        ow   = words[i] if i < len(words) else pw
        prob = lex_probs.get((pos, pw.lower()), 1e-10)
        surp = -math.log2(prob)
        flag = " *" if rep[i] else ""
        results.append((ow + flag, pos, surp))

    return results, tree, oovs, None


# =============================================================================
# AFFICHAGE
# =============================================================================

def afficher_resultats(words, ng_res, pcfg_res, oovs, pcfg_error):
    print("\n" + "━"*62)
    print(f"  RÉSULTATS — \"{' '.join(words)}\"")
    print("━"*62)

    # N-gramme
    print("\n  ► MODÈLE N-GRAMME (surprise lexicale / statistique)")
    print(f"  {'Mot':<22} {'Surprisal (bits)':>16}   Visualisation")
    print("  " + "-"*54)
    total_ng = 0
    for w, s in ng_res:
        if s == float('inf'):
            print(f"  {w:<22} {'inf':>16}")
        else:
            print(f"  {w:<22} {s:>16.2f}   {'█'*min(int(s/1.5),26)}")
            total_ng += s
    avg_ng = total_ng / len(ng_res)
    print(f"  {'─'*54}")
    print(f"  Surprisal moyen : {avg_ng:.2f} bits/mot")

    # PCFG
    print("\n  ► MODÈLE PCFG (surprise syntaxique — Hale 2001)")
    if oovs:
        print(f"  Mots hors-vocabulaire remplacés : {', '.join(oovs)}")

    if pcfg_error:
        print(f"  ✗ Impossible de parser : {pcfg_error}")
        print(f"    → Essaie une phrase plus courte ou avec des mots plus courants.")
    elif pcfg_res:
        print(f"  {'Mot':<22} {'POS':<8} {'Surprisal (bits)':>16}   Visualisation")
        print("  " + "-"*58)
        total_pc = 0
        max_surp = max(s for _,_,s in pcfg_res)
        for w, pos, s in pcfg_res:
            bar  = '█' * min(int(s / 1.5), 26)
            peak = " ◄ PIC" if s == max_surp else ""
            print(f"  {w:<22} {pos:<8} {s:>16.2f}   {bar}{peak}")
            total_pc += s
        avg_pc = total_pc / len(pcfg_res)
        print(f"  {'─'*58}")
        print(f"  Surprisal moyen : {avg_pc:.2f} bits/mot")
        print(f"  (* = OOV remplacé  |  ◄ PIC = mot syntaxiquement le plus inattendu)")

        # Interprétation
        print("\n  ► INTERPRÉTATION")
        print(f"  N-gramme → {avg_ng:.2f} bits/mot  (surprise lexicale)")
        print(f"  PCFG     → {avg_pc:.2f} bits/mot  (surprise syntaxique)")
        pic = max(pcfg_res, key=lambda x: x[2])
        print(f"  Mot le plus coûteux : '{pic[0].strip()}' ({pic[2]:.2f} bits, POS={pic[1]})")
        if avg_pc > 8.5:
            print(f"  → Phrase syntaxiquement COMPLEXE")
        elif avg_pc < 6.5:
            print(f"  → Phrase syntaxiquement SIMPLE")
        else:
            print(f"  → Complexité syntaxique MODÉRÉE")

    print("━"*62 + "\n")


# =============================================================================
# BOUCLE INTERACTIVE
# =============================================================================

def main():
    print("\n" + "═"*62)
    print("  SURPRISAL INTERACTIF — NLTK")
    print("  Tape une phrase en anglais pour obtenir son surprisal.")
    print("═"*62 + "\n")

    ng, parser, grammar, vocab_pcfg, best_pos, lex_probs = load_models()

    print("  Commandes : phrase en anglais → résultats")
    print("              'quit' → quitter  |  'help' → aide\n")

    while True:
        try:
            user_input = input("  Phrase > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Au revoir !"); break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("  Au revoir !"); break
        if user_input.lower() == 'help':
            print("""
  Tape n'importe quelle phrase en anglais.
  Exemples :
    The company reported a loss .
    The bank the court had sued failed .
    The old man the boat .

  Surprisal = −log₂ P(mot | contexte)
  Plus la valeur est haute, plus le mot est inattendu.
  N-gramme = surprise statistique | PCFG = surprise syntaxique
""")
            continue

        words = tokenize(user_input)
        print(f"\n  Tokens : {words}")

        ng_res = calc_ngram(words, ng)
        pcfg_res, tree, oovs, error = calc_pcfg(
            words, parser, vocab_pcfg, best_pos, lex_probs)
        afficher_resultats(words, ng_res, pcfg_res, oovs, error)


if __name__ == "__main__":
    main()
