import nltk
import re
import math

for r in ['treebank', 'punkt', 'punkt_tab',
          'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(r, quiet=True)

from nltk.corpus import treebank
from nltk import induce_pcfg, Nonterminal
from nltk.parse import ViterbiParser, EarleyChartParser


# =============================================================================
# CHARGEMENT
# =============================================================================

def load_parsers():
    print("=" * 60)
    print("  Chargement de la PCFG (Penn Treebank)...")
    print("  (~ 30 secondes, une seule fois)")
    print("=" * 60)

    prods = []
    for t in treebank.parsed_sents():
        t.collapse_unary(collapsePOS=False)
        t.chomsky_normal_form(horzMarkov=2)
        prods += t.productions()

    grammar = induce_pcfg(Nonterminal('S'), prods)

    vocab, best_pos = set(), {}
    for p in grammar.productions():
        if len(p.rhs()) == 1 and isinstance(p.rhs()[0], str):
            w, pos = p.rhs()[0], str(p.lhs())
            vocab.add(w)
            if pos not in best_pos or p.prob() > best_pos[pos][1]:
                best_pos[pos] = (w, p.prob())
    best_pos = {k: v[0] for k, v in best_pos.items()}

    # Probabilites lexicales - somme des variantes de casse (bug fix)
    lex_probs = {}
    for p in grammar.productions():
        if len(p.rhs()) == 1 and isinstance(p.rhs()[0], str):
            key = (str(p.lhs()), p.rhs()[0].lower())
            lex_probs[key] = lex_probs.get(key, 0.0) + p.prob()

    viterbi = ViterbiParser(grammar, trace=0)
    earley  = EarleyChartParser(grammar, trace=0)

    print(f"  {len(grammar.productions())} productions  OK")
    print("=" * 60)
    print("  Parsers prets !\n")

    return viterbi, earley, grammar, vocab, best_pos, lex_probs


# =============================================================================
# TOKENISATION + OOV
# =============================================================================

def tokenize(sentence):
    sentence = sentence.strip()
    sentence = re.sub(r"([.,!?;:])", r" \1", sentence)
    if not sentence.rstrip().endswith(('.', '!', '?')):
        sentence = sentence.rstrip() + " ."
    return sentence.split()


def fix_oov(words, vocab, best_pos):
    tagged = nltk.pos_tag(words)
    out, replacements = [], []
    for w, pos in tagged:
        if w in vocab:
            out.append(w); replacements.append(None)
        elif w.lower() in vocab:
            out.append(w.lower()); replacements.append(None)
        elif pos in best_pos:
            out.append(best_pos[pos]); replacements.append(f"{w}->{best_pos[pos]}")
        else:
            out.append(w); replacements.append(f"{w}=OOV")
    return out, replacements


# =============================================================================
# SURPRISAL MOT PAR MOT depuis un arbre Viterbi
# =============================================================================

def calc_surprisal(tree, words, replacements, lex_probs):
    subtrees = [s for s in tree.subtrees(lambda t: t.height() == 2)]
    results = []
    for i, st in enumerate(subtrees):
        pos  = st.label()
        pw   = st.leaves()[0]
        ow   = words[i] if i < len(words) else pw
        prob = lex_probs.get((pos, pw.lower()), 1e-10)
        surp = -math.log2(prob)
        flag = " *" if (i < len(replacements) and replacements[i]) else ""
        results.append((ow + flag, pos, surp))
    return results


def afficher_surprisal(results, prob_parse):
    print(f"\n  Surprisal total  : {-math.log2(prob_parse):.2f} bits")
    print(f"  P(parse)         : {prob_parse:.2e}\n")
    print(f"  {'Mot':<22} {'POS':<8} {'Surprisal (bits)':>16}   Visualisation")
    print("  " + "-" * 60)
    total = 0
    max_s = max(s for _, _, s in results)
    for w, pos, s in results:
        bar  = 'X' * min(int(s / 1.5), 26)
        peak = " <- PIC" if s == max_s else ""
        print(f"  {w:<22} {pos:<8} {s:>16.2f}   {bar}{peak}")
        total += s
    print("  " + "-" * 60)
    print(f"  Surprisal moyen  : {total/len(results):.2f} bits/mot")
    print(f"  (* = OOV remplace  |  <- PIC = mot le plus inattendu)")


# =============================================================================
# AFFICHAGE ARBRE
# =============================================================================

def afficher_arbre(tree, label):
    print(f"\n  +-- {label}")
    for line in tree.pformat(margin=60).split('\n'):
        print(f"  |   {line}")


# =============================================================================
# BOUCLE INTERACTIVE
# =============================================================================

def main():
    print("\n" + "=" * 62)
    print("  NLTK PARSERS INTERACTIF")
    print("  Viterbi (probabiliste) + Earley (non-probabiliste)")
    print("=" * 62 + "\n")

    viterbi, earley, grammar, vocab, best_pos, lex_probs = load_parsers()

    print("  Commandes :")
    print("  * n'importe quelle phrase en anglais")
    print("  * 'quit' -> quitter  |  'help' -> aide\n")

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
  Exemples :
    The company reported a loss .
    The plan that the board approved was rejected .
    The old man the boat .
    The bank the court had sued failed .
""")
            continue

        words     = tokenize(user_input)
        processed, replacements = fix_oov(words, vocab, best_pos)
        oovs = [r for r in replacements if r]

        print(f"\n  Tokens  : {words}")
        if oovs:
            print(f"  OOV     : {', '.join(oovs)}")
        print(f"  Traites : {processed}")

        # Viterbi
        print("\n" + "-" * 62)
        print("  VITERBI -- parse le plus probable")
        print("-" * 62)
        try:
            parses_v = list(viterbi.parse(processed))
            if parses_v:
                t = parses_v[0]
                afficher_arbre(t, "Arbre syntaxique")
                print()
                surp = calc_surprisal(t, words, replacements, lex_probs)
                afficher_surprisal(surp, t.prob())
            else:
                print("  X Aucun parse trouve.")
        except Exception as e:
            print(f"  X Erreur : {e}")

        # Earley
        print("\n" + "-" * 62)
        print("  EARLEY - affiche 3 parses possibles")
        print("-" * 62)
        try:
            parses_e = list(earley.parse(processed))
            if parses_e:
                print(f"  {len(parses_e)} parse(s) trouve(s)")
                for i, t in enumerate(parses_e[:3]):
                    afficher_arbre(t, f"Parse #{i+1}")
                if len(parses_e) > 3:
                    print(f"\n  ... ({len(parses_e)-3} autre(s) non affiche(s))")
            else:
                print("  X Aucun parse trouve.")
        except Exception as e:
            print(f"  X Erreur : {e}")

        print("\n" + "=" * 62 + "\n")


if __name__ == "__main__":
    main()
