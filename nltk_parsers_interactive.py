import nltk
import re
import math

for r in ['treebank', 'punkt', 'punkt_tab',
          'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(r, quiet=True)

from nltk.corpus import treebank
from nltk import induce_pcfg, Nonterminal
from nltk.parse import ViterbiParser, EarleyChartParser


def load_parsers():
    print("=" * 60)
    print("  Chargement de la PCFG (Penn Treebank)... (~30 s)")
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

    lex_probs = {}
    for p in grammar.productions():
        if len(p.rhs()) == 1 and isinstance(p.rhs()[0], str):
            key = (str(p.lhs()), p.rhs()[0].lower())
            lex_probs[key] = lex_probs.get(key, 0.0) + p.prob()

    viterbi = ViterbiParser(grammar, trace=0)
    earley  = EarleyChartParser(grammar, trace=0)
    print(f"  {len(grammar.productions())} productions  OK\n")
    return viterbi, earley, grammar, vocab, best_pos, lex_probs


def tokenize(sentence):
    # BUG FIX : "it's" -> ["it", "'s"]  (avant : un seul token)
    words = nltk.word_tokenize(sentence.strip())
    if not words or words[-1] not in ('.', '!', '?'):
        words.append('.')
    return words


def fix_oov(words, vocab, best_pos):
    """Retourne (tokens_pour_parser, infos) ; infos[i]=(mot, POS_predit, est_OOV)."""
    tagged = nltk.pos_tag(words)
    proc, info = [], []
    for w, pos in tagged:
        if w in vocab:
            proc.append(w);             info.append((w, pos, False))
        elif w.lower() in vocab:
            proc.append(w.lower());     info.append((w, pos, False))
        elif pos in best_pos:
            proc.append(best_pos[pos]); info.append((w, pos, True))
        else:
            proc.append(w);             info.append((w, pos, True))
    return proc, info


def restore_leaves(tree, original_words):
    """Remet les vrais mots dans l'arbre (supprime 'Mr.'/'%' a l'affichage)."""
    for i in range(len(tree.leaves())):
        if i < len(original_words):
            tree[tree.leaf_treeposition(i)] = original_words[i]
    return tree


def calc_surprisal(tree, info, lex_probs):
    subtrees = [s for s in tree.subtrees(lambda t: t.height() == 2)]
    results = []
    for i, st in enumerate(subtrees):
        tree_pos = st.label(); pw = st.leaves()[0]
        if i < len(info):
            ow, opos, is_oov = info[i]
        else:
            ow, opos, is_oov = pw, tree_pos, False
        prob = lex_probs.get((tree_pos, pw.lower()), 1e-10)
        show_pos = opos if is_oov else tree_pos
        results.append((ow + (" *" if is_oov else ""), show_pos, -math.log2(prob)))
    return results


def afficher_surprisal(results, prob_parse):
    print(f"\n  Surprisal total  : {-math.log2(prob_parse):.2f} bits")
    print(f"  P(parse)         : {prob_parse:.2e}\n")
    print(f"  {'Mot':<22} {'POS':<8} {'Surprisal (bits)':>16}   Visualisation")
    print("  " + "-" * 60)
    total = 0; max_s = max(s for _, _, s in results)
    for w, pos, s in results:
        bar = 'X' * min(int(s / 1.5), 26)
        peak = " <- PIC" if s == max_s else ""
        print(f"  {w:<22} {pos:<8} {s:>16.2f}   {bar}{peak}")
        total += s
    print("  " + "-" * 60)
    print(f"  Surprisal moyen  : {total/len(results):.2f} bits/mot")
    print(f"  (* = mot hors-vocabulaire, surprisal approximee  |  <- PIC = plus inattendu)")


def afficher_arbre(tree, label):
    print(f"\n  +-- {label}")
    for line in tree.pformat(margin=60).split('\n'):
        print(f"  |   {line}")


def main():
    print("\n" + "=" * 62)
    print("  NLTK PARSERS INTERACTIF  (Viterbi + Earley)")
    print("=" * 62 + "\n")
    viterbi, earley, grammar, vocab, best_pos, lex_probs = load_parsers()
    print("  Phrase en anglais  |  'quit' -> quitter  |  'help' -> aide\n")

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
            print("\n  Exemples : The company reported a loss . | It's an apple . | The old man the boat .\n")
            continue

        words = tokenize(user_input)
        processed, info = fix_oov(words, vocab, best_pos)
        oovs = [ow for (ow, _, is_oov) in info if is_oov]
        print(f"\n  Tokens  : {words}")
        if oovs:
            print(f"  OOV (hors-vocabulaire) : {', '.join(oovs)}")

        print("\n" + "-" * 62 + "\n  VITERBI -- parse le plus probable\n" + "-" * 62)
        try:
            parses_v = list(viterbi.parse(processed))
            if parses_v:
                t = parses_v[0]
                surp = calc_surprisal(t, info, lex_probs)   # avant restauration
                restore_leaves(t, words)
                afficher_arbre(t, "Arbre syntaxique"); print()
                afficher_surprisal(surp, t.prob())
            else:
                print("  X Aucun parse trouve.")
        except Exception as e:
            print(f"  X Erreur : {e}")

        print("\n" + "-" * 62 + "\n  EARLEY - jusqu'a 3 parses\n" + "-" * 62)
        try:
            parses_e = list(earley.parse(processed))
            if parses_e:
                print(f"  {len(parses_e)} parse(s) trouve(s)")
                for i, t in enumerate(parses_e[:3]):
                    restore_leaves(t, words)
                    afficher_arbre(t, f"Parse #{i+1}")
            else:
                print("  X Aucun parse trouve.")
        except Exception as e:
            print(f"  X Erreur : {e}")
        print("\n" + "=" * 62 + "\n")


if __name__ == "__main__":
    main()
