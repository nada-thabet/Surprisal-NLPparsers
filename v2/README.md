# Version 2 of the surprisal measures

This folder recomputes both surprisal measures (syntactic PCFG and GPT-2) for both corpora. The files of the internship report are unchanged in `../version_rapport/`.

## Differences with the report version

| | Report version | v2 |
|---|---|---|
| Log base | bits, -log2 P | nats, -ln P, as used in the lab. bits = nats / ln 2, so correlations and p-values do not change |
| Book, syntactic | the 16 sentences over 40 words were skipped (741 of 11,416 words, e.g. one in the middle of `audio19`) | every word has a value |
| PCFG | learned from NLTK's Penn Treebank sample with punctuation, empty elements (`-NONE-` traces) and function tags (`NP-SBJ-1`) kept, even though the input words have none of them | cleaned treebank: no punctuation, no empty elements, bare labels (153 non-terminals instead of 939). `--grammaire originale` gives back the report's grammar |
| Book, GPT-2 | the TextGrid words were given in capitals ("HE WAS AN OLD MAN"), without punctuation, and GPT-2 restarted at each prosodic sentence | each TextGrid word is aligned to the book text (`oldmansea.pdf`, normal case and punctuation; 11,414 of 11,416 words found) and GPT-2 reads the story continuously (sliding 1,024-token window, at least 512 tokens of context) |
| AMI, GPT-2 | restarted at each utterance | each meeting is read continuously: all speakers' utterances in time order, one per line |
| AMI frequency | 341 acronym tokens (`l_c_d_`, `t_v_`) had a Zipf frequency of 0 | frequency of the written form (`lcd`, `tv`) |
| New columns | | `hors_vocab` (1 = word not in the grammar, replaced by the most likely word of its POS), `aligne_texte` (book) |

Unchanged: the words and their timings, the prosodic sentence split of the book (same as `stimuli_livre/phrases_prosodiques.txt`), the AMI sentence split (transcript punctuation, cut into chunks of at most 40 words), frequencies (except acronyms), phone and letter counts, and the parsing algorithm. With `--grammaire originale`, v2 gives the report's values to within 1.2e-4 bits (CSV rounding).

Negative values: a surprisal -ln P can't be negative, and none of the surprisal files has one. Negative values only appear in derived columns, which are regression residuals and so have a mean of 0: `surprise_*_ctrlfreq` (surprisal after removing frequency) and `surprise_*_apres_controle` (after removing length, frequency, their interaction and word class), in `indices_eeg/` for v2 and in `../version_rapport/5_indices_eeg/` for the report version. The `*_scaled` columns are z-scores.

## Data (`donnees/`)

| File | Rows | Content |
|---|---|---|
| `surprise_livre_v2.csv` | 11,416 | The Old Man and the Sea, 20 TextGrids: `surprise_syn` and `surprise_llm` (nats) |
| `surprise_ami_v2.csv` | 67,381 | AMI, the 12 meetings ES2002a–ES2004d used in the report: `surprise_syn` and `surprise_llm` (nats) |
| `surprise_syn_ami_171reunions_v2.csv.gz` | 978,212 | AMI, all 171 meetings, `surprise_syn` only (nats). 806 words (0.08 %) have no value because the grammar gives their prefix probability 0 |
| `dialogpt/` (7 files) | 67,381 each | AMI, 12 meetings: `surprise_ami_v2.csv` with `surprise_llm` recomputed by GPT-2 or DialoGPT in the reading setups of `10_dialogpt.py` (context window, one utterance at a time, hyphens kept or not) |

The two measures are in the same file because the scripts run one after the other on it: step 1 writes the words, step 2 adds `surprise_syn`, step 3 adds `surprise_llm`. Both measures are therefore computed for the same words, with the same timings and sentences. In the analyses, each measure leaves out the few rows where it has no value (see "Missing values" below), so n differs slightly between the two (book 11,413 and 11,411; AMI 60,277 and 60,285).

`indices_eeg/` holds the four files for the EEG team (section 5.4 of the report), recomputed from `surprise_livre_v2.csv` by `8_indices_eeg.py`: `surprise_syntaxique_ctrlfreq_v2.csv`, `surprise_llm_ctrlfreq_v2.csv`, `surprise_syntaxique_apres_controle_v2.csv` and `surprise_llm_apres_controle_v2.csv` (11,416 rows each, in nats). The report's four indices are syntactic and GPT-2 surprisal, with and without the frequency control: the raw columns and the two `*_ctrlfreq` files.

`python lancer_gpt2.py` adds the `surprise_llm` column (GPT-2 is downloaded from Hugging Face). `python lancer_gpt2.py gpt2-xl` (or any Hugging Face causal model, or a local folder) writes `..._v2_gpt2-xl.csv` files next to them.

Common columns: `onset`, `offset` (s), `mot`, `POS` (NLTK perceptron tagger), `freq_zipf` (wordfreq), `n_phones` (book) or `n_letters` (AMI), `hors_vocab`. Book rows are identified by `fichier, phrase_id, position`, AMI rows by `meeting, locuteur, phrase_id, position`.

## Code

```
pip install nltk scipy numpy wordfreq pandas statsmodels pymupdf torch transformers
python 1_mots_livre.py csv ../version_rapport/2_csv_du_rapport/surprisal_llm_textgrid.csv mots_livre.csv
python 2_surprise_syntaxique.py mots_livre.csv livre_syn.csv
python 3_surprise_llm_livre.py livre_syn.csv ../stimuli_livre/oldmansea.pdf donnees/surprise_livre_v2.csv gpt2
python 1_mots_ami.py csv ../version_rapport/2_csv_du_rapport/surprisal_llm_ami.csv mots_ami.csv
python 2_surprise_syntaxique.py mots_ami.csv ami_syn.csv
python 3_surprise_llm_ami.py ami_syn.csv donnees/surprise_ami_v2.csv gpt2
python 4_analyse.py donnees/surprise_livre_v2.csv donnees/surprise_ami_v2.csv
python 5_resultats_complets.py
python 6_figures.py
python 7_robustesse.py
python 8_indices_eeg.py
python 9_dernier_decile.py
python 10_dialogpt.py
python test_prefixe.py
```

- `1_mots_*.py`: one row per word. `csv` mode reuses the report's words and sentences; `textgrid` / `xml` mode rebuilds them from the TextGrids (with the `.wav` files for pitch) or from the AMI `*.words.xml` files.
- `2_surprise_syntaxique.py`: incremental PCFG surprisal (Hale 2001). About 15 s for the book on 4 cores, a few minutes for the 12 AMI meetings.
- `3_surprise_llm_*.py`: GPT-2 surprisal. The last argument can be any Hugging Face causal model (`gpt2-xl`, `meta-llama/Llama-3.2-1B`, ...) or a local folder. `3_surprise_llm_ami.py` also takes `--fenetre N` (window size in tokens, 1,024 by default), `--tours` (speaker turns separated by the end-of-text token, as in DialoGPT's training data), `--enonce` (each utterance read alone, as in the report version) and `--sans-tiret` (compound words written with a space). The defaults give the v2 values.
- `lancer_gpt2.py`: runs both step-3 scripts on the files in `donnees/` and writes `resultats_v2.txt`.
- `4_analyse.py`: the report's duration models on the v2 files (coefficients in seconds per nat).
- `5_resultats_complets.py`: all the numbers of the comparison for the 4 cases (syntactic and GPT-2, book and AMI) and for the syntactic surprisal on all 171 AMI meetings: correlations, coefficients without and with frequency and control by control, mixed model (with and without frequency), EXTRA and partial correlation, then the 171 meetings against the 12 matched meetings (section 4.3). Output: `resultats_complets_v2.txt`. Same content as `resultats_comparaison.py` in the report version, plus the "AMI complet" line of Annex C.
- `6_figures.py`: the same 16 figures as `version_rapport/4_figures` (decile figures, surprisal vs frequency, EXTRA), recomputed with the v2 values, plus 3 AMI figures: syntactic surprisal on 171 meetings against the 12 matched meetings (`cmp_ami171_*`) and the AMI residual figure with the speaker also controlled (`cmp_ami_rouge_locuteur`). Output in `figures/`.
- `7_robustesse.py`: the additional analyses and robustness checks for the 5 cases (`resultats_robustesse_v2.txt`): cleaning step by step, the full model with every coefficient and the speaker variance, AIC and likelihood-ratio test, Demberg's two-step method against the joint model (section 4.1), a quadratic frequency term, the model without the top surprisal decile, the AMI mixed model with each optimizer in ML and REML, both measures on exactly the same AMI words, and the AMI cleaning variants: hesitations kept, no 4-word threshold, no speech rate, and all three at once (the 0-2 s duration filter is kept in every variant).
- `8_indices_eeg.py`: the four EEG files in `indices_eeg/` (same computation as `version_rapport/5_indices_eeg/`, which it reproduces to within 5e-7 on the report CSVs).
- `9_dernier_decile.py`: the top surprisal decile for the 4 cases (size, which words, residuals by frequency, effect of a quadratic frequency term and of removing it), in `resultats_dernier_decile_v2.txt`, and the decile figures with 10 and with 9 deciles (`figures/dec_*`).
- `10_dialogpt.py`: the test with a conversational model (DialoGPT, section 3.5 of the report) continued with v2: GPT-2 and DialoGPT with the same context and the same text, one utterance at a time as in the report, and both measures in the same model (`resultats_dialogpt_v2.txt`, `figures/cmp_ami_dialogpt.png` and `cmp_ami_dialogpt_enonce.png`). The commands that make the `donnees/dialogpt/` files are at the top of the script.
- `test_prefixe.py`: checks the incremental parser against a brute-force enumeration on two small grammars (section 3.4 of the report). The largest gap is 5e-14 nats.
- `commun.py`: parser, grammar, out-of-vocabulary rule, acronyms, sliding-window scoring for the language model.

## Reports (`comptes_rendus/`)

- `compte_rendu_v2.docx`: what changed in v2 and what it changes to the results, following the earlier comparison report, and the DialoGPT test on AMI (in French).
- `analyse_extra_v2.docx`: the EXTRA analysis redone with the v2 values (in French).
- `figures/`: the figures of both documents.

## Results

Same models as in the report (duration ~ length × frequency + word class [+ speech rate] + surprisal; 0 < duration ≤ 2 s; AMI without hesitations and with utterances of at least 4 words). Coefficients are in s/nat; multiply by ln 2 = 0.693 to compare with the report's s/bit.

| | Report (s/bit) | v2 (s/nat) | v2 in s/bit |
|---|---|---|---|
| Syntactic, book, OLS | +0.00168, p = 4.2e-10 (n = 10,672) | +0.00288, p = 8.2e-13 (n = 11,413) | +0.00200 |
| Syntactic, AMI 12 meetings, mixed model (1 \| speaker) | +0.00077, p = 0.028 (fit did not converge) | +0.00129, p = 0.019 (converged) | +0.00089 |
| GPT-2, book, OLS | -0.00022, p = 0.27 (n = 11,413) | -0.00092, p = 0.0064 (n = 11,411) | -0.00064 |
| GPT-2, AMI 12 meetings, mixed model | -0.00180, p = 7.8e-14 | -0.00408, p = 9.6e-24 (converged) | -0.00283 |
| Syntactic, AMI 171 meetings, OLS ("AMI complet") | +0.00110, p = 8.5e-32 (about 846,000 words) | +0.00149, p = 2e-24 (n = 845,392) | +0.00103 |

The main result of the report still holds: with frequency and length controlled, syntactic surprisal predicts longer durations and GPT-2 surprisal does not. The difference is that the GPT-2 coefficient for the book is now small but significantly negative instead of null. The new GPT-2 input changes the values a lot: v2 GPT-2 surprisal correlates r = 0.66 (Spearman 0.63) with the report's values for the book, and r = 0.73 for AMI. GPT-2 surprisal is also more correlated with frequency than syntactic surprisal (book r = -0.63 against -0.55). Full output: `resultats_v2.txt`.

Caveat for AMI: if only the words in the grammar's vocabulary are kept, the AMI syntactic effect is negative (-0.00848 s/nat, p = 3.6e-22). About 14 % of the analysed AMI words (12.6 % in the book) are out of vocabulary and get the surprisal of a stand-in word, and they carry the positive effect. This was already the case in the report version. The AMI syntactic effect on the 12 matched meetings is also sensitive to the frequency control: with a quadratic frequency term it is +0.00108 (p = 0.075, mixed model), and without the top surprisal decile +0.00037 (p = 0.58). On the 171 meetings it stays positive in both cases (+0.00090 and +0.00035, p = 0.048). The book syntactic effect stays positive and significant in every check (`resultats_robustesse_v2.txt`).

DialoGPT (section 3.5 of the report): during the internship, a test with DialoGPT, a conversational model, gave a coefficient close to 0 on AMI (-0.0003 s/bit, p = 0.10, in an internship progress report), which suggested that GPT-2's negative coefficient came from its written training data. v2 takes this test further. In the same setup as the internship (one utterance at a time, hyphens kept), DialoGPT-small again gives a small coefficient (-0.00063 s/nat, p = 0.005). DialoGPT, however, gives very high surprisal values to words written with a hyphen (flip-top, jog-dial: 73 nats on average for DialoGPT-small, 14 for GPT-2), because its training texts contain almost no hyphens, and these 138 analysed words are long. With compound words written with a space for both models, DialoGPT gives the same negative coefficient as GPT-2: -0.00408 (p = 1.4e-30) against -0.00400 with the same 64-token context, -0.00332 against -0.00352 one utterance at a time, and -0.00375 for DialoGPT-medium. With both measures in the same model, DialoGPT keeps the negative effect (-0.00335, p = 7e-7) and GPT-2 does not (-0.00093, p = 0.2). The negative coefficient on AMI is therefore also found with a conversational model, and is not specific to a model trained on written text. The DialoGPT values of the internship were not kept, so the two runs cannot be compared word by word. DialoGPT is used with a short context (64 tokens for small, 32 for medium) because it predicts the meetings much worse with a long one (9.1 nats per word on average with 1,024 tokens against 5.0 with 64), whereas GPT-2 is best with 1,024 tokens (4.2 nats).

Two-step method (section 4.1): with Demberg's two steps, the coefficients are smaller than with the joint model but have the same sign (book syntactic +0.00152, p = 1.9e-7; book GPT-2 -0.00053, p = 0.039; AMI syntactic +0.00075, p = 0.074; AMI GPT-2 -0.00263, p = 7e-16).

Missing values: 19 AMI words have no syntactic value in v2 (the report version had a value for every word). The tagger gives the quote tag '' to a few words ('kay 11 times, 'ca, 'cause). The cleaned grammar has no such category, so there is no stand-in word, and the prefix probability is 0 from that word to the end of its sentence. For the same reason, 806 words of the 171-meeting file have no value. 8 of the 19 are among the analysed words; with these 8 also removed from the GPT-2 analysis, so that both measures use the same 60,277 words, the GPT-2 coefficient is unchanged (-0.00408 s/nat, p = 8.4e-24). Two book words have no GPT-2 value: "not angry" in `audio01`, which the reader says but which is not in the book text.
