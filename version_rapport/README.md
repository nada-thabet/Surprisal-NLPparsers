# Report version: the CSVs and the code behind the internship report

This folder reproduces the numbers of the report's Annexes C.1, C.2 and D, and all its figures. The segmenter evaluation (Annex C.3) is done by separate scripts (see the end of this file).

## 1_scripts_generation_csv: scripts that generate the CSVs

| Script | Output | Corpus / measure |
|---|---|---|
| `surprise_incrementale.py` | `surprisal_incrementale.csv` | The Old Man and the Sea, syntactic surprisal: NLTK PCFG learned from the Penn Treebank, prefix probabilities with Jelinek–Lafferty |
| `surprise_llm_textgrid.py` | `surprisal_llm_textgrid.csv` | The Old Man and the Sea, GPT-2 surprisal (token surprisals summed per word) |
| `surprise_incrementale_ami.py` | `surprisal_incrementale_ami.csv` | AMI, syntactic surprisal |
| `surprise_llm_ami.py` | `surprisal_llm_ami.csv` | AMI, GPT-2 surprisal |
| `segmentation_prosodique.py`, `constituants_dataset.py` | (helpers) | Prosodic sentence segmentation, out-of-vocabulary words, Zipf frequency, phone counts. `constituants_dataset.py` comes from the constituency analysis I dropped: that analysis is no longer used, but the final scripts import four of its functions (`load_sentences`, `fix_oov`, `word_freq`, `count_phones`) in `surprise_incrementale.py`, `surprise_incrementale_ami.py` and `surprise_llm_textgrid.py`, and so does `v2/1_mots_livre.py` in `textgrid` mode. It has to stay in this folder. |
| `ami_dataset.py` | (helper) | Reads the AMI `*.words.xml` files. Used by the two AMI scripts. |

Sentences: for the book, the TextGrids have no punctuation, so sentences come from my prosodic segmenter (pause, final lengthening and pitch reset, then POS rules). For AMI, sentences come from the punctuation of the transcript, and sentences over 40 words are cut into chunks of 40.

Run from a folder that contains the seven scripts:
```
pip install nltk scipy numpy wordfreq praat-parselmouth torch transformers
python surprise_incrementale.py  ../../stimuli_livre   (+ the .wav files for the pitch cue)
python surprise_llm_textgrid.py  <same folder>
python surprise_incrementale_ami.py <folder with the .words.xml of the 12 meetings>
python surprise_llm_ami.py          <same folder>
```

Without the `.wav` files there is no pitch cue, so about 3 of the ~800 sentence boundaries change (in audio03, audio13 and audio17). The segmentation used in the CSVs was made with the audio and is saved in `../stimuli_livre/phrases_prosodiques.txt`.

## 2_csv_du_rapport: the CSVs used in the report

| File | Rows | After cleaning (words analysed) |
|---|---|---|
| `surprisal_incrementale.csv` | 10,675 | 10,672 (Annex D) |
| `surprisal_llm_textgrid.csv` | 11,416 | 11,413 (Annex D) |
| `surprisal_incrementale_ami.csv` | 67,381 | 60,285, 48 speakers (Annex D) |
| `surprisal_llm_ami.csv` | 67,381 | 60,285, 48 speakers (Annex D) |
| `surprisal_incrementale_ami_171reunions.csv.gz` | 978,212 | all 171 meetings ("AMI complet" line of Annex C) |

The two AMI files cover the same 12 meetings (ES2002a–d, ES2003a–d, ES2004a–d): these are the matched data of section 4.3. The file for all 171 meetings is gzip-compressed because of GitHub's size limit (`pandas.read_csv` reads it directly).

Cleaning is done in the analysis scripts: 0 < duration ≤ 2 s; for AMI, fillers removed and utterances of at least 4 words.

## 3_analyse_et_figures: numbers and figures

Put the CSVs next to these scripts and run:

| Script | Output |
|---|---|
| `resultats_comparaison.py` | Coefficients, p-values, confidence intervals, correlations and EXTRA values (Annex C). Expected output: `resultats_attendus.txt` |
| `figures_comparaisons.py` | The 10 decile figures (Fig. 1 and 6–14) |
| `frequence_surprise.py` | Surprisal vs frequency figures and the \|r\| bar chart (Fig. 2–4) |
| `figure_extra.py` | The EXTRA figure (Fig. 5) |

### Values

| Case | Value |
|---|---|
| Read, syntactic (OLS) | +0.00168, p = 4.2e-10, [+0.00115 ; +0.00221] |
| Read, GPT-2 (OLS) | -0.00022, p = 0.27, [-0.00062 ; +0.00017] |
| AMI matched, syntactic (mixed) | +0.00077, p = 0.028, [+0.00008 ; +0.00145] |
| AMI matched, GPT-2 (mixed) | -0.00179, p = 1.0e-13, [-0.00227 ; -0.00132] |
| AMI matched, OLS syntactic / GPT-2 | +0.00069 (p = 0.050) / -0.00186 (p = 1.2e-14) |
| EXTRA (read syn, read GPT-2, AMI syn, AMI GPT-2) | +0.066, +0.009, +0.034, -0.022 (Annex C.2 gives -0.023 because it subtracts rounded values) |

Note on the AMI mixed models: the two lines above come from statsmodels fits with `method="lbfgs"` (ML for the syntactic line, REML for the GPT-2 line) that do not converge properly: the speaker variance is 0 and the log-likelihood is infinite. Fits that converge (for example `method="powell"` or `"nm"`) give: syntactic +0.00076, p = 0.030, CI [+0.00007 ; +0.00144]; GPT-2 -0.00180, p ≈ 8e-14, CI [-0.00228 ; -0.00133]. The conclusions are the same. The v2 analysis uses the converged fits.

## 4_figures

The figures produced by the scripts above.

## 5_indices_eeg: surprisal after control (section 5.4)

`surprise_apres_controle_total.py` and `surprise_apres_controle.py` create the four CSVs of this folder from the two read-speech CSVs (run them from `2_csv_du_rapport/`). The new columns `surprise_syn_apres_controle`, `surprise_llm_apres_controle` (length, frequency, their interaction and word class removed) and `surprise_*_ctrlfreq` (frequency removed) are regression residuals: their mean is 0, so about half of the values are negative. The raw surprisal columns are never negative.

## Segmenter evaluation

The evaluation of the segmenter (Annex C.3: precision, recall, F1 and the comparison with Whisper) is done by `comparer_segmentation.py` and `comparer_whisper.py`, which are not in this folder.
