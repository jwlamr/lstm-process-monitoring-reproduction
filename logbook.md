# Logbook — Paper reproduction (Tax et al. 2017; first attempt: ProcessTransformer 2021)

Format: date — what I did — result / problem — decision.

---

## 18/11/2026 — Project brief
- Brief received: choose a paper related to my professional objective, understand it and its state of the art, reproduce the experimental results. Deliverables: one repository with slides, report and logbook (end of December); 5-minute pitch on 24/11; peer evaluation of the other pitches.

## 21/11/2026 — Paper selection
- Theme: **AI applied to business**, more precisely AI inside business processes.
- Shortlist:
  1. ProcessTransformer (Bukhsh et al., 2021): predicting how a business process will continue, from event logs.
  2. LayoutLMv3 on receipts (CORD): invoice data extraction.
  3. Chronos (Amazon, 2024): demand forecasting with pretrained models.
- **Decision: ProcessTransformer.** It is the closest to "AI in processes", the code and data are public, the model is small enough for Colab/CPU, and the business story (SLAs, staffing, customer service) fits an AI-consulting profile.

## 21/11/2026 — Reading the paper and state of the art
- Read the paper; notes in `docs/paper_notes.md`.
- Built the state-of-the-art timeline (process mining → classical ML → LSTM/CNN → Transformer) in `docs/state_of_the_art.md`.
- Target numbers for Helpdesk: next activity accuracy 85.63%, next event time MAE 2.98 days, remaining time MAE 3.72 days.
- Noted differences between the paper and the code before running anything:
  - paper: learning rate 10⁻², global max-pooling, dense layers 32/128;
  - code: learning rate 10⁻³ (default), global *average* pooling, dense layer 64 for next activity, 10 epochs by default (README says 100).
  - Decision: follow the **README commands** (official code, 100 epochs, code defaults), because this is what the authors published for reproduction.

## 21/11/2026 — Getting the data
- The paper uses the 4TU Research Data repository. Direct download from 4TU was blocked in my environment.
- First copy found on GitHub (Tax et al. repository) had only 3,804 cases / 13,710 events / 9 activities: it is a filtered version, **not** the one in the paper → rejected.
- Second copy (GitHub, LFS file) has exactly 4,580 cases / 21,348 events / 14 activities and the column names expected by the code (`Case ID`, `Activity`, `Complete Timestamp`) → matches Table 1 of the paper → used.

## 21/11/2026 — Running the official code: compatibility problems
Environment: Python 3.13, TensorFlow 2.21, pandas 3, NumPy 2 (CPU, 2 cores).
1. **Keras 3:** the code was written for TensorFlow 2.4 / Keras 2. Fix without touching the code: install `tf-keras` and set `TF_USE_LEGACY_KERAS=1`.
2. **Date parsing crash:** `pd.to_datetime(..., dayfirst=True)` on `2012/10/09`-style timestamps → pandas guesses `%Y-%d-%m` and crashes on the first date with day > 12. With other versions, day and month could be silently swapped, which would corrupt all time features. Fix: `dayfirst=False`.
3. **NumPy 2:** `np.array_split(DataFrame)` now returns arrays → preprocessing crashes. Fix: split row indices instead.
- After these 2 one-line fixes, preprocessing runs (16 s for next activity, 36 s for each time task).
- Dataset after preprocessing: 13,443 train / 3,325 test prefixes (next activity); 17,107 / 4,241 (time tasks).

## 21/11/2026 — Observations on the evaluation code
- Metrics are computed per prefix length k and then **averaged with equal weight** for every k. The test set has 916 samples for k=0 but only 1 sample for the longest prefix → a single sample counts as much as 916. I will report both the paper's average and an average weighted by number of samples.
- Bug in `next_activity.py`: the summary row appended to the results CSV uses the accuracy of the **last** prefix instead of the mean, and this row is then included in the printed "average accuracy". I recompute the averages from the per-prefix rows.
- The train/test split takes the first 80% of cases in file order (`sort_temporally=False` by default); the paper describes a chronological split.

## 21/11/2026 — Extension: simple baselines
- Question: how much does the Transformer add compared with simple rules a consultant could build in Excel?
- Wrote `code/baselines.py`: majority class, "most frequent next activity after the last activity", "most frequent next activity after the full prefix"; and for time tasks, mean time by last activity / by (prefix length, last activity). Same split and same metrics as the paper.
- Next activity, paper protocol: last-activity rule **79.3%** accuracy, prefix lookup 79.1% (vs majority 26.2%).
- Next event time MAE: 3.31 days; remaining time MAE: 4.38 days (best simple baseline).
- First insight: the simple rules already beat all the LSTM/CNN baselines reported in the paper for Helpdesk (best: 76.51% accuracy, 5.77 / 10.38 days MAE). The Helpdesk process is very repetitive.

## 21/11/2026 — Training (official scripts, 100 epochs)
- Launched next activity, next time and remaining time training (README commands, 100 epochs, batch size 12, lr 0.001) on CPU.
- CPU with 2 cores: 9 runs in parallel were far too slow (1 epoch per minute), so I reduced to 1 run per task first, then added more runs.
- **Run 1 results (paper protocol):** next activity 84.27% accuracy (paper 85.63%), next event time MAE 2.87 days (paper 2.98), remaining time MAE 3.83 days (paper 3.72) → reproduction successful.
- The script printed 85.48% for next activity, but this value includes the summary-row bug. Recomputed correctly: 84.27%.
- **Weighted by number of test cases:** 80.15% accuracy, 4.03 / 5.34 days MAE.
- Accuracy by prefix length: for prefixes of 1–3 events (81% of test cases) the Transformer and the last-activity rule are almost identical. The Transformer only wins on long, rare cases.

## 21/11/2026 — Variance
- Next activity: 3 runs → 83.69% ± 0.77 (paper protocol), 80.07% ± 0.25 (weighted). Printed by the script: 85.48 / 84.14 / 85.21%.
- Time tasks: 2 runs → next time 2.88 ± 0.02 days, remaining time 3.98 ± 0.21 days.
- Conclusion: the results are stable; the averaging method has a much bigger effect (≈3.6 points) than randomness between runs (<1 point).

## 21/11/2026 — Deliverables
- Pitch slides for 24/11 (8 slides, speaker notes with timing).
- Report draft with results, discussion and business perspective (`report/`).
- Colab notebook that replays everything end to end.

## 23/11/2026 — Change of paper: a "pillar" paper is required
- Clarification of the brief: the paper must be a well-known, foundational ("pillar") paper.
- Checked citations on OpenAlex: ProcessTransformer (2021) = 152 citations (preprint); **Tax et al. (2017), CAiSE = 738 citations**, the reference baseline of predictive process monitoring; Evermann et al. (2017) = 518; Camargo et al. (2019) = 319.
- **Decision: switch to Tax et al. (2017).** Same business problem and same Helpdesk process, so the work on ProcessTransformer is kept as an extension ("recent model") in `extension/processtransformer/`.
- Pitch for 24/11 rebuilt around Tax et al.: business problem, state of the art (van der Aalst 2011, Breuker 2016, Evermann 2016), the paper, reproduction plan, planning. Results kept out of the pitch on purpose.
- Repository renamed `lstm-process-monitoring-reproduction`.

## 23/11/2026 — Reading Tax et al. (2017)
- Notes in `docs/paper_notes.md`, state of the art in `docs/state_of_the_art.md`.
- Targets (Table 1, configuration of the published code: LSTM, 2 layers, 1 shared, 100 neurons): Helpdesk 0.7119 accuracy / 3.77 days MAE; BPI'12 W 0.7600 / 1.56 days.
- Protocol: chronological split, first 2/3 of cases for training, last 1/3 for testing; prefixes 2 ≤ k < case length.
- Paper vs code: the paper mentions Adam, the code uses Nadam (lr 0.002, gradient clipping), early stopping with patience 42, up to 500 epochs, batch size = longest case. I follow the code.

## 23/11/2026 — Porting the 2017 code
- Official repository: Python 2.7, Keras 2.0 (`izip`, `unichr`, `unicode`, `keras.layers.recurrent`, `Nadam(schedule_decay=…)`, files opened in binary mode). It does not run with today's Python.
- Minimal port to Python 3 + `tf-keras` (Keras 2 API): imports, Python 2 syntax, file modes, log name and model file passed as arguments. 48 changed lines in `train.py`, 28 in the evaluation script (`code/tax2017/python3_port.diff`). No change to model, features, split, training or evaluation logic.
- The data in the authors' repository matches the paper: Helpdesk 3,804 cases / 13,710 events / 9 activities; BPI'12 W 9,658 cases / 72,413 events / 6 activities.

## 23/11/2026 — Bug found in the official evaluation script
- In `evaluate_next_activity_and_time.py`, the lists `times2` and `times3` are never reset when a new case starts. All cases share the same list, so every test case gets the timestamps of the first events of the log → wrong time-of-day and day-of-week features at test time (checked with a small script: all 3,804 cases point to the same list).
- Wrote `evaluate_fixed.py` (2 lines added) to measure the effect.

## 23/11/2026 — Simple baselines on the same split
- `baselines_tax_split.py`: same split, same evaluated prefixes, same metrics.
- First version wrongly scored the end of the case as a "next activity"; the official evaluation does not (no row is written when the case has ended). Fixed → 1,993 test predictions on Helpdesk, exactly as the official script.
- Helpdesk: last-activity rule 69.44% accuracy; median time after the last activity 3.66 days MAE (means are worse: MAE is minimised by the median).
- BPI'12 W: last-activity rule **78.70%** accuracy (paper's LSTM: 76.00%; Breuker et al.: 71.9%); median by (prefix length, last activity) 1.56 days MAE (paper's LSTM: 1.56).

## 23/11/2026 — Training and evaluation on Helpdesk
- Run 1: training interrupted at epoch 53 by an environment restart. Evaluation of its best checkpoint (epoch 26): **71.00% accuracy, 3.62 days MAE**.
- Run 2: full training, early stopping at epoch 62 (best epoch 20, ~8 s/epoch on CPU). Evaluation: **70.75% accuracy, 3.62 days MAE** (paper: 71.19%, 3.77). → **Reproduction successful.**
- With the evaluation bug fixed: 70.70%, 3.62 days → the bug has almost no effect; the model hardly relies on these features.
- By prefix length: LSTM and simple rule are almost identical for k = 2 (76.9% vs 76.4%, 1,200 of 1,993 predictions); the LSTM's advantage comes from k = 3 (59.3% vs 50.3%); the rule is better for k = 4–5.
- BPI'12 W training started but stopped: ~150 s per epoch on 2 CPU cores, so several hours with early stopping → to be run on a Colab GPU (notebook section 7).

---

## Next steps (planned)
- [ ] 23/11: rehearse the 5-minute pitch on Tax et al. (target: under 5 min).
- [ ] 24/11: pitch + summarise and evaluate the other students (`docs/peer_evaluation_template.md`).
- [ ] Run `notebooks/Tax2017_reproduction.ipynb` myself on Google Colab (GPU), including BPI'12 W, and compare with these results.
- [x] ProcessTransformer extension: 3 runs next activity, 2 runs time tasks.
- [ ] ProcessTransformer extension: third run for the time tasks.
- [ ] Several runs on Helpdesk for variance; optional: suffix and remaining-time prediction.
- [ ] Write the final report on Tax et al. and finalise slides (end of December).
