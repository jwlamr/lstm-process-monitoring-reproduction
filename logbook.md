# Logbook — ProcessTransformer reproduction

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

---

## Next steps (planned)
- [ ] 22–23/11: rehearse the 5-minute pitch (target: 50 s per slide).
- [ ] 24/11: pitch + summarise and evaluate the other students (`docs/peer_evaluation_template.md`).
- [ ] Re-run the notebook myself on Google Colab and compare with these results (variance between runs).
- [x] Repeat training to measure variance (3 runs next activity, 2 runs time tasks).
- [ ] Third run for the time tasks.
- [ ] Optional: try the paper's learning rate (10⁻²) and a second dataset (BPIC 2012).
- [ ] Write the final report and finalise slides (end of December).
