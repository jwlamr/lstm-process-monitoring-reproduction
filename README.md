# Reproducing Tax et al. (2017) — Predictive Business Process Monitoring with LSTM Neural Networks

**Student:** Jules Amimer — MSc AI for Business (NEOMA x INSA)
**Paper:** N. Tax, I. Verenich, M. La Rosa, M. Dumas (2017). *Predictive Business Process Monitoring with LSTM Neural Networks.* CAiSE 2017, Springer LNCS, pp. 477–492. [arXiv:1612.02130](https://arxiv.org/abs/1612.02130) — [official code](https://github.com/verenich/ProcessSequencePrediction)

## Why this paper
Companies record every step of their processes (support tickets, loan applications, orders) in **event logs**. *Predictive business process monitoring* uses these logs to predict how a running case will continue: **what happens next, and when?** Tax et al. (2017) showed that a single LSTM neural network can replace the tailor-made methods used before for each prediction task. With 738 citations (OpenAlex, Nov. 2026) it is the **reference paper of the field**: later work (CNNs, GANs, Transformers) uses it as its baseline. It connects directly to my professional objective, AI consulting: putting AI inside real business processes.

## What I did
1. Read the paper and mapped its state of the art → [`docs/paper_notes.md`](docs/paper_notes.md), [`docs/state_of_the_art.md`](docs/state_of_the_art.md)
2. **Ported the authors' 2017 code** (Python 2.7 / Keras 2.0) to Python 3 / `tf-keras` with minimal changes → [`code/tax2017/`](code/tax2017/), [`python3_port.diff`](code/tax2017/python3_port.diff)
3. **Reproduced Table 1 on Helpdesk** (next activity accuracy, next event time MAE) with the configuration published in the repository → [`results/tax2017/`](results/tax2017/)
4. **Found and tested a bug** in the official evaluation script (time features of the test cases) → [`code/tax2017/evaluate_fixed.py`](code/tax2017/evaluate_fixed.py)
5. **Extension:** compared the LSTM with simple frequency rules on exactly the same split → [`code/tax2017/baselines_tax_split.py`](code/tax2017/baselines_tax_split.py)
6. **Extension:** reproduction of a recent model of the field, ProcessTransformer (Bukhsh et al., 2021) → [`extension/processtransformer/`](extension/processtransformer/)
7. Documented everything day by day → [`logbook.md`](logbook.md)

## Results — Helpdesk (3,804 cases, 13,710 events, 9 activities)

| | Next activity accuracy | Next event time MAE |
|---|---|---|
| Paper — LSTM, 2 layers, 1 shared (published configuration) | 71.19% | 3.77 days |
| **Our reproduction** — run 2 (full training, early stopping at epoch 62) | **70.75%** | **3.62 days** |
| Our reproduction — run 1 (training interrupted at epoch 53, best checkpoint) | 71.00% | 3.62 days |
| Our run 2 with the evaluation bug fixed | 70.70% | 3.62 days |
| Simple rule: most frequent next activity after the last one | 69.44% | — |
| Simple rule: median time to next event after the last activity | — | 3.66 days |
| Process mining baseline reported in the paper (van der Aalst et al.) | — | 5.67 days |

![Helpdesk comparison](results/tax2017/helpdesk_comparison.png)

**Key findings**
- **The reproduction is successful:** 70.75% vs 71.19% accuracy and 3.62 vs 3.77 days MAE, with the authors' code and data.
- **The 2017 code no longer runs as-is** (Python 2, Keras 2.0 APIs); a minimal port was needed.
- **Evaluation bug:** the official evaluation script never resets the list of timestamps between cases, so every test case receives the time-of-day and day-of-week features of the first events of the log. Fixing it changes the results by less than 0.1 point: the model hardly uses these features.
- **A simple frequency rule is almost as good as the LSTM** on Helpdesk: 69.4% vs 70.8% accuracy, and the same time error (3.66 vs 3.62 days). The LSTM's advantage comes mostly from prefixes of length 3. On BPI'12 W, the same rule reaches 78.7% accuracy (paper's LSTM: 76.0%), to be confirmed by running the LSTM ourselves.

![Accuracy by prefix length](results/tax2017/helpdesk_accuracy_by_prefix.png)

**Not done yet:** BPI'12 W with the LSTM (several hours on CPU; planned on a Colab GPU), suffix and remaining-time prediction, several runs for variance.

## Repository structure
```
├── README.md
├── logbook.md                      ← day-by-day work log
├── docs/                           ← paper notes, state of the art, peer-evaluation template
├── notebooks/Tax2017_reproduction.ipynb   ← Colab notebook, end to end
├── code/tax2017/                   ← Python 3 port of the official code, bug fix, baselines, figures
├── results/tax2017/                ← predictions, summaries, logs, figures, trained model
├── extension/processtransformer/   ← reproduction of ProcessTransformer (2021) on Helpdesk
├── slides/                         ← presentations
└── report/                         ← final report (end of December)
```

## How to reproduce
Open [`notebooks/Tax2017_reproduction.ipynb`](notebooks/Tax2017_reproduction.ipynb) in Google Colab and run all cells (Helpdesk ≈ 15–25 min on CPU; use a GPU for BPI'12 W).

Locally:
```bash
git clone https://github.com/verenich/ProcessSequencePrediction.git && cd ProcessSequencePrediction/code
pip install tf-keras distance jellyfish && export TF_USE_LEGACY_KERAS=1
cp <this repo>/code/tax2017/*.py .
python train.py helpdesk.csv
python evaluate_next_activity_and_time.py helpdesk.csv output_files/models/<best model>.h5
python summarize_results.py output_files/results/next_activity_and_time_helpdesk.csv
python baselines_tax_split.py ../data/helpdesk.csv
```

## Credits
Model code and data: Tax, Verenich, La Rosa & Dumas (2017). My additions (port, bug fix, baselines, analysis, documents) are released under MIT.
