# Paper notes — ProcessTransformer (Bukhsh, Saeed & Dijkman, 2021)

**Reference:** Z. A. Bukhsh, A. Saeed, R. M. Dijkman. *ProcessTransformer: Predictive Business Process Monitoring with Transformer Network.* arXiv:2104.00721, 2021. Eindhoven University of Technology.
**Code:** https://github.com/Zaharah/processtransformer (MIT licence, also on PyPI as `processtransformer`)

## 1. Business problem

Companies record every step of their processes in IT systems (ERP, CRM, ticketing tools). Each record is an *event*: which case (ticket, loan application, order), which activity, and when. A list of events is called an **event log**.

**Predictive Business Process Monitoring (PBPM)** uses these logs to answer questions about cases that are still running:

| Task | Business question | Example (IT helpdesk) |
|---|---|---|
| Next activity | What will happen next? | Will this ticket be closed, or escalated to another team? |
| Next event time | When will it happen? | How many days before the next action? |
| Remaining time | When will the case be finished? | Will we meet the 7-day SLA? |

Value for a company: anticipate delays, allocate staff, warn customers early, respect SLAs.

## 2. Gap identified by the authors

Earlier deep-learning approaches were mostly based on LSTMs (recurrent networks). According to the authors, they:
- struggle with long traces (long-range dependencies);
- often use one-hot encoding, which grows with the number of activities;
- process events one by one, which limits parallel training;
- often need hand-crafted features or a different architecture for each task.

## 3. Method

- **Input:** only the sequence of activity names of the running case (the *prefix*), encoded as integers and zero-padded to the longest trace.
- **Embedding:** token embedding + positional embedding (36 dimensions).
- **One Transformer block:** multi-head self-attention (4 heads) + feed-forward layer, with residual connections, dropout and layer normalisation.
- **Pooling + dense layers + output:** softmax over activities (next activity), or a single linear output (time tasks).
- **Time tasks:** three extra numeric features are concatenated: time since previous event, time since the event before that, time since case start.
- **Losses:** cross-entropy (classification), log-cosh (regression).
- **Training:** Adam, 100 epochs.

## 4. Experimental protocol

- **9 public event logs** from the 4TU repository (Helpdesk, BPIC 2012 and variants, BPIC 2013, BPIC 2020, Hospital, Traffic fines).
- **Split:** 80% of cases for training, 20% for testing (described as chronological); 20% of training held out for validation.
- **All prefix lengths** are used, no filtering.
- **Metrics:** accuracy and weighted F1 for next activity; MAE in days for time tasks. Metrics are computed for each prefix length k and then **averaged over k**.

## 5. Key results (Helpdesk = the dataset we reproduce)

| Task | ProcessTransformer | Best baseline cited |
|---|---|---|
| Next activity accuracy | **85.63%** (F1 0.82) | 76.51% (Camargo et al.) |
| Next event time MAE | **2.98 days** | 5.77 days (Tax et al.) |
| Remaining time MAE | **3.72 days** | 10.38 days (Navarin et al.) |

Across the 5 logs used for comparison, average next-activity accuracy is 80.58% vs 76.11% for the best baseline. The model is weaker on BPIC13 (62.11%), which the authors explain by fewer but longer cases.

## 6. Limitations stated by the authors

- Baselines used different preprocessing and extra features, so comparisons are not perfectly fair.
- A GAN-based method was not compared.
- Each task is trained separately (no multi-task model).

## 7. Points to check during reproduction (my own questions)

1. Does the official code run today (2026 library versions)?
2. Does the code match the paper (architecture, learning rate, epochs)?
3. Is the "average over prefix length" metric a fair summary? Large k values have very few test samples.
4. How much does deep learning add compared with a very simple baseline (e.g. "predict the most frequent next step after the last activity")?
