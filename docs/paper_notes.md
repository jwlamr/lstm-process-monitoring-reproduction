# Paper notes — Tax, Verenich, La Rosa & Dumas (2017)

**Reference:** N. Tax, I. Verenich, M. La Rosa, M. Dumas. *Predictive Business Process Monitoring with LSTM Neural Networks.* Proceedings of the 29th International Conference on Advanced Information Systems Engineering (CAiSE 2017), Springer LNCS, pp. 477–492. [arXiv:1612.02130](https://arxiv.org/abs/1612.02130)
**Code:** https://github.com/verenich/ProcessSequencePrediction
**Why a "pillar" paper:** cited 738 times (OpenAlex, November 2026). It is the reference baseline of predictive process monitoring: later approaches (Camargo et al. 2019, CNN-based models, ProcessTransformer 2021…) all compare to it.

## 1. Business problem

Companies record each step of their processes (tickets, loan applications, orders) in an **event log**: case ID, activity, timestamp. **Predictive business process monitoring** uses completed cases to make predictions about running cases:

| Task | Business question |
|---|---|
| Next activity | What will happen next in this case? |
| Next event time | When will it happen? |
| Suffix | What is the full remaining sequence of activities? |
| Remaining time | When will the case be finished? (e.g. to prioritise cases and avoid deadline violations) |

## 2. Gap identified by the authors

Existing methods were **tailor-made for one specific task** (one method for time, another for next activity…), and their accuracy varied a lot from one log to another and from one point in the case to another.

## 3. Method

- **Input per event:** one-hot encoding of the activity + time features: time since previous event, time within the day, time within the week (the code also adds the position in the case and the time since case start).
- **Architecture:** LSTM network(s), 100 neurons per layer. Three variants: separate models for activity and time, fully shared multi-task model, and a hybrid with **shared layers followed by task-specific layers** (best overall).
- **Outputs:** softmax over activities (+ an "end of case" symbol) and a regression output for the time until the next event.
- **Losses:** cross-entropy (activity) and mean absolute error (time).
- **Suffix and remaining time:** predicted by applying the model iteratively until "end of case" is predicted.

## 4. Experimental protocol

- **Logs:** Helpdesk (ticketing process of an Italian software company; 3,804 cases, 13,710 events, 9 activities) and BPI'12 W (work items of a loan application process of a Dutch bank, "complete" events only; 9,658 cases, 72,413 events, 6 activities in the file used by the code).
- **Split:** chronological, first 2/3 of cases for training, last 1/3 for testing.
- **Evaluation:** every prefix of length 2 ≤ k < case length; accuracy for next activity, MAE in days for next event time, averaged over all predictions ("All" column).

## 5. Key results (Table 1, next activity and next event time)

| Model | Helpdesk MAE (days) | Helpdesk accuracy | BPI'12 W MAE (days) | BPI'12 W accuracy |
|---|---|---|---|---|
| LSTM, 2 layers, 1 shared (**configuration of the published code**) | 3.77 | 0.7119 | 1.56 | **0.7600** |
| LSTM, 3 layers, 1 shared | 3.76 | **0.7123** | – | – |
| LSTM, 1 layer, 1 shared | **3.75** | 0.7072 | 1.57 | 0.7486 |
| Simple RNN, 2 layers, 1 shared | 3.98 | 0.6624 | – | – |
| van der Aalst et al., sequence abstraction (time) | 5.67 | – | 1.91 | – |
| Breuker et al. (next activity) | – | – | – | 0.719 |
| Evermann et al. (next activity) | – | – | – | 0.623 |

## 6. Points to check during reproduction (my questions)

1. Does the 2017 code (Python 2, Keras 2.0) still run, and with how many changes?
2. Do we obtain the published numbers with the published configuration?
3. Does the code match the paper (optimizer, features)?
4. How much does the LSTM add compared with a simple frequency rule on the same split?
