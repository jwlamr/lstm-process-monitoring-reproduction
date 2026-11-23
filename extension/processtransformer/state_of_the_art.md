# State of the art — Predictive Business Process Monitoring

## Timeline

```
Process mining (2000s–)      Classical ML (2010s)        Deep learning (2017–2020)          Attention (2021–)
──────────────────────       ────────────────────        ──────────────────────────         ─────────────────
Discover process models   →  Hand-crafted features   →   LSTMs on event sequences       →   Transformer
from event logs              + decision trees,           CNNs on "images" of traces          (ProcessTransformer)
(e.g. transition systems,    random forests              Memory networks, GANs
 annotated with times)
```

## 1. Classical process mining

Process mining discovers a process model (e.g. a Petri net or transition system) from an event log. Early predictive methods annotated these models with statistics, for example the average time remaining from each state. They are interpretable, but they depend on the quality of the discovered model and do not generalise well to complex, variable processes.

## 2. Classical machine learning

Prefixes of cases are turned into fixed-size feature vectors (counts of activities, last state, elapsed time…) and fed into models such as random forests or gradient boosting. Performance depends a lot on feature engineering.

## 3. Deep learning on sequences (the main baselines of the paper)

| Work | Idea |
|---|---|
| Evermann et al. (2017) | First LSTM for next-event prediction, with embeddings (inspired by language models). |
| Tax et al. (2017) | LSTM with one-hot encoding; predicts next activity, next timestamp, remaining time and the full suffix. Became the reference baseline. |
| Navarin et al. (2017) | LSTM for remaining time, using extra event attributes. |
| Khan et al. (2018) | Memory-augmented neural networks. |
| Camargo et al. (2019) | LSTM using both categorical (activity, role) and numeric features. |
| Di Mauro et al. (2019) | Inception-style CNN for next activity. |
| Pasquadibisceglie et al. (2019) | Event data transformed into image-like matrices for a CNN. |
| Taymouri et al. (2020) | GANs for suffix and remaining-time prediction. |

Common limits pointed out by the paper: sequential processing (slow), difficulty with long traces, heavy preprocessing or task-specific design.

## 4. ProcessTransformer (2021)

Applies the Transformer of Vaswani et al. (2017) — the architecture behind modern language models — to event logs. Self-attention looks at all events of the prefix at once, so it can capture long-range dependencies and be trained in parallel. One architecture is used for three tasks, with only the activity sequence as input (plus three time features for time tasks).

## 5. What this paper adds

- First Transformer-based approach for PBPM tasks, evaluated on 9 real logs.
- Higher average accuracy than LSTM/CNN baselines on next activity and lower MAE on time tasks.
- Simple input (no hand-crafted features), open-source code.

## 6. Open issues in the field (relevant for the discussion)

- **Comparability:** papers use different preprocessing, splits and metrics, so numbers from different papers are hard to compare. The authors themselves acknowledge this. Benchmark studies in the field (e.g. Rama-Maneiro, Vidal & Lama's review and benchmark of deep learning for PBPM) make the same point.
- **Simple baselines:** many processes are very repetitive, so a simple frequency-based model can already be strong. We test this in our reproduction.
- **Business adoption:** explainability and integration into operational tools remain key barriers for companies.
