# State of the art — around Tax et al. (2017)

## Timeline

```
Before 2017: one method per task                     2017                       After 2017: Tax et al. as baseline
─────────────────────────────────                    ────                       ──────────────────────────────────
Time:     van der Aalst et al. (2011)                Tax et al.:                Camargo et al. (2019): LSTM + more attributes
          transition systems + time annotations      one LSTM, shared +         Di Mauro / Pasquadibisceglie (2019): CNNs
Activity: Breuker et al. (2016) probabilistic models specialised layers,        Taymouri et al. (2020): GANs
          Le et al., Markov / sequence models        several tasks              Bukhsh et al. (2021): ProcessTransformer
Deep learning: Evermann et al. (2016-2017)
          first LSTM, word embeddings
```

## 1. Predicting time-related properties

**van der Aalst, Schonenberg & Song (2011)** build a *transition system* from the event log: each state is an abstraction of the prefix (the *set*, *bag* or *sequence* of activities seen so far) and is annotated with the times observed in the log. The prediction for a running case is the average time stored in its current state. Other approaches use stochastic Petri nets or queueing models. These methods are interpretable but were designed for one task (time).

## 2. Predicting the next activity

**Breuker, Matzner, Delfmann & Becker (2016)** learn probabilistic models of the process (inspired by grammatical inference) to predict the next event, with a focus on comprehensible models. Other works used Markov models or sequence-pattern methods.

## 3. First deep-learning approaches

**Evermann, Rehse & Fettke (2016, journal version 2017)** were the first to use an LSTM for next-event prediction, inspired by language models (activities treated like words, with embeddings). They showed the potential of deep learning but did not predict time.

## 4. Contribution of Tax et al. (2017)

- One LSTM architecture for several tasks (next activity, next time, suffix, remaining time) instead of tailor-made methods.
- Multi-task learning with shared and specialised layers.
- Better results than all the baselines above on both logs (e.g. BPI'12 W: 76.0% accuracy vs 71.9% for Breuker et al.; 1.56 vs 1.91 days MAE for van der Aalst et al.).
- Public code, which made it the standard baseline.

## 5. After 2017

Most later work in predictive process monitoring compares to Tax et al.: richer LSTMs with more event attributes (Camargo, Dumas & González-Rojas, 2019), convolutional networks (Di Mauro et al., 2019; Pasquadibisceglie et al., 2019), generative adversarial nets (Taymouri et al., 2020) and Transformers (Bukhsh, Saeed & Dijkman, 2021 — reproduced as an extension in `extension/processtransformer/`). Benchmark studies later pointed out that differences in preprocessing, data splits and metrics make published numbers hard to compare.

## 6. Open questions relevant for my reproduction

- **Comparability:** results depend strongly on the split and on which prefixes are evaluated.
- **Simple baselines:** many business processes are repetitive; how far does a frequency table get?
- **Business adoption:** interpretability and maintenance (code written in 2017 no longer runs as-is).

*Note: publication years of the baselines are taken from the reference list of Tax et al. (2017) and should be double-checked against the PDF.*
