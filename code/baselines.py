"""Simple non-deep-learning baselines for next-activity prediction on Helpdesk.
Uses the SAME train/test files produced by the official ProcessTransformer
preprocessing, and the SAME evaluation protocol (metrics per prefix length k,
then averaged over k), so results are directly comparable.

Baselines:
  1. Majority class      : always predict the most frequent next activity.
  2. Last-activity (1-gram Markov): most frequent next activity given the last event.
  3. Full-prefix lookup  : most frequent next activity given the whole prefix,
                           falling back to the last-activity model for unseen prefixes.
"""
import sys, json
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from sklearn import metrics

d = sys.argv[1] if len(sys.argv) > 1 else "datasets/helpdesk/processed"
out = sys.argv[2] if len(sys.argv) > 2 else "results/baselines_next_activity.csv"
tr = pd.read_csv(f"{d}/next_activity_train.csv")
te = pd.read_csv(f"{d}/next_activity_test.csv")

majority = tr.next_act.value_counts().idxmax()
last_tab = defaultdict(Counter); pref_tab = defaultdict(Counter)
for p, y in zip(tr.prefix, tr.next_act):
    last_tab[p.split()[-1]][y] += 1
    pref_tab[p][y] += 1

def pred_last(p):
    c = last_tab.get(p.split()[-1]);  return c.most_common(1)[0][0] if c else majority
def pred_prefix(p):
    c = pref_tab.get(p);  return c.most_common(1)[0][0] if c else pred_last(p)

models = {"majority": lambda p: majority, "last_activity": pred_last, "prefix_lookup": pred_prefix}
rows = []
for name, f in models.items():
    te[name] = [f(p) for p in te.prefix]
    accs, f1s = [], []
    for k, g in te.groupby("k"):
        accs.append(metrics.accuracy_score(g.next_act, g[name]))
        f1s.append(metrics.precision_recall_fscore_support(g.next_act, g[name], average="weighted", zero_division=0)[2])
    rows.append({"model": name,
                 "acc_mean_over_k": np.mean(accs), "f1_mean_over_k": np.mean(f1s),
                 "acc_overall": metrics.accuracy_score(te.next_act, te[name]),
                 "f1_overall": metrics.precision_recall_fscore_support(te.next_act, te[name], average="weighted", zero_division=0)[2]})
res = pd.DataFrame(rows); res.to_csv(out, index=False); print(res.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# Time tasks: naive baselines (same files, MAE per prefix length then averaged)
#   global_mean      : mean target of the training set
#   by_last_activity : mean target given the last activity of the prefix
#   by_k_last        : mean target given (prefix length k, last activity), fallback to by_last_activity
# ---------------------------------------------------------------------------
rows_t = []
for task, ycol in [("next_time", "next_time"), ("remaining_time", "remaining_time_days")]:
    tr_t = pd.read_csv(f"{d}/{task}_train.csv"); te_t = pd.read_csv(f"{d}/{task}_test.csv")
    for df_ in (tr_t, te_t):
        df_["last"] = df_.prefix.str.split().str[-1]
    gm = tr_t[ycol].mean()
    by_last = tr_t.groupby("last")[ycol].mean()
    by_kl = tr_t.groupby(["k", "last"])[ycol].mean()
    te_t["global_mean"] = gm
    te_t["by_last_activity"] = te_t["last"].map(by_last).fillna(gm)
    te_t["by_k_last"] = [by_kl.get((k, l), np.nan) for k, l in zip(te_t.k, te_t["last"])]
    te_t["by_k_last"] = te_t["by_k_last"].fillna(te_t["by_last_activity"])
    for name in ["global_mean", "by_last_activity", "by_k_last"]:
        maes = [metrics.mean_absolute_error(g[ycol], g[name]) for _, g in te_t.groupby("k")]
        rows_t.append({"task": task, "model": name, "mae_mean_over_k": np.mean(maes),
                       "mae_overall": metrics.mean_absolute_error(te_t[ycol], te_t[name])})
res_t = pd.DataFrame(rows_t); res_t.to_csv(out.replace("next_activity", "time_tasks"), index=False)
print(res_t.round(3).to_string(index=False))
