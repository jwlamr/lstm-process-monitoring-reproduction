"""Simple baselines on EXACTLY the split and evaluation protocol of Tax et al. (2017).

Split: cases in file order, first 2/3 = training (folds 1+2), last 1/3 = test (fold 3).
Test prefixes: length k with 2 <= k < case length, i.e. only real next events are scored (the evaluation script
writes no row when the case has ended), as in evaluate_next_activity_and_time.py. Metrics: accuracy (next activity) and MAE in days (time to next event),
averaged over all test predictions ("All" column of Table 1 in the paper).

Baselines
  majority        : most frequent next activity in training
  last_activity   : most frequent next activity after the last observed activity
  prefix_lookup   : most frequent next activity after the full prefix (fallback: last_activity)
  time: mean or median training time-to-next-event, globally, given the last activity, or given (prefix length, last activity)

Usage: python baselines_tax_split.py ../data/helpdesk.csv
"""
import csv, sys
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

path = sys.argv[1]
rows = list(csv.reader(open(path)))[1:]
cases, order = defaultdict(list), []
for cid, act, ts in rows:
    if cid not in cases:
        order.append(cid)
    cases[cid].append((act, datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")))
traces = [cases[c] for c in order]
n = len(traces) + 1                      # the original code counts one extra case (numlines += 1 at the end)
fold = int(round(n / 3))
train, test = traces[:2 * fold], traces[2 * fold:]

def samples(trace, min_k, with_end):
    acts = [a for a, _ in trace] + ["END"]
    times = [0.0] + [(trace[i][1] - trace[i - 1][1]).total_seconds() for i in range(1, len(trace))] + [0.0]
    for k in range(min_k, len(trace) + (1 if with_end else 0)):
        yield tuple(acts[:k]), acts[k], times[k] / 86400

tr = [s for t in train for s in samples(t, 1, True)]   # training prefixes include the end of case, as in train.py
te = [s for t in test for s in samples(t, 2, False)]   # evaluation script only scores real next events (no end-of-case)
majority = Counter(y for _, y, _ in tr).most_common(1)[0][0]
by_last, by_prefix = defaultdict(Counter), defaultdict(Counter)
t_last, t_klast = defaultdict(list), defaultdict(list)
for p, y, dt in tr:
    by_last[p[-1]][y] += 1; by_prefix[p][y] += 1
    t_last[p[-1]].append(dt); t_klast[(len(p), p[-1])].append(dt)
gm = np.mean([dt for _, _, dt in tr])

def last(p): return by_last[p[-1]].most_common(1)[0][0] if p[-1] in by_last else majority
def prefix(p): return by_prefix[p].most_common(1)[0][0] if p in by_prefix else last(p)
def tl(p): return np.mean(t_last[p[-1]]) if p[-1] in t_last else gm
def tkl(p): return np.mean(t_klast[(len(p), p[-1])]) if (len(p), p[-1]) in t_klast else tl(p)

print(f"{path}: {len(traces)} cases, train {len(train)}, test {len(test)}, test predictions {len(te)}")
for name, f in [("majority", lambda p: majority), ("last_activity", last), ("prefix_lookup", prefix)]:
    print(f"  accuracy {name:15s}: {np.mean([f(p) == y for p, y, _ in te]):.4f}")
gmed = np.median([dt for _, _, dt in tr])
def tl_med(p): return np.median(t_last[p[-1]]) if p[-1] in t_last else gmed
def tkl_med(p): return np.median(t_klast[(len(p), p[-1])]) if (len(p), p[-1]) in t_klast else tl_med(p)
# MAE is minimised by the median, so median versions are the fair "simple rule" for an MAE-trained model
for name, f in [("global_mean", lambda p: gm), ("by_last_activity", tl), ("by_k_last_activity", tkl),
                ("global_median", lambda p: gmed), ("median_by_last", tl_med), ("median_by_k_last", tkl_med)]:
    print(f"  next-time MAE {name:20s}: {np.mean([abs(f(p) - dt) for p, _, dt in te]):.3f} days")
