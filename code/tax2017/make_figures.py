"""Figures for the Helpdesk reproduction: paper vs our run vs simple rules, and accuracy by prefix length."""
import sys, csv
from collections import Counter, defaultdict
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
pred_csv, log_csv, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = pd.read_csv(pred_csv, quotechar="|")
ours_acc, ours_mae = d["Damerau"].mean() * 100, d["MAE"].mean() / 86400
# simple rule (same split as baselines_tax_split.py): most frequent next activity after the last activity
rows = list(csv.reader(open(log_csv)))[1:]
cases, order = defaultdict(list), []
for c, a, t in rows:
    if c not in cases: order.append(c)
    cases[c].append(a)
tr = [cases[c] for c in order]; fold = int(round((len(tr) + 1) / 3))
tab = defaultdict(Counter)
for t in tr[:2 * fold]:
    acts = t + ["END"]
    for k in range(1, len(acts)): tab[acts[k - 1]][acts[k]] += 1
rule = defaultdict(list)
for t in tr[2 * fold:]:
    for k in range(2, len(t)): rule[k].append(tab[t[k - 1]].most_common(1)[0][0] == t[k])
rule_acc = np.mean([x for v in rule.values() for x in v]) * 100
C = {"paper": "#9aa0a6", "ours": "#2a6fdb", "rule": "#e8833a", "old": "#cfd3d7"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
v1 = [71.19, ours_acc, rule_acc]; l1 = ["Paper\n(LSTM)", "Our run\n(LSTM)", "Simple rule"]
ax[0].bar(range(3), v1, color=[C["paper"], C["ours"], C["rule"]]); ax[0].set_xticks(range(3)); ax[0].set_xticklabels(l1, fontsize=9)
ax[0].set_title("Next activity accuracy (%) ↑", fontsize=10); ax[0].set_ylim(0, 100)
v2 = [5.67, 3.77, ours_mae, float(sys.argv[4])]; l2 = ["Process mining\n(van der Aalst)", "Paper\n(LSTM)", "Our run\n(LSTM)", "Simple rule\n(median)"]
ax[1].bar(range(4), v2, color=[C["old"], C["paper"], C["ours"], C["rule"]]); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(l2, fontsize=9)
ax[1].set_title("Next event time MAE (days) ↓", fontsize=10)
for a, v in ((ax[0], v1), (ax[1], v2)):
    for i, x in enumerate(v): a.text(i, x, f"{x:.2f}", ha="center", va="bottom", fontsize=8)
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle("Helpdesk — Tax et al. (2017): paper vs our reproduction vs simple rules (same split)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{out}/helpdesk_comparison.png", dpi=200)
g = d.groupby("Prefix length")["Damerau"].agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(g.index, g["size"], color="#e6e6e6"); ax.set_ylabel("# test predictions")
ax2 = ax.twinx(); ax2.plot(g.index, g["mean"] * 100, "o-", color=C["ours"], label="LSTM (our run)")
ax2.plot(sorted(rule), [np.mean(rule[k]) * 100 for k in sorted(rule)], "s--", color=C["rule"], label="Simple rule (last activity)")
ax2.set_ylim(0, 105); ax2.set_ylabel("Accuracy (%)"); ax.set_xlabel("Prefix length (events observed)"); ax2.legend(frameon=False, loc="lower right")
ax.set_title("Helpdesk — next-activity accuracy by prefix length (bars = test predictions)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{out}/helpdesk_accuracy_by_prefix.png", dpi=200)
print(f"ours {ours_acc:.2f}% {ours_mae:.3f} d | rule {rule_acc:.2f}%")
