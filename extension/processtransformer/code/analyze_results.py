"""Recompute summary metrics from the per-prefix CSVs of the official scripts,
compare with the paper and the simple baselines, and draw figures."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
run_dir, data_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]   # results/helpdesk, datasets/helpdesk/processed, output folder
na = pd.read_csv(f"{run_dir}/results_next_activity.csv").iloc[:-1]
nt = pd.read_csv(f"{run_dir}/results_next_time.csv").iloc[:-1]
rt = pd.read_csv(f"{run_dir}/results_remaining_time.csv").iloc[:-1]
w_na = pd.read_csv(f"{data_dir}/next_activity_test.csv").k.value_counts().reindex(na.k).values
w_t = pd.read_csv(f"{data_dir}/next_time_test.csv").k.value_counts().reindex(nt.k).values
b = pd.read_csv(f"{out}/baselines_next_activity.csv").set_index("model")
bt = pd.read_csv(f"{out}/baselines_time_tasks.csv")
bt_best = bt[bt.model == "by_k_last"].set_index("task")
s = pd.DataFrame([
 ["Next activity accuracy (%)", 85.63, 76.51, na.accuracy.mean()*100, np.average(na.accuracy, weights=w_na)*100, b.loc["last_activity","acc_mean_over_k"]*100, b.loc["last_activity","acc_overall"]*100],
 ["Next activity weighted F1", 0.82, np.nan, na.fscore.mean(), np.average(na.fscore, weights=w_na), b.loc["last_activity","f1_mean_over_k"], b.loc["last_activity","f1_overall"]],
 ["Next event time MAE (days)", 2.98, 5.77, nt.mean_absolute_error.mean(), np.average(nt.mean_absolute_error, weights=w_t), bt_best.loc["next_time","mae_mean_over_k"], bt_best.loc["next_time","mae_overall"]],
 ["Remaining time MAE (days)", 3.72, 10.38, rt.mean_absolute_error.mean(), np.average(rt.mean_absolute_error, weights=w_t), bt_best.loc["remaining_time","mae_mean_over_k"], bt_best.loc["remaining_time","mae_overall"]],
], columns=["metric","paper_ProcessTransformer","paper_best_baseline","ours_mean_over_k","ours_weighted","simple_rule_mean_over_k","simple_rule_weighted"])
s.to_csv(f"{out}/summary_helpdesk.csv", index=False); print(s.round(3).to_string(index=False))

C_PT, C_RULE, C_PAPER = "#2a6fdb", "#e8833a", "#9aa0a6"
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(na.k + 1, w_na, color="#e6e6e6"); ax.set_ylabel("# test samples"); ax.set_xlabel("Prefix length (events observed so far)")
ax2 = ax.twinx(); ax2.plot(na.k + 1, na.accuracy*100, "o-", color=C_PT, label="ProcessTransformer (our run)")
te = pd.read_csv(f"{data_dir}/next_activity_test.csv"); tr = pd.read_csv(f"{data_dir}/next_activity_train.csv")
from collections import Counter, defaultdict
tab = defaultdict(Counter)
for p, y in zip(tr.prefix, tr.next_act): tab[p.split()[-1]][y] += 1
maj = tr.next_act.value_counts().idxmax()
te["rule"] = [tab[p.split()[-1]].most_common(1)[0][0] if tab.get(p.split()[-1]) else maj for p in te.prefix]
acc_rule = te.groupby("k").apply(lambda g: (g.rule == g.next_act).mean()).reindex(na.k)
ax2.plot(na.k + 1, acc_rule*100, "s--", color=C_RULE, label="Simple rule (last activity)")
ax2.set_ylabel("Accuracy (%)"); ax2.set_ylim(0, 105); ax2.legend(loc="lower left", frameon=False)
ax.set_title("Next-activity accuracy by prefix length — Helpdesk (bars = test samples)")
for a in (ax, ax2): a.spines["top"].set_visible(False)
fig.tight_layout(); fig.savefig(f"{out}/accuracy_by_prefix.png", dpi=200, facecolor="#FBFAF7")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
items = [("Next activity\naccuracy (%) ↑", [85.63, 76.51, s.iloc[0,3], s.iloc[0,5]]),
         ("Next event time\nMAE (days) ↓", [2.98, 5.77, s.iloc[2,3], s.iloc[2,5]]),
         ("Remaining time\nMAE (days) ↓", [3.72, 10.38, s.iloc[3,3], s.iloc[3,5]])]
labels = ["Paper", "Paper's best\nbaseline", "Our\nreproduction", "Simple\nrule"]
cols = [C_PAPER, "#cfd3d7", C_PT, C_RULE]
for a, (t, v) in zip(axes, items):
    a.bar(range(4), v, color=cols); a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=8); a.set_title(t, fontsize=10)
    for i, x in enumerate(v): a.text(i, x, f"{x:.2f}", ha="center", va="bottom", fontsize=8)
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle("Helpdesk — paper vs our reproduction vs a simple rule (paper protocol: mean over prefix lengths)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{out}/summary_comparison.png", dpi=200, facecolor="#FBFAF7")
