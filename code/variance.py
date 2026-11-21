"""Aggregate several runs: mean ± std of each metric (paper protocol and sample-weighted)."""
import sys, glob, numpy as np, pandas as pd
res_root, data_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
w = {t: pd.read_csv(f"{data_dir}/{t}_test.csv").k.value_counts() for t in ["next_activity", "next_time"]}
rows = []
for f in sorted(glob.glob(f"{res_root}/run*/helpdesk/results_*.csv")):
    run = f.split("/")[-3]; task = f.split("results_")[-1][:-4]
    d = pd.read_csv(f).iloc[:-1]
    if task == "next_activity":
        ww = w["next_activity"].reindex(d.k).values
        rows += [(run, "Next activity accuracy (%)", d.accuracy.mean()*100, np.average(d.accuracy, weights=ww)*100),
                 (run, "Next activity weighted F1", d.fscore.mean(), np.average(d.fscore, weights=ww))]
    else:
        ww = w["next_time"].reindex(d.k).values
        name = "Next event time MAE (days)" if task == "next_time" else "Remaining time MAE (days)"
        rows.append((run, name, d.mean_absolute_error.mean(), np.average(d.mean_absolute_error, weights=ww)))
df = pd.DataFrame(rows, columns=["run", "metric", "mean_over_k", "weighted"])
df.to_csv(f"{out}/all_runs.csv", index=False)
agg = df.groupby("metric").agg(n_runs=("run", "count"), mean_over_k=("mean_over_k", "mean"), std_over_k=("mean_over_k", "std"),
                               weighted=("weighted", "mean"), std_weighted=("weighted", "std"))
agg.to_csv(f"{out}/runs_summary.csv"); print(df.round(3).to_string(index=False)); print(agg.round(3).to_string())
