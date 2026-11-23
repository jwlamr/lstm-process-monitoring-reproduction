"""Summarise the output of evaluate_next_activity_and_time.py (one row per test prediction).

With predict_size = 1, the Damerau-Levenshtein similarity of a row is 1 if the predicted next activity is correct
and 0 otherwise, so its mean is the next-activity accuracy. The MAE column is in seconds.
Usage: python summarize_results.py output_files/results/next_activity_and_time_helpdesk.csv
"""
import sys
import pandas as pd

for path in sys.argv[1:]:
    d = pd.read_csv(path, quotechar="|")
    acc = d["Damerau"].mean()
    mae = d["MAE"].mean() / 86400
    print(f"{path}\n  predictions: {len(d)}   accuracy: {acc:.4f}   next-event-time MAE: {mae:.3f} days")
    by_k = d.groupby("Prefix length").agg(n=("Damerau", "size"), accuracy=("Damerau", "mean"), mae_days=("MAE", lambda x: x.mean() / 86400))
    print(by_k.round(3).to_string())
