"""Scan the benchmark suite and write results/catalog.csv (datasets with n<=max_n, d<=max_d)."""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from mstcut import data

ap = argparse.ArgumentParser()
ap.add_argument("--max_n", type=int, default=1000); ap.add_argument("--max_d", type=int, default=3)
a = ap.parse_args()
rows = data.catalog(a.max_n, a.max_d)
df = pd.DataFrame(rows); df["tier"] = df.n.map(data.tier)
# role flags tie each dataset to the research gaps
df["rg1_noise"] = df.noise > 0                    # labelled noise / outliers -> exception budget B matters
df["rg2_imbalance"] = df.sizes_ratio >= 3          # unequal cluster sizes -> m_min vs balance assumptions
df["control"] = (df.noise == 0) & (df.sizes_ratio < 3)   # clean, balanced: model should not lose to baselines
df["tier"] = df.n.map(data.tier)
os.makedirs("results", exist_ok=True); df.to_csv("results/catalog.csv", index=False)
print(df.groupby(["battery", "tier"]).size().unstack(fill_value=0))
df.loc[df.K<=1, "K"]=1
print(len(df), "dataset/labeling pairs;", df[["battery", "name"]].drop_duplicates().shape[0], "datasets")
