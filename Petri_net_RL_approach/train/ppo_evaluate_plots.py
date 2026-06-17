"""
ppo_eval_plots.py
-----------------
Read the PPO evaluation CSV produced by ppo_evaluate.py and generate
computational-efficiency diagrams comparable to what would be produced
for the A* baseline.

Plots produced
--------------
1. alignment_time_sec  vs  prefix_length   (mean ± std band)
2. steps_taken         vs  prefix_length   (mean ± std band)
3. Distribution of alignment_time_sec      (histogram + KDE)
4. Distribution of steps_taken             (histogram + KDE)
5. cost distribution by prefix_length      (box-plot)
6. Move-type breakdown                     (stacked bar, averaged across all prefixes)

All figures are saved as PNG files next to the CSV.

Usage
-----
    python ppo_eval_plots.py  [path/to/ppo_eval_results.csv]

If no path is given the script looks for ppo_eval_results.csv in the
same directory as this script.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Resolve CSV path ──────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\LENONVO\OneDrive\Desktop\preprocessing\weights\data_stage\ppo_eval_results.csv"

OUT_DIR = os.path.dirname(os.path.abspath(CSV_PATH))
print(f"Reading  : {CSV_PATH}")
print(f"Saving to: {OUT_DIR}")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Rows loaded: {len(df)}")
print(df.dtypes)
print(df.head(3))

# Ensure numeric types
for col in ["prefix_length", "alignment_time_sec", "steps_taken",
            "cost", "sync_moves", "log_moves", "model_moves"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["alignment_time_sec", "steps_taken", "prefix_length"])

# ── Shared style ──────────────────────────────────────────────────────────────
PLT_STYLE  = "seaborn-v0_8-whitegrid"
COLOR_TIME = "#2E86AB"
COLOR_STEP = "#E84855"
COLOR_COST = "#6A994E"
ALPHA_BAND = 0.20
FIG_DPI    = 150

try:
    plt.style.use(PLT_STYLE)
except OSError:
    pass   # fallback to default


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _mean_std_by_prefix(series_name):
    g = df.groupby("prefix_length")[series_name]
    return g.mean(), g.std().fillna(0), g.size()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 – alignment_time_sec vs prefix_length
# ─────────────────────────────────────────────────────────────────────────────

mean_t, std_t, cnt = _mean_std_by_prefix("alignment_time_sec")
x = mean_t.index.values

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(x, mean_t.values * 1_000, color=COLOR_TIME, linewidth=2,
        label="Mean alignment time (ms)")
ax.fill_between(
    x,
    (mean_t - std_t).values * 1_000,
    (mean_t + std_t).values * 1_000,
    color=COLOR_TIME, alpha=ALPHA_BAND,
    label="±1 std dev (alignment time ms)"
)
ax.set_xlabel("Prefix length", fontsize=12)
ax.set_ylabel("Alignment time (ms)", fontsize=12)
ax.set_title("PPO – Alignment Time vs Prefix Length", fontsize=13, fontweight="bold")
ax.legend()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
_save(fig, "plot1_time_vs_prefix.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 – steps_taken vs prefix_length
# ─────────────────────────────────────────────────────────────────────────────

mean_s, std_s, _ = _mean_std_by_prefix("steps_taken")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(x, mean_s.values, color=COLOR_STEP, linewidth=2,
        label="Mean decoding steps")
ax.fill_between(
    x,
    (mean_s - std_s).values,
    (mean_s + std_s).values,
    color=COLOR_STEP, alpha=ALPHA_BAND, label="±1 std dev (decoding steps)"
)
ax.set_xlabel("Prefix length", fontsize=12)
ax.set_ylabel("Decoding steps", fontsize=12)
ax.set_title("PPO – Steps Taken vs Prefix Length", fontsize=13, fontweight="bold")
ax.legend()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
_save(fig, "plot2_steps_vs_prefix.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 – Distribution of alignment_time_sec
# ─────────────────────────────────────────────────────────────────────────────

times_ms = df["alignment_time_sec"].values * 1_000

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(times_ms, bins=50, color=COLOR_TIME, edgecolor="white", alpha=0.85,
    density=True, label="Histogram (alignment time ms)")

# Simple KDE via numpy
from scipy.stats import gaussian_kde         # optional but almost always present
try:
    kde = gaussian_kde(times_ms, bw_method="scott")
    t_range = np.linspace(times_ms.min(), times_ms.max(), 300)
    ax.plot(t_range, kde(t_range), color="navy", linewidth=1.8,
            label="KDE — estimated density (alignment time ms)")
except Exception:
    pass   # skip KDE if scipy not available

ax.set_xlabel("Alignment time (ms)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("PPO – Distribution of Alignment Time", fontsize=13, fontweight="bold")
ax.legend()
_save(fig, "plot3_time_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 – Distribution of steps_taken
# ─────────────────────────────────────────────────────────────────────────────

steps = df["steps_taken"].values

fig, ax = plt.subplots(figsize=(8, 4))
bins = range(int(steps.min()), int(steps.max()) + 2)
ax.hist(steps, bins=bins, color=COLOR_STEP, edgecolor="white", alpha=0.85,
    density=True, label="Histogram (decoding steps)")

try:
    kde2 = gaussian_kde(steps, bw_method="scott")
    s_range = np.linspace(steps.min(), steps.max(), 300)
    ax.plot(s_range, kde2(s_range), color="darkred", linewidth=1.8,
            label="KDE — estimated density (decoding steps)")
except Exception:
    pass

ax.set_xlabel("Decoding steps", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("PPO – Distribution of Steps Taken", fontsize=13, fontweight="bold")
ax.legend()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
_save(fig, "plot4_steps_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 – Cost distribution by prefix_length  (box-plot)
# ─────────────────────────────────────────────────────────────────────────────

# Group into equal-width bins for readability
# Use percentiles but ensure bin edges are unique. If all prefix lengths
# are identical, create a small artificial bin so pd.cut can operate.
raw_edges = np.percentile(df["prefix_length"], [0, 25, 50, 75, 100])
bin_edges = np.unique(raw_edges)
if len(bin_edges) < 2:
    v = float(bin_edges[0])
    bin_edges = np.array([v - 0.5, v + 0.5])

bin_labels = [
    f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
    for i in range(len(bin_edges) - 1)
]

df["pl_bin"] = pd.cut(df["prefix_length"], bins=bin_edges,
                      labels=bin_labels, include_lowest=True)

groups = [grp["cost"].dropna().values for _, grp in df.groupby("pl_bin", observed=True)]

fig, ax = plt.subplots(figsize=(8, 4))
bp = ax.boxplot(
    groups,
    labels=bin_labels,
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    boxprops=dict(facecolor=COLOR_COST, alpha=0.75),
    whiskerprops=dict(color=COLOR_COST),
    capprops=dict(color=COLOR_COST),
    flierprops=dict(marker="o", markersize=3, alpha=0.4,
                    markerfacecolor=COLOR_COST, markeredgecolor="none"),
)
ax.set_xlabel("Prefix length (quartile bins)", fontsize=12)
ax.set_ylabel("Alignment cost", fontsize=12)
ax.set_title("PPO – Cost Distribution by Prefix Length", fontsize=13, fontweight="bold")
_save(fig, "plot5_cost_by_prefix.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 – Move-type breakdown (stacked bar per prefix-length bin)
# ─────────────────────────────────────────────────────────────────────────────

move_df = df.groupby("pl_bin", observed=True)[["sync_moves", "log_moves", "model_moves"]].mean()

fig, ax = plt.subplots(figsize=(8, 4))
bottom = np.zeros(len(move_df))

for col, color, label in [
    ("sync_moves",  "#2A9D8F", "Synchronized moves (S)"),
    ("log_moves",   "#E9C46A", "Log-only moves (L)"),
    ("model_moves", "#E76F51", "Model-only moves (M)"),
]:
    vals = move_df[col].values
    ax.bar(move_df.index, vals, bottom=bottom,
           color=color, label=label, alpha=0.9, edgecolor="white")
    bottom += vals

ax.set_xlabel("Prefix length (quartile bins)", fontsize=12)
ax.set_ylabel("Avg move count", fontsize=12)
ax.set_title("PPO – Move-Type Breakdown by Prefix Length", fontsize=13, fontweight="bold")
ax.legend(loc="upper left")
_save(fig, "plot6_move_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics (printed to console)
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Summary ──────────────────────────────────────────────")
print(f"Total prefixes evaluated : {len(df)}")
print(f"Unique cases             : {df['case_id'].nunique()}")
print(f"\nalignment_time_sec")
print(df["alignment_time_sec"].describe().to_string())
print(f"\nsteps_taken")
print(df["steps_taken"].describe().to_string())
print(f"\ncost")
print(df["cost"].describe().to_string())
conforming = df["is_conforming"].sum() if "is_conforming" in df else "N/A"
print(f"\nConforming prefixes      : {conforming} / {len(df)}")