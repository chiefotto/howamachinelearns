# evaluate_holdout_2025.py
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

MODEL_PATH = Path("totals_model_thru_2024.joblib")
DATA_PATH  = Path("data/raw/model_dataset.parquet")
OUT_DIR    = Path("reports"); OUT_DIR.mkdir(exist_ok=True, parents=True)

mdl  = joblib.load(MODEL_PATH)
pipe = mdl["pipe"]
CAT  = mdl["CAT"]
NUM  = mdl["NUM"]

df = pd.read_parquet(DATA_PATH).sort_values(["season","week"]).reset_index(drop=True)

# ---------- schema repair (non-destructive) ----------
def ensure_col(df, want, candidates):
    """If 'want' not in df, copy first available candidate; else create 'UNK'."""
    if want in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            df[want] = df[c]
            print(f"[schema] created '{want}' from '{c}'")
            return df
    df[want] = "UNK"
    print(f"[schema] created missing '{want}' filled with 'UNK'")
    return df

df = ensure_col(df, "home_team", ["team_home", "home", "homeTeam", "HTeam"])
df = ensure_col(df, "away_team", ["team_away", "away", "awayTeam", "ATeam"])

# pair features (needed by CAT list if included there)
if "pair_home_off_vs_away_def" not in df.columns and \
   {"home_off_cluster","away_def_cluster"}.issubset(df.columns):
    df["pair_home_off_vs_away_def"] = (
        df["home_off_cluster"].astype("Int64").astype(str) + "_" +
        df["away_def_cluster"].astype("Int64").astype(str)
    )
    print("[schema] created 'pair_home_off_vs_away_def' from clusters")

if "pair_away_off_vs_home_def" not in df.columns and \
   {"away_off_cluster","home_def_cluster"}.issubset(df.columns):
    df["pair_away_off_vs_home_def"] = (
        df["away_off_cluster"].astype("Int64").astype(str) + "_" +
        df["home_def_cluster"].astype("Int64").astype(str)
    )
    print("[schema] created 'pair_away_off_vs_home_def' from clusters")

# Ensure all CAT columns exist (ColumnTransformer requires presence)
for c in CAT:
    if c not in df.columns:
        df[c] = "UNK"
        print(f"[schema] created missing categorical '{c}' as 'UNK'")

# Ensure all NUM columns exist
for c in NUM:
    if c not in df.columns:
        df[c] = np.nan
        print(f"[schema] created missing numeric '{c}' as NaN")

# quick sanity preview
missing_core = [c for c in ["season","week","combined_pts","home_team","away_team"] if c not in df.columns]
if missing_core:
    raise ValueError(f"Core columns missing in model_dataset: {missing_core}\nHave: {list(df.columns)}")

# ---------- filter to 2025 holdout ----------
test = df[df["season"] == 2025].copy()
if test.empty:
    raise ValueError("No rows for season 2025 in model_dataset.parquet")

X = test[CAT + NUM]
y = test["combined_pts"].astype(float).values

# ---------- predictions ----------
test["pred_total"] = pipe.predict(X).astype(float)
test["error"] = test["pred_total"] - test["combined_pts"]
test["abs_error"] = test["error"].abs()

cols = [
    "season","week","home_team","away_team",
    "combined_pts","pred_total","error","abs_error",
]
# include cluster names if they exist
for c in ["home_off_cluster_name","home_def_cluster_name","away_off_cluster_name","away_def_cluster_name"]:
    if c in test.columns: cols.append(c)

pred_table = test[cols].sort_values(["week","home_team","away_team"])

pred_csv = OUT_DIR / "predictions_2025.csv"
pred_table.to_csv(pred_csv, index=False)
print(f"[saved] {pred_csv.resolve()}")

# ---------- baseline vs model ----------
train_avg = df[df["season"] <= 2024]["combined_pts"].mean()
test["baseline_pred"] = train_avg
test["baseline_ae"] = (test["baseline_pred"] - test["combined_pts"]).abs()

mae_model = test["abs_error"].mean()
mae_base  = test["baseline_ae"].mean()
rmse_model = np.sqrt(((test["pred_total"] - test["combined_pts"])**2).mean())
rmse_base  = np.sqrt(((test["baseline_pred"] - test["combined_pts"])**2).mean())

print(f"2025 Holdout  |  MAE(model): {mae_model:.2f}  vs  MAE(baseline): {mae_base:.2f}")
print(f"               RMSE(model): {rmse_model:.2f} vs  RMSE(baseline): {rmse_base:.2f}")
print(f"n={len(test)}  train_avg_total≈{train_avg:.2f}")

# ---------- by-week metrics ----------
by_week = (test
           .groupby("week")
           .agg(n=("combined_pts","size"),
                MAE=("abs_error","mean"),
                Baseline_MAE=("baseline_ae","mean"))
           .reset_index()
           .sort_values("week"))

by_week_csv = OUT_DIR / "by_week_mae_2025.csv"
by_week.to_csv(by_week_csv, index=False)
print(f"[saved] {by_week_csv.resolve()}")

# ---------- charts ----------
# 1) Scatter: Pred vs Actual
plt.figure()
plt.scatter(test["combined_pts"], test["pred_total"])
mn = float(min(test["combined_pts"].min(), test["pred_total"].min()))
mx = float(max(test["combined_pts"].max(), test["pred_total"].max()))
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Actual total (2025)")
plt.ylabel("Predicted total")
plt.title("Predicted vs Actual Totals — 2025 Holdout")
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_pred_vs_actual_2025.png", dpi=160)

# 2) MAE by week
plt.figure()
plt.bar(by_week["week"], by_week["MAE"])
plt.xlabel("Week (2025)")
plt.ylabel("MAE")
plt.title("MAE by Week — 2025 Holdout")
plt.tight_layout()
plt.savefig(OUT_DIR / "mae_by_week_2025.png", dpi=160)

# 3) Residuals histogram
plt.figure()
plt.hist(test["error"], bins=30)
plt.xlabel("Error (Pred - Actual)")
plt.ylabel("Count")
plt.title("Residuals — 2025 Holdout")
plt.tight_layout()
plt.savefig(OUT_DIR / "residuals_hist_2025.png", dpi=160)

print("[done] tables + charts written to 'reports/'")
