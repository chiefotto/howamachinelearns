# calibrate_confidence.py
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = Path("data/raw/model_dataset.parquet")
MODEL_PATH = Path("totals_model_thru_2024.joblib")
SLATE_PATH = Path("slate2.csv")                    # with market_total column
PRED_PATH  = Path("reports/preds_from_list.csv")  # output from your predictor
OUT_PATH   = Path("reports/preds_with_confidence.csv")

ALPHA = 0.2   # 80% interval (change to 0.1 for 90%)
MIN_PAIR_RESID = 30  # samples needed to use the most-specific residual bucket

# ----- 1) Load data & model -----
df = pd.read_parquet(DATA_PATH).sort_values(["season","week"]).reset_index(drop=True)
mdl = joblib.load(MODEL_PATH)
pipe: Pipeline = mdl["pipe"]
CAT = mdl["CAT"]; NUM = mdl["NUM"]

# safety: ensure cols exist
for c in CAT:
    if c not in df.columns: df[c] = "UNK"
for n in NUM:
    if n not in df.columns: df[n] = np.nan

# ----- 2) Out-of-fold predictions on <=2024 to get honest residuals -----
train_mask = df["season"] <= 2024
train = df.loc[train_mask].copy()

X = train[CAT + NUM]; y = train["combined_pts"].astype(float).values

# we’ll rebuild a small copy of the pipeline each split to mimic training-time behavior
def fit_predict_oof(train_df):
    Xall = train_df[CAT + NUM]; yall = train_df["combined_pts"].astype(float).values
    tscv = TimeSeriesSplit(n_splits=5)
    oof = pd.Series(index=train_df.index, dtype=float)

    for tr_idx, te_idx in tscv.split(Xall):
        # fresh pipeline
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ("num", "passthrough", NUM),
        ])
        mdl_inner = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=500, random_state=42)
        inner_pipe = Pipeline([("prep", pre), ("mdl", mdl_inner)])
        inner_pipe.fit(Xall.iloc[tr_idx], yall[tr_idx])
        oof.iloc[te_idx] = inner_pipe.predict(Xall.iloc[te_idx])
    return oof

train["oof_pred"] = fit_predict_oof(train)
train["residual"] = train["oof_pred"] - train["combined_pts"]

# keys for conditioning residuals
def ensure_pairs(df_):
    if "pair_home_off_vs_away_def" not in df_.columns and {"home_off_cluster","away_def_cluster"}.issubset(df_.columns):
        df_["pair_home_off_vs_away_def"] = (
            df_["home_off_cluster"].astype("Int64").astype(str) + "_" +
            df_["away_def_cluster"].astype("Int64").astype(str)
        )
    if "pair_away_off_vs_home_def" not in df_.columns and {"away_off_cluster","home_def_cluster"}.issubset(df_.columns):
        df_["pair_away_off_vs_home_def"] = (
            df_["away_off_cluster"].astype("Int64").astype(str) + "_" +
            df_["home_def_cluster"].astype("Int64").astype(str)
        )
ensure_pairs(train)

# build residual banks at 3 levels of specificity
bank_pair = train.groupby("pair_home_off_vs_away_def")["residual"].apply(list).to_dict()
bank_hod  = train.groupby(["home_off_cluster","away_def_cluster"])["residual"].apply(list).to_dict()
bank_all  = train["residual"].tolist()

def residuals_for_row(row):
    pair = row.get("pair_home_off_vs_away_def")
    hoc  = row.get("home_off_cluster")
    adc  = row.get("away_def_cluster")
    # try pair-specific
    rs = bank_pair.get(pair, [])
    if len(rs) >= MIN_PAIR_RESID: return np.array(rs)
    # fallback to (home_off, away_def)
    rs = bank_hod.get((hoc, adc), [])
    if len(rs) >= MIN_PAIR_RESID: return np.array(rs)
    # global fallback
    return np.array(bank_all)

# ----- 3) Load your slate predictions (already contains pred_total) -----
slate = pd.read_csv(SLATE_PATH)
pred  = pd.read_csv(PRED_PATH)

# merge market totals in
if "market_total" not in slate.columns:
    raise ValueError("Add a 'market_total' column to slate.csv to compute P(over).")

pred = pred.merge(slate[["season","week","home_team","away_team","market_total"]],
                  on=["season","week","home_team","away_team"], how="left")

# ensure pair keys exist in pred too (for picking residual bank)
ensure_pairs(pred)

# ----- 4) Compute intervals & probabilities -----
lo_q = 100 * (ALPHA/2)         # e.g., 10th for 80% PI
hi_q = 100 * (1 - ALPHA/2)     # e.g., 90th

rows = []
for _, r in pred.iterrows():
    rs = residuals_for_row(r)
    qlo, qhi = np.percentile(rs, [lo_q, hi_q])
    pi_low  = r["pred_total"] + qlo
    pi_high = r["pred_total"] + qhi

    # P(over market_total) = P(pred + resid > line) = P(resid > line - pred)
    thresh = r["market_total"] - r["pred_total"]
    p_over = float(np.mean(rs > thresh))

    rows.append({**r.to_dict(),
                 "pi_low": pi_low, "pi_high": pi_high,
                 "p_over": p_over, "edge": r["pred_total"] - r["market_total"]})

out = pd.DataFrame(rows)

# nice formatting
out["p_over_pct"] = (out["p_over"] * 100).round(1)
out["pi_80"] = out.apply(lambda s: f"[{s['pi_low']:.1f}, {s['pi_high']:.1f}]", axis=1)

# save
out_cols = ["season","week","home_team","away_team",
            "pred_total","market_total","edge","pi_low","pi_high","p_over","p_over_pct"]
out[out_cols].to_csv(OUT_PATH, index=False)
print(f"[saved] {OUT_PATH.resolve()}")
print(out[out_cols].to_string(index=False))
