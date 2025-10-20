# predict_from_list.py
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = Path("totals_model_thru_2024.joblib")
DATA_ROOT  = Path("data/raw")
SLATE_CSV  = Path("slate.csv")   # <-- your manual list

# ---- helpers ----
def last_known(labels, team, season, week, col):
    row = labels.query("season==@season and team==@team and week==@week")
    if not row.empty: return row.iloc[0][col]
    prev = labels.query("season==@season and team==@team and week<@week").sort_values("week")
    return prev.iloc[-1][col] if not prev.empty else np.nan

def add_rolls_shifted(df, group, order, cols, wins=(1,3,5)):
    df = df.sort_values(order)
    g = df.groupby(group, group_keys=False)
    out = df.copy()
    for w in wins:
        rolled = g[cols].shift(1).rolling(window=w, min_periods=1).mean()
        rolled.columns = [f"{c}_last{w}" for c in rolled.columns]
        out = pd.concat([out, rolled], axis=1)
    return out

def build_rows_for_game(season, week, home, away):
    ydir = DATA_ROOT / str(season)

    # labels
    off_labels = pd.read_parquet(ydir / "off_labels.parquet")
    def_labels = pd.read_parquet(ydir / "def_labels.parquet")

    # weekly sources (for rolling)
    off_weekly = pd.read_parquet(sorted(ydir.glob("*weekly_offense*.parquet"))[0])
    def_weekly = pd.read_parquet(sorted(ydir.glob("*weekly_defense*.parquet"))[0])

    OFF_FORM = [c for c in ["epa_per_play","pass_rate","yards_per_att","rush_ypc","explosive_proxy","adot","yac_per_rec"]
                if c in off_weekly.columns]
    DEF_FORM = [c for c in ["epa_per_play_allowed","pass_rate_allowed","yards_per_att_allowed","rush_ypc_allowed",
                            "explosive_allowed","adot_allowed","yac_per_rec_allowed"]
                if c in def_weekly.columns]

    off_form = off_weekly[["season","week","team"] + OFF_FORM].copy()
    def_form = def_weekly[["season","week","team"] + DEF_FORM].copy()

    off_rolls = add_rolls_shifted(off_form, ["season","team"], ["season","week"], OFF_FORM)
    def_rolls = add_rolls_shifted(def_form, ["season","team"], ["season","week"], DEF_FORM)

    # keep rows up to the target week; take the target week row (which is shifted already)
    ho_off = off_rolls.query("season==@season and week==@week and team==@home").add_prefix("home_off_")
    ho_def = def_rolls.query("season==@season and week==@week and team==@home").add_prefix("home_def_")
    aw_off = off_rolls.query("season==@season and week==@week and team==@away").add_prefix("away_off_")
    aw_def = def_rolls.query("season==@season and week==@week and team==@away").add_prefix("away_def_")

    # If week rows don’t exist yet (e.g., you haven’t ingested last week), fallback to last prior week
    if ho_off.empty:
        ho_off = off_rolls.query("season==@season and week<@week and team==@home").sort_values("week").tail(1).add_prefix("home_off_")
    if ho_def.empty:
        ho_def = def_rolls.query("season==@season and week<@week and team==@home").sort_values("week").tail(1).add_prefix("home_def_")
    if aw_off.empty:
        aw_off = off_rolls.query("season==@season and week<@week and team==@away").sort_values("week").tail(1).add_prefix("away_off_")
    if aw_def.empty:
        aw_def = def_rolls.query("season==@season and week<@week and team==@away").sort_values("week").tail(1).add_prefix("away_def_")

    base = pd.DataFrame([{"season":season,"week":week,"home_team":home,"away_team":away}])

    row = base.join(ho_off.reset_index(drop=True), how="left") \
              .join(ho_def.reset_index(drop=True), how="left") \
              .join(aw_off.reset_index(drop=True), how="left") \
              .join(aw_def.reset_index(drop=True), how="left")

    # clusters + names (current week if present; else fallback)
    row["home_off_cluster"]       = last_known(off_labels, home, season, week, "off_cluster")
    row["home_def_cluster"]       = last_known(def_labels, home, season, week, "def_cluster")
    row["away_off_cluster"]       = last_known(off_labels, away, season, week, "off_cluster")
    row["away_def_cluster"]       = last_known(def_labels, away, season, week, "def_cluster")
    row["home_off_cluster_name"]  = last_known(off_labels, home, season, week, "off_cluster_name")
    row["home_def_cluster_name"]  = last_known(def_labels, home, season, week, "def_cluster_name")
    row["away_off_cluster_name"]  = last_known(off_labels, away, season, week, "off_cluster_name")
    row["away_def_cluster_name"]  = last_known(def_labels, away, season, week, "def_cluster_name")

    # pair IDs
    row["pair_home_off_vs_away_def"] = row["home_off_cluster"].astype("Int64").astype(str) + "_" + row["away_def_cluster"].astype("Int64").astype(str)
    row["pair_away_off_vs_home_def"] = row["away_off_cluster"].astype("Int64").astype(str) + "_" + row["home_def_cluster"].astype("Int64").astype(str)
    return row

# ---- main ----
if __name__ == "__main__":
    mdl  = joblib.load(MODEL_PATH)
    pipe = mdl["pipe"]; CAT = mdl["CAT"]; NUM = mdl["NUM"]

    slate = pd.read_csv(SLATE_CSV)
    rows = []
    for _, r in slate.iterrows():
        rows.append(build_rows_for_game(int(r.season), int(r.week), r.home_team, r.away_team))
    feat = pd.concat(rows, ignore_index=True)

    # make sure expected columns exist (tolerant)
    for c in CAT:
        if c not in feat.columns: feat[c] = "UNK"
    for n in NUM:
        if n not in feat.columns: feat[n] = np.nan

    feat["pred_total"] = pipe.predict(feat[CAT + NUM]).astype(float)

    out_cols = ["season","week","home_team","away_team",
                "home_off_cluster_name","home_def_cluster_name",
                "away_off_cluster_name","away_def_cluster_name",
                "pred_total"]
    out_cols = [c for c in out_cols if c in feat.columns] + ["pred_total"]
    out = feat[out_cols].sort_values(["week","home_team","away_team"])

    out_path = Path("reports/preds_from_list.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[saved] {out_path.resolve()}")
    print(out.to_string(index=False))
