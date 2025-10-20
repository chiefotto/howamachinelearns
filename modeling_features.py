# make_model_dataset.py
from pathlib import Path
import pandas as pd
import numpy as np

DATA_ROOT = Path("data/raw")
YEARS = list(range(2015, 2026))

def add_rolls(df, grp_cols, sort_cols, cols, spec):
    """
    spec: list of (new_suffix, window, agg)
    Example: [("last1","1","mean"), ("last3","3","mean"), ("last5","5","mean")]
    """
    df = df.sort_values(sort_cols)
    g = df.groupby(grp_cols, group_keys=False)
    out = df.copy()
    for suf, win, agg in spec:
        rolled = g[cols].shift(1).rolling(window=int(win), min_periods=1).agg(agg)  # shift(1) prevents leakage
        rolled.columns = [f"{c}_{suf}" for c in rolled.columns]
        out = pd.concat([out, rolled], axis=1)
    return out

def build_year(year: int) -> pd.DataFrame:
    ydir = DATA_ROOT / str(year)
    matchups = pd.read_parquet(ydir / "matchups.parquet")  # you just built this

    # Bring raw weekly offense/defense to compute rolling form
    off_weekly = pd.read_parquet(sorted(ydir.glob("*weekly_offense*.parquet"))[0])
    def_weekly = pd.read_parquet(sorted(ydir.glob("*weekly_defense*.parquet"))[0])

    # Keep just the “process/efficiency” anchors for rolls (expand as you like)
    OFF_FORM = ["epa_per_play", "pass_rate", "yards_per_att", "rush_ypc", "explosive_proxy", "adot", "yac_per_rec"]
    DEF_FORM = ["epa_per_play_allowed", "pass_rate_allowed", "yards_per_att_allowed", "rush_ypc_allowed", "explosive_allowed", "adot_allowed", "yac_per_rec_allowed"]

    off_form = off_weekly[["season","week","team"] + [c for c in OFF_FORM if c in off_weekly.columns]].copy()
    def_form = def_weekly[["season","week","team"] + [c for c in DEF_FORM if c in def_weekly.columns]].copy()

    # Compute rolling (shifted) for teams (offense) and teams (defense)
    roll_spec = [("last1", 1, "mean"), ("last3", 3, "mean"), ("last5", 5, "mean")]

    off_rolls = add_rolls(off_form, grp_cols=["season","team"], sort_cols=["season","week"], cols=[c for c in OFF_FORM if c in off_form.columns], spec=roll_spec)
    def_rolls = add_rolls(def_form, grp_cols=["season","team"], sort_cols=["season","week"], cols=[c for c in DEF_FORM if c in def_form.columns], spec=roll_spec)

    # Merge HOME rolls
    home = matchups.merge(
        off_rolls.add_prefix("home_off_"),
        left_on=["season","week","home_team"],
        right_on=["home_off_season","home_off_week","home_off_team"],
        how="left"
    ).merge(
        def_rolls.add_prefix("home_def_"),
        left_on=["season","week","home_team"],
        right_on=["home_def_season","home_def_week","home_def_team"],
        how="left"
    )

    # Merge AWAY rolls
    full = home.merge(
        off_rolls.add_prefix("away_off_"),
        left_on=["season","week","away_team"],
        right_on=["away_off_season","away_off_week","away_off_team"],
        how="left"
    ).merge(
        def_rolls.add_prefix("away_def_"),
        left_on=["season","week","away_team"],
        right_on=["away_def_season","away_def_week","away_def_team"],
        how="left"
    )

    # Clean helper columns
    drop_like = [c for c in full.columns if c.endswith("_season") or c.endswith("_week") or c.endswith("_team")]
    full = full.drop(columns=drop_like)

    # Time-safe target encodings for cluster pairs (expanding mean of combined_pts)
    full = full.sort_values(["season","week"])
    for col in ["pair_home_off_vs_away_def","pair_away_off_vs_home_def"]:
        # expanding past mean (no current week)
        grp = full.groupby(col, group_keys=False)["combined_pts"]
        full[f"te_{col}"] = grp.apply(lambda s: s.shift(1).expanding().mean())

    return full

def main():
    dfs = []
    for y in YEARS:
        try:
            dfs.append(build_year(y))
            print(f"[{y}] model rows: {len(dfs[-1])}")
        except Exception as e:
            print(f"[{y}] SKIP: {e}")
    if not dfs:
        raise SystemExit("No data")
    model_df = pd.concat(dfs, ignore_index=True)
    out = DATA_ROOT / "model_dataset.parquet"
    model_df.to_parquet(out, index=False)
    print(f"[ALL] saved {len(model_df)} → {out}")

if __name__ == "__main__":
    main()
