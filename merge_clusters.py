# build_matchups.py
from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data/raw")
YEARS = list(range(2015, 2026))
OFF_PAT = "*weekly_offense*.parquet"
DEF_PAT = "*weekly_defense*.parquet"

def load_one(glob_dir: Path, pattern: str) -> Path:
    files = sorted(glob_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {glob_dir}")
    return files[0]

def make_year_matchups(year: int) -> pd.DataFrame:
    ydir = DATA_ROOT / str(year)
    if not ydir.exists():
        raise FileNotFoundError(f"Missing folder {ydir}")

    # Base tables
    off_weekly_path = load_one(ydir, OFF_PAT)
    def_weekly_path = load_one(ydir, DEF_PAT)
    off_labels_path = ydir / "off_labels.parquet"
    def_labels_path = ydir / "def_labels.parquet"

    offense = pd.read_parquet(off_weekly_path)
    defense = pd.read_parquet(def_weekly_path)
    off_labels = pd.read_parquet(off_labels_path)
    def_labels = pd.read_parquet(def_labels_path)
    
    if "off_team_loc" not in offense.columns:
        if "loc" in offense.columns:
            offense = offense.rename(columns={"loc": "off_team_loc"})


    # 1) Start with OFFENSE table (your base), add this team's OFF cluster
    df = offense.merge(
        off_labels, on=["season","week","team"], how="left"
    )  # adds: off_cluster, off_cluster_name

    # 2) Add THIS TEAM's own DEF cluster (use def_labels keyed by defensive team == this team)
    df = df.merge(
        def_labels.rename(columns={
            "def_cluster": "team_def_cluster",
            "def_cluster_name": "team_def_cluster_name"
        }),
        on=["season","week","team"],
        how="left"
    )

    # 3) Add OPPONENT OFFENSE cluster (join off_labels on opponent_team)
    df = df.merge(
        off_labels.rename(columns={
            "team":"opponent_team",
            "off_cluster":"opp_off_cluster",
            "off_cluster_name":"opp_off_cluster_name"
        }),
        on=["season","week","opponent_team"],
        how="left"
    )

    # 4) Add OPPONENT DEFENSE cluster **by linking DEF weekly via off_team**
    #    Rule: off_table.team == def_table.off_team
    #    => find which defensive team faced our offense team, then map its def cluster
    opp_def_map = (
        defense[["season","week","off_team","team"]]
        .drop_duplicates()
        .rename(columns={
            "off_team":"offense_team_faced",   # this is our offense 'team'
            "team":"defense_team_faced"        # opponent's defensive team
        })
        .merge(
            def_labels.rename(columns={
                "team":"defense_team_faced",
                "def_cluster":"opp_def_cluster",
                "def_cluster_name":"opp_def_cluster_name"
            }),
            on=["season","week","defense_team_faced"],
            how="left"
        )
        [["season","week","offense_team_faced","opp_def_cluster","opp_def_cluster_name"]]
    )

    df = df.merge(
        opp_def_map,
        left_on=["season","week","team"],
        right_on=["season","week","offense_team_faced"],
        how="left"
    ).drop(columns=["offense_team_faced"])

    # 5) Keep only the columns you asked to preserve (+ the opponent clusters)
    keep = [
        "season","week","team","opponent_team","off_team_loc","combined_pts",
        "off_cluster","off_cluster_name",
        "team_def_cluster","team_def_cluster_name",
        "opp_off_cluster","opp_off_cluster_name",
        "opp_def_cluster","opp_def_cluster_name",
    ]
    df = df[keep]

    # 6) Guarantee one row per full game:
    #    Use home rows only (off_team_loc == 'home') → ensures team != opponent_team inherently.
    #    If your values are 'home'/'away' lowercase, this works; otherwise normalize.
    df["off_team_loc"] = df["off_team_loc"].str.lower()
    games = df[df["off_team_loc"] == "home"].copy()

    # 7) Rename to explicit home/away semantics
    games = games.rename(columns={
        "team":"home_team",
        "opponent_team":"away_team",
        "off_cluster":"home_off_cluster",
        "off_cluster_name":"home_off_cluster_name",
        "team_def_cluster":"home_def_cluster",
        "team_def_cluster_name":"home_def_cluster_name",
        "opp_off_cluster":"away_off_cluster",
        "opp_off_cluster_name":"away_off_cluster_name",
        "opp_def_cluster":"away_def_cluster",
        "opp_def_cluster_name":"away_def_cluster_name",
    })

    # 8) Basic sanity: drop any accidental rows with same teams
    games = games[games["home_team"] != games["away_team"]].reset_index(drop=True)

    # 9) Optional: pair-feature IDs (handy for modeling)
    games["pair_home_off_vs_away_def"] = games["home_off_cluster"].astype("Int64").astype(str) + "_" + games["away_def_cluster"].astype("Int64").astype(str)
    games["pair_away_off_vs_home_def"] = games["away_off_cluster"].astype("Int64").astype(str) + "_" + games["home_def_cluster"].astype("Int64").astype(str)

    return games

def main():
    all_years = []
    for y in YEARS:
        try:
            g = make_year_matchups(y)
            out_path = DATA_ROOT / str(y) / "matchups.parquet"
            g.to_parquet(out_path, index=False)
            print(f"[{y}] matchups: {len(g)} rows → {out_path}")
            all_years.append(g)
        except Exception as e:
            print(f"[{y}] SKIP: {e}")

    if all_years:
        big = pd.concat(all_years, ignore_index=True)
        big_out = DATA_ROOT / "matchups_all_years.parquet"
        big.to_parquet(big_out, index=False)
        print(f"[ALL] {len(big)} rows → {big_out}")

if __name__ == "__main__":
    main()
