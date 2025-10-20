from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# ==========================================
# CONFIG
# ==========================================
DATA_ROOT = Path("data/raw")
YEARS = [y for y in range(2015, 2026)]  # 2015–2025 except 2024
PATTERN = "*weekly_defense*.parquet"                 # your defense files

# Use only process/style "allowed/forced" features (no lines/scores)
DEF_FEATS = [
    "epa_per_play_allowed",
    "def_epa_allowed",
    "pass_rate_allowed",
    "yards_per_att_allowed",
    "rush_ypc_allowed",
    "explosive_allowed",
    "td_rate_allowed",
    "int_rate_forced",
    "sack_rate_forced",
    "adot_allowed",
    "yac_per_rec_allowed",
    # Optional raw counting (can add slight volume flavor):
    # "def_sacks","def_qb_hits","def_interceptions","def_fumbles_forced",
    # "def_pass_defended","def_tds","def_safeties"
]
INCLUDE_COUNTS = False  # set True to append the count stats above

# ==========================================
# HELPERS
# ==========================================
def winsorize(x: pd.Series, lo=0.01, hi=0.99):
    a, b = x.quantile(lo), x.quantile(hi)
    return x.clip(a, b)

def zprofiles(df: pd.DataFrame, feat_cols, cluster_col):
    z = (df[feat_cols] - df[feat_cols].mean()) / df[feat_cols].std(ddof=0)
    z[cluster_col] = df[cluster_col].values
    return z.groupby(cluster_col).mean().sort_index()

def name_def_cluster(row: pd.Series) -> str:
    # row is z-scored mean per cluster
    epa, passrate, ypa, ypc, xpl, tdr, intr, sackr, adot, yac = (
        row.get("epa_per_play_allowed",0),
        row.get("pass_rate_allowed",0),
        row.get("yards_per_att_allowed",0),
        row.get("rush_ypc_allowed",0),
        row.get("explosive_allowed",0),
        row.get("td_rate_allowed",0),
        row.get("int_rate_forced",0),
        row.get("sack_rate_forced",0),
        row.get("adot_allowed",0),
        row.get("yac_per_rec_allowed",0),
    )
    tags = []
    # overall efficiency (lower epa_allowed is better)
    if epa <= -0.5: tags.append("Stingy")
    elif epa >= 0.5: tags.append("Leaky")
    else: tags.append("Average")

    # coverage / explosive suppression
    if ypa <= -0.4 and xpl <= -0.4 and adot <= -0.3: tags.append("Strong Pass")
    elif ypa >= 0.5 or xpl >= 0.5: tags.append("Big-Play Risk")

    # run defense
    if ypc <= -0.5: tags.append("Strong Run")
    elif ypc >= 0.5: tags.append("Soft Run")

    # pressure/turnovers
    if sackr >= 0.5: tags.append("High Pressure")
    if intr >= 0.4: tags.append("Ball-Hawking")

    # endzone resistance
    if tdr <= -0.5: tags.append("Red-Zone Tough")
    elif tdr >= 0.5: tags.append("Red-Zone Leaky")

    return " ".join(tags) if tags else "Neutral"

def process_defense_file(parquet_path: Path):
    print(f"\n[LOAD] {parquet_path}")
    df = pd.read_parquet(parquet_path)

    feats = DEF_FEATS.copy()
    if INCLUDE_COUNTS:
        extra = ["def_sacks","def_qb_hits","def_interceptions",
                 "def_fumbles_forced","def_pass_defended","def_tds","def_safeties"]
        feats += [c for c in extra if c in df.columns]

    # Prefer the defensive team key as 'team'
    # Your file already has 'team' (def team) and 'off_team' (opponent offense)
    use = df[["season","week","team"] + feats].copy()

    # robustify tails
    for c in feats:
        use[c] = winsorize(use[c].astype(float), 0.01, 0.99)
    use[feats] = use[feats].replace([np.inf, -np.inf], np.nan)
    use = use.dropna(subset=feats).reset_index(drop=True)

    # scale
    scaler = StandardScaler().fit(use[feats])
    Xs = scaler.transform(use[feats])

    # auto-pick k
    best = {"k": None, "sil": -1, "model": None}
    for k in [5, 6, 7]:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(Xs)
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(Xs, labels)
        if sil > best["sil"]:
            best = {"k": k, "sil": sil, "model": km}

    if best["model"] is None:
        raise RuntimeError("Could not find valid k (check data size).")

    k = best["k"]; km = best["model"]
    print(f"[CLUSTERS] {parquet_path.parent.name}: k={k}, silhouette={best['sil']:.3f}")

    use["def_cluster"] = km.predict(Xs)

    # profiles + names
    prof = zprofiles(use, feats, "def_cluster")
    cluster_names = {cid: name_def_cluster(prof.loc[cid]) for cid in prof.index}
    use["def_cluster_name"] = use["def_cluster"].map(cluster_names)

    # outputs
    save_dir = parquet_path.parent
    def_labels = use[["season","week","team","def_cluster","def_cluster_name"]].drop_duplicates()

    def_labels.to_parquet(save_dir / "def_labels.parquet", index=False)
    prof.round(3).to_csv(save_dir / "def_cluster_profiles.csv")
    joblib.dump(
        {"scaler": scaler, "kmeans": km, "DEF_FEATS": feats, "cluster_names": cluster_names},
        save_dir / "defense_clusters.joblib"
    )
    print(f"[SAVED] → {save_dir.name}/def_labels.parquet")

# ==========================================
# RUNNERS
# ==========================================
def run_single_season(season: int):
    season_dir = DATA_ROOT / str(season)
    files = sorted(season_dir.glob(PATTERN))
    if not files:
        print(f"[SKIP] No defense parquet in {season_dir}")
        return
    process_defense_file(files[0])

def batch_all_defense_seasons():
    for y in YEARS:
        if not (DATA_ROOT / str(y)).exists():
            continue
        try:
            run_single_season(y)
        except Exception as e:
            print(f"[ERROR] {y}: {e}")

if __name__ == "__main__":
    # 1) Try a single season first (e.g., 2025)
    # run_single_season(2025)

    # 2) Then batch all seasons:
    batch_all_defense_seasons()
