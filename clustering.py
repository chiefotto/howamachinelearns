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
DATA_ROOT = Path("data/raw")             # your folder root
YEARS = [y for y in range(2015, 2026) if y != 2024]  # 2015–2025 except 2024
PATTERN = "*weekly_offense*.parquet"     # your offense files

OFF_FEATS = [
    "pass_rate",
    "yards_per_att",
    "td_rate",
    "int_rate",
    "sack_rate",
    "rush_ypc",
    "explosive_proxy",
    "epa_per_play",
    "adot",
    "yac_per_rec",
]
INCLUDE_OFF_EPA = False  # change to True if you want volume included


# ==========================================
# HELPERS
# ==========================================
def winsorize(x, lo=0.01, hi=0.99):
    a, b = x.quantile(lo), x.quantile(hi)
    return x.clip(a, b)


def zprofiles(df, feat_cols, cluster_col):
    z = (df[feat_cols] - df[feat_cols].mean()) / df[feat_cols].std(ddof=0)
    z[cluster_col] = df[cluster_col].values
    return z.groupby(cluster_col).mean().sort_index()


def name_off_cluster(row):
    pr, ypa, tdr, intr, sackr, ypc, xpl, epa, adot, yac = \
        row.get("pass_rate", 0), row.get("yards_per_att", 0), row.get("td_rate", 0), row.get("int_rate", 0), row.get("sack_rate", 0), \
        row.get("rush_ypc", 0), row.get("explosive_proxy", 0), row.get("epa_per_play", 0), row.get("adot", 0), row.get("yac_per_rec", 0)

    tags = []
    if pr > 0.6: tags.append("Pass-Heavy")
    elif pr < -0.5: tags.append("Run-Heavy")
    else: tags.append("Balanced")

    if epa > 0.5: tags.append("Efficient")
    elif epa < -0.5: tags.append("Inefficient")

    if xpl > 0.5 or adot > 0.5: tags.append("Explosive")
    elif ypc > 0.6 and pr < 0.0: tags.append("Ground-Centric")

    if sackr > 0.6: tags.append("Pressure-Prone")
    if intr > 0.6: tags.append("Risky")

    return " ".join(tags) if tags else "Neutral"


def process_offense_file(parquet_path: Path, include_off_epa=False):
    print(f"\n[LOAD] {parquet_path}")
    df = pd.read_parquet(parquet_path)

    feats = OFF_FEATS.copy()
    if include_off_epa and "off_epa" in df.columns:
        feats.append("off_epa")

    use = df[["season", "week", "team"] + feats].copy()
    for c in feats:
        use[c] = winsorize(use[c].astype(float))

    use[feats] = use[feats].replace([np.inf, -np.inf], np.nan)
    use = use.dropna(subset=feats)

    scaler = StandardScaler().fit(use[feats])
    Xs = scaler.transform(use[feats])

    # Auto-select k via silhouette
    best = {"k": None, "sil": -1, "model": None}
    for k in [5, 6, 7]:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(Xs)
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(Xs, labels)
        if sil > best["sil"]:
            best = {"k": k, "sil": sil, "model": km}

    k = best["k"]
    km = best["model"]
    print(f"[CLUSTERS] {parquet_path.parent.name}: k={k}, silhouette={best['sil']:.3f}")

    use["off_cluster"] = km.predict(Xs)
    prof = zprofiles(use, feats, "off_cluster")
    cluster_names = {cid: name_off_cluster(prof.loc[cid]) for cid in prof.index}
    use["off_cluster_name"] = use["off_cluster"].map(cluster_names)

    # Save outputs in the same folder
    save_dir = parquet_path.parent
    off_labels = use[["season", "week", "team", "off_cluster", "off_cluster_name"]].drop_duplicates()
    off_labels.to_parquet(save_dir / "off_labels.parquet", index=False)
    prof.round(3).to_csv(save_dir / "off_cluster_profiles.csv")
    joblib.dump(
        {"scaler": scaler, "kmeans": km, "OFF_FEATS": feats, "cluster_names": cluster_names},
        save_dir / "offense_clusters.joblib"
    )

    print(f"[SAVED] → {save_dir.name}/off_labels.parquet")


# ==========================================
# MAIN LOOP
# ==========================================
def batch_all_offense_seasons():
    for year in YEARS:
        season_dir = DATA_ROOT / str(year)
        if not season_dir.exists():
            continue
        files = sorted(season_dir.glob(PATTERN))
        if not files:
            print(f"[SKIP] No offense parquet in {season_dir}")
            continue
        try:
            process_offense_file(files[0], include_off_epa=INCLUDE_OFF_EPA)
        except Exception as e:
            print(f"[ERROR] {season_dir.name}: {e}")


if __name__ == "__main__":
    batch_all_offense_seasons()
