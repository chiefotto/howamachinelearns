# predict_totals.py
import joblib, pandas as pd
from pathlib import Path

MODEL = joblib.load("totals_model_thru_2024.joblib")
PIPE = MODEL["pipe"]

def predict_one(model_df: pd.DataFrame, season:int, week:int, home:str, away:str) -> float:
    row = model_df.query("season==@season and week==@week and home_team==@home and away_team==@away")
    if row.empty:
        raise ValueError("Game row not found; ensure model_dataset includes this week.")
    return float(PIPE.predict(row[MODEL["CAT"] + MODEL["NUM"]])[0])

if __name__ == "__main__":
    df = pd.read_parquet(Path("data/raw/model_dataset.parquet"))
    print(predict_one(df, 2025, 1, "Ravens", "Cardinals"))
