import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# 1) Load the master file (built earlier)
df = pd.read_parquet("data/raw/model_dataset.parquet").sort_values(["season","week"]).reset_index(drop=True)

# 2) Feature sets (same as before)
CAT = [
    "home_team","away_team",
    "home_off_cluster","home_def_cluster","away_off_cluster","away_def_cluster",
    "pair_home_off_vs_away_def","pair_away_off_vs_home_def"
]
for c in CAT:
    if c not in df.columns: df[c] = "UNK"

NUM = [c for c in df.columns if any(s in c for s in ["_last1","_last3","_last5","te_pair_"])]

X = df[CAT + NUM]
y = df["combined_pts"]

# 3) Train/holdout split by YEAR
train_years = list(range(2015, 2025))   # <= 2024
test_years  = [2025]

X_train = X[df["season"].isin(train_years)]
y_train = y[df["season"].isin(train_years)]
X_test  = X[df["season"].isin(test_years)]
y_test  = y[df["season"].isin(test_years)]

# 4) Pipeline + fit on TRAIN ONLY (<=2024)
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ("num", "passthrough", NUM),
])
model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=500, random_state=42)
pipe = Pipeline([("prep", pre), ("mdl", model)])
pipe.fit(X_train, y_train)

# 5) Evaluate on strict 2025 holdout
pred_2025 = pipe.predict(X_test)
mae_2025 = mean_absolute_error(y_test, pred_2025)
print(f"Holdout 2025 MAE: {mae_2025:.2f}  (n={len(y_test)})")

# 6) Save the <=2024-trained model
joblib.dump({"pipe": pipe, "CAT": CAT, "NUM": NUM, "train_years": train_years}, "totals_model_thru_2024.joblib")
