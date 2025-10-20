import pandas as pd
import json

# Input and output files
input_file = "data/raw/odds_data.txt"
output_file = "nfl_betting_data.parquet"

# Read JSON
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Remove metadata column
if 'DT_RowId' in df.columns:
    df = df.drop('DT_RowId', axis=1)

# Convert numeric columns
numeric_cols = ['week', 'spread', 'total', 'money_line', 'mov', 'ats_margin', 
                'combined_pts', 'OU_margin', 'days_of_rest', 'opp_days_of_rest', 
                'rest_advantage']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert date column
if 'game_date' in df.columns:
    df['game_date'] = pd.to_datetime(df['game_date'])

# Convert season to integer
if 'season' in df.columns:
    df['season'] = df['season'].astype(int)

# Convert Yes/No to boolean
bool_cols = ['conf_game', 'div_game']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': True, 'No': False})

# Save to Parquet
df.to_parquet(output_file, index=False, compression='snappy')

print(f"✓ Converted {len(df):,} rows to {output_file}")