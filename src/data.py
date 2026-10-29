import pandas as pd
from .config import CLIENT_FILE, RECORD_FILE, ID_COLUMN, TARGET

def load_data(client_path=CLIENT_FILE, record_path=RECORD_FILE):
    
    if not client_path.exists():
        raise FileNotFoundError(f"Could not find {client_path}. Keep Client.csv and Record.csv inside telecom/")
    if not record_path.exists():
        raise FileNotFoundError(f"Could not find {record_path}. Keep Client.csv and Record.csv inside telecom/")

    
    client = pd.read_csv(client_path, low_memory=False)
    record = pd.read_csv(record_path, low_memory=False)

    
    if ID_COLUMN not in client.columns:
        raise ValueError("Client.csv is missing required column " + ID_COLUMN)
        
    if ID_COLUMN not in record.columns:
        raise ValueError(f"Record.csv is missing required column {ID_COLUMN!r}")

    
    merged = record.merge(
        client,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
        suffixes=('_record', '_client')
    )

    if TARGET not in merged.columns:
        raise ValueError(f"Target column {TARGET!r} was not found.")
        
    return client, record, merged

def dataset_profile(df):
    
    dtypes = df.dtypes.astype(str)
    missing_count = df.isna().sum()
    missing_percent = (df.isna().mean() * 100).round(2)
    unique_vals = df.nunique(dropna=True)
    
   
    summary_df = pd.DataFrame({
        "dtype": dtypes,
        "missing_count": missing_count,
        "missing_percent": missing_percent,
        "unique_values": unique_vals,
    })
    
    return summary_df.sort_values(by=["missing_percent", "unique_values"], ascending=[False, True])

def check_target(df, target=TARGET):
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} is missing.")
        
    y_target = df[target].dropna()
    
    unique_vals = sorted(y_target.unique().tolist())
    
    if len(unique_vals) != 2:
        raise ValueError(f"Expected a binary target, but found {len(unique_vals)} values: {unique_vals}")
        
    counts = y_target.value_counts(dropna=False)
    proportions = y_target.value_counts(normalize=True).round(4)
    
    return counts, proportions