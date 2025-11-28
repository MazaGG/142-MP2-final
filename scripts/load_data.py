import pandas as pd

def load_pois(csv_path):
    df = pd.read_csv(csv_path)
    expected = ['name', 'lat', 'lon']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.dropna(subset=['name','lat','lon'])
    df = df.drop_duplicates(subset=['name','lat','lon']).reset_index(drop=True)
    return df
