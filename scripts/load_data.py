import numpy as np
import pandas as pd
import hnswlib

def load_pois(csv_path):
    df = pd.read_csv(csv_path)
    expected = ['name', 'lat', 'lon']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.dropna(subset=['name','lat','lon'])
    df = df.drop_duplicates(subset=['name','lat','lon']).reset_index(drop=True)
    return df

def load_hnsw(path, dim, space):
    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(path)
    return index

def generate_random_dataset(n_points, dim, seed=1):
    rng = np.random.default_rng(seed)
    data = rng.random((n_points, dim)).astype(np.float32)
    return data
