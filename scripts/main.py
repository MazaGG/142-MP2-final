import os
import numpy as np
import argparse
from load_data import load_pois
from vectorize import geo_vectors, semantic_vectors
from evaluate import brute_force_nn_geo, brute_force_nn_semantic
from sweep import sweep_params

def run_experiment(csv_path, hnsw_bin=None, out_folder="results", num_queries=1000, sample_queries=500, k=10):
    os.makedirs(out_folder, exist_ok=True)
    print("Loading POIs from:", csv_path)
    df = load_pois(csv_path)
    print("Total POIs:", len(df))

    # GEO vectors
    print("Creating geographic vectors...")
    geo = geo_vectors(df)

    # semantic vectors
    print("Creating semantic vectors...")
    sem = semantic_vectors(df, model_name="all-MiniLM-L6-v2")

    # generate random geo query points within bbox
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    print("Generating random geo queries...")
    q_geo = np.column_stack([
        np.random.uniform(lat_min, lat_max, size=num_queries),
        np.random.uniform(lon_min, lon_max, size=num_queries)
    ]).astype(np.float32)

    # semantic query vectors: pick random POI names as queries
    print("Generating semantic queries (random POI names)...")
    rng = np.random.default_rng()
    sample_idxs = rng.choice(len(sem), size=num_queries, replace=True)
    q_sem = sem[sample_idxs]

    # Sweep parameters
    M_vals = [8, 16, 32]
    efC_vals = [50, 100, 200]
    efS_vals = [50, 100, 200]

    print("\n=== Sweep: GEO (low-dimensional) ===")
    df_geo = sweep_params(geo, q_geo, brute_force_nn_geo, space='l2', M_vals=M_vals, efC_vals=efC_vals, efS_vals=efS_vals, k=k, sample_queries=sample_queries, hnsw_bin=hnsw_bin)
    geo_out = os.path.join(out_folder, "sweep_geo_results.csv")
    df_geo.to_csv(geo_out, index=False)
    print("Saved GEO sweep results to:", geo_out)

    print("\n=== Sweep: SEMANTIC (high-dimensional) ===")
    df_sem = sweep_params(sem, q_sem, brute_force_nn_semantic, space='cosine', M_vals=M_vals, efC_vals=efC_vals, efS_vals=efS_vals, k=k, sample_queries=sample_queries, hnsw_bin=hnsw_bin)
    sem_out = os.path.join(out_folder, "sweep_semantic_results.csv")
    df_sem.to_csv(sem_out, index=False)
    print("Saved SEMANTIC sweep results to:", sem_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str)
    parser.add_argument("--hnsw_bin", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--num_queries", type=int)
    parser.add_argument("--sample_queries", type=int)
    parser.add_argument("--k", type=int)
    args = parser.parse_args()

    run_experiment(args.csv, hnsw_bin=args.hnsw_bin, out_folder=args.out, num_queries=args.num_queries, sample_queries=args.sample_queries, k=args.k)
