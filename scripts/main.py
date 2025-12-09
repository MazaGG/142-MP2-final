import os
import time
import numpy as np
import pandas as pd
import argparse
from sklearn.neighbors import KDTree
from load_data import load_pois, generate_random_dataset
from vectorize import geo_vectors, semantic_vectors
from evaluate import brute_force_nn_geo, brute_force_nn_semantic, recall_at_k
from sweep import sweep_params


def run_comparison(vectors, queries, brute_force_fn, hnsw_bin, space='l2', k=10):
    results = []

    # Brute Force
    print("\nRunning Brute Force...")
    start = time.time()
    gt_labels = brute_force_fn(queries, vectors, k=k)
    bf_time = time.time() - start
    print(f"Brute Force done in {bf_time:.6f}s")
    results.append({
        'method': 'BruteForce',
        'build_time': 0.0,
        'query_time': bf_time,
        'recall_at_k': 1.0,
        'M': 'n/a',
        'efConstruction': 'n/a',
        'efSearch': 'n/a',
    })

    # KDTree
    print("\nBuilding KDTree...")
    start = time.time()
    kd = KDTree(vectors, metric='euclidean')
    build_time = time.time() - start
    print(f"KDTree built in {build_time:.6f}s")

    print("Querying KDTree...")
    start = time.time()
    dist, idx = kd.query(queries, k=k)
    query_time = time.time() - start
    recall = recall_at_k(gt_labels, idx)
    print(f"KDTree query done in {query_time:.6f}s, recall@{k}={recall:.4f}")
    results.append({
        'method': 'KDTree',
        'build_time': build_time,
        'query_time': query_time,
        'recall_at_k': recall,
        'M': 'n/a',
        'efConstruction': 'n/a',
        'efSearch': 'n/a',
    })

    # HNSW
    M_vals = [8, 16, 32]
    efC_vals = [50, 100, 200]
    efS_vals = [10, 100, 500]
    if len(vectors) >= 100000:
        M_vals = [8, 16, 32, 64]
        efC_vals = [10, 100, 200, 400]
        efS_vals = [10, 100, 500, 1000]
    dim = vectors.shape[1]
    results = sweep_params(dim, results, vectors, queries, brute_force_fn, space, M_vals, efC_vals, efS_vals, k, hnsw_bin)

    return pd.DataFrame(results)


def run_experiment(csv_path, out_folder="results", hnsw_bin="bin", num_queries=1000, k=10, n_points=None):
    os.makedirs(out_folder, exist_ok=True)

    if n_points is None:
        print("Loading POIs from:", csv_path)
        df = load_pois(csv_path)
        print("Total POIs:", len(df))

        # GEO vectors
        print("\nCreating geographic vectors...")
        geo = geo_vectors(df)
        lat_min, lat_max = df['lat'].min(), df['lat'].max()
        lon_min, lon_max = df['lon'].min(), df['lon'].max()
        q_geo = np.column_stack([
            np.random.uniform(lat_min, lat_max, size=num_queries),
            np.random.uniform(lon_min, lon_max, size=num_queries)
        ]).astype(np.float32)

        # SEMANTIC vectors 
        print("\nCreating semantic vectors...")
        sem = semantic_vectors(df, model_name="all-MiniLM-L6-v2")
        rng = np.random.default_rng()
        sample_idxs = rng.choice(len(sem), size=num_queries, replace=True)
        q_sem = sem[sample_idxs]

    else:
        # GEO vectors
        print("\nCreating geographic vectors...")
        geo = generate_random_dataset(csv_path, n_points=n_points, dim=2)
        rng = np.random.default_rng()
        sample_idxs = rng.choice(len(geo), size=num_queries, replace=True)
        q_geo = geo[sample_idxs]

        # SEMANTIC vectors
        print("\nCreating semantic vectors...")
        sem = generate_random_dataset(csv_path, n_points=n_points, dim=384)
        sample_idxs = rng.choice(len(sem), size=num_queries, replace=True)
        q_sem = sem[sample_idxs]

    print("\n----- GEO Comparison -----")
    df_geo = run_comparison(geo, q_geo, brute_force_nn_geo, hnsw_bin, space='l2', k=k)
    geo_out = os.path.join(out_folder, "comparison_geo.csv")
    df_geo.to_csv(geo_out, index=False)
    print("Saved GEO comparison results to:", geo_out)

    print("\n----- SEMANTIC Comparison -----")
    df_sem = run_comparison(sem, q_sem, brute_force_nn_semantic, hnsw_bin, space='cosine', k=k)
    sem_out = os.path.join(out_folder, "comparison_semantic.csv")
    df_sem.to_csv(sem_out, index=False)
    print("Saved SEMANTIC comparison results to:", sem_out)


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--hnsw_bin", type=str, default="bin")
    parser.add_argument("--num_queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--synthetic_data", type=int)
    args = parser.parse_args()

    if args.synthetic_data is None:
        run_experiment(
            csv_path=args.csv,
            out_folder=args.out,
            hnsw_bin=args.hnsw_bin,
            num_queries=args.num_queries,
            k=args.k
        )
    else:
        run_experiment(
            csv_path=args.csv,
            out_folder=args.out,
            hnsw_bin=args.hnsw_bin,
            num_queries=args.num_queries,
            k=args.k,
            n_points=args.synthetic_data
        )
