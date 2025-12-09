import numpy as np
import pandas as pd
import time
from build_hnsw import build_index, query_index
from evaluate import recall_at_k

def sweep_params(dim, results, vectors, queries, gt_fn, space, M_vals, efC_vals, efS_vals, k=10, hnsw_bin=None):

    # compute ground truth once for queries
    gt = gt_fn(queries, vectors, k=k)        

    for M in M_vals:
        for efC in efC_vals:
            print(f"\nBuilding HNSW M{M}_efC{efC}...")
            index, build_time = build_index(vectors, space=space, M=M, ef_construction=efC)
            # print(f"HNSW M{M}_efC{efC} built in {build_time:.6f}s")
            if hnsw_bin is not None:
                save_path = f"{hnsw_bin}/{dim}/M{M}_efC{efC}.bin"
                index.save_index(save_path)
            for efS in efS_vals:
                print(f"Querying HNSW M{M}_efC{efC}_efS{efS}...")
                start_q = time.time()
                labels, distances, query_time = query_index(index, queries, k=k, ef_search=efS)
                recall = recall_at_k(gt, labels)
                results.append({
                    'method': 'HNSW',
                    'build_time': build_time,
                    'query_time': query_time,
                    'recall_at_k': recall,
                    'M': M,
                    'efConstruction': efC,
                    'efSearch': efS,
                })
                # print(f"HNSW M{M}_efC{efC}_efS{efS} query done in {query_time:.6f}s, recall@{k}={recall:.4f}")
    return results
