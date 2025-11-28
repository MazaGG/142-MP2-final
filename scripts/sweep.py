import numpy as np
import pandas as pd
import time
from build_hnsw import build_index, query_index
from evaluate import recall_at_k

def sweep_params(vectors, queries, gt_fn, space, M_vals, efC_vals, efS_vals, k=10, sample_queries=None, hnsw_bin=None):
    results = []
    Q_total = queries.shape[0]
    if sample_queries is not None and sample_queries < Q_total:
        idxs = np.random.choice(Q_total, sample_queries, replace=False)
        queries_sample = queries[idxs]
    else:
        queries_sample = queries

    # compute ground truth once for sampled queries
    gt = gt_fn(queries_sample, vectors, k=k)        

    for M in M_vals:
        for efC in efC_vals:
            index, build_time = build_index(vectors, space=space, M=M, ef_construction=efC)
            if hnsw_bin is not None:
                hnsw_bin = f"{hnsw_bin}/M{M}_efC{efC}.bin"
                index.save_index(hnsw_bin)
            for efS in efS_vals:
                start_q = time.time()
                labels, distances, qtime = query_index(index, queries_sample, k=k, ef_search=efS)
                recall = recall_at_k(gt, labels)
                results.append({
                    'M': M,
                    'efConstruction': efC,
                    'efSearch': efS,
                    'recall@k': recall,
                    'build_time_s': build_time,
                    'query_time_s': qtime,
                    'qps': len(queries_sample) / qtime
                })
                print(f"M={M} efC={efC} efS={efS} recall={recall:.4f} qtime={qtime:.3f}s")
    return pd.DataFrame(results)
