import hnswlib
import time
import numpy as np

def build_index(vectors, space, M=16, ef_construction=200, max_elements=None):
    n, dim = vectors.shape
    if max_elements is None:
        max_elements = n
    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
    start = time.time()
    p.add_items(vectors, ids=np.arange(n))
    build_time = time.time() - start
    return p, build_time

def query_index(index, queries, k=10, ef_search=50):
    index.set_ef(ef_search)
    start = time.time()
    labels, distances = index.knn_query(queries, k=k)
    qtime = time.time() - start
    return labels, distances, qtime
