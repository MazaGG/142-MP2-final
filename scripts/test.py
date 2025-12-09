import numpy as np
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from evaluate import brute_force_nn_semantic 
from load_data import load_pois, load_hnsw

def load_semantic_vectors(df, model_name="all-MiniLM-L6-v2"):
  model = SentenceTransformer(model_name)
  texts = df["name"].astype(str).tolist()
  vectors = model.encode(texts, convert_to_numpy=True).astype(np.float32)
  return vectors, model

def interactive_session(csv_path, index_path, k=5):
  print("Loading POIs...")
  df = load_pois(csv_path)

  print("Loading semantic vectors...")
  vectors, model = load_semantic_vectors(df)

  dim = vectors.shape[1]

  print("Loading HNSW index...")
  index = load_hnsw(index_path, dim=dim, space='cosine')
  index.set_ef(200) 

  print("\nType a place or description (or type 'exit')\n")

  while True:
    query = input("Query: ").strip()
    if query.lower() == "exit":
      break

    q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    start = time.time()
    labels, dist = index.knn_query(q_vec, k=k)
    runtime = time.time() - start
    labels = labels[0]
    dist = dist[0]

    print("\n--- Approximate Nearest Neighbors (HNSW) ---")
    print(f"(search time: {runtime:.4f} seconds)")
    for i, idx in enumerate(labels):
      print(f"{i+1}. {df.iloc[idx]['name']}  (dist={dist[i]:.4f})")

    print("\n--- True Nearest Neighbors (Brute Force) ---")
    start = time.time()
    true_ids = brute_force_nn_semantic(q_vec, vectors, k=k)[0]
    runtime = time.time() - start
    print(f"(search time: {runtime:.4f} seconds)")
    for i, idx in enumerate(true_ids):
      print(f"{i+1}. {df.iloc[idx]['name']}")

    print("\n--------------------------------------------\n")


if __name__ == "__main__":
  CSV_PATH = "data/sea_tourist_pois.csv"          
  INDEX_PATH = "bin/sea/M32_efC200.bin"         
  interactive_session(CSV_PATH, INDEX_PATH, k=10)
