import numpy as np
from sentence_transformers import SentenceTransformer

def geo_vectors(df):
    return df[['lat','lon']].to_numpy(dtype=np.float32)

def semantic_vectors(df, model_name="all-MiniLM-L6-v2", cache_path=None, batch_size=64):
    texts = df['name'].astype(str).tolist()
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb.astype(np.float32) / norms.astype(np.float32)
    return emb
