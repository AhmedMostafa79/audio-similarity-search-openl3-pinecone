import numpy as np
import pandas as pd

def create_vectors(X: pd.DataFrame, y: pd.DataFrame):
    X = X.copy()
    X['id'] = [str(i) for i in range(len(X))]
    vectors = []
    for idx, row in X.iterrows():
        vectors.append({
            "id": row['id'],
            "values": row['features'],
            "metadata": {
                "sheikh_name": y.loc[idx, 'name'],
                "surah_number": int(y.loc[idx, 'surah_number']),
                "ayah_number": int(y.loc[idx, 'ayah_number']),
            },
        })
    return vectors

def predict_vector(feature_vector, pinecone_vdb):
    return pinecone_vdb.search(feature_vector.tolist(), top_k=1)
