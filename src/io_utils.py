import os
import numpy as np
import pandas as pd

def save_into_files(df: pd.DataFrame, csv_name: str, np_name: str):
    df.drop('features', axis=1).to_csv(csv_name, index=False)
    np.save(np_name, np.array(df['features'].tolist()))

def load_metadata(metadata_file_name: str, features_file_name: str):
    csv_data = pd.read_csv(metadata_file_name)
    meta_data = pd.DataFrame(columns=csv_data.columns)
    for col in csv_data.columns:
        if col in csv_data:
            meta_data[col] = csv_data[col]
    features = np.load(features_file_name, allow_pickle=True)
    if len(features) != len(meta_data):
        raise ValueError("Length mismatch between features and metadata")
    meta_data['features'] = list(features)
    return meta_data
