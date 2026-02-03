import json
import pandas as pd
import numpy as np
from tqdm import tqdm

MODEL_LIST_PATH = "../../Benchmark/model_list.json"
GROUND_TRUTH_PATH = "../../Benchmark/cluster.json"
CREATED_AT_CSV = "./model_created_at.csv"
OUTPUT_MATRIX_CSV = "./chron_result.csv"

def normalize_to_underscore(name):
    if not isinstance(name, str):
        return name
    return name.replace("/", "_")

def main():
    print("Loading model list...")
    with open(MODEL_LIST_PATH, 'r') as f:
        model_data = json.load(f)
        raw_models = model_data.get("models", [])
    
    all_models_norm = [normalize_to_underscore(m) for m in raw_models]
    
    n = len(all_models_norm)
    matrix_df = pd.DataFrame(
        np.zeros((n, n), dtype=int), 
        index=all_models_norm, 
        columns=all_models_norm
    )

    print("Loading creation timestamps...")
    time_df = pd.read_csv(CREATED_AT_CSV)
    time_df = time_df[~time_df['createdAt'].str.contains("Error", na=False)]
    
    time_df['modelId_norm'] = time_df['modelId'].apply(normalize_to_underscore)
    time_dict = pd.to_datetime(time_df.set_index('modelId_norm')['createdAt']).to_dict()

    print("Processing clusters and filling directed edges...")
    with open(GROUND_TRUTH_PATH, 'r') as f:
        clusters = json.load(f)

    for cluster in tqdm(clusters, desc="Processing Clusters"):
        cluster_info = []
        for m in cluster:
            m_norm = normalize_to_underscore(m)
            if m_norm in time_dict:
                cluster_info.append({"name": m_norm, "time": time_dict[m_norm]})
            else:
                continue

        sorted_cluster = sorted(cluster_info, key=lambda x: x['time'])
        
        num_in_cluster = len(sorted_cluster)
        
        for i in range(num_in_cluster - 1):
            early_model = sorted_cluster[i]['name']
            late_model = sorted_cluster[i+1]['name']
            
            if late_model in matrix_df.index and early_model in matrix_df.columns:
                matrix_df.at[late_model, early_model] = 1
                print(f"  Link: {early_model} (Col) -> {late_model} (Row)")

    print(f"Saving adjacency matrix (shape: {matrix_df.shape}) to {OUTPUT_MATRIX_CSV}...")
    matrix_df.to_csv(OUTPUT_MATRIX_CSV)
    print("Task completed!")

if __name__ == "__main__":
    main()