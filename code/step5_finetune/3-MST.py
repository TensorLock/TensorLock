import pandas as pd
import json
import networkx as nx
import numpy as np

CLUSTERS_JSON = "../step1_cluster/clusters.json"
INPUT_MATRIX_CSV = "./finetune_result.csv"
OUTPUT_MATRIX_CSV = "./final_result.csv"
METRICS_SUMMARY = "./metrics_summary.csv"
DIRECTION_CACHE = "./entropy_cache.csv" 

def ensure_cluster_connectivity_mst(df_matrix, cluster_models, dist_df, entropy_dict):
    """
    Ensure that models within a cluster form a connected graph using Liu & Zhu algorithm idea:
    - Construct a fully connected weighted graph with L2 distance as weight
    - Compute minimum spanning tree (MST) to connect all nodes
    - Set direction based on entropy (lower entropy -> higher entropy)
    """
    cluster_models = [m for m in cluster_models if m in df_matrix.index]
    if len(cluster_models) < 2:
        return df_matrix

    G = nx.Graph()
    for i, u in enumerate(cluster_models):
        for j in range(i+1, len(cluster_models)):
            v = cluster_models[j]
            d_row = dist_df[((dist_df['Model_A'] == u) & (dist_df['Model_B'] == v)) |
                            ((dist_df['Model_A'] == v) & (dist_df['Model_B'] == u))]
            if not d_row.empty:
                attn_dist = d_row['mean_attn_L2_dist'].values[0]
                mlp_dist = d_row['mean_mlp_L2_dist'].values[0]
                dist = np.nanmean([attn_dist, mlp_dist])
                if pd.notna(dist):
                    G.add_edge(u, v, weight=dist)

    # Compute MST
    mst = nx.minimum_spanning_tree(G, weight='weight')
    
    for u, v, data in mst.edges(data=True):
        ent_u = entropy_dict.get(u, 0)
        ent_v = entropy_dict.get(v, 0)
        if ent_u < ent_v:
            src, tgt = u, v
        else:
            src, tgt = v, u
        if df_matrix.loc[tgt, src] in ["unknown", 0, "0", np.nan]:
            df_matrix.loc[tgt, src] = "finetune"
            print(f"    ✨ MST Connected nodes: {src} -> {tgt} (L2 Distance: {data['weight']:.4f})")
    
    return df_matrix


def main():
    full_matrix = pd.read_csv(INPUT_MATRIX_CSV, index_col=0).fillna("unknown")
    with open(CLUSTERS_JSON, "r") as f:
        clusters = json.load(f)
    
    dist_df = pd.read_csv(METRICS_SUMMARY)
    entropy_df = pd.read_csv(DIRECTION_CACHE)
    entropy_dict = entropy_df.set_index("model")["RankEff"].to_dict()
    
    for i, cluster_models in enumerate(clusters):
        if len(cluster_models) < 2:
            continue
            
        cluster_models = [m for m in cluster_models if m in full_matrix.index]
        
        full_matrix = ensure_cluster_connectivity_mst(full_matrix, cluster_models, dist_df, entropy_dict)

    full_matrix.to_csv(OUTPUT_MATRIX_CSV)
    print(f"\n✅ MST-based correction completed! Result saved to: {OUTPUT_MATRIX_CSV}")


if __name__ == "__main__":
    main()