import json
import os
import numpy as np
from sklearn.metrics import adjusted_rand_score

PRED_PATH = "../step1_cluster/clusters.json"
GT_PATH = "../../dataset/cluster.json"

def evaluate_clustering_ari():
    if not os.path.exists(PRED_PATH) or not os.path.exists(GT_PATH):
        print("❌ File path does not exist.")
        return

    with open(PRED_PATH, 'r', encoding='utf-8') as f:
        pred_clusters_raw = json.load(f)
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_clusters_raw = json.load(f)

    def normalize_clusters(clusters):
        return [[m.replace("/", "_") for m in c] for c in clusters]

    pred_clusters = normalize_clusters(pred_clusters_raw)
    gt_clusters = normalize_clusters(gt_clusters_raw)

    all_gt_models = sorted(list(set(m for c in gt_clusters for m in c)))
    model_to_idx = {model: i for i, model in enumerate(all_gt_models)}
    
    num_models = len(all_gt_models)
    print(f"Total ground-truth models: {num_models}")

    gt_labels = np.zeros(num_models, dtype=int)
    for cluster_id, cluster in enumerate(gt_clusters):
        for model in cluster:
            if model in model_to_idx:
                gt_labels[model_to_idx[model]] = cluster_id

    pred_labels = np.full(num_models, -1, dtype=int)
    max_pred_id = len(pred_clusters)

    for cluster_id, cluster in enumerate(pred_clusters):
        for model in cluster:
            if model in model_to_idx:
                pred_labels[model_to_idx[model]] = cluster_id

    missing_mask = (pred_labels == -1)
    num_missing = np.sum(missing_mask)
    
    if num_missing > 0:
        pred_labels[missing_mask] = np.arange(max_pred_id, max_pred_id + num_missing)

    ari_score = adjusted_rand_score(gt_labels, pred_labels)

    print("-" * 50)
    print(f"Number of predicted clusters: {len(pred_clusters)} (+{num_missing} isolated models)")
    print(f"ARI: {ari_score:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    evaluate_clustering_ari()