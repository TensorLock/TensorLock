import pandas as pd
import json
import numpy as np
from collections import deque

GT_CLUSTER_PATH = "../../evaluation/Benchmark/cluster.json"
GT_MATRIX_PATH = "../step0_ground_truth/ground_truth_matrix.csv"
PRED_MATRIX_OUT = "../step5_finetune/final_result.csv"

SYNONYM_MAP = {
    "Gensyn_Qwen2.5-0.5B-Instruct": "Qwen_Qwen2.5-0.5B-Instruct",
    "Gensyn_Qwen2.5-1.5B-Instruct": "Qwen_Qwen2.5-1.5B-Instruct",
}

def apply_synonyms(df):
    """
    Merge equivalent models' rows and columns.
    Uses numpy's logical_or for compatibility with bool/int types.
    """
    df = df.copy()
    for syn, target in SYNONYM_MAP.items():
        if syn in df.index and target in df.index:            
            df = df.astype(int)
            
            df.loc[:, target] = np.logical_or(df.loc[:, target].values, df.loc[:, syn].values).astype(int)
            df.loc[target, :] = np.logical_or(df.loc[target, :].values, df.loc[syn, :].values).astype(int)
            
            df = df.drop(index=syn, columns=syn)
        elif syn in df.index:
            df = df.rename(index={syn: target}, columns={syn: target})
    return df

def convert_to_binary(val):
    s_val = str(val).strip().lower()
    if s_val in ['0', '0.0', 'none', 'nan', 'self', '', 'unknown']:
        return 0
    return 1

def normalize_name(name):
    return name.replace("/", "_")

def get_reachable_matrix(df):
    """
    Compute the reachability matrix of models.
    For each model, mark all models reachable via directed edges.
    """
    nodes = df.index.tolist()
    adj = {node: [] for node in nodes}
    for col in df.columns:
        for row in df.index:
            if df.loc[row, col] == 1:
                adj[col].append(row)
    
    reachable = pd.DataFrame(0, index=nodes, columns=nodes)
    for start_node in nodes:
        visited = set()
        queue = deque([start_node])
        while queue:
            curr = queue.popleft()
            if curr not in visited:
                if curr != start_node:
                    reachable.loc[start_node, curr] = 1
                visited.add(curr)
                queue.extend(adj[curr])
    return reachable

def calculate_metrics(tp, fp, fn):
    """Compute precision, recall, and F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def run_evaluation():
    """Run evaluation of predicted relations against ground-truth."""
    gt_df_raw = pd.read_csv(GT_MATRIX_PATH, index_col=0).fillna(0)
    pred_df_raw = pd.read_csv(PRED_MATRIX_OUT, index_col=0).fillna(0)

    gt_df_raw.index = [normalize_name(str(x)) for x in gt_df_raw.index]
    gt_df_raw.columns = [normalize_name(str(x)) for x in gt_df_raw.columns]

    pred_df_raw.index = [normalize_name(str(x)) for x in pred_df_raw.index]
    pred_df_raw.columns = [normalize_name(str(x)) for x in pred_df_raw.columns]

    with open(GT_CLUSTER_PATH, 'r') as f:
        gt_clusters_raw = json.load(f)
        gt_clusters = []

        for cluster in gt_clusters_raw:
            norm_c = []
            for m in cluster:
                name = normalize_name(m)
                final_name = SYNONYM_MAP.get(name, name)
                norm_c.append(final_name)
            gt_clusters.append(list(set(norm_c)))

    root_models_all = [SYNONYM_MAP.get(normalize_name(c[0]), normalize_name(c[0])) 
                       for c in gt_clusters_raw if len(c) > 0]

    gt_df_raw = gt_df_raw.applymap(convert_to_binary)
    pred_df_raw = pred_df_raw.applymap(convert_to_binary)

    gt_df_raw = apply_synonyms(gt_df_raw)
    pred_df_raw = apply_synonyms(pred_df_raw)

    all_gt_models = sorted(list(gt_df_raw.index))
    gt_df = gt_df_raw.loc[all_gt_models, all_gt_models].astype(int)
    pred_df = pred_df_raw.reindex(index=all_gt_models, columns=all_gt_models).fillna(0).astype(int)

    np.fill_diagonal(gt_df.values, 0)
    np.fill_diagonal(pred_df.values, 0)
    
    missing_in_pred = set(all_gt_models) - set(pred_df_raw.index)
    if missing_in_pred:
        print(f"Warning: {len(missing_in_pred)} models in GT are MISSING in predictions!")

    common_models = all_gt_models
    if not common_models:
        print("Error: No common models found between GT and Pred!")
        return

    total_pred_edges = np.sum(pred_df.values)
    print(f"Total relations found in prediction: {total_pred_edges}")
    if total_pred_edges == 0:
        print("Warning: Prediction matrix has ZERO relations after processing!")

    gt_reach = get_reachable_matrix(gt_df)
    pred_reach = get_reachable_matrix(pred_df)

    micro_tp = np.sum((pred_df.values == 1) & (gt_df.values == 1))
    micro_fp = np.sum((pred_df.values == 1) & (gt_df.values == 0))
    micro_fn = np.sum((pred_df.values == 0) & (gt_df.values == 1))
    
    cluster_stats = []
    for cluster in gt_clusters:
        c_models = [m for m in cluster if m in common_models]
        if len(c_models) < 2: continue
        
        c_gt = gt_df.loc[c_models, c_models]
        c_pred = pred_df.loc[c_models, c_models]
        
        tp = np.sum((c_pred.values == 1) & (c_gt.values == 1))
        fp = np.sum((c_pred.values == 1) & (c_gt.values == 0))
        fn = np.sum((c_pred.values == 0) & (c_gt.values == 1))
        
        reach_scores = []
        for m in c_models:
            gt_downstream = np.sum(gt_reach.loc[m, c_models] == 1)
            if gt_downstream == 0: continue
            
            pred_downstream = np.sum((gt_reach.loc[m, c_models] == 1) & (pred_reach.loc[m, c_models] == 1))
            reach_scores.append(pred_downstream / gt_downstream)
            
        cluster_reach = np.mean(reach_scores) if reach_scores else 0
        p, r, f1 = calculate_metrics(tp, fp, fn)
        cluster_stats.append({'p': p, 'r': r, 'reach': cluster_reach})

    macro_p = np.mean([s['p'] for s in cluster_stats]) if cluster_stats else 0
    macro_r = np.mean([s['r'] for s in cluster_stats]) if cluster_stats else 0
    macro_reach = np.mean([s['reach'] for s in cluster_stats]) if cluster_stats else 0
    
    all_reach_scores = []
    for m in common_models:
        gt_ds = np.sum(gt_reach.loc[m, :] == 1)
        if gt_ds > 0:
            pred_ds = np.sum((gt_reach.loc[m, :] == 1) & (pred_reach.loc[m, :] == 1))
            all_reach_scores.append(pred_ds / gt_ds)
    micro_reach = np.mean(all_reach_scores) if all_reach_scores else 0

    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Micro-Avg':<12} | {'Macro-Avg':<12}")
    print("-" * 60)
    p_mic, r_mic, _ = calculate_metrics(micro_tp, micro_fp, micro_fn)
    print(f"{'Precision':<20} | {p_mic:<12.4f} | {macro_p:<12.4f}")
    print(f"{'Recall':<20} | {r_mic:<12.4f} | {macro_r:<12.4f}")
    print(f"{'Reachability':<20} | {micro_reach:<12.4f} | {macro_reach:<12.4f}")
    print("-" * 60)
    print(f"Total Common Models: {len(common_models)}")
    print(f"Total Clusters Evaluated: {len(cluster_stats)}")
    print("="*60)

    print("\n🌳 Root Models Reachability Analysis 🌳")
    print("-" * 70)
    print(f"{'Root Model Name':<50} | {'Reachability':<10}")
    print("-" * 70)
    
    root_reach_scores = []
    for root in root_models_all:
        if root not in common_models:
            print(f"{root[:50]:<50} | {'MISSING':<10}")
            continue
            
        gt_downstream_nodes = gt_reach.columns[gt_reach.loc[root] == 1].tolist()
        num_gt = len(gt_downstream_nodes)
        
        if num_gt == 0:
            print(f"{root[:50]:<50} | {'NO CHILD':<10}")
            continue
            
        num_pred_reachable = np.sum(pred_reach.loc[root, gt_downstream_nodes] == 1)
        score = num_pred_reachable / num_gt
        root_reach_scores.append(score)
        print(f"{root[:50]:<50} | {score:<10.4f}")

    avg_root_reach = np.mean(root_reach_scores) if root_reach_scores else 0
    print("-" * 70)
    print(f"{'ROOT MODELS AVERAGE REACHABILITY':<50} | {avg_root_reach:<10.4f}")
    print("="*70)

if __name__ == "__main__":
    run_evaluation()