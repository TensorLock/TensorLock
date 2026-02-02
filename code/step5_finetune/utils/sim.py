import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
from load_model import load_any_model

MATRIX_PATH = "../../step4_peft/peft_result.csv"
MAX_DIST = 10000


def is_real_weight(key: str):
    """Check if a weight key corresponds to a real weight, not auxiliary or quantization params"""
    bad_keywords = ["base_layer", "zeros", "absmax", "scaling", "quant", "bnb", "quant_state"]
    return not any(bad in key for bad in bad_keywords)


def filter_real_weights(sd):
    """Filter out non-real weights from state dict"""
    return {k: v for k, v in sd.items() if is_real_weight(k)}


def calculate_dist_metrics(vec_A: np.ndarray, vec_B: np.ndarray):
    """Compute normalized L1 and L2 distances between two numpy vectors"""
    if vec_A is None or vec_B is None or vec_A.size != vec_B.size:
        return None, None
    
    diff = vec_A - vec_B
    l1 = np.linalg.norm(diff, ord=1) / vec_A.size  
    l2 = np.linalg.norm(diff, ord=2) 
    return l1, l2


def calculate_dist_metrics_gpu(tensor_A: torch.Tensor, tensor_B: torch.Tensor):
    """Compute L1 and L2 distances on GPU using torch"""
    if tensor_A is None or tensor_B is None or tensor_A.shape != tensor_B.shape:
        return None, None
    
    diff = tensor_A - tensor_B
    l1 = torch.norm(diff, p=1).item() / tensor_A.numel()
    l2 = torch.norm(diff, p=2).item()
    
    return l1, l2


def calculate_module_metrics(A_name, B_name, A_sd, B_sd, out_csv):
    """Compute per-layer metrics for attention and MLP layers and save to CSV"""
    keysA = list(A_sd.keys())
    keysB = list(B_sd.keys())

    if len(keysA) != len(keysB):
        return []

    results = []
    for i, (kA, kB) in enumerate(zip(keysA, keysB)):
        attn_metrics = [None, None]
        if "self_attn" in kA and A_sd[kA].shape == B_sd[kB].shape:
            attn_metrics = calculate_dist_metrics_gpu(A_sd[kA], B_sd[kB])

        mlp_metrics = [None, None]
        if "mlp" in kA and A_sd[kA].shape == B_sd[kB].shape:
            mlp_metrics = calculate_dist_metrics_gpu(A_sd[kA], B_sd[kB])

        if any(m is not None for m in list(attn_metrics) + list(mlp_metrics)):
            results.append([i] + list(attn_metrics) + list(mlp_metrics))

    with open(out_csv, "a", newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Model_A", "Model_B", "layer", "attn_L1", "attn_L2", "mlp_L1", "mlp_L2"])
        for row in results:
            writer.writerow([A_name, B_name] + row)
    
    return results


def load_existing_results(summary_file):
    """Load existing computed pairs to avoid redundant computation"""
    existing_pairs = set()
    if os.path.exists(summary_file):
        try:
            df = pd.read_csv(summary_file)
            if not df.empty:
                for _, row in df.iterrows():
                    existing_pairs.add((str(row["Model_A"]), str(row["Model_B"])))
                    existing_pairs.add((str(row["Model_B"]), str(row["Model_A"])))
        except Exception as e:
            print(f"Failed to load cache file: {e}")
    return existing_pairs


def main_metrics(matrix_path=MATRIX_PATH, summary_file="metrics_summary_full.csv",
                 out_csv="module_metrics_results_full.csv", model_list="model_list.txt"):
    """Compute module-level L1/L2 distances between model pairs"""
    df_matrix = pd.read_csv(matrix_path, index_col=0).fillna("unknown")
    
    with open(model_list) as f:
        all_cluster_models = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    existing_pairs = load_existing_results(summary_file)
    if existing_pairs:
        print(f"Loaded cache for {len(existing_pairs)//2} computed model pairs.")

    models_to_compare = []
    for m in all_cluster_models:
        if m in df_matrix.index:
            row = df_matrix.loc[m]
            if (row == "unknown").all():
                models_to_compare.append(m)
        else:
            models_to_compare.append(m) 

    print(f"Total models in cluster: {len(all_cluster_models)}, models needing parent search: {len(models_to_compare)}")

    model_pairs = []
    for m_a in models_to_compare:
        for m_b in all_cluster_models:
            if m_a != m_b:
                if (str(m_a), str(m_b)) in existing_pairs:
                    continue
                model_pairs.append((m_a, m_b))

    if not model_pairs:
        print("All model pairs already computed or not required, skipping.")
        return

    for idx, (A_path, B_path) in enumerate(tqdm(model_pairs, desc="Pair-wise Sim")):

        if (str(A_path), str(B_path)) in existing_pairs:
            print(f"Skipping {A_path} and {B_path}, already calculated.")
            continue
        try:
            A_model = load_any_model(A_path)
            B_model = load_any_model(B_path)
            
            A_clean = filter_real_weights(A_model.state_dict())
            B_clean = filter_real_weights(B_model.state_dict())
            
            min_len = min(len(A_clean), len(B_clean))
            A_sd = dict(list(A_clean.items())[:min_len])
            B_sd = dict(list(B_clean.items())[:min_len])

            results = calculate_module_metrics(A_path, B_path, A_sd, B_sd, out_csv)
            
            if results:
                res_arr = np.array([r[1:] for r in results], dtype=float)
                means = np.nanmean(res_arr, axis=0).tolist()
            else:
                print(f"⚠️ No common layers between {A_path} and {B_path}. Recording as MAX_DIST.")
                means = [MAX_DIST, MAX_DIST, MAX_DIST, MAX_DIST]

            file_exists = os.path.isfile(summary_file)
            with open(summary_file, "a", newline='') as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(summary_file) == 0:
                    writer.writerow([
                        "Model_A", "Model_B", 
                        "mean_attn_L1_dist", "mean_attn_L2_dist", 
                        "mean_mlp_L1_dist", "mean_mlp_L2_dist"
                    ])
                writer.writerow([A_path, B_path] + means)

            existing_pairs.add((str(A_path), str(B_path)))
            existing_pairs.add((str(B_path), str(A_path)))

            del A_model, B_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error comparing {A_path} and {B_path}: {e}")
            continue


if __name__ == "__main__":
    main_metrics()