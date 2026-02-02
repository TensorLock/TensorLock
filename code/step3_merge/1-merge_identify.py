import os
import torch
import json
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from utils.load_model import load_any_model, get_safe_path
from utils.ties_recovery import ties_recovery
from utils.moe_recovery import moe_recovery

MODEL_BASE_DIR = "../../dataset/model"
CLUSTERS_JSON = "../step1_cluster/clusters.json"
INPUT_CSV = "../step2_quantize/quantize_result.csv"
OUTPUT_CSV = "./merge_result.csv"

RESIDUAL_THRESHOLD = 0.2


def load_any_model_sd(model_name):
    """Load model and return filtered state_dict with normalized keys"""
    try:
        model_path = get_safe_path(model_name)

        gguf_file = None
        if os.path.isdir(model_path):
            gguf_file = next((f for f in os.listdir(model_path) if f.endswith((".gguf"))), None)
        if gguf_file:
            return None

        skip_keywords = ["GGUF", "gguf", "GPTQ", "gptq", "AWQ", "awq", "Int4", "Int8"]
        for keyword in skip_keywords:
            if keyword in model_path:
                return None

        adapter_file = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_file):
            return None

        print(f"Loading {model_name}")
        model = load_any_model(model_name)
        sd = model.state_dict()

        clean_sd = {}
        for k, v in sd.items():
            if any(bad in k for bad in ["zeros", "absmax", "scaling", "quant", "bnb", "lora_"]):
                continue
            norm_k = k.replace(".base_layer.", ".").replace("base_model.model.", "")
            clean_sd[norm_k] = v.detach().cpu().float()

        del model
        return clean_sd

    except Exception as e:
        print(f" ❌ [Error] Failed to load {model_name}: {e}")
        return None


def solve_alpha_beta(A, B, C):
    X = np.stack([A, B], axis=1)
    sol, _, _, _ = np.linalg.lstsq(X, C, rcond=None)
    return float(sol[0]), float(sol[1])


def check_linear_merge(sd_A, sd_B, sd_C):
    """Check whether C = aA + bB"""
    keys = [
        k for k in sd_A.keys()
        if "self_attn.o_proj.weight" in k or "mlp.down_proj.weight" in k
    ]
    if not keys:
        return False, 0, 0

    results = []

    pbar = tqdm(keys, desc="  └─ Layers", leave=False, disable=len(keys) < 5)

    for k in pbar:
        if k not in sd_B or k not in sd_C:
            continue
        try:
            vec_A = sd_A[k].cpu().numpy().flatten()
            vec_B = sd_B[k].cpu().numpy().flatten()
            vec_C = sd_C[k].cpu().numpy().flatten()

            a, b = solve_alpha_beta(vec_A, vec_B, vec_C)
            res = np.linalg.norm(vec_C - (a * vec_A + b * vec_B))

            is_hard = (
                (abs(a - 1) < 0.05 and abs(b) < 0.05) or
                (abs(a) < 0.05 and abs(b - 1) < 0.05)
            )

            results.append({'res': res, 'is_hard': is_hard})
            pbar.set_postfix({"last_res": f"{res:.4f}"})

        except Exception:
            continue

    if not results:
        return False, 10, 100

    avg_res = np.mean([r['res'] for r in results])
    split_pct = 100 * sum([1 for r in results if r['is_hard']]) / len(results)

    return (avg_res < RESIDUAL_THRESHOLD and split_pct < 50), avg_res, split_pct


CACHE_FILE = "linear_merge_cache.json"


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache_data):
    with open(CACHE_FILE, 'w') as f:
        print("Cache saved successfully!")
        json.dump(cache_data, f, indent=4)


def get_trio_key(a, b, c):
    """Generate unique key so that A+B->C and B+A->C map to the same result"""
    parents = sorted([a, b])
    return f"{parents[0]} + {parents[1]} -> {c}"


def check_cluster_cache_status(cluster, linear_cache, df_matrix):
    """
    Check whether this cluster requires loading model weights.

    Condition:
    1. Linear merge: if any trio combination is missing in linear_cache,
       model weights must be loaded.
    """

    if len(cluster) < 3:
        return False

    all_trios_keys = []
    cluster_trios = list(itertools.combinations(cluster, 3))

    for trio in cluster_trios:
        for i in range(3):
            C = trio[i]
            A, B = [m for m in trio if m != C]
            all_trios_keys.append(get_trio_key(A, B, C))

    missing_keys = [k for k in all_trios_keys if k not in linear_cache]

    if missing_keys:
        print(f"  [Cache Miss] {len(missing_keys)} new trio directions detected, loading weights required.")
        return True
    else:
        print(f"  [Cache Hit] All trios already cached, stock decision already completed, skipping loading.")
        return False


def main():
    with open(CLUSTERS_JSON, 'r') as f:
        clusters = json.load(f)

    df_matrix = pd.read_csv(INPUT_CSV, index_col=0)
    linear_cache = load_cache()

    save_counter = 0

    for cluster_idx, cluster in enumerate(clusters):
        if len(cluster) < 2:
            continue

        print(f"\n[{cluster_idx+1}/{len(clusters)}] Scanning cluster: {cluster[0]} (size: {len(cluster)})")

        need_load = False
        loaded_names = cluster

        print("\n Begin ties analysis..... ")
        df_matrix = ties_recovery(loaded_names, df_matrix)

        print("\n Begin moe analysis..... ")
        df_matrix = moe_recovery(loaded_names, df_matrix)

        print("\n Begin linear analysis ..... ")
        if len(loaded_names) >= 3:
            need_load = check_cluster_cache_status(cluster, linear_cache, df_matrix)

            all_sds = {}
            if need_load:
                print(f"Loading {len(cluster)} models...")
                for m in cluster:
                    sd = load_any_model_sd(m)
                    if sd:
                        all_sds[m] = sd

            loaded_names = list(all_sds.keys()) if all_sds else cluster

            all_trios = list(itertools.combinations(loaded_names, 3))
            pbar = tqdm(all_trios, desc="Scanning Trios", unit="trio", leave=True)

            all_possible_candidates = []

            for trio in pbar:
                trio_candidates = []
                trio_updated = False

                for i in range(3):
                    C = trio[i]
                    A, B = [m for m in trio if m != C]

                    trio_key = get_trio_key(A, B, C)
                    if trio_key in linear_cache:
                        res, pct = linear_cache[trio_key]['res'], linear_cache[trio_key]['pct']
                    else:
                        is_m_raw, res, pct = check_linear_merge(all_sds[A], all_sds[B], all_sds[C])
                        linear_cache[trio_key] = {'res': float(res), 'pct': float(pct)}
                        trio_updated = True

                    if res < RESIDUAL_THRESHOLD and pct < 50:
                        score = res * (1 + pct / 100.0)
                        trio_candidates.append({
                            'target': C,
                            'parent1': A,
                            'parent2': B,
                            'res': res,
                            'pct': pct,
                            'score': score
                        })

                if trio_updated:
                    save_counter += 1
                    if save_counter >= 1:
                        save_cache(linear_cache)
                        save_counter = 0

                if trio_candidates:
                    trio_candidates.sort(key=lambda x: x['score'])
                    all_possible_candidates.append(trio_candidates[0])

            all_possible_candidates.sort(key=lambda x: x['score'])

            confirmed_targets = set()

            for candidate in all_possible_candidates:
                C = candidate['target']
                A, B = candidate['parent1'], candidate['parent2']

                row_c_exists = df_matrix.loc[C].drop(index=C, errors='ignore').notna().any()
                if not row_c_exists and C not in confirmed_targets:
                    pbar.write(
                        f"  ✨ [Global Best Match] {C} <- {A} + {B} "
                        f"(Score: {candidate['score']:.4f}, Res: {candidate['res']:.4f})"
                    )

                    df_matrix.at[C, A] = "merge"
                    df_matrix.at[C, B] = "merge"
                    confirmed_targets.add(C)
                else:
                    pbar.write(
                        f"  [Skip] {C} already has a better relation, skipping suboptimal pair {A}+{B}"
                    )
                    pass

        save_cache(linear_cache)
        del all_sds
        torch.cuda.empty_cache()

    df_matrix.to_csv(OUTPUT_CSV)
    print(f"\nTask completed! Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()