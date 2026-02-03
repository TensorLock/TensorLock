import os
import json
import glob
import torch
import numpy as np
import re
import pandas as pd
from safetensors.torch import load_file as safe_load_file
from sklearn.metrics import normalized_mutual_info_score

BASE_MODEL_DIR = "../../evaluation/Benchmark/models/"
CONVERTED_GGUF_DIR = "../step1_cluster/converted/"
CLUSTERS_JSON_PATH = "../step1_cluster/clusters.json"
QUANTIZED_CSV_PATH = "../step1_cluster/quantized_models.csv"
OUTPUT_CSV = "./quantize_result.csv"

NMI_SAMPLE_POINTS = 500000
RELAXED_TOLERANCE = 0.02
B = 256

class DataLoader:
    def __init__(self):
        self.weights_cache = {}

    def _get_full_path(self, model_name):
        dir_name = model_name.replace("/", "_")
        
        converted_path = os.path.join(CONVERTED_GGUF_DIR, dir_name)
        if os.path.exists(converted_path):
            return converted_path, "Converted"
        
        original_path = os.path.join(BASE_MODEL_DIR, dir_name)
        if os.path.exists(original_path):
            return original_path, "Original"
            
        return None, None

    def _normalize_layer_name(self, name):
        name = name.lower()
        prefixes_to_remove = ["base_model.model.model.", "base_model.model.", "model.language_model.", "model.encoder.", "model.decoder."]
        for prefix in prefixes_to_remove:
            if name.startswith(prefix):
                name = name[len(prefix):]

        match = re.search(r'(layers|h|blk)\.(\d+)\..*?\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|c_attn|c_proj|attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down|fc1|fc2|out_proj)', name)
        if match:
            layer_idx, layer_type = int(match.group(2)), match.group(3)
            type_map = {'attn_q': 'q_proj', 'attn_k': 'k_proj', 'attn_v': 'v_proj', 'attn_output': 'o_proj', 'ffn_gate': 'gate_proj', 'ffn_up': 'up_proj', 'ffn_down': 'down_proj', 'c_attn': 'q_proj', 'c_proj': 'o_proj', 'fc1': 'gate_proj', 'fc2': 'down_proj', 'out_proj': 'o_proj'}
            return (layer_idx, type_map.get(layer_type, layer_type))
        return None

    def get_aligned_weights(self, model1_name, model2_name):
        
        def find_layers(name):
            path, source = self._get_full_path(name)
            if not path: return {}, f"Path not found for {name}"
            
            cache_key = f"{name}_{source}"
            if cache_key in self.weights_cache:
                return self.weights_cache[cache_key], source

            layers = {}
            files = glob.glob(os.path.join(path, "**", "*.safetensors"), recursive=True) + glob.glob(os.path.join(path, "**", "*.bin"), recursive=True)
            if not files: return {}, f"No weight files in {path}"
            
            for f in files:
                try:
                    data = {}
                    if f.endswith(".safetensors"): data = safe_load_file(f, device="cpu")
                    else: data = torch.load(f, map_location="cpu")

                    for k, v in data.items():
                        if "weight" in k and isinstance(v, torch.Tensor) and v.dim() == 2 and v.is_floating_point():
                            norm_name = self._normalize_layer_name(k)
                            if norm_name:
                                layers[norm_name] = v
                except: continue
            
            self.weights_cache[cache_key] = layers
            return layers, source

        m1_layers, m1_source = find_layers(model1_name)
        m2_layers, m2_source = find_layers(model2_name)
        
        if not m1_layers: return {"status": "fail", "reason": f"No valid layers in {model1_name} ({m1_source})"}
        if not m2_layers: return {"status": "fail", "reason": f"No valid layers in {model2_name} ({m2_source})"}
        
        common_keys = set(m1_layers.keys()) & set(m2_layers.keys())
        if not common_keys:
            return {"status": "fail", "reason": f"No common layers. M1 keys: {list(m1_layers.keys())[:3]}, M2 keys: {list(m2_layers.keys())[:3]}"}
        
        sorted_keys = sorted(list(common_keys), key=lambda x: (x[1] != 'down_proj', x[0]))
        target_key = sorted_keys[len(sorted_keys) // 2]
        
        w1 = m1_layers[target_key]
        w2 = m2_layers[target_key]
        
        if w1.shape != w2.shape:
            if w1.T.shape == w2.shape: w1 = w1.T
            else: return {"status": "fail", "reason": f"Shape mismatch for key {target_key}: {w1.shape} vs {w2.shape}"}
        
        return {"status": "success", "w1": w1, "w2": w2}

def calculate_nmi(w_p, w_c):
    w_p, w_c = w_p.detach().cpu().float(), w_c.detach().cpu().float()
    w_p_flat, w_c_flat = w_p.flatten(), w_c.flatten()
    
    num_elements = w_p_flat.numel()
    if num_elements > NMI_SAMPLE_POINTS:
        step = num_elements // NMI_SAMPLE_POINTS
        indices = torch.arange(0, num_elements, step)[:NMI_SAMPLE_POINTS]
        w_p_flat, w_c_flat = w_p_flat[indices], w_c_flat[indices]
    
    w_p_np = w_p_flat.numpy()
    w_c_np = w_c_flat.numpy()
    
    p_binned = np.digitize(w_p_np, np.linspace(w_p_np.min(), w_p_np.max(), B))
    c_binned = np.digitize(w_c_np, np.linspace(w_c_np.min(), w_c_np.max(), B))
    return normalized_mutual_info_score(p_binned, c_binned)

def main():
    try:
        with open(CLUSTERS_JSON_PATH, 'r') as f:
            clusters = json.load(f)
        quantized_target_list = pd.read_csv(QUANTIZED_CSV_PATH, header=None)[0].tolist()
    except Exception as e:
        print(f"Failed to load input files: {e}")
        return

    all_models = sorted(list(set([m for cluster in clusters for m in cluster])))
    matrix_strict = pd.DataFrame("", index=all_models, columns=all_models)
    
    model_to_cluster = {m: i for i, cluster in enumerate(clusters) for m in cluster}
    loader = DataLoader()

    for i, child_model in enumerate(quantized_target_list):
        print(f"\n[{i+1}/{len(quantized_target_list)}] Target: {child_model}")
        
        cluster_idx = model_to_cluster.get(child_model)
        if cluster_idx is None:
            print(f"  ! Skip: {child_model} not found in clusters.json")
            continue
            
        candidates = [m for m in clusters[cluster_idx] if m not in quantized_target_list]
        if not candidates:
            print(f"  ! Skip: No non-quantized candidates in cluster {cluster_idx}")
            continue

        results = []
        for parent_model in candidates:
            try:
                res = loader.get_aligned_weights(parent_model, child_model)
                if res["status"] == "success":
                    score = calculate_nmi(res["w1"], res["w2"])
                    results.append((parent_model, score))
                    print(f"    - {parent_model}: NMI={score:.4f}")
                else:
                    print(f"    - {parent_model}: Failed ({res['reason']})")
            except Exception as e:
                print(f"    - {parent_model}: Unexpected Error ({str(e)})")
                continue

        if not results:
            continue

        results.sort(key=lambda x: x[1], reverse=True)
        max_nmi = results[0][1]
        best_parent = results[0][0]

        matrix_strict.at[child_model, best_parent] = "QTZ"

    matrix_strict.to_csv(OUTPUT_CSV)

    print(f"\n" + "="*40)
    print(f"SUCCESS: Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()