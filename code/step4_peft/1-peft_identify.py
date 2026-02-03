import os
import torch
import json
import pandas as pd
from utils.load_model import load_any_model

MODEL_BASE_DIR = "../../evaluation/Benchmark/models"
CLUSTERS_JSON = "../step1_cluster/clusters.json"
MERGE_CSV = "../step3_merge/merge_result.csv"
OUTPUT_CSV = "./peft_result.csv"

def get_safe_path(model_name):
    """Replace / with _ in model name and join with base path"""
    safe_name = model_name.replace("/", "_")
    return os.path.join(MODEL_BASE_DIR, safe_name)

def check_if_peft(model_name):
    """Check if the model is an explicit PEFT without loading large weights"""
    model_path = get_safe_path(model_name)
    adapter_file = os.path.join(model_path, "adapter_config.json")
    return os.path.exists(adapter_file)

def load_and_extract(model_name):
    """Load model and extract base state dict, return (state_dict, is_explicit_peft)"""
    model_path = get_safe_path(model_name)
    
    if not os.path.exists(model_path):
        print(f"  [Warning] Path does not exist, skipping: {model_path}")
        return None, False

    gguf_file = next((f for f in os.listdir(model_path) if f.endswith(".gguf")), None)
    if gguf_file:
        print(f"{model_name} is a GGUF model! Skipping")
        return None, False

    is_peft = check_if_peft(model_name)
    try:
        model = load_any_model(model_name)
        
        sd = model.state_dict()
        clean_sd = {}
        has_adapter_layers = False
        
        for k, v in sd.items():
            if any(x in k.lower() for x in ["lora", "adapter", "peft", "lora_a", "lora_b"]):
                has_adapter_layers = True
                continue
            ck = k.replace("base_model.model.", "").replace(".base_layer.", ".")
            clean_sd[ck] = v.detach().cpu()
            
        del model
        return clean_sd, (is_peft or has_adapter_layers)
        
    except Exception as e:
        print(f"  [Error] Failed to process {model_name}: {e}")
        return None, False

def detect_peft_relation(sd_A, sd_B):
    """Detect if there is a PEFT derivation relationship A -> B"""
    if sd_A is None or sd_B is None: 
        return False
    if set(sd_A.keys()) != set(sd_B.keys()): 
        return False
    
    total_params = 0
    diff_params = 0
    total_l1_dist = 0.0

    for k in sd_A.keys():
        wa, wb = sd_A[k], sd_B[k]
        if wa.shape != wb.shape: 
            return False
        
        total_params += wa.numel()
        if not torch.equal(wa, wb):
            diff = (wa - wb).abs()
            diff_params += (diff > 0).sum().item()
            total_l1_dist += diff.sum().item()

            if any(x in k for x in ["embed_tokens", "lm_head", "wte"]):
                if not torch.allclose(wa, wb, atol=1e-7):
                    return False

    if total_params == 0: 
        return False
    return (diff_params / total_params < 0.05) and (total_l1_dist < 0.1)

def main():
    with open(CLUSTERS_JSON, 'r') as f:
        clusters = json.load(f)
    df_matrix = pd.read_csv(MERGE_CSV, index_col=0)
    
    print(f"=== Starting PEFT relationship detection ===")

    for cluster_idx, cluster in enumerate(clusters):
        print(f"\n[{cluster_idx+1}/{len(clusters)}] Scanning cluster (first model: {cluster[0]})")
        
        peft_candidates = []
        
        for model_name in cluster:
            if model_name in df_matrix.index and df_matrix.loc[model_name].notna().any():
                print(model_name)
                continue
            
            _, is_explicit = load_and_extract(model_name)
            if is_explicit:
                peft_candidates.append(model_name)
                print(f" ✅ [Found PEFT Candidate] {model_name}")

        for model_B in peft_candidates:
            print(f"  Searching for base model for {model_B}...")
            sd_B, _ = load_and_extract(model_B)
            if sd_B is None: 
                continue

            found_base = False
            for model_A in cluster:
                if model_A == model_B: 
                    continue
                if model_A in peft_candidates: 
                    continue

                sd_A, _ = load_and_extract(model_A)
                if sd_A is None: 
                    continue

                if detect_peft_relation(sd_A, sd_B):
                    print(f"    [Match!] {model_A} (Base) -> {model_B} (Derived)")
                    df_matrix.at[model_B, model_A] = "peft"
                    found_base = True
                
                del sd_A
                if found_base: 
                    break 
            
            del sd_B
            torch.cuda.empty_cache()

    df_matrix.to_csv(OUTPUT_CSV)
    print(f"\nTask completed! Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()