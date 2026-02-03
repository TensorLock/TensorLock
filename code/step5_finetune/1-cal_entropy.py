import os
import csv
import json
import torch
import numpy as np
from tqdm import tqdm
from utils.load_model import load_any_model

MODEL_BASE_DIR = "../../evaluation/Benchmark/models"
INPUT_JSON = "../../evaluation/Benchmark/model_list.json"
CACHE_CSV = "./entropy_cache.csv"

def load_existing_cache(cache_csv):
    cache_models = set()
    if os.path.exists(cache_csv):
        with open(cache_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cache_models.add(row["model"])
    return cache_models

def append_to_cache(cache_csv, model_name, rank_eff):
    file_exists = os.path.exists(cache_csv)
    with open(cache_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "RankEff"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"model": model_name, "RankEff": rank_eff})

def compute_effective_rank(model, device="cuda"):
    entropy_list = []
    
    target_params = []
    for name, p in model.named_parameters():
        lname = name.lower()
        if p.ndim == 2 and not any(k in lname for k in ["embed", "lm_head", "norm"]):
            target_params.append((name, p))

    pbar = tqdm(target_params, desc="Calculating Effective Rank", leave=False)
    
    for name, p in pbar:
        pbar.set_postfix({"layer": name.split('.')[-2]}) 

        W = p.detach().to(device).to(torch.float32)

        try:
            s = torch.linalg.svdvals(W)
            if s.numel() == 0 or s.sum() == 0:
                continue
            
            s_norm = s / s.sum()
            s_norm = s_norm + 1e-12
            H = -(s_norm * torch.log(s_norm)).sum()
            
            rank_eff = torch.exp(H).item()
            entropy_list.append(rank_eff)
            
        except Exception as e:
            print(f"  ⚠️ SVD failed on {name}: {e}")
            continue
        finally:
            del W
            if "cuda" in device:
                torch.cuda.empty_cache()

    return float(np.mean(entropy_list)) if entropy_list else float("nan")

def main():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
        model_list = data.get("models", [])

    processed_models = load_existing_cache(CACHE_CSV)
    print(f"Total models to process: {len(model_list)} (Already cached: {len(processed_models)})")

    for model_name in tqdm(model_list, desc="Overall Progress"):
        if model_name in processed_models:
            continue

        print(f"\n🚀 Processing: {model_name}")
        try:
            model = load_any_model(model_name)
            model = model.to("cuda").eval()

            rank_eff = compute_effective_rank(model)
            
            if not np.isnan(rank_eff):
                append_to_cache(CACHE_CSV, model_name, rank_eff)
                print(f"  ✅ RankEff: {rank_eff:.4f}")
            else:
                print(f"  ❌ No valid layers found for {model_name}")

            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  🔥 Error processing {model_name}: {e}")
            continue

    print(f"\n🏁 Finished! Results saved in {CACHE_CSV}")

if __name__ == "__main__":
    main()