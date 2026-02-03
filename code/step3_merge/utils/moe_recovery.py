import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm

MODEL_BASE_DIR = "../../../evaluation/Benchmark/models"


def load_any_model(model_path):
    full_path = os.path.join(MODEL_BASE_DIR, model_path)

    skip_keywords = ["GGUF", "gguf", "GPTQ", "gptq", "AWQ", "awq", "Int4", "Int8"]
    for keyword in skip_keywords:
        if keyword in full_path:
            return None

    print(f"== Loading model: {full_path} ==")
    
    try:
        return AutoModelForCausalLM.from_pretrained(
            full_path,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def is_real_weight(key: str):
    bad_keywords = [
        "base_layer", "zeros", "absmax",
        "scaling", "quant", "bnb",
        "embed_tokens.weight.quant_state",
        "lm_head.weight.quant_state",
    ]
    return not any(bad in key for bad in bad_keywords)

def filter_real_weights(sd):
    return {k: v for k, v in sd.items() if is_real_weight(k)}


def categorize_parameters(state_dict):
    """
    Categorize parameters into three groups:
    1. shared: Attention, LayerNorm, Embedding, LM Head
    2. experts: FFN/MLP expert parameters
    3. router: routing/gating parameters
    """
    shared_keys = []
    expert_keys = []
    router_keys = []
    
    for key in state_dict.keys():
        if any(k in key.lower() for k in ["gate", "router"]):
            router_keys.append(key)
        elif any(k in key for k in ["experts", "block_sparse_moe"]):
            expert_keys.append(key)
        elif any(k in key for k in [
            "self_attn", "input_layernorm", "post_attention_layernorm",
            "embed_tokens", "lm_head", "norm"
        ]):
            shared_keys.append(key)
        else:
            shared_keys.append(key)
    
    return shared_keys, expert_keys, router_keys


def compute_cosine_similarity(p1, p2):
    """Compute cosine similarity between two tensors"""
    if p1.shape != p2.shape:
        return 0.0
    p1_flat = p1.flatten().float()
    p2_flat = p2.flatten().float()
    cos_sim = torch.nn.functional.cosine_similarity(p1_flat.unsqueeze(0), p2_flat.unsqueeze(0))
    return cos_sim.item()

def parameters_equal(p1, p2, rtol=1e-5, atol=1e-8):
    """Check whether two tensors are exactly equal (for base model matching)"""
    if p1.shape != p2.shape:
        return False
    return torch.allclose(p1.float(), p2.float(), rtol=rtol, atol=atol)

def parameters_similar(p1, p2, threshold=0.95):
    """
    Check whether two tensors are similar (for expert matching)
    Uses cosine similarity, default threshold = 0.95
    """
    if p1.shape != p2.shape:
        return False
    similarity = compute_cosine_similarity(p1, p2)
    return similarity >= threshold

def identify_base_model(moe_sd, candidate_models_sd, shared_keys):
    """
    Identify base model by comparing shared parameters
    Returns: (base_model_name, match_info)
    """
    print("\n=== Identifying Base Model ===")
    results = {}
    
    for model_name, candidate_sd in candidate_models_sd.items():
        matched = 0
        total = 0
        
        for key in tqdm(shared_keys, desc=f"Comparing with {model_name}"):
            if key in candidate_sd:
                total += 1
                if parameters_equal(moe_sd[key], candidate_sd[key]):
                    matched += 1
            else:
                alt_key = key.replace("model.", "") if "model." in key else f"model.{key}"
                if alt_key in candidate_sd:
                    total += 1
                    if parameters_equal(moe_sd[key], candidate_sd[alt_key]):
                        matched += 1
        
        if total > 0:
            match_ratio = matched / total
            results[model_name] = {
                "matched": matched,
                "total": total,
                "match_ratio": match_ratio
            }
            print(f"{model_name}: {matched}/{total} = {match_ratio:.2%}")
    
    if results:
        base_model = max(results.items(), key=lambda x: x[1]["match_ratio"])
        return base_model[0], base_model[1]
    
    return None, None


def extract_expert_ffn(moe_sd, expert_keys, layer_idx, expert_idx):
    """
    Extract FFN parameters of a specific expert in a specific layer
    Returns a dict of FFN weights
    """
    expert_params = {}
    
    for key in expert_keys:
        if f"layers.{layer_idx}." in key and f"experts.{expert_idx}." in key:
            expert_params[key] = moe_sd[key]
    
    return expert_params


def identify_moe_model(models_info):
    """
    Identify MoE model by structure (presence of experts parameters)
    Returns: (moe_model_name, moe_state_dict, candidate_models_info)
    """
    print("\n=== Identifying MoE Model by Structure ===")
    
    for model_name, model_sd in models_info.items():
        has_experts = any("experts" in key or "block_sparse_moe" in key for key in model_sd.keys())
        has_router = any("gate" in key.lower() or "router" in key.lower() for key in model_sd.keys())
        
        if has_experts:
            print(f"✅ MoE model detected: {model_name}")
            print(f"   - Contains expert params: {has_experts}")
            print(f"   - Contains router params: {has_router}")
            
            candidate_models_sd = {k: v for k, v in models_info.items() if k != model_name}
            return model_name, model_sd, candidate_models_sd
    
    return None, None, None


def identify_expert_models(moe_sd, candidate_models_sd, expert_keys, num_layers=22, num_experts=3, base_model_name=None):
    """
    Identify expert models by comparing FFN parameters
    Strategy: each expert is assigned to only one model, allow partial missing experts
    """
    print("\n=== Identifying Expert Models ===")
    
    model_expert_layer_scores = {}  # {model_name: {expert_idx: [(layer_idx, similarity)]}}
    
    for layer_idx in tqdm(range(num_layers), desc="Analyzing layers"):
        layer_experts = {}
        for expert_idx in range(num_experts):
            expert_params = extract_expert_ffn(moe_sd, expert_keys, layer_idx, expert_idx)
            if expert_params:
                layer_experts[expert_idx] = expert_params
        
        if not layer_experts:
            continue
        
        for model_name, candidate_sd in candidate_models_sd.items():
            if model_name not in model_expert_layer_scores:
                model_expert_layer_scores[model_name] = {}
            
            candidate_ffn = {}
            for key in candidate_sd.keys():
                if f"layers.{layer_idx}." in key or f".{layer_idx}." in key:
                    if any(k in key for k in ["mlp", "gate_proj", "up_proj", "down_proj"]):
                        candidate_ffn[key] = candidate_sd[key]
            
            if not candidate_ffn:
                continue
            
            for expert_idx, expert_params in layer_experts.items():
                similarities = []
                
                for expert_key, expert_param in expert_params.items():
                    parts = expert_key.split(".")
                    weight_type = ".".join(parts[-2:])
                    
                    for cand_key, cand_param in candidate_ffn.items():
                        if weight_type in cand_key or self_match_ffn_keys(expert_key, cand_key):
                            similarity = compute_cosine_similarity(expert_param, cand_param)
                            similarities.append(similarity)
                            break
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    if expert_idx not in model_expert_layer_scores[model_name]:
                        model_expert_layer_scores[model_name][expert_idx] = []
                    model_expert_layer_scores[model_name][expert_idx].append((layer_idx, avg_similarity))
    
    model_best_expert = {}  # {model_name: (expert_idx, avg_similarity, matched_layers)}
    
    print("\n=== Debug Info: Expert Matching Per Model ===")
    for model_name, expert_scores in model_expert_layer_scores.items():
        print(f"\n{model_name}:")
        for expert_idx, layer_similarities in expert_scores.items():
            high_sim_layers = [(l, s) for l, s in layer_similarities if s >= 0.95]
            
            if high_sim_layers:
                avg_sim = np.mean([s for l, s in high_sim_layers])
                matched_layers = len(high_sim_layers)
                print(f"  Expert {expert_idx}: matched_layers={matched_layers}/{num_layers}, avg_similarity={avg_sim:.4f}")
    
    for model_name, expert_scores in model_expert_layer_scores.items():
        best_expert = None
        best_avg_sim = 0
        best_matched_layers = 0
        
        for expert_idx, layer_similarities in expert_scores.items():
            high_sim_layers = [(l, s) for l, s in layer_similarities if s >= 0.95]
            
            if high_sim_layers:
                avg_sim = np.mean([s for l, s in high_sim_layers])
                matched_layers = len(high_sim_layers)
                
                if matched_layers > best_matched_layers or (matched_layers == best_matched_layers and avg_sim > best_avg_sim):
                    best_expert = expert_idx
                    best_avg_sim = avg_sim
                    best_matched_layers = matched_layers
        
        if best_expert is not None and best_matched_layers >= num_layers * 0.5: 
            model_best_expert[model_name] = (best_expert, best_avg_sim, best_matched_layers)

    expert_to_model = {}  # {expert_idx: model_name}
    final_assignments = {}
    
    if base_model_name and base_model_name in model_best_expert:
        expert_idx, avg_sim, matched_layers = model_best_expert[base_model_name]
        expert_to_model[expert_idx] = base_model_name
        final_assignments[base_model_name] = {
            "expert_idx": expert_idx,
            "avg_match_ratio": avg_sim,
            "matched_layers": matched_layers,
            "total_layers": num_layers
        }
    
    sorted_models = sorted(
        [(name, info) for name, info in model_best_expert.items() if name != base_model_name],
        key=lambda x: (x[1][2], x[1][1]),
        reverse=True
    )
    
    for model_name, (expert_idx, avg_sim, matched_layers) in sorted_models:
        if expert_idx not in expert_to_model:
            expert_to_model[expert_idx] = model_name
            final_assignments[model_name] = {
                "expert_idx": expert_idx,
                "avg_match_ratio": avg_sim,
                "matched_layers": matched_layers,
                "total_layers": num_layers
            }
    
    print("\n=== Expert Matching Results ===")
    for model_name, info in final_assignments.items():
        print(f"\n{model_name}:")
        print(f"  Assigned expert: Expert {info['expert_idx']}")
        print(f"  Average similarity: {info['avg_match_ratio']:.2%}")
        print(f"  Matched layers: {info['matched_layers']}/{info['total_layers']}")
    
    unmatched_experts = [i for i in range(num_experts) if i not in expert_to_model]
    if unmatched_experts:
        print(f"\n⚠️  Unmatched expert indices: {unmatched_experts} (possibly missing models)")
    
    return final_assignments

def self_match_ffn_keys(moe_key, candidate_key):
    """
    Custom matching rules between MoE expert keys and candidate FFN keys
    Examples:
    - experts.0.w1.weight <-> gate_proj.weight
    - experts.0.w2.weight <-> down_proj.weight
    - experts.0.w3.weight <-> up_proj.weight
    """
    moe_mapping = {
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj"
    }
    
    for moe_name, cand_name in moe_mapping.items():
        if moe_name in moe_key and cand_name in candidate_key:
            return True
    
    return False


def moe_recovery(model_list, matrix_df):
    models = model_list

    print(f"Loaded {len(models)} models:")
    for m in models:
        print(f"  - {m}")
    
    print("\n" + "="*60)
    print("Step 1: Load all models")
    print("="*60)
    all_models_sd = {}
    
    for model_name in models:
        model_obj = load_any_model(model_name.replace("/","_"))
        if model_obj is not None:
            all_models_sd[model_name] = filter_real_weights(model_obj.state_dict())
            del model_obj
        else:
            print(f"Warning: failed to load {model_name}, skipped.")
    
    print("\n" + "="*60)
    print("Step 2: Identify MoE model (by structure)")
    print("="*60)
    moe_model, moe_sd, candidate_models_sd = identify_moe_model(all_models_sd)
    
    if not moe_model:
        print("Error: No MoE model found!")
        del all_models_sd
        torch.cuda.empty_cache()
        return matrix_df
    
    print(f"Candidate parent models: {list(candidate_models_sd.keys())}")
    
    shared_keys, expert_keys, router_keys = categorize_parameters(moe_sd)
    
    print(f"\nParameter categories:")
    print(f"  Shared params: {len(shared_keys)}")
    print(f"  Expert params: {len(expert_keys)}")
    print(f"  Router params: {len(router_keys)}")
    
    if expert_keys:
        print(f"\nExpert param examples:")
        for key in expert_keys[:5]:
            print(f"    {key}")
    
    if not expert_keys:
        print("Warning: No expert parameters found. Model may not be valid MoE.")
        del all_models_sd
        torch.cuda.empty_cache()
        return matrix_df
    
    num_layers_detected = 0
    num_experts_detected = 0
    for key in expert_keys:
        if "layers." in key and "experts." in key:
            layer_idx = int(key.split("layers.")[1].split(".")[0])
            num_layers_detected = max(num_layers_detected, layer_idx + 1)
            expert_idx = int(key.split("experts.")[1].split(".")[0])
            num_experts_detected = max(num_experts_detected, expert_idx + 1)
    
    print(f"\nDetected layers: {num_layers_detected}")
    print(f"Detected experts: {num_experts_detected}")
    if num_experts_detected < 1:
        print("Warning: No expert structure detected. Not a valid MoE merge.")
        del all_models_sd
        torch.cuda.empty_cache()
        return matrix_df

    print("\n" + "="*60)
    print("Step 3: Identify base model")
    print("="*60)
    base_model_name, base_match_info = identify_base_model(
        moe_sd, candidate_models_sd, shared_keys
    )
    
    if base_model_name:
        print(f"\n✅ Base model identified: {base_model_name}")
        print(f"   Match ratio: {base_match_info['match_ratio']:.2%}")
        print(f"   Matched params: {base_match_info['matched']}/{base_match_info['total']}")
    else:
        print("\n❌ Base model not identified")
    
    print("\n" + "="*60)
    print("Step 4: Identify expert models")
    print("="*60)
    
    all_candidates_for_experts = dict(candidate_models_sd)
    if base_model_name and base_model_name in all_models_sd:
        all_candidates_for_experts[base_model_name] = all_models_sd[base_model_name]
    
    expert_summary = identify_expert_models(
        moe_sd, all_candidates_for_experts, expert_keys,
        num_layers=num_layers_detected,
        num_experts=num_experts_detected,
        base_model_name=base_model_name
    )
    
    del all_models_sd
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("Final Identification Results")
    print("="*60)
    
    parent_models_set = set()
    if base_model_name:
        parent_models_set.add(base_model_name)
    parent_models_set.update(expert_summary.keys())
    parent_models = sorted(list(parent_models_set))
    

    for p in parent_models:
        matrix_df.at[moe_model, p] = "merge"
        print(f"Identified {p} -> {moe_model}")
    
    return matrix_df