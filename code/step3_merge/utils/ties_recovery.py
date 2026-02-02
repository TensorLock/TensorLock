import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm


MODEL_BASE_DIR = "../../../dataset/models"


def load_any_model(model_path):
    full_path = os.path.join(MODEL_BASE_DIR, model_path)
    
    skip_keywords = ["GGUF", "gguf", "GPTQ", "gptq", "AWQ", "awq", "Int4", "Int8"]
    for keyword in skip_keywords:
        if keyword in model_path:
            return None
    
    try:
        return AutoModelForCausalLM.from_pretrained(
            full_path,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    except:
        return None


def is_real_weight(key: str):
    bad_keywords = [
        "zeros", "absmax",
        "scaling", "quant", "bnb",
        "embed_tokens.weight.quant_state",
        "lm_head.weight.quant_state",
        "lora_A", "lora_B", 
    ]
    return not any(bad in key for bad in bad_keywords)

def filter_real_weights(sd):
    result = {}
    for k, v in sd.items():
        if not is_real_weight(k):
            continue
        normalized_key = k.replace(".base_layer.", ".")
        result[normalized_key] = v
    return result


def extract_representative_vectors(state_dict, sample_layers=None, max_params_per_layer=10000):
    vectors = {}
    
    candidate_keys = []
    for key in state_dict.keys():
        if any(k in key for k in ["gate_proj", "up_proj", "down_proj", "o_proj"]):
            if "weight" in key:
                candidate_keys.append(key)
    
    if sample_layers and len(candidate_keys) > sample_layers:
        indices = np.linspace(0, len(candidate_keys)-1, sample_layers, dtype=int)
        candidate_keys = [candidate_keys[i] for i in indices]
    
    for key in candidate_keys:
        weight = state_dict[key].flatten().float()
        
        if weight.numel() > max_params_per_layer:
            rng = np.random.RandomState(42)
            indices = rng.choice(weight.numel(), max_params_per_layer, replace=False)
            indices = torch.from_numpy(np.sort(indices))
            weight = weight[indices]
        
        vectors[key] = weight
    
    return vectors

def compute_cosine_similarity(v1, v2):
    if v1.shape != v2.shape:
        return 0.0
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    cos_sim = torch.dot(v1, v2) / (norm1 * norm2)
    return cos_sim.item()


def compute_l2_distance(v1, v2):
    if v1.shape != v2.shape:
        return float('inf')
    return torch.norm(v1 - v2).item()


def normalize_key(key):
    if key.startswith("model."):
        return key[6:]
    return key

def get_common_keys(all_vectors, model_names, max_keys=15):
    if not model_names:
        return []
    
    common_keys = set(all_vectors[model_names[0]].keys())
    for name in model_names[1:]:
        common_keys &= set(all_vectors[name].keys())
    
    if common_keys:
        return sorted(list(common_keys))[:max_keys]
  
    normalized_keys_map = {}  
    for name in model_names:
        normalized_keys_map[name] = {normalize_key(k): k for k in all_vectors[name].keys()}
    
    common_normalized = set(normalized_keys_map[model_names[0]].keys())
    for name in model_names[1:]:
        common_normalized &= set(normalized_keys_map[name].keys())
    
    if not common_normalized:
        return []
    
    result = []
    for norm_key in sorted(list(common_normalized))[:max_keys]:
        result.append(normalized_keys_map[model_names[0]][norm_key])
    
    return result

def get_corresponding_key(all_vectors, model_name, reference_key):
    if reference_key in all_vectors[model_name]:
        return reference_key
    
    norm_ref = normalize_key(reference_key)
    for key in all_vectors[model_name].keys():
        if normalize_key(key) == norm_ref:
            return key
    
    return None


def find_merge_model_by_linear_dependency(all_vectors, model_names):

    common_keys = get_common_keys(all_vectors, model_names)
    if not common_keys:
        return None, {}
    
    reconstruction_errors = {}
    model_diversity_scores = {}  
    
    for target_model in tqdm(model_names, desc="Testing linear dependency"):
        other_models = [m for m in model_names if m != target_model]
        
        if len(other_models) < 2:
            continue
        
        total_relative_error = 0.0
        valid_keys = 0
        all_alphas = [] 
        
        for key in common_keys:
            target_key = get_corresponding_key(all_vectors, target_model, key)
            if not target_key:
                continue
            
            target_vec = all_vectors[target_model][target_key].numpy()
            
            other_vecs = []
            for m in other_models:
                m_key = get_corresponding_key(all_vectors, m, key)
                if m_key:
                    other_vecs.append(all_vectors[m][m_key].numpy())
            
            if len(other_vecs) < 2:
                continue
            
            A = np.column_stack(other_vecs)
            
            try:
                ATA = A.T @ A
                ATb = A.T @ target_vec
                
                reg = 1e-6 * np.trace(ATA) / ATA.shape[0]
                alpha = np.linalg.solve(ATA + reg * np.eye(ATA.shape[0]), ATb)
                
                reconstructed = A @ alpha
                
                error = np.linalg.norm(target_vec - reconstructed)
                norm = np.linalg.norm(target_vec)
                
                if norm > 1e-8:
                    relative_error = error / norm
                    total_relative_error += relative_error
                    valid_keys += 1
                    
                    all_alphas.append(np.abs(alpha))
            except Exception as e:
                continue
        
        if valid_keys > 0:
            avg_relative_error = total_relative_error / valid_keys
            reconstruction_errors[target_model] = avg_relative_error
            
            if all_alphas:
                avg_alpha = np.mean(all_alphas, axis=0)
                normalized_alpha = avg_alpha / (np.sum(avg_alpha) + 1e-8)
                
                entropy = -np.sum(normalized_alpha * np.log(normalized_alpha + 1e-8))
                
                significant_contributors = np.sum(normalized_alpha > 0.1)
                
                diversity_score = entropy * significant_contributors
                model_diversity_scores[target_model] = {
                    'entropy': entropy,
                    'n_contributors': int(significant_contributors),
                    'diversity_score': diversity_score,
                    'avg_weights': normalized_alpha.tolist()
                }
            else:
                model_diversity_scores[target_model] = {
                    'entropy': 0,
                    'n_contributors': 0,
                    'diversity_score': 0,
                    'avg_weights': []
                }
            
            n_contrib = model_diversity_scores[target_model]['n_contributors']
            div_score = model_diversity_scores[target_model]['diversity_score']
    
    sorted_by_contributors = sorted(
        model_diversity_scores.items(),
        key=lambda x: (x[1]['n_contributors'], -reconstruction_errors[x[0]]),
        reverse=True
    )
    
    merge_candidates = []
    for model, div_info in model_diversity_scores.items():
        error = reconstruction_errors[model]
        if div_info['n_contributors'] >= 4 and \
           0.0008 < error < 0.003 and \
           div_info['entropy'] > 1.5:
            merge_candidates.append((model, div_info['n_contributors'], error, div_info['entropy']))
    
    if not merge_candidates:
        return None, {}
    
    merge_candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    
    merge_model = merge_candidates[0][0]
    
    return merge_model, reconstruction_errors


def find_base_model_by_collinearity(all_vectors, model_names, merge_model):
    candidates = [m for m in model_names if m != merge_model]
    common_keys = get_common_keys(all_vectors, [merge_model] + candidates)
    
    if len(candidates) < 2:
        return None, {}
    
    collinearity_scores = {}
    
    for base_candidate in tqdm(candidates, desc="Testing collinearity"):
        other_candidates = [m for m in candidates if m != base_candidate]
        
        if len(other_candidates) < 1:
            continue
        
        cos_sims = []
        
        for key in common_keys:
            merge_key = get_corresponding_key(all_vectors, merge_model, key)
            base_key = get_corresponding_key(all_vectors, base_candidate, key)
            
            if not merge_key or not base_key:
                continue
            
            other_vecs = []
            for m in other_candidates:
                m_key = get_corresponding_key(all_vectors, m, key)
                if m_key:
                    other_vecs.append(all_vectors[m][m_key])
            
            if not other_vecs:
                continue
            
            avg_vec = torch.stack(other_vecs).mean(dim=0)
            
            v_dir = avg_vec - all_vectors[base_candidate][base_key]
            
            v_merge = all_vectors[merge_model][merge_key] - all_vectors[base_candidate][base_key]
            
            cos_sim = compute_cosine_similarity(v_dir, v_merge)
            cos_sims.append(cos_sim)
        
        if cos_sims:
            avg_cos_sim = np.mean(cos_sims)
            collinearity_scores[base_candidate] = avg_cos_sim
    
    sorted_scores = sorted(collinearity_scores.items(), key=lambda x: x[1], reverse=True)
    
    if collinearity_scores:
        base_model = sorted_scores[0][0]
        base_score = sorted_scores[0][1]
        
        if base_score < 0.99:
            return None, {}
        
        return base_model, collinearity_scores
    
    return None, {}


def identify_parent_models(all_vectors, model_names, merge_model, base_model, threshold=0.3):

    candidates = [m for m in model_names if m != merge_model and m != base_model]
    
    base_variants_to_exclude = set()
    
    finetune_suffixes = ['-Instruct', '-Chat', '-IT', '_Instruct', '_Chat', '_IT']
    for suffix in finetune_suffixes:
        if suffix in base_model:
            base_name = base_model.replace(suffix, '')
            for candidate in candidates:
                if candidate == base_name or candidate.startswith(base_name + '-') or candidate.startswith(base_name + '_'):
                    has_finetune_marker = any(marker in candidate.lower() for marker in ['lora', 'finetune', 'sft', 'dpo', 'chat', 'instruct'])
                    if not has_finetune_marker:
                        base_variants_to_exclude.add(candidate)
    
    candidates = [c for c in candidates if c not in base_variants_to_exclude]
    
    if not candidates:
        return [], {}
    
    common_keys = get_common_keys(all_vectors, [merge_model, base_model] + candidates)
    
    parent_scores = {}
    
    for candidate in candidates:
        contributions = []
        distances = []
        
        for key in common_keys:
            merge_key = get_corresponding_key(all_vectors, merge_model, key)
            base_key = get_corresponding_key(all_vectors, base_model, key)
            cand_key = get_corresponding_key(all_vectors, candidate, key)
            
            if not merge_key or not base_key or not cand_key:
                continue
            
            v_merge = all_vectors[merge_model][merge_key] - all_vectors[base_model][base_key]
            
            v_parent = all_vectors[candidate][cand_key] - all_vectors[base_model][base_key]
            
            cos_sim = compute_cosine_similarity(v_parent, v_merge)
            contributions.append(cos_sim)
            
            dist = compute_l2_distance(all_vectors[candidate][cand_key], all_vectors[base_model][base_key])
            distances.append(dist)
        
        if contributions:
            avg_contribution = np.mean(contributions)
            avg_distance = np.mean(distances)
            
            parent_scores[candidate] = {
                "contribution": avg_contribution,
                "distance_to_base": avg_distance
            }
    
    parents = [name for name, scores in parent_scores.items() if scores["contribution"] > threshold]
    
    return parents, parent_scores


def verify_model_stock_relationship(all_vectors, merge_model, base_model, parent_models):
    if not parent_models:
        return 0.0, 0.0
    
    common_keys = get_common_keys(all_vectors, [merge_model, base_model] + parent_models)
    
    cos_sims = []
    t_values = []
    reconstruction_errors = []
    
    for key in common_keys:
        merge_key = get_corresponding_key(all_vectors, merge_model, key)
        base_key = get_corresponding_key(all_vectors, base_model, key)
        
        if not merge_key or not base_key:
            continue
        
        base_vec = all_vectors[base_model][base_key]
        merge_vec = all_vectors[merge_model][merge_key]
        
        parent_vecs = []
        for m in parent_models:
            m_key = get_corresponding_key(all_vectors, m, key)
            if m_key:
                parent_vecs.append(all_vectors[m][m_key])
        
        if not parent_vecs:
            continue
        
        avg_parents = torch.stack(parent_vecs).mean(dim=0)
        
        v_dir = avg_parents - base_vec
        
        v_merge = merge_vec - base_vec
        
        cos_sim = compute_cosine_similarity(v_dir, v_merge)
        cos_sims.append(cos_sim)
        
        v_dir_norm = torch.norm(v_dir).item()
        v_merge_norm = torch.norm(v_merge).item()
        if v_dir_norm > 1e-8 and cos_sim > 0.5:
            t = v_merge_norm / v_dir_norm
            t_values.append(t)
        
        if t_values:
            avg_t = np.mean(t_values)
            reconstructed = base_vec + avg_t * v_dir
            error = torch.norm(merge_vec - reconstructed).item() / (torch.norm(merge_vec).item() + 1e-8)
            reconstruction_errors.append(error)
    
    avg_cos_sim = np.mean(cos_sims) if cos_sims else 0.0
    avg_t = np.mean(t_values) if t_values else 0.0
    
    return avg_cos_sim, avg_t


def find_base_by_distance(all_vectors, model_names, merge_model):
    
    candidates = [m for m in model_names if m != merge_model]
    common_keys = get_common_keys(all_vectors, [merge_model] + candidates)
    merge_distances = {}
    for candidate in candidates:
        total_dist = 0.0
        count = 0
        for key in common_keys:
            merge_key = get_corresponding_key(all_vectors, merge_model, key)
            cand_key = get_corresponding_key(all_vectors, candidate, key)
            if merge_key and cand_key:
                dist = compute_l2_distance(all_vectors[merge_model][merge_key], all_vectors[candidate][cand_key])
                total_dist += dist
                count += 1
        if count > 0:
            merge_distances[candidate] = total_dist / count
    
    avg_inter_distances = {}
    for candidate in candidates:
        others = [m for m in candidates if m != candidate]
        total_dist = 0.0
        count = 0
        for other in others:
            for key in common_keys:
                cand_key = get_corresponding_key(all_vectors, candidate, key)
                other_key = get_corresponding_key(all_vectors, other, key)
                if cand_key and other_key:
                    dist = compute_l2_distance(all_vectors[candidate][cand_key], all_vectors[other][other_key])
                    total_dist += dist
                    count += 1
        avg_inter_distances[candidate] = total_dist / count if count > 0 else 0
    
    scores = {}
    for candidate in candidates:
        if candidate not in merge_distances:
            continue
        merge_dist = merge_distances[candidate]
        inter_dist = avg_inter_distances.get(candidate, 0)
        
        if merge_dist > 1e-8:
            scores[candidate] = inter_dist / merge_dist
        else:
            scores[candidate] = 0
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_scores:
        return sorted_scores[0][0]
    return None

def comprehensive_analysis(all_vectors, model_names):
    merge_model, reconstruction_errors = find_merge_model_by_linear_dependency(all_vectors, model_names)
    
    if not merge_model:
        return None
    
    base_model, collinearity_scores = find_base_model_by_collinearity(all_vectors, model_names, merge_model)
    
    if not base_model:
        base_model = find_base_by_distance(all_vectors, model_names, merge_model)
    
    if not base_model:
        return None
    
    parent_models, parent_scores = identify_parent_models(
        all_vectors, model_names, merge_model, base_model, threshold=0.3
    )
    
    verification_score, avg_t = verify_model_stock_relationship(
        all_vectors, merge_model, base_model, parent_models
    )
    if verification_score < 0.98:
        return None
    
    return {
        "merge_model": merge_model,
        "base_model": base_model,
        "parent_models": parent_models,
        "verification_score": verification_score,
        "avg_interpolation_t": avg_t,
        "reconstruction_errors": reconstruction_errors,
        "collinearity_scores": collinearity_scores,
        "parent_scores": parent_scores
    }


def ties_recovery(model_list, matrix_df):

    models = model_list

    lora_models = [m for m in models if 'lora' in m.lower() or 'LoRA' in m]
    print(f"Detected {len(lora_models)} LoRA models")
    
    if len(lora_models) < 4:
        print(f"❌ Not enough LoRA models ({len(lora_models)} < 4), exiting script")
        return matrix_df
        
    print("\nLoading models...")
    
    all_vectors = {}
    skipped_models = []
    
    for model_name in models:
        print(model_name)
        model_obj = load_any_model(model_name.replace("/", "_"))
        if model_obj is not None:
            sd = filter_real_weights(model_obj.state_dict())
            vectors = extract_representative_vectors(sd, sample_layers=20, max_params_per_layer=5000)
            
            if vectors:
                all_vectors[model_name] = vectors
            else:
                skipped_models.append(model_name)
            
            del model_obj
            del sd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            skipped_models.append(model_name)
    
    valid_models = list(all_vectors.keys())

    if len(valid_models) < 3:
        print("❌ Error: At least 3 models are required for analysis")
        return matrix_df
    
    print("Analyzing model relationships...")
    result = comprehensive_analysis(all_vectors, valid_models)
    
    if not result:
        print("❌ Analysis failed: Could not identify a Model Stock relationship")
        return matrix_df
    
    merge_model = result["merge_model"]
    base_model = result["base_model"]
    parent_models = result["parent_models"]
    
    all_parent_models = [base_model] + parent_models
    
    for p in all_parent_models:
        matrix_df.at[merge_model, p] = "merge"
        print(f"Identified {p} -> {merge_model}")

    return matrix_df