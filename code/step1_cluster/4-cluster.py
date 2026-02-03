import os
import logging
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
import json
from collections import defaultdict
import hdbscan
import networkx as nx


FINGERPRINT_CACHE_PATH = "./fingerprint/"
INPUT_JSON_FILE = "../../evaluation/Benchmark/model_list.json"
CLUSTERS_OUTPUT_JSON = "./clusters.json"

SIMILARITY_THRESHOLD = 0.6
MIN_CLUSTER_SIZE = 2

LOGGING_CONFIG = { "level": "INFO", "format": "%(asctime)s - [%(levelname)s] - %(message)s" }

def setup_logging():
    """Configure the logging system."""
    logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"], datefmt="%Y-%m-%d %H:%M:%S")

def compare_fingerprints(fp1, fp2):
    """Calculate similarity using Spearman rank correlation + equal weights."""
    if fp1 is None or fp2 is None: return 0.0
    def _corr(v1, v2):
        v1, v2 = v1.numpy(), v2.numpy()
        if len(v1) != len(v2):
            if len(v1) < len(v2): v1 = np.interp(np.linspace(0, len(v1)-1, len(v2)), np.arange(len(v1)), v1)
            else: v2 = np.interp(np.linspace(0, len(v2)-1, len(v1)), np.arange(len(v2)), v2)
        if len(v1) < 2: return 0.0
        corr, _ = spearmanr(v1, v2)
        return corr if not np.isnan(corr) else 0.0
    total_similarity = 0.0
    for comp in ['q', 'k', 'v', 'o']:
        if comp in fp1 and comp in fp2:
            component_similarity = 0.0
            num_features = len(fp1[comp])
            for feat in fp1[comp]:
                if feat in fp2[comp]:
                    correlation = _corr(fp1[comp][feat], fp2[comp][feat])
                    component_similarity += (1.0 / num_features) * correlation
            total_similarity += 0.25 * component_similarity
    return total_similarity


def main():
    """Main function: Build graph -> Compute shortest path distances -> Cluster using HDBSCAN."""
    setup_logging()

    if not os.path.exists(INPUT_JSON_FILE):
        logging.error(f"Input model file not found: {INPUT_JSON_FILE}")
        return
    
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        target_model_list = data.get("models", [])
        if not target_model_list:
            logging.error("The specified model list is empty.")
            return
        logging.info(f"Read {len(target_model_list)} target models from {INPUT_JSON_FILE}.")
    except Exception as e:
        logging.error(f"Failed to parse model list JSON: {e}")
        return

    available_fingerprints = {}
    if not os.path.isdir(FINGERPRINT_CACHE_PATH):
        logging.error(f"Fingerprint directory does not exist: {FINGERPRINT_CACHE_PATH}")
        return

    logging.info("Loading fingerprint files for specified models from directory...")
    for model_name in target_model_list:
        filename = f"{model_name.replace('/', '__')}.pt"
        file_path = os.path.join(FINGERPRINT_CACHE_PATH, filename)
        
        if os.path.exists(file_path):
            try:
                available_fingerprints[model_name] = torch.load(file_path, map_location='cpu')
            except Exception as e:
                logging.warning(f"Failed to load fingerprint file {filename}: {e}")
        else:
            logging.warning(f"Skipping: Fingerprint file for model '{model_name}' not found in directory.")

    if len(available_fingerprints) < 1:
        logging.warning("No valid fingerprints loaded. Clustering terminated.")
        return
    logging.info(f"Number of models actually available for clustering: {len(available_fingerprints)}")

    logging.info(f"Building model relationship graph using threshold {SIMILARITY_THRESHOLD}...")
    model_names = sorted(list(available_fingerprints.keys()))
    G = nx.Graph()
    G.add_nodes_from(model_names)
    
    for model_a, model_b in tqdm(list(combinations(model_names, 2)), desc="Computing similarity and building graph"):
        score = compare_fingerprints(available_fingerprints[model_a], available_fingerprints[model_b])
        if score >= SIMILARITY_THRESHOLD:
            G.add_edge(model_a, model_b, weight=1)

    logging.info("Calculating shortest path distances for all node pairs in the graph...")
    num_models = len(model_names)
    distance_matrix = np.full((num_models, num_models), np.inf)
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    for i, model_a in enumerate(model_names):
        distance_matrix[i, i] = 0
        if model_a in path_lengths:
            for j, model_b in enumerate(model_names):
                if model_b in path_lengths[model_a]:
                    distance_matrix[i, j] = path_lengths[model_a][model_b]

    logging.info(f"Performing clustering via HDBSCAN on graph distances (min_cluster_size={MIN_CLUSTER_SIZE})...")
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=MIN_CLUSTER_SIZE,
        allow_single_cluster=True
    )
    clusterer.fit(distance_matrix)
    
    labels = clusterer.labels_
    n_clusters_raw = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    logging.info(f"HDBSCAN Raw Results: Found {n_clusters_raw} clusters and {n_noise} noise points.")

    clusters_dict = defaultdict(list)
    for i, label in enumerate(labels):
        if label != -1:
            clusters_dict[label].append(model_names[i])
            
    final_clusters = [
        sorted(models) for models in clusters_dict.values() if len(models) >= 2
    ]
    
    final_clusters = [c for c in final_clusters if len(c) >= MIN_CLUSTER_SIZE]

    logging.info(f"After filtering, outputting {len(final_clusters)} meaningful clusters (size >= {MIN_CLUSTER_SIZE}).")

    try:
        with open(CLUSTERS_OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(sorted(final_clusters, key=len, reverse=True), f, indent=2, ensure_ascii=False)
        logging.info(f"Clustering results successfully saved to: {CLUSTERS_OUTPUT_JSON}")
    except Exception as e:
        logging.error(f"Error occurred while saving JSON file: {e}")

    print("\nClustering analysis completed.")

if __name__ == "__main__":
    main()