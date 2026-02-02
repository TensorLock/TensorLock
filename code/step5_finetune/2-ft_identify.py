import os
import json
import pandas as pd
import utils.sim as sim
import utils.analyze as analyze

CLUSTERS_JSON = "../step1_cluster/clusters.json"
INPUT_MATRIX_CSV = "../step4_peft/peft_result.csv"
OUTPUT_MATRIX_CSV = "./finetune_result.csv"
DIRECTION_CACHE = "./entropy_cache.csv"

MODEL_LIST = "model_list.txt"
METRICS_SUMMARY = "metrics_summary.csv"
METRICS_ALL = "module_metrics.csv"
NEAREST_FILE = "temp_model_nearest.csv"

METRIC = "mean_L2"

def build_finetune_relations(df_matrix, nearest_path, entropy_path, metric):

    with open(MODEL_LIST, 'r') as f:
        model_list = [line.strip() for line in f if line.strip()]

    nearest_df = pd.read_csv(nearest_path).set_index("Model")
    entropy_df = pd.read_csv(entropy_path)
    entropy = entropy_df.set_index("model")["RankEff"].to_dict()

    metric_col = f"Nearest_{metric}_Model"
    dist_col = f"Nearest_{metric}_Dist"

    for derived_candidate in model_list:

        if derived_candidate not in nearest_df.index:
            print(f"Warning: {derived_candidate} not found in nearest neighbors file, skipping.")
            continue

        row = nearest_df.loc[derived_candidate]

        existing_relations = df_matrix.loc[derived_candidate].dropna()
        mask = (existing_relations != "unknown") & (existing_relations != "0") & (existing_relations != 0)
        valid_existing = existing_relations[mask]
        if not valid_existing.empty:
            print(f"{derived_candidate} already has a parent, skipping.")
            continue

        base_candidate = row[metric_col]
        dist = row[dist_col]

        if pd.isna(base_candidate) or pd.isna(dist) or dist > 10:
            continue

        entropy_m = entropy.get(derived_candidate)
        entropy_n = entropy.get(base_candidate)
        if entropy_m is None or entropy_n is None:
            print(f"{derived_candidate} or {base_candidate} missing entropy value!")
            continue

        if entropy_m < entropy_n:
            src, tgt = derived_candidate, base_candidate
        else:
            src, tgt = base_candidate, derived_candidate

        existing_parent = df_matrix.loc[tgt].dropna()
        null_labels = ["unknown", "0", 0]
        existing_parent = existing_parent[~existing_parent.isin(null_labels)]

        is_empty_direct = df_matrix.loc[tgt, src] in null_labels
        is_empty_reverse = df_matrix.loc[src, tgt] in null_labels

        if existing_parent.empty and is_empty_direct and is_empty_reverse:
            df_matrix.loc[tgt, src] = "finetune"
        else:
            print(f"Result skipped: {tgt} already has a parent, ignoring connection from {src}")

    return df_matrix


def main():
    if not os.path.exists(INPUT_MATRIX_CSV):
        print(f"Error: {INPUT_MATRIX_CSV} not found.")
        return

    full_matrix = pd.read_csv(INPUT_MATRIX_CSV, index_col=0)
    full_matrix = full_matrix.fillna("unknown")

    with open(CLUSTERS_JSON, "r") as f:
        clusters = json.load(f)

    for i, cluster_models in enumerate(clusters):
        print(f"\n--- Processing Cluster {i+1}/{len(clusters)} ---")

        with open(MODEL_LIST, "w") as f:
            for m in cluster_models:
                f.write(m + "\n")

        try:
            sim.main_metrics(
                matrix_path=INPUT_MATRIX_CSV,
                summary_file=METRICS_SUMMARY,
                out_csv=METRICS_ALL,
                model_list=MODEL_LIST
            )
            print("Similarity metrics calculation completed.")

            cluster_step5 = full_matrix.loc[cluster_models, cluster_models]
            cluster_step5_tmp = "temp_cluster_step5.csv"
            cluster_step5.to_csv(cluster_step5_tmp)

            analyze.run_analyze(
                input_csv=METRICS_SUMMARY,
                peft_csv=cluster_step5_tmp,
                output_csv=NEAREST_FILE
            )
            print("Analyze completed.")

            full_matrix = build_finetune_relations(
                full_matrix,
                NEAREST_FILE,
                DIRECTION_CACHE,
                METRIC
            )

        except Exception as e:
            print(f"Error processing cluster {i}: {e}")
            continue

    os.makedirs(os.path.dirname(OUTPUT_MATRIX_CSV), exist_ok=True)
    full_matrix.to_csv(OUTPUT_MATRIX_CSV)

    print(f"\n🏁 All clusters processed.")
    print(f"Final matrix saved to: {OUTPUT_MATRIX_CSV}")


if __name__ == "__main__":
    main()