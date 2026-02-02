import pandas as pd
import os

def run_analyze(
    input_csv="metrics_summary_full.csv",
    peft_csv="peft_result.csv",
    output_csv="model_nearest_each_metric.csv"
):
    if not os.path.exists(input_csv) or not os.path.exists(peft_csv):
        print("Error: Input files not found.")
        return

    df = pd.read_csv(input_csv)
    step2 = pd.read_csv(peft_csv, index_col=0)

    all_models = sorted(set(df["Model_A"]) | set(df["Model_B"]))
    results = []

    SKIP_KEYWORDS = ["gguf", "gptq", "awq", "onnx", "lora", "adapter"]

    for model in all_models:
        sub_A = df[df["Model_A"] == model].copy()
        sub_B = df[df["Model_B"] == model].copy()

        if not sub_B.empty:
            sub_B = sub_B.rename(columns={"Model_A": "Model_B", "Model_B": "Model_A"})

        sub = pd.concat([sub_A, sub_B], ignore_index=True)
        sub = sub[sub["Model_B"] != model]

        if sub.empty:
            results.append({
                "Model": model,
                "Nearest_attn_L1_Model": None, "Nearest_attn_L1_Dist": None,
                "Nearest_mlp_L1_Model": None, "Nearest_mlp_L1_Dist": None,
                "Nearest_attn_L2_Model": None, "Nearest_attn_L2_Dist": None,
                "Nearest_mlp_L2_Model": None, "Nearest_mlp_L2_Dist": None,
                "Nearest_mean_L2_Model": None, "Nearest_mean_L2_Dist": None,
            })
            continue

        def is_invalid_connection(m1, m2):
            m2_lower = str(m2).lower()
            if any(k in m2_lower for k in SKIP_KEYWORDS):
                return True
            
            if m2 in step2.index:
                if (step2.loc[m2] == "peft").any():
                    return True
            
            if m1 in step2.index and m2 in step2.columns:
                val = step2.loc[m1, m2]
                if pd.notna(val) and val != "unknown":
                    return True
            
            if m2 in step2.index and m1 in step2.columns:
                val = step2.loc[m2, m1]
                if pd.notna(val) and val != "unknown":
                    return True
            
            return False

        def nearest_with_filters(col):
            sub2 = sub.dropna(subset=[col]).sort_values(col)
            for _, row in sub2.iterrows():
                tgt = row["Model_B"]
                if is_invalid_connection(model, tgt):
                    continue
                return tgt, row[col]
            return None, None

        a1, a1d = nearest_with_filters("mean_attn_L1_dist")
        m1, m1d = nearest_with_filters("mean_mlp_L1_dist")
        a2, a2d = nearest_with_filters("mean_attn_L2_dist")
        m2, m2d = nearest_with_filters("mean_mlp_L2_dist")

        sub_mean = sub.copy()
        sub_mean["mean_L2_dist"] = sub_mean[
            ["mean_attn_L2_dist", "mean_mlp_L2_dist"]
        ].mean(axis=1, skipna=True)
        sub_mean = sub_mean.dropna(subset=["mean_L2_dist"]).sort_values("mean_L2_dist")

        mean_model, mean_dist = None, None
        for _, row in sub_mean.iterrows():
            tgt = row["Model_B"]
            if is_invalid_connection(model, tgt):
                continue
            mean_model = tgt
            mean_dist = row["mean_L2_dist"]
            break

        results.append({
            "Model": model,
            "Nearest_attn_L1_Model": a1, "Nearest_attn_L1_Dist": a1d,
            "Nearest_mlp_L1_Model": m1, "Nearest_mlp_L1_Dist": m1d,
            "Nearest_attn_L2_Model": a2, "Nearest_attn_L2_Dist": a2d,
            "Nearest_mlp_L2_Model": m2, "Nearest_mlp_L2_Dist": m2d,
            "Nearest_mean_L2_Model": mean_model, "Nearest_mean_L2_Dist": mean_dist,
        })

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"analyze2 finished: {output_csv}")

if __name__ == "__main__":
    run_analyze()