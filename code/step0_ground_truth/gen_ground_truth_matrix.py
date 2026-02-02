import pandas as pd

INPUT_CSV = f"../../dataset/model_relation.csv"
OUTPUT_CSV = f"ground_truth_matrix.csv"

def main():
    df = pd.read_csv(INPUT_CSV)

    base_models = set(df["Base Model"].astype(str))
    derived_models = set(df["Derived Model"].astype(str))
    model_set = sorted(base_models | derived_models)

    matrix = pd.DataFrame(
        "none",
        index=model_set,
        columns=model_set
    )

    for m in model_set:
        matrix.loc[m, m] = "self"

    for _, row in df.iterrows():
        base = str(row["Base Model"])
        derived = str(row["Derived Model"])
        relation = str(row["Relation"])

        matrix.loc[derived, base] = relation

    matrix.to_csv(OUTPUT_CSV)

    print(f"✅ Relation matrix saved to: {OUTPUT_CSV}")
    print(f"   Matrix size: {len(model_set)} x {len(model_set)}")


if __name__ == "__main__":
    main()