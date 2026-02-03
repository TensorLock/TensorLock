import os
import json
from huggingface_hub import snapshot_download
from tqdm import tqdm

MODEL_LIST_JSON = "../../evaluation/Benchmark/model_list.json" 
local_dir = "../../evaluation/Benchmark/models"

os.makedirs(local_dir, exist_ok=True)

# Load model list
with open(MODEL_LIST_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

models = data.get("models", [])

print(f"Total models: {len(models)}")

# Download models
for model_id in tqdm(models, desc="Downloading models"):
    print(f"\nDownloading: {model_id}")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(local_dir, model_id.replace("/", "_")),
            resume_download=True,
        )
        print(f"\nDownload {model_id} successfully!\n")
    except Exception as e:
        print(f"\nDownload failed: {model_id}, Error: {e}\n")