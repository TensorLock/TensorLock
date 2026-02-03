import os
import json
import glob
import torch
from safetensors.torch import load_file as safe_load_file
import onnx

BASE_MODEL_DIR = "../../evaluation/Benchmark/models"
INPUT_JSON_FILE = "../../evaluation/Benchmark/model_list.json"
OUTPUT_CSV_FILE = "./quantized_models.csv"

class QuantizationDetector:
    """
    A standalone, metadata-free quantization detector.
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir
        print("Quantization Detector initialized.")

    def _get_full_path(self, model_name):
        dir_name = model_name.replace("/", "_")
        full_path = os.path.join(self.base_dir, dir_name)
        if not os.path.exists(full_path):
            search_pattern = os.path.join(self.base_dir, f"*{model_name.split('/')[-1]}*")
            possible_dirs = glob.glob(search_pattern)
            if possible_dirs:
                return possible_dirs[0]
            return None
        return full_path

    def analyze_onnx(self, model_path):
        onnx_files = glob.glob(os.path.join(model_path, '**', '*.onnx'), recursive=True)
        if not onnx_files:
            return False, "Default", "No ONNX files"

        target_file = onnx_files[0]
        for f in onnx_files:
            if "int8" in f.lower() or "quantized" in f.lower():
                target_file = f
                break
        
        try:
            model = onnx.load(target_file)
            for tensor in model.graph.initializer:
                if tensor.data_type in [2, 3]:  # UINT8, INT8
                    return True, "Dtype (ONNX)", f"INT8/UINT8 in {os.path.basename(target_file)}"
            
            return False, "Format", ".onnx (FP32/FP16)"

        except Exception as e:
            return False, "Scan Failed", f"ONNX load error: {e}"

    def detect(self, model_name):
        model_path = self._get_full_path(model_name)
        if not model_path:
            return False, "Not Found", "Path does not exist"

        # Format-based detection
        if glob.glob(os.path.join(model_path, '**', '*.gguf'), recursive=True):
            return True, "Format", ".gguf"
        if glob.glob(os.path.join(model_path, '**', '*.npz'), recursive=True):
            return True, "Format", ".npz (MLX)"
        
        # ONNX-based detection
        if glob.glob(os.path.join(model_path, '**', '*.onnx'), recursive=True):
            return self.analyze_onnx(model_path)

        # Weight file scanning
        files_to_scan = glob.glob(os.path.join(model_path, '**', '*.safetensors'), recursive=True) + \
                        glob.glob(os.path.join(model_path, '**', '*.bin'), recursive=True)
        
        if not files_to_scan:
            return False, "Default", "No weight files"

        for fpath in files_to_scan[:1]: 
            try:
                state_dict = {}
                if fpath.endswith(".safetensors"):
                    state_dict = safe_load_file(fpath, device="cpu")
                else:
                    state_dict = torch.load(fpath, map_location="cpu")

                found_quant_keys = set()
                found_lora = False
                found_int_dtype = None

                for k, v in state_dict.items():
                    if k.endswith((".qweight", ".qzeros", ".scales", ".g_idx")):
                        feature_key = "." + k.split('.')[-1]
                        found_quant_keys.add(feature_key)
                    
                    if "lora_a" in k.lower() or "lora_b" in k.lower():
                        found_lora = True
                    
                    if isinstance(v, torch.Tensor) and not v.is_floating_point():
                        found_int_dtype = str(v.dtype)
                
                if found_quant_keys:
                    return True, "Key Name", ", ".join(sorted(list(found_quant_keys)))
                if found_int_dtype:
                    return True, "Dtype", found_int_dtype
                if found_lora:
                    return False, "Key Name", "LoRA"
                        
            except Exception as e:
                return False, "Scan Failed", str(e)

        return False, "Default", "No explicit quantization features found"

def main():
    print("\nPreprocessing - Quantization Detection")
    
    # Load model list
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model_list = data.get("models", [])
        if not model_list:
            print(f"FATAL: No models found in '{INPUT_JSON_FILE}'")
            return
        print(f"Loaded {len(model_list)} models from '{INPUT_JSON_FILE}'")
    except Exception as e:
        print(f"FATAL: Could not load or parse '{INPUT_JSON_FILE}'. Error: {e}")
        return

    detector = QuantizationDetector(BASE_MODEL_DIR)
    quantized_models = []
    
    total_models = len(model_list)
    for i, model_name in enumerate(model_list):
        print(f"[{i+1}/{total_models}] Processing: {model_name} ...", end=" ")
        try:
            is_quant, method, details = detector.detect(model_name)
            if is_quant:
                print(f"Quantized (Reason: {method} - {details})")
                quantized_models.append(model_name)
            else:
                print(f"Not Quantized (Reason: {method} - {details})")
        except Exception as e:
            print(f"🔥 ERROR: {e}")
            
    try:
        with open(OUTPUT_CSV_FILE, 'w') as f:
            for model_name in quantized_models:
                f.write(f"{model_name}\n")
        print(f"\nSuccessfully wrote {len(quantized_models)} quantized models to '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"\nERROR writing to CSV: {e}")

if __name__ == "__main__":
    main()