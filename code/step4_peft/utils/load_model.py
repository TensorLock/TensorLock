import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
from gguf import GGUFReader
import numpy as np

MODEL_BASE_DIR = "../../../evaluation/Benchmark/models"

def get_safe_path(model_id_or_path):
    """Convert model ID to a safe local path"""
    safe_name = model_id_or_path.replace("/", "_")
    path = os.path.join(MODEL_BASE_DIR, safe_name) if not os.path.isabs(safe_name) else safe_name
    return path if os.path.exists(path) else model_id_or_path

class GGUFWeightWrapper:
    """
    Simulate a model object for GGUF weights access and dequantization
    """
    def __init__(self, gguf_path):
        self.reader = GGUFReader(gguf_path)
        self.path = gguf_path

    def state_dict(self):
        """
        Iterate GGUF tensors and convert them to a PyTorch-like state_dict format
        """
        weights = {}
        print(f"--- Parsing GGUF weights and converting to PyTorch format ---")
        for tensor in self.reader.tensors:
            name = tensor.name
            try:
                # Attempt to convert to float32 array
                weights[name] = torch.from_numpy(tensor.data.astype(np.float32))
            except Exception as e:
                print(f"⚠️ Failed to convert tensor {name}: {e}")
                weights[name] = torch.from_numpy(tensor.data)
        return weights

    def to(self, device):
        return self

    def eval(self):
        return self


def load_any_model(model_name):
    """
    Intelligent model loader:
    1. Auto-detect GGUF (vLLM)
    2. Auto-detect PEFT (Adapter)
    3. Automatically distinguish CausalLM and Seq2Seq (T5/BART etc.)
    """
    model_path = get_safe_path(model_name)
    
    if not os.path.exists(model_path):
        model_path = model_name 

    gguf_file = None
    if os.path.isdir(model_path):
        gguf_file = next((f for f in os.listdir(model_path) if f.endswith((".gguf"))), None)

    if gguf_file:
        gguf_full_path = os.path.join(model_path, gguf_file)
        print(f"== Detected GGUF model: {gguf_full_path} ==")
        return GGUFWeightWrapper(gguf_full_path)

    adapter_file = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_file):
        print(f"== Loading PEFT model: {model_path} ==")
        from peft import PeftConfig, PeftModel
        
        perf_config = PeftConfig.from_pretrained(model_path)
        base_model_path = perf_config.base_model_name_or_path
        
        base_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        model_type = base_config.model_type.lower()
        
        load_kwargs = {
            "torch_dtype": torch.float32,
            "trust_remote_code": True
        }

        if model_type in ["t5", "mt5", "bart", "mbart", "pegasus", "marian", "prophetnet"]:
            print(f"[INFO] Loading Seq2Seq Base Model for PEFT: {model_type}")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path, **load_kwargs)

        elif "vl" in model_type or "mllama" in model_type:
            base_model = AutoModel.from_pretrained(base_model_path, torch_dtype=torch.float32, trust_remote_code=True)
        else:
            print(f"[INFO] Loading Causal Base Model for PEFT: {model_type}")
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
            
        return PeftModel.from_pretrained(base_model, model_path)

    print(f"== Loading standard HF model: {model_path} ==")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = config.model_type.lower()
    
    common_kwargs = {
        "ignore_mismatched_sizes": True,
        "torch_dtype": torch.float32,
        "trust_remote_code": True
    }

    if model_type in ["t5", "mt5", "bart", "mbart", "pegasus", "marian", "prophetnet"]:
        print(f"[INFO] Detected Seq2Seq architecture: {model_type}")
        return AutoModelForSeq2SeqLM.from_pretrained(model_path, **common_kwargs)

    elif "vl" in model_type or "mllama" in model_type or "vision" in model_type:
        print(f"[INFO] Detected multimodal architecture ({model_type}), loading with AutoModel: {model_name}")
        return AutoModel.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True, ignore_mismatched_sizes=True)
    else:
        print(f"[INFO] Detected Causal architecture: {model_type}")
        return AutoModelForCausalLM.from_pretrained(model_path, **common_kwargs)