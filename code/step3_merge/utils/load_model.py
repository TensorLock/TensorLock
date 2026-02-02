import os
import torch
from gguf import GGUFReader
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel

MODEL_BASE_DIR = "../../../dataset/models"


def get_safe_path(model_id_or_path):
    """Convert model ID to actual local filesystem path"""
    safe_name = model_id_or_path.replace("/", "_")
    path = os.path.join(MODEL_BASE_DIR, safe_name) if not os.path.isabs(safe_name) else safe_name
    return path if os.path.exists(path) else model_id_or_path


class GGUFWeightWrapper:
    """
    Simulated model wrapper specifically for GGUF weight access and dequantization
    """
    def __init__(self, gguf_path):
        self.reader = GGUFReader(gguf_path)
        self.path = gguf_path

    def state_dict(self):
        """
        Iterate over GGUF tensors and convert them into a PyTorch-like state_dict format
        """
        weights = {}
        for tensor in self.reader.tensors:
            name = tensor.name
            try:
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
    model_path = get_safe_path(model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Path not found: {model_path}")

    skip_keywords = ["GGUF", "gguf", "GPTQ", "gptq", "AWQ", "awq", "Int4", "Int8"]
    for keyword in skip_keywords:
        if keyword in model_path:
            return None

    adapter_file = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_file):
        print(f"🧬 [PEFT Mode] Loading adapter from: {model_path}")
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path

        base_conf = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        m_type = base_conf.model_type.lower()

        common_kwargs = {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }

        if m_type in ["t5", "mt5", "bart", "mbart", "pegasus", "marian", "prophetnet"]:
            print(f"   └─ [Seq2Seq Base] Loading {m_type}...")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path, **common_kwargs)

        elif any(x in m_type for x in ["vl", "mllama", "vision", "vit", "clip", "siglip", "dinov2", "deit"]):
            print(f"   └─ [Multimodal Base] Loading {m_type}...")
            base_model = AutoModel.from_pretrained(base_model_path, **common_kwargs)

        else:
            print(f"   └─ [Causal Base] Loading {m_type}...")
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **common_kwargs)

        return PeftModel.from_pretrained(base_model, model_path)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    m_type = config.model_type.lower()

    final_kwargs = {
        "ignore_mismatched_sizes": True,
        "torch_dtype": torch.float32,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True
    }

    if m_type in ["t5", "mt5", "bart", "mbart", "pegasus", "marian", "prophetnet"]:
        print(f"🚀 [Standard Mode] Loading Seq2Seq model: {m_type}")
        return AutoModelForSeq2SeqLM.from_pretrained(model_path, **final_kwargs)

    elif any(x in m_type for x in ["vl", "mllama", "vision", "vit", "clip", "siglip", "dinov2", "deit"]):
        print(f"🚀 [Standard Mode] Loading Vision/Multimodal model: {m_type}")
        return AutoModel.from_pretrained(model_path, **final_kwargs)

    else:
        print(f"🚀 [Standard Mode] Loading Causal model: {m_type}")
        return AutoModelForCausalLM.from_pretrained(model_path, **final_kwargs)