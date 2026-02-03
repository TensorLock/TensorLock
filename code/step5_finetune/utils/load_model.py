import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel

def get_safe_path(model_name, base_dir="../../evaluation/Benchmark/models"):
    safe_name = model_name.replace("/", "_")
    return os.path.join(base_dir, safe_name)

def load_any_model(model_name):
    MODEL_BASE_DIR = "../../../evaluation/Benchmark/models"
    DEQUANT_BASE_DIR = "../../step1_cluster/converted"

    model_path = get_safe_path(model_name, MODEL_BASE_DIR)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Path not found: {model_path}")

    skip_keywords = ["GGUF", "gguf", "GPTQ", "gptq", "AWQ", "awq", "Int4", "Int8"]
    if any(keyword in model_path for keyword in skip_keywords):
        safe_name = model_name.replace("/", "_")
        model_path = os.path.join(DEQUANT_BASE_DIR, safe_name)
        print(f"\n 📦 [GGUF Redirect] Detected GGUF, using de-quantized HF path: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Dequantized model path not found: {model_path}")

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

    # Standard loading
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