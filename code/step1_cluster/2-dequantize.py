import os
import csv
import glob
import shutil
import gc
import torch
import torch.nn as nn
import packaging.version
import numpy as np
from transformers import GenerationConfig, activations

INPUT_CSV = "./quantized_models.csv"
SOURCE_ROOT = "../../dataset/models"
DEST_ROOT = "./converted"


_original_parse = packaging.version.parse
def _safe_version_parse(version):
    if str(version).strip().upper() == "N/A": 
        return _original_parse("4.0.0")
    return _original_parse(version)
packaging.version.parse = _safe_version_parse

_original_gen_from_pretrained = GenerationConfig.from_pretrained
@classmethod
def _safe_gen_from_pretrained(cls, pretrained_model_name, **kwargs):
    try: 
        return _original_gen_from_pretrained(pretrained_model_name, **kwargs)
    except (OSError, TypeError): 
        return cls()
GenerationConfig.from_pretrained = _safe_gen_from_pretrained

activations.PytorchGELUTanh = activations.GELUTanh

from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer, 
    AutoConfig
)


try:
    import onnx
    from onnx import numpy_helper
    from safetensors.torch import save_file as safe_save_file
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def get_clean_model_id(model_id):
    if "/tree/main" in model_id: 
        return model_id.split("/tree/main")[0]
    return model_id.strip()

def find_file_recursive(directory, extension):
    files = glob.glob(os.path.join(directory, "**", f"*.{extension}"), recursive=True)
    if not files: 
        return None
    files.sort(key=lambda x: (x.count(os.sep), -os.path.getsize(x)))
    return files[0]

def copy_auxiliary_files(src_path, dest_path):
    """
    Enhanced auxiliary file reconstruction:
    - Extract metadata from GGUF
    - Physically copy tokenizer/config/scripts/text files
    """
    print(f"    [*] Reconstructing auxiliary files from {src_path}...")
    
    gguf_file = find_file_recursive(src_path, "gguf")
    if gguf_file:
        try:
            filename = os.path.basename(gguf_file)
            print(f"    [*] GGUF detected: {filename}. Attempting metadata extraction...")
            tokenizer = AutoTokenizer.from_pretrained(src_path, gguf_file=filename, trust_remote_code=True)
            tokenizer.save_pretrained(dest_path)
            config = AutoConfig.from_pretrained(src_path, gguf_file=filename, trust_remote_code=True)
            config.save_pretrained(dest_path)
            print("    [+] Successfully extracted Tokenizer/Config from GGUF.")
        except Exception as e:
            print(f"    [!] GGUF metadata extraction failed: {e}")

    if not os.path.exists(os.path.join(dest_path, "tokenizer_config.json")):
        try:
            tokenizer = AutoTokenizer.from_pretrained(src_path, local_files_only=True, trust_remote_code=True)
            tokenizer.save_pretrained(dest_path)
            print("    [+] Tokenizer saved via AutoTokenizer.")
        except:
            pass

    if not os.path.exists(os.path.join(dest_path, "config.json")):
        try:
            config = AutoConfig.from_pretrained(src_path, local_files_only=True, trust_remote_code=True)
            config.save_pretrained(dest_path)
            print("    [+] Config saved via AutoConfig.")
        except:
            pass

    extensions = ["*.json", "*.model", "*.txt", "*.py", "*.spm", "*.tiktoken",
                  "preprocessor_config.json", "generation_config.json"]
    copied = 0
    for ext in extensions:
        for file in glob.glob(os.path.join(src_path, "**", ext), recursive=True):
            if any(x in file.lower() for x in ["safetensors", "bin", "pth", "onnx", "gguf"]): 
                continue
            filename = os.path.basename(file)
            target_file = os.path.join(dest_path, filename)
            if not os.path.exists(target_file):
                try:
                    shutil.copy(file, dest_path)
                    copied += 1
                except:
                    pass
    print(f"    [+] Manually copied {copied} auxiliary files.")

def sanitize_config(config):
    for key, value in config.__dict__.items():
        if isinstance(value, torch.dtype):
            setattr(config, key, str(value).split('.')[-1])
    if hasattr(config, 'torch_dtype') and isinstance(config.torch_dtype, torch.dtype):
        config.torch_dtype = str(config.torch_dtype).split('.')[-1]
    return config


def load_model_smart(src_path, device_map="auto"):
    path_lower = src_path.lower()
    candidates = [
        (AutoModelForCausalLM, "CausalLM"),
        (AutoModelForImageTextToText, "ImageTextToText"),
        (AutoModelForVision2Seq, "Vision2Seq"),
        (AutoModelForSpeechSeq2Seq, "SpeechSeq2Seq"),
        (AutoModelForSeq2SeqLM, "Seq2SeqLM"),
        (AutoModel, "Generic AutoModel")
    ]

    priority = []
    if "whisper" in path_lower or "speech" in path_lower:
        priority.append((AutoModelForSpeechSeq2Seq, "SpeechSeq2Seq"))
    elif "vl" in path_lower or "vision" in path_lower or "moondream" in path_lower:
        priority.append((AutoModelForImageTextToText, "ImageTextToText"))
        priority.append((AutoModelForVision2Seq, "Vision2Seq"))
    elif "t5" in path_lower or "bart" in path_lower:
        priority.append((AutoModelForSeq2SeqLM, "Seq2SeqLM"))

    final_order = priority + [c for c in candidates if c not in priority]
    
    last_err = None
    for Cls, name in final_order:
        try:
            model = Cls.from_pretrained(
                src_path, device_map=device_map, torch_dtype=torch.float16, 
                trust_remote_code=True, local_files_only=True
            )
            print(f"    [+] Loaded successfully as {name}")
            return model
        except Exception as e:
            last_err = e
    raise last_err


def process_onnx(model_id, src_path, dest_path, onnx_file):
    if not HAS_ONNX: 
        raise ImportError("ONNX library not installed.")
    model_proto = onnx.load(onnx_file)
    onnx_weights = {}
    for initializer in model_proto.graph.initializer:
        if "onnx::" in initializer.name or initializer.name.startswith("/"): 
            continue
        w_np = numpy_helper.to_array(initializer)
        w_torch = torch.from_numpy(w_np).to(torch.float16)
        onnx_weights[initializer.name] = w_torch

    try:
        config = AutoConfig.from_pretrained(src_path, local_files_only=True, trust_remote_code=True)
        with torch.device("meta"): 
            skeleton = AutoModel.from_config(config)
        hf_keys = set(skeleton.state_dict().keys())
    except:
        hf_keys = set()

    possible_prefixes = ["", "model.", "encoder.", "decoder.", "transformer.", 
                         "model.encoder.", "model.decoder."]
    best_prefix = ""
    max_match = 0
    if hf_keys:
        for prefix in possible_prefixes:
            match_count = sum(1 for k in list(onnx_weights.keys())[:50] if (prefix + k) in hf_keys)
            if match_count > max_match:
                max_match = match_count
                best_prefix = prefix

    final_tensors = {best_prefix + k: v for k, v in onnx_weights.items()}
    safe_save_file(final_tensors, os.path.join(dest_path, "model.safetensors"))
    copy_auxiliary_files(src_path, dest_path)

def process_gguf(model_id, src_path, dest_path, gguf_file):
    gguf_dir = os.path.dirname(gguf_file)
    gguf_filename = os.path.basename(gguf_file)
    
    model = None
    for Cls in [AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel]:
        try:
            model = Cls.from_pretrained(
                gguf_dir, gguf_file=gguf_filename, torch_dtype=torch.float16,
                local_files_only=True, trust_remote_code=True
            )
            break
        except:
            continue
    
    if model is None: 
        raise ValueError("Could not load GGUF via any class")
    
    model.save_pretrained(dest_path)
    copy_auxiliary_files(src_path, dest_path)
    del model


def manual_unpack_gptq(module, in_features, out_features):
    qweight = getattr(module, "qweight", None) or getattr(module, "weight_packed", None)
    scales = getattr(module, "scales", None) or getattr(module, "weight_scale", None)
    qzeros = getattr(module, "qzeros", None) or getattr(module, "weight_zero_point", None)
    
    iweights = torch.zeros((in_features, out_features), dtype=torch.int32)
    bits = 4
    mask = (1 << bits) - 1
    qweight_cpu = qweight.cpu()
    for i in range(8):
        iweights[i::8, :] = (qweight_cpu >> (bits * i)) & mask
    
    if qzeros is not None:
        zeros = torch.zeros((scales.shape[0], scales.shape[1]), dtype=torch.float16)
        qzeros_cpu = qzeros.cpu()
        for i in range(8):
            zeros[:, i::8] = ((qzeros_cpu >> (bits * i)) & mask).to(torch.float16)
        zeros = zeros + 1
    else:
        zeros = torch.zeros_like(scales.cpu())
        
    group_size = in_features // scales.shape[0]
    scales_ext = scales.cpu().repeat_interleave(group_size, dim=0)
    zeros_ext = zeros.repeat_interleave(group_size, dim=0)
    
    weights = (iweights.to(torch.float16) - zeros_ext) * scales_ext
    return weights.t()

def recursive_replace_linear(module):
    count = 0
    for name, child in module.named_children():
        is_quant = (
            hasattr(child, "qweight") or 
            "QuantLinear" in child.__class__.__name__ or 
            "WQLinear" in child.__class__.__name__
        )
        if is_quant:
            in_f = getattr(child, "in_features", None) or getattr(child, "infeatures", None)
            out_f = getattr(child, "out_features", None) or getattr(child, "outfeatures", None)
            if in_f is None: 
                continue
            
            bias = hasattr(child, 'bias') and child.bias is not None
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                child.to(device)
                eye = torch.eye(in_f, dtype=torch.float16, device=device)
                with torch.no_grad():
                    out = child(eye)
                    if bias: 
                        out = out - child.bias
                w_val = out.t().contiguous().to("cpu")
            except:
                w_val = manual_unpack_gptq(child, in_f, out_f)
            
            new_layer = nn.Linear(in_f, out_f, bias=bias)
            new_layer.weight.data = w_val
            if bias: 
                new_layer.bias.data = child.bias.data.to(torch.float16).to("cpu")
            setattr(module, name, new_layer)
            count += 1
        else:
            count += recursive_replace_linear(child)
    return count

def process_linear_projection(model_id, src_path, dest_path):
    model = load_model_smart(src_path, device_map="auto")
    recursive_replace_linear(model)
    if hasattr(model.config, "quantization_config"): 
        delattr(model.config, "quantization_config")
    model.config = sanitize_config(model.config)
    model.save_pretrained(dest_path)
    copy_auxiliary_files(src_path, dest_path)
    del model


def recursive_dequantize_bnb(module):
    count = 0
    for name, child in module.named_children():
        cls_name = child.__class__.__name__
        if "Linear4bit" in cls_name or "Linear8bit" in cls_name:
            if "Linear4bit" in cls_name:
                w_dequant = bnb.functional.dequantize_4bit(child.weight.data, child.weight.quant_state)
            else:
                w_dequant = bnb.functional.dequantize_8bit(child.weight.data, child.weight.CB)
            new_layer = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
            new_layer.weight.data = w_dequant.to(torch.float16).to("cpu")
            if child.bias is not None:
                new_layer.bias.data = child.bias.data.to(torch.float16).to("cpu")
            setattr(module, name, new_layer)
            count += 1
        else:
            count += recursive_dequantize_bnb(child)
    return count

def process_bnb_dequantize(model_id, src_path, dest_path):
    model = load_model_smart(src_path, device_map="auto")
    recursive_dequantize_bnb(model)
    if hasattr(model.config, "quantization_config"): 
        delattr(model.config, "quantization_config")
    model.config = sanitize_config(model.config)
    model.save_pretrained(dest_path)
    copy_auxiliary_files(src_path, dest_path)
    del model


def process_one_model(model_id):
    model_id = get_clean_model_id(model_id)
    dir_name = model_id.replace("/", "_")
    src_path = os.path.join(SOURCE_ROOT, dir_name)
    dest_path = os.path.join(DEST_ROOT, dir_name)
    
    print(f"\n{'='*60}")
    print(f"Processing: {model_id}")
    
    if not os.path.exists(src_path):
        print(f"[!] Source not found: {src_path}")
        return

    weight_files = glob.glob(os.path.join(dest_path, "*.safetensors")) + \
                   glob.glob(os.path.join(dest_path, "pytorch_model*.bin"))
    has_weights = len(weight_files) > 0
    has_tokenizer = any(os.path.exists(os.path.join(dest_path, f)) 
                        for f in ["tokenizer_config.json", "tokenizer.model", "tokenizer.json"])
    has_config = os.path.exists(os.path.join(dest_path, "config.json"))

    if has_weights and has_tokenizer and has_config:
        print(f"[*] Full model (weights + tokenizer + config) already exists. Skipping.")
        return

    if has_weights:
        print(f"[*] Weights already exist. Reconstructing auxiliary files only...")
        copy_auxiliary_files(src_path, dest_path)
        return

    os.makedirs(dest_path, exist_ok=True)

    strategies = [
        ("GGUF Dequantization", process_gguf, "gguf"),
        ("ONNX Extraction", process_onnx, "onnx"),
        ("AWQ/GPTQ Projection", process_linear_projection, None),
        ("BnB Dequantization", process_bnb_dequantize, None)
    ]

    success = False
    for name, func, ext_req in strategies:
        target_file = find_file_recursive(src_path, ext_req) if ext_req else None
        if ext_req and not target_file: 
            continue
        
        print(f"[*] Attempting strategy: {name}...")
        try:
            if ext_req: 
                func(model_id, src_path, dest_path, target_file)
            else: 
                func(model_id, src_path, dest_path)
            print(f"[+] Successfully converted {model_id} via {name}")
            success = True
            break
        except Exception as e:
            print(f"[-] Strategy {name} failed: {str(e).splitlines()[0]}")
            if os.path.exists(os.path.join(dest_path, "model.safetensors")):
                try: 
                    os.remove(os.path.join(dest_path, "model.safetensors"))
                except: 
                    pass
            gc.collect()
            torch.cuda.empty_cache()

    if not success:
        print(f"[!] All strategies failed for {model_id}.")
        if os.path.exists(dest_path) and not os.listdir(dest_path): 
            shutil.rmtree(dest_path)

def main():
    if not os.path.exists(INPUT_CSV): 
        print(f"CSV not found: {INPUT_CSV}")
        return
    if not os.path.exists(DEST_ROOT): 
        os.makedirs(DEST_ROOT)

    models_to_process = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row: 
                models_to_process.append(row[0].strip())

    print(f"[*] Total models to process: {len(models_to_process)}")
    for model_id in models_to_process:
        process_one_model(model_id)

    print("\n[+] Done.")

if __name__ == "__main__":
    main()