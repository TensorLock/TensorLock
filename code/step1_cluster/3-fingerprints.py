import os
import logging
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import skew, kurtosis
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoConfig, AutoModelForImageClassification
from peft import PeftModel
import glob
import json
import pandas as pd
from transformers import activations

activations.PytorchGELUTanh = activations.GELUTanh

MODEL_BASE_PATH = "../../evaluation/Benchmark/models/"
DEQUANTIZED_MODEL_PATH = "./converted/"
FINGERPRINT_CACHE_PATH = "./fingerprint/"
QUANTIZED_MODELS_CSV = "./quantized_models.csv"
INPUT_JSON_FILE = "../../evaluation/Benchmark/model_list.json"

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - [%(levelname)s] - %(message)s"
}


def setup_logging():
    """Configure logging system"""
    logging.basicConfig(
        level=LOGGING_CONFIG["level"],
        format=LOGGING_CONFIG["format"],
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_lora_base_model(adapter_path: str):
    """Dynamically read adapter_config.json to obtain base model name"""
    config_path = os.path.join(adapter_path, 'adapter_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"adapter_config.json not found in LoRA directory: {adapter_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    base_model_name = config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"'base_model_name_or_path' field missing in adapter_config.json: {config_path}")

    return base_model_name


def load_model_and_tokenizer(model_path: str, model_name_for_log: str):
    """Load model and tokenizer from a given path with robust support for vision models (e.g. ViT)"""
    logging.info(f"Attempting to load model from path: {model_path}")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Path does not exist: {model_path}")

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "trust_remote_code": True,
        "ignore_mismatched_sizes": True
    }

    model_name_lower = model_name_for_log.lower()

    if any(k in model_name_lower for k in ['distil', 't5', 'opt', 'vit']):
        logging.info(f"Detected {model_name_for_log} model, disabling device_map='auto'.")
    else:
        model_kwargs["device_map"] = "auto"

    if not glob.glob(os.path.join(model_path, '*.safetensors')):
        logging.info("No .safetensors files found, forcing .bin loading.")
        model_kwargs['use_safetensors'] = False

    model = None
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if config.model_type == 'vit':
            logging.info("Detected ViT model type, using AutoModelForImageClassification...")
            model = AutoModelForImageClassification.from_pretrained(model_path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except Exception:
        logging.warning("Task-specific loading failed, falling back to generic AutoModel...")
        model = AutoModel.from_pretrained(model_path, **model_kwargs)

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    except Exception as e:
        logging.warning(f"Tokenizer loading failed (normal for vision models): {e}")

    logging.info(f"Successfully loaded model from '{model_path}'.")
    return model, tokenizer


def get_model_layers(model):
    """Robustly extract transformer layers with support for multimodal models (e.g. Qwen2-VL, Jina-VL)"""
    model_type_name = type(model).__name__

    if hasattr(model, 'language_model'):
        logging.info(f"Detected internal language_model in {model_type_name}, redirecting...")
        return get_model_layers(model.language_model)

    if 'BeeForConditionalGeneration' in model_type_name:
        logging.info("Detected Bee multimodal model, accessing internal language_model...")
        return get_model_layers(model.language_model)

    layers = []
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
        layers.extend(model.encoder.block)
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
        layers.extend(model.decoder.block)
    if layers:
        return layers

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return model.model.decoder.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
        return model.transformer.layer
    elif hasattr(model, 'layers'):
        return model.layers

    if hasattr(model, 'vit') and hasattr(model.vit, 'encoder') and hasattr(model.vit.encoder, 'layer'):
        return model.vit.encoder.layer
    elif model_type_name == 'ViTModel' and hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return model.encoder.layer

    raise TypeError(f"Unrecognized model structure type: {model_type_name}")


def calculate_fingerprint(model):
    """
    Compute enhanced fingerprint (std, skew, kurtosis) using a universal
    attention-weight extraction parser with multi-schema support.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced fingerprint computation (std, skew, kurtosis)...")

    try:
        layers = get_model_layers(model)
        stats = {comp: {'std': [], 'skew': [], 'kurtosis': []} for comp in ['q', 'k', 'v', 'o']}

        name_schemas = [
            {'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj', 'o': 'o_proj'},
            {'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj', 'o': 'out_proj'},
            {'q': 'query',  'k': 'key',    'v': 'value',  'o': 'dense'},
            {'q': 'query',  'k': 'key',    'v': 'value',  'o': 'output'},
            {'q': 'q',      'k': 'k',      'v': 'v',      'o': 'o'},
        ]

        for layer in layers:
            weights = {}
            found_schema = False

            attn_module = None
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'attention'):
                attn_module = layer.attention.attention
            elif hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
            elif hasattr(layer, 'attn'):
                attn_module = layer.attn
            elif hasattr(layer, 'attention'):
                attn_module = layer.attention
            elif hasattr(layer, 'layer') and hasattr(layer.layer[0], 'SelfAttention'):
                attn_module = layer.layer[0].SelfAttention
            else:
                continue

            for schema in name_schemas:
                try:
                    if hasattr(attn_module, 'c_attn') and hasattr(attn_module, 'c_proj'):
                        q, k, v = torch.chunk(attn_module.c_attn.weight, 3, dim=0)
                        weights = {'q': q, 'k': k, 'v': v, 'o': attn_module.c_proj.weight}
                    elif schema['o'] == 'dense' and hasattr(attn_module, 'output') and hasattr(attn_module.output, 'dense'):
                        weights = {
                            'q': getattr(attn_module, schema['q']).weight,
                            'k': getattr(attn_module, schema['k']).weight,
                            'v': getattr(attn_module, schema['v']).weight,
                            'o': attn_module.output.dense.weight
                        }
                    else:
                        weights = {key: getattr(attn_module, name).weight for key, name in schema.items()}

                    found_schema = True
                    break
                except AttributeError:
                    continue

            if not found_schema:
                logging.warning(
                    f"No known attention weight schema found in layer {type(layer).__name__} "
                    f"(attention module: {type(attn_module).__name__}), skipping."
                )
                continue

            for key, weight_tensor in weights.items():
                w_flat = weight_tensor.float().detach().cpu().numpy().flatten()
                stats[key]['std'].append(np.std(w_flat))
                stats[key]['skew'].append(skew(w_flat))
                stats[key]['kurtosis'].append(kurtosis(w_flat))

        fingerprints = {}
        for comp, comp_stats in stats.items():
            fingerprints[comp] = {}
            for feat, stat_list in comp_stats.items():
                if not stat_list:
                    raise ValueError(f"Failed to extract feature '{comp}/{feat}'.")
                tensor = torch.tensor(stat_list, dtype=torch.float32)
                mean, std = torch.mean(tensor), torch.std(tensor)
                fingerprints[comp][feat] = (tensor - mean) / std if std > 1e-9 else tensor - mean

        logger.info("Enhanced fingerprint computation successful.")
        return fingerprints

    except Exception as e:
        logger.error(f"Error during fingerprint computation: {e}", exc_info=True)
        return None


def main():
    """Main pipeline: load model list from JSON, generate fingerprints with dual fallback logic"""
    setup_logging()
    os.makedirs(FINGERPRINT_CACHE_PATH, exist_ok=True)

    if not os.path.exists(INPUT_JSON_FILE):
        logging.error(f"Model input file not found: {INPUT_JSON_FILE}")
        return

    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model_list_from_file = data.get("models", [])
        if not model_list_from_file:
            logging.warning("'models' list in JSON is empty.")
            return

        models_to_process = sorted(list(set(model_list_from_file)))
        logging.info(f"Loaded {len(models_to_process)} models from JSON.")
    except Exception as e:
        logging.error(f"Failed to parse JSON file: {e}")
        return

    quantized_models = set()
    if os.path.exists(QUANTIZED_MODELS_CSV):
        try:
            df = pd.read_csv(QUANTIZED_MODELS_CSV, header=None)
            quantized_models = set(df[0].tolist())
            logging.info(f"Loaded {len(quantized_models)} quantized models.")
        except Exception as e:
            logging.error(f"Failed to read '{QUANTIZED_MODELS_CSV}': {e}.")
    else:
        logging.warning(f"Quantized model list file not found: '{QUANTIZED_MODELS_CSV}'.")

    for model_name in tqdm(models_to_process, desc="Generating fingerprints"):
        cache_file_name = f"{model_name.replace('/', '__')}.pt"
        cache_file_path = os.path.join(FINGERPRINT_CACHE_PATH, cache_file_name)

        if os.path.exists(cache_file_path):
            logging.debug(f"Fingerprint for '{model_name}' already exists, skipping.")
            continue

        logging.info(f"--- Processing model: {model_name} ---")

        fingerprint = None
        model_dir_name = model_name.replace('/', '_')

        is_lora = os.path.exists(os.path.join(MODEL_BASE_PATH, model_dir_name, 'adapter_config.json'))
        is_quantized = model_name in quantized_models

        paths_to_try = []

        if is_lora:
            try:
                base_model_identifier = get_lora_base_model(os.path.join(MODEL_BASE_PATH, model_dir_name))
                base_model_path = base_model_identifier
                if not os.path.isdir(base_model_path):
                    base_model_path = os.path.join(MODEL_BASE_PATH, base_model_identifier.replace('/', '_'))
                paths_to_try.append(('lora', base_model_path, model_name))
            except Exception as e:
                logging.error(f"Failed to resolve base model for LoRA '{model_name}': {e}")

        elif is_quantized:
            paths_to_try.append(('full', os.path.join(DEQUANTIZED_MODEL_PATH, model_dir_name), model_name))
            paths_to_try.append(('full', os.path.join(MODEL_BASE_PATH, model_dir_name), model_name))
        else:
            paths_to_try.append(('full', os.path.join(MODEL_BASE_PATH, model_dir_name), model_name))
            paths_to_try.append(('full', os.path.join(DEQUANTIZED_MODEL_PATH, model_dir_name), model_name))

        for model_type, path, log_name in paths_to_try:
            try:
                model, tokenizer = None, None

                if model_type == 'lora':
                    base_model_name_for_log = os.path.basename(path).replace('_', '/', 1)
                    model, tokenizer = load_model_and_tokenizer(path, base_model_name_for_log)
                    adapter_path = os.path.join(MODEL_BASE_PATH, model_name.replace('/', '_'))
                    model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
                else:
                    model, tokenizer = load_model_and_tokenizer(path, log_name)

                if model:
                    fingerprint = calculate_fingerprint(model)

                del model, tokenizer
                torch.cuda.empty_cache()

                if fingerprint is not None:
                    logging.info(f"Fingerprint successfully generated from '{path}'.")
                    break

            except Exception as e:
                logging.warning(f"Failed on path '{path}': {e}. Trying next fallback.")
                torch.cuda.empty_cache()

        if fingerprint:
            torch.save(fingerprint, cache_file_path)
            logging.info(f"Fingerprint saved: {model_name}")
        else:
            logging.error(f"All paths failed. Fingerprint generation failed for model '{model_name}'.")

    logging.info("All models processed.")


if __name__ == "__main__":
    main()