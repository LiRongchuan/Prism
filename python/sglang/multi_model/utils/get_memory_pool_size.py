import json
import os
from typing import Dict, List

import torch


def get_static_memory_pool_size(
    model_names_to_req_rates,
    model_names_to_model_paths=None,
    total_gpu_mem=None,
    mem_frac=0.85,
):
    model_info = load_model_info()
    if not model_names_to_model_paths:
        model_names_to_model_paths = {
            model_name: model_name for model_name in model_names_to_req_rates
        }

    if not total_gpu_mem:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        total_gpu_mem = gpu_memory / (1024**3)
        print(f"Total GPU Memory: {total_gpu_mem:.2f} GB")

    model_paths = set(model_names_to_model_paths.values())
    for model_path in model_paths:
        assert (
            model_path in model_info
        ), f"Model {model_path} not found in model_info. Please re-generate model_info.json with profile_model_info.py"

    kv_cache_ratios = []
    total_model_size = 0
    for model_name in model_names_to_req_rates:
        req_rate = model_names_to_req_rates[model_name]
        cell_size = model_info[model_names_to_model_paths[model_name]]["cell_size"]
        kv_cache_ratios.append(req_rate * cell_size)
        model_path = model_names_to_model_paths[model_name]
        total_model_size += model_info[model_path]["model_size"]
    total_kv_cache_ratio = sum(kv_cache_ratios)
    kv_cache_fracs = [kv / total_kv_cache_ratio for kv in kv_cache_ratios]

    total_kv_cache_mem = total_gpu_mem * mem_frac - total_model_size
    print(
        f"Max Mem Usage: {total_gpu_mem * mem_frac:.2f} GB. Total KV Cache Memory: {total_kv_cache_mem:.2f} GB. Total Model Size: {total_model_size:.2f} GB"
    )

    model_name_to_kv_pool_size = {
        model_name: kv_cache_fracs[i] * total_kv_cache_mem
        for i, model_name in enumerate(model_names_to_req_rates)
    }
    return model_name_to_kv_pool_size


def load_model_info():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_info_path = os.path.join(cur_dir, "model_info.json")
    with open(model_info_path, "r") as f:
        model_info = json.load(f)
    return model_info


def get_model_path_to_cell_size(model_paths: List[str]) -> Dict[str, int]:
    """Get the mapping between model paths and their cell sizes.

    Args:
        model_paths: A list of model paths.

    Returns:
        A dictionary mapping model paths to their cell sizes in bytes.
    """
    model_info = load_model_info()

    model_path_to_cell_size = {}
    for model_path in model_paths:
        if model_path not in model_info:
            raise ValueError(
                f"Model path {model_path} not found in the profiled model info file."
            )
        model_path_to_cell_size[model_path] = model_info[model_path]["cell_size"]

    return model_path_to_cell_size


def get_model_path_to_model_size(model_paths: List[str]) -> Dict[str, int]:
    model_info = load_model_info()
    model_path_to_model_size = {}
    for model_path in model_paths:
        if model_path not in model_info:
            raise ValueError(
                f"Model path {model_path} not found in the profiled model info file."
            )
        model_path_to_model_size[model_path] = model_info[model_path]["model_size"]
    return model_path_to_model_size


if __name__ == "__main__":
    # model_paths_to_req_rates = {
    #     "meta-llama/Llama-3.2-3B": 100,
    #     "mistralai/Mistral-Nemo-Instruct-2407": 100,
    # }

    model_names_to_model_paths = {
        "model_1": "meta-llama/Llama-3.1-8B",
        "model_2": "mistralai/Mistral-7B-Instruct-v0.2",
        "model_3": "meta-llama/Llama-3.2-3B",
        "model_4": "meta-llama/Llama-3.2-3B",
    }
    model_names_to_req_rates = {
        "model_1": 199,
        "model_2": 262,
        "model_3": 22,
        "model_4": 31,
    }

    model_name_to_kv_pool_size = get_static_memory_pool_size(
        model_names_to_model_paths=model_names_to_model_paths,
        model_names_to_req_rates=model_names_to_req_rates,
        mem_frac=0.85,
    )
    print(model_name_to_kv_pool_size)
