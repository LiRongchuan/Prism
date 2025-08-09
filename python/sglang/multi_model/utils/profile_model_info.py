# profile model info including model size, kv cache cell size, etc.
import gc
import json
import os
import time

import torch
from vllm.config import DeviceConfig, LoadConfig
from vllm.config import ModelConfig as VllmModelConfig
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.model_executor.model_loader import get_model

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.utils import (
    get_available_gpu_memory,
    monkey_patch_vllm_dummy_weight_loader,
)


def init_torch_distributed():
    # This can reduce thread conflicts and speed up weight loading.
    torch.set_num_threads(1)
    dist_init_method = f"tcp://127.0.0.1:6444"
    init_distributed_environment(
        backend="nccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=dist_init_method,
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


def load_model(model_path):
    # Prepare the vllm model config
    available_memory_start = get_available_gpu_memory(device="cuda", gpu_id=0)
    monkey_patch_vllm_dummy_weight_loader()
    load_config = LoadConfig(load_format="auto")
    vllm_model_config = VllmModelConfig(
        model=model_path,
        quantization=None,
        tokenizer=None,
        tokenizer_mode=None,
        trust_remote_code=True,
        dtype="auto",
        seed=42,
        skip_tokenizer_init=True,
    )

    dtype = vllm_model_config.dtype

    tic = time.time()
    available_mem_before_load = get_available_gpu_memory(device="cuda", gpu_id=0)
    # Load the model
    model = get_model(
        model_config=vllm_model_config,
        load_config=load_config,
        device_config=DeviceConfig(device="cuda"),
        parallel_config=None,
        scheduler_config=None,
        lora_config=None,
        cache_config=None,
    )

    available_mem_after_load = get_available_gpu_memory(device="cuda", gpu_id=0)
    model_size = available_mem_before_load - available_mem_after_load
    print(
        f"Load {model_path} end. Time cost: {time.time() - tic:.4f}s. Model size: {model_size:.2f} GB"
    )
    print(
        f"model_size computed from start: {available_memory_start - available_mem_after_load:.2f} GB"
    )
    del model
    clean_up()
    return model_size, dtype


def clean_up():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_cell_size(model_path, dtype):
    model_config = ModelConfig(
        name=None,
        path=model_path,
    )
    tp_size = 1
    cell_size = (
        model_config.get_num_kv_heads(tp_size)
        * model_config.head_dim
        * model_config.num_hidden_layers
        * 2
        * torch._utils._element_size(dtype)
    )
    print(
        f"Cell size: {cell_size:.2f} bytes, {cell_size / 1024 ** 2:.2f} MB. num_kv_heads: {model_config.get_num_kv_heads(tp_size)}, head_dim: {model_config.head_dim}, num_hidden_layers: {model_config.num_hidden_layers}. dtype: {dtype}"
    )
    return cell_size


if __name__ == "__main__":
    save_path = "model_info.json"
    skip_load = False
    save = True

    if os.path.exists(save_path) and skip_load:
        with open(save_path, "r") as f:
            model_info = json.load(f)
    else:
        model_info = {}
    # model paths to profile
    model_paths = [
        "meta-llama/Llama-3.2-1B",  # 1.24B
        "google/gemma-2-2b-it",  # 2.61B
        "meta-llama/Llama-3.2-3B",  # 3.21B
        "mistralai/Mistral-7B-Instruct-v0.2",  # 7.24B
        "meta-llama/Meta-Llama-3.1-8B",  # 8.03B
        "google/gemma-2-9b-it",  # 9.24B
        "mistralai/Mistral-Nemo-Instruct-2407",  # 12.2B
        "meta-llama/Llama-2-7b-hf",
    ]
    init_torch_distributed()
    for model_path in model_paths:
        if model_path in model_info:
            continue
        clean_up()
        model_size, dtype = load_model(model_path)
        cell_size = get_cell_size(model_path, dtype)
        model_info[model_path] = {
            "model_size": model_size,
            "cell_size": cell_size,
        }

    # save the model info
    if save:
        with open(save_path, "w") as f:
            json.dump(model_info, f)
