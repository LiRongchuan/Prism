from collections import Counter, defaultdict
from trace import TraceConfig, generate_synthetic_reqs

import numpy as np

MODEL_TO_SIZE = {
    "meta-llama/Llama-2-7b-chat-hf": 3.5,
    "meta-llama/Llama-2-7b-hf": 9.5,
    "mistralai/Mistral-Nemo-Instruct-2407": 23.1,
    "meta-llama/Llama-3.2-3B": 6.78,
    "google/gemma-2-9b-it": 17.39,
    "mistralai/Mistral-7B-Instruct-v0.2": 13.51,
}

MODEL_TO_KV_TOKEN_SIZE = {
    "mistralai/Mistral-Nemo-Instruct-2407": 51200,
    "google/gemma-2-9b-it": 75264,
    "mistralai/Mistral-7B-Instruct-v0.2": 32768,
    "meta-llama/Llama-3.2-3B": 28672,
}


def compute_model_request_ratios(num_models, alpha):
    ratios = []

    for i in range(num_models):
        current_ratio = (float(i + 1) / num_models) ** alpha
        if i > 0:
            # Subtract the cumulative sum of previous ratios
            current_ratio -= sum(ratios)
        ratios.append(current_ratio)

    # Ensure ratios sum up to 1
    total_sum = sum(ratios)
    ratios = [r / total_sum for r in ratios]

    return ratios


def compute_memory_frac_per_gpu(
    model_names,
    req_ratios,
    total_memory=78,
    mem_frac=0.85,
    static=True,
):
    model_weights = [MODEL_TO_SIZE[model_name] for model_name in model_names]
    model_kv_token_sizes = [
        MODEL_TO_KV_TOKEN_SIZE[model_name] for model_name in model_names
    ]

    total_memory = total_memory
    total_kv_cache_memory = total_memory * mem_frac - sum(model_weights)
    # print(f"Total KV Cache Memory: {total_kv_cache_memory}, model_weights: {sum(model_weights)}\n")

    num_models = len(model_weights)
    kv_cache_ratios = [r * s for r, s in zip(req_ratios, model_kv_token_sizes)]
    total_kv_cache_ratio = sum(kv_cache_ratios)
    kv_cache_fracs = [kv / total_kv_cache_ratio for kv in kv_cache_ratios]

    model_memory = [
        model_weights[i] + kv_cache_fracs[i] * total_kv_cache_memory
        for i in range(num_models)
    ]
    # for model_name, memory in zip(model_names, model_memory):
    # print(f"Model: {model_name}, Memory: {memory}, model_weights: {MODEL_TO_SIZE[model_name]}, kv_cache_fracs: {kv_cache_fracs[model_names.index(model_name)]}\n")

    if static:
        model_memory_fracs = []
        # next model memory frac is based on the rest of memory after all previous models are placed
        for i, m in enumerate(model_memory):
            if i == 0:
                model_memory_fracs.append(m / total_memory)
            else:
                previous_model_memory = sum(model_memory[:i])
                remaining_memory = total_memory - previous_model_memory
                model_memory_fracs.append(m / remaining_memory)

    else:
        model_memory_fracs = [m / total_memory for m in model_memory]

    return model_memory_fracs


def get_collocate_memory_frac(
    model_names,
    placements,
    trace_config,
    total_memory=78,
    mem_frac=0.85,
):
    requests = generate_synthetic_reqs(trace_config)
    model_requests = Counter()
    for req in requests:
        model_requests[req.model] += 1

    mem_frac_each_model = {}
    gpu_to_model_requests = defaultdict(dict)
    for model, placement in zip(model_names, placements):
        gpu_to_model_requests[placement][model] = model_requests[model]
    for gpu_id in gpu_to_model_requests:
        model_request_this_gpu = gpu_to_model_requests[gpu_id]
        model_names_this_gpu = list(model_request_this_gpu.keys())
        req_counts = list(model_request_this_gpu.values())
        req_ratios = [req / sum(req_counts) for req in req_counts]
        # print(f"GPU: {gpu_id}, Model: {model_names_this_gpu}, Request Ratios: {req_ratios}")
        memory_fracs_this_gpu = compute_memory_frac_per_gpu(
            model_names_this_gpu, req_ratios, total_memory, mem_frac
        )
        for model, memory_frac in zip(model_names_this_gpu, memory_fracs_this_gpu):
            mem_frac_each_model[model] = memory_frac
    mem_fracs = [mem_frac_each_model[model] for model in model_names]
    return mem_fracs


if __name__ == "__main__":
    n_models = 4
    alpha = 1.7
    req_rate = 16
    cv = 1
    seed = 42
    on_off_ratio = 0.5
    models = [
        "mistralai/Mistral-Nemo-Instruct-2407",
        "meta-llama/Llama-3.2-3B",
        "google/gemma-2-9b-it",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ]
    trace_config = TraceConfig(
        req_rate=req_rate,
        duration=60,
        input_range=[8, 1024],
        output_range=[8, 512],
        model_paths=models,
        seed=seed,
        alpha=alpha,
        cv=cv,
        on_off_cycle_len=30,
        on_off_ratio=on_off_ratio,
    )
    placements = [0, 0, 1, 1]
    print(
        f"Req_rate = {req_rate}, alpha = {alpha}, cv = {cv}, on_off_ratio = {on_off_ratio}"
    )
    memory_fracs = get_collocate_memory_frac(models, placements, trace_config)
    print("\nMemory Fractions:")
    for model, memory_frac in zip(models, memory_fracs):
        print(f"{model}: {memory_frac}")
