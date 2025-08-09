import json

import torch
from vllm.config import DeviceConfig, LoadConfig
from vllm.config import ModelConfig as VllmModelConfig
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.model_loader import get_model

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_port_available, monkey_patch_vllm_dummy_weight_loader


def init_torch_distributed_tp_1(device="cpu"):
    tp_size = 1
    tp_rank = 0
    local_rank = 0

    if device == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"

    port = 29999
    while True:
        if is_port_available(port):
            break
        port -= 1

    dist_init_method = f"tcp://127.0.0.1:{port}"
    init_distributed_environment(
        backend=backend,
        world_size=tp_size,
        rank=tp_rank,
        local_rank=local_rank,
        distributed_init_method=dist_init_method,
    )
    initialize_model_parallel(tensor_model_parallel_size=tp_size)


def destroy_torch_distributed_tp_1():
    destroy_model_parallel()
    destroy_distributed_environment()


def monkey_patch_dist_tp_size(tp_size: int):
    """
    Monkey patch the get world size method in dist module.
    """

    import vllm.distributed.parallel_state as dist

    setattr(dist._TP, "world_size", tp_size)


def monkey_patch_dist_tp_rank(tp_rank: int):
    """
    Monkey patch the get tp rank method in dist module.
    """

    import vllm.distributed.parallel_state as dist

    setattr(dist._TP, "rank_in_group", tp_rank)


def load_shared_cpu_model(server_args: ServerArgs):
    model_config = ModelConfig(
        server_args.model_name,
        server_args.model_path,
        server_args.trust_remote_code,
        context_length=server_args.context_length,
        model_override_args=json.loads(server_args.json_model_override_args),
    )

    # need distributed environment to load model in vllm
    torch.set_num_threads(1)

    # Prepare the vllm model config
    monkey_patch_vllm_dummy_weight_loader()
    load_config = LoadConfig(load_format=server_args.load_format)
    vllm_model_config = VllmModelConfig(
        model=server_args.model_path,
        quantization=server_args.quantization,
        tokenizer=None,
        tokenizer_mode=None,
        trust_remote_code=server_args.trust_remote_code,
        dtype=server_args.dtype,
        seed=server_args.random_seed,
        skip_tokenizer_init=True,
    )
    if model_config.model_override_args is not None:
        vllm_model_config.hf_config.update(model_config.model_override_args)

    monkey_patch_dist_tp_size(server_args.tp_size)
    models = []
    # Load models for different TP ranks
    for tp_rank in range(server_args.tp_size):
        monkey_patch_dist_tp_rank(tp_rank)
        model = get_model(
            model_config=vllm_model_config,
            load_config=load_config,
            device_config=DeviceConfig(device="cpu"),
            parallel_config=None,
            scheduler_config=None,
            lora_config=None,
            cache_config=None,
        )

        # make the model pickleable. (PP relavant params)
        model.make_empty_intermediate_tensors = None
        model.model.make_empty_intermediate_tensors = None
        model.share_memory()
        models.append(model)
    return models
