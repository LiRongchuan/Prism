"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""ModelRunner runs the forward passes of the models."""
import math
import atexit
import copy
import dataclasses
import gc
import glob
import importlib
import importlib.resources
import json
import logging
import multiprocessing.shared_memory as shared_memory
import os
import pkgutil
import signal
import time
from functools import lru_cache
from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from tensordict import TensorDict
from vllm.config import DeviceConfig, LoadConfig
from vllm.config import ModelConfig as VllmModelConfig
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import ModelRegistry

from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.constrained import disable_cache
from sglang.srt.layers.attention.double_sparsity_backend import DoubleSparseAttnBackend
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.managers.io_struct import MemoryUsage
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.mem_cache.memory_pool import (
    DoubleSparseTokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    enable_show_time_cost,
    get_available_gpu_memory,
    is_attention_free_model,
    is_embedding_model,
    is_generation_model,
    is_multimodal_model,
    model_has_inner_state,
    monkey_patch_vllm_dummy_weight_loader,
    monkey_patch_vllm_p2p_access_check,
)

logger = logging.getLogger(__name__)

import io
import pickle

from torch.multiprocessing.queue import ForkingPickler


@dataclasses.dataclass
class MemoryPoolInfo:
    memory_pool_memory: float
    req_to_token_pool_memory: float
    token_to_kv_pool_memory: float

    def to_dict(self):
        return dataclasses.asdict(self)


class BaseModelRunner:
    """Base class for model runners with common functionality."""

    def __init__(
        self,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        shared_cpu_models: Dict[
            str, List[nn.Module]
        ],  # model name -> list of different ranks of shared cpu models
        engine_id: Optional[str] = None,
        input_queue: Optional[torch.multiprocessing.Queue] = None,
        output_queue: Optional[torch.multiprocessing.Queue] = None,
    ):
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dist_port = nccl_port
        self.server_args = server_args
        self.shared_cpu_models = shared_cpu_models
        self.is_multimodal_model = False
        self.enable_elastic_memory = server_args.enable_elastic_memory
        self.engine_id = engine_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.use_model_service = input_queue is not None
        if tp_size > 1:
            logger.warning(
                "Tensor parallelism is enabled, model service will not be used."
            )
            self.use_model_service = False
        self._init_shared_memory()

        # Global vars
        if server_args.show_time_cost:
            enable_show_time_cost()
        if server_args.disable_disk_cache:
            disable_cache()

        global_server_args_dict.update(
            {
                "attention_backend": server_args.attention_backend,
                "sampling_backend": server_args.sampling_backend,
                "triton_attention_reduce_in_fp32": server_args.triton_attention_reduce_in_fp32,
                "disable_mla": server_args.disable_mla,
                "torchao_config": server_args.torchao_config,
                "disable_penalizer": server_args.disable_penalizer,
                "disable_nan_detection": server_args.disable_nan_detection,
            }
        )

        # Init components
        min_per_gpu_memory = self.init_torch_distributed()
        self.max_mem_usage = self._get_max_mem_usage(
            min_per_gpu_memory, server_args.max_mem_usage
        )
        self.min_reserve_mem = (
            self.max_mem_usage * (1 - server_args.mem_fraction_static) * 0.8
        )  # leave room for other model's activations
        self.sampler = Sampler()

        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _set_device(self):
        if self.device == "cuda":
            torch.cuda.set_device(self.gpu_id)
        elif self.device == "xpu":
            torch.xpu.set_device(self.gpu_id)

    def _init_shared_memory(self):
        assert hasattr(
            self, "ipc_name"
        ), "ipc_name must be set before initializing shared memory"
        try:
            # First try to clean up any existing shared memory with this name
            try:
                existing_shm = shared_memory.SharedMemory(name=self.ipc_name)
                existing_shm.close()
                existing_shm.unlink()
                logger.info(f"Cleaned up existing shared memory: {self.ipc_name}")
            except FileNotFoundError:
                # This is expected if the shared memory doesn't exist yet
                pass

            # Now create the new shared memory
            self.shm = shared_memory.SharedMemory(
                create=True, size=8, name=self.ipc_name
            )
        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            self.shm = None

    def init_torch_distributed(self):
        logger.info("Init torch distributed begin.")
        # Init torch distributed
        if self.device == "cuda":
            torch.cuda.set_device(self.gpu_id)
            backend = "nccl"
        # ToDO(liangan1):Just use gloo to bypass the initilization fail
        # Need to use xccl for xpu backend in the future
        elif self.device == "xpu":
            torch.xpu.set_device(self.gpu_id)
            backend = "gloo"

        if not self.server_args.enable_p2p_check:
            monkey_patch_vllm_p2p_access_check(self.gpu_id)
        if self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        init_distributed_environment(
            backend=backend,
            world_size=self.tp_size,
            rank=self.tp_rank,
            local_rank=self.gpu_id,
            distributed_init_method=dist_init_method,
        )
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        min_per_gpu_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.tp_size > 1
        )
        self.tp_group = get_tp_group()

        # Currently, there is a bug with mulit-node tensor parallelsim + padded cuda graph,
        # so we disable padding in cuda graph.
        if self.device == "cuda" and not all(
            in_the_same_node_as(self.tp_group.cpu_group, source_rank=0)
        ):
            self.server_args.disable_cuda_graph_padding = True
            logger.info(
                "Setting disable_cuda_graph_padding to True because of multi-node tensor parallelism."
            )

        # Check memory for tensor parallelism
        # if self.tp_size > 1:
        #     local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        #     if min_per_gpu_memory < local_gpu_memory * 0.9:
        #         raise ValueError(
        #             f"The memory capacity is unbalanced. Some GPUs may be occupied by other processes. min_per_gpu_memory={min_per_gpu_memory:.2f} GB, local_gpu_memory={local_gpu_memory:.2f} GB"
        #         )

        return min_per_gpu_memory

    def delete_gpu_model(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

    def destroy(self):
        destroy_model_parallel()
        destroy_distributed_environment()

    def _get_cpu_model_ref(self, model_name: str):
        model_path = self.model_config.path
        model_key = (model_path, self.tp_size)
        if model_key in self.shared_cpu_models:
            return self.shared_cpu_models[model_key][self.tp_rank]
        else:
            raise ValueError(
                "Model {} not found in shared cpu models".format(model_name)
            )

    def load_model_from_cpu_model_ref(self):
        logger.info(
            f"Load weight from CPU model ref begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
        tic = time.time()
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(self.cpu_model_ref)

        self.model = pickle.loads(buf.getvalue())
        state_dict_host = TensorDict(self.model.state_dict())
        state_dict_device = state_dict_host.to(
            self.device, non_blocking_pin=True, num_threads=4
        )
        self.model.load_state_dict(state_dict_device, assign=True)

        logger.info(
            f"Load weight from CPU model ref end. It takes {time.time() - tic:.4f}s. "
        )
        self.sliding_window_size = (
            self.model.get_attention_sliding_window_size()
            if hasattr(self.model, "get_attention_sliding_window_size")
            else None
        )
        self.is_generation = is_generation_model(
            self.model_config.hf_config.architectures, self.server_args.is_embedding
        )
        # TODO: find a better way to get dtype
        # self.dtype = self.model.parameters().__next__().dtype
        self.load_config = LoadConfig(load_format=self.server_args.load_format)
        vllm_model_config = VllmModelConfig(
            model=self.server_args.model_path,
            quantization=self.server_args.quantization,
            tokenizer=None,
            tokenizer_mode=None,
            trust_remote_code=self.server_args.trust_remote_code,
            dtype=self.server_args.dtype,
            seed=self.server_args.random_seed,
            skip_tokenizer_init=True,
        )
        if self.model_config.model_override_args is not None:
            vllm_model_config.hf_config.update(self.model_config.model_override_args)
        self.dtype = vllm_model_config.dtype

    def update_weights(self, model_path: str, load_format: str):
        """Update weights in-place."""
        from vllm.model_executor.model_loader.loader import (
            DefaultModelLoader,
            device_loading_context,
            get_model_loader,
        )
        from vllm.model_executor.model_loader.utils import set_default_torch_dtype

        logger.info(
            f"Update weights begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        target_device = torch.device(self.device)

        try:
            # TODO: Use a better method to check this
            vllm_model_config = VllmModelConfig(
                model=model_path,
                quantization=self.server_args.quantization,
                tokenizer=None,
                tokenizer_mode=None,
                trust_remote_code=self.server_args.trust_remote_code,
                dtype=self.server_args.dtype,
                seed=self.server_args.random_seed,
                skip_tokenizer_init=True,
            )
        except Exception as e:
            message = f"Failed to load model config: {e}."
            return False, message

        load_config = LoadConfig(load_format=load_format)

        # Only support vllm DefaultModelLoader for now
        loader = get_model_loader(load_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source(
                    config.model,
                    revision=config.revision,
                    fall_back_to_pt=getattr(
                        self.model, "fall_back_to_pt_during_load", True
                    ),
                )
            )
            return iter

        def model_load_weights(model, iter):
            model.load_weights(iter)
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
            return model

        with set_default_torch_dtype(vllm_model_config.dtype):
            try:
                iter = get_weight_iter(vllm_model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.vllm_model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.vllm_model_config = vllm_model_config
        self.load_config = load_config
        self.model_config.path = model_path

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            lora_paths=self.server_args.lora_paths,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
        )
        logger.info("LoRA manager ready.")

    def _get_cell_size(self):
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not self.server_args.disable_mla
        ):
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * self.model_config.num_hidden_layers
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        else:
            cell_size = (
                self.model_config.get_num_kv_heads(self.tp_size)
                * self.model_config.head_dim
                * self.model_config.num_hidden_layers
                * 2
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        return cell_size

    def _get_max_mem_usage(
        self, min_per_gpu_memory: int, max_mem_usage: Optional[float] = None
    ):
        if max_mem_usage is not None:
            return max_mem_usage
        else:
            logger.info(
                f"Using mem_fraction_static: {self.server_args.mem_fraction_static} and min_per_gpu_memory: {min_per_gpu_memory:.2f} GB"
            )
            return min_per_gpu_memory * self.server_args.mem_fraction_static

    def profile_max_num_token(self, total_gpu_memory: int):
        available_gpu_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.tp_size > 1
        )
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        logger.info(
            f"avail mem={available_gpu_memory:.2f} GB, "
            f"total mem={total_gpu_memory:.2f} GB, "
            f"rest mem={rest_memory:.2f} GB, "
            f"cell_size={self.cell_size} bytes"
        )
        max_num_token = int(rest_memory * (1 << 30) // self.cell_size)
        return max_num_token

    def _get_max_total_num_tokens(self, memory_pool_size: float):
        """使用预估请求数，计算可运行token数"""
        # TODO: 针对模型修改最大请求
        approx_max_num_reqs = self.server_args.max_num_reqs
        approx_max_num_reqs = 512 if approx_max_num_reqs is None else approx_max_num_reqs
        approx_req_to_token_pool_size = (
            (approx_max_num_reqs + 1)                   # 请求数量
            * (self.model_config.context_len + 4)       # 请求长度
            * torch._utils._element_size(torch.int32)   # 元素大小
            / (1 << 30)                                 # 单位GB
        )
        logger.info(f"Max context length: {self.model_config.context_len}, request to token size: {approx_req_to_token_pool_size} GB")
        token_to_kv_pool_size = memory_pool_size - approx_req_to_token_pool_size

        max_num_token = int(token_to_kv_pool_size * (1 << 30) // self.cell_size) # cell_size：每token占用cache大小
        max_total_tokens = self.server_args.max_total_tokens
        if max_total_tokens is not None:
            if max_total_tokens > max_num_token:
                logger.warning(
                    f"max_total_tokens={max_total_tokens} in server_args is larger than the profiled value "
                    f"{max_num_token}. Use the profiled value instead."
                )
            max_num_token = min(max_num_token, max_total_tokens)
        if max_num_token <= 0:
            # raise RuntimeError(
            #     "Not enough memory. Please try to increase --mem-fraction-static or --max-memory-pool-size or --max-mem-usage."
            # )
            logger.warning("Not enough memory. Please try to increase --max-memory-pool-size or decrease --max-num-reqs")
            # TODO: 修改比例
            max_num_token = int(token_to_kv_pool_size * 0.5 * (1 << 30) // self.cell_size)
        return max_num_token

    def get_memory_usage(self):
        total_used_memory = torch.cuda.memory_allocated(device=self.device) / (1 << 30)
        memory_usage = MemoryUsage(
            total_used_memory=total_used_memory,
            model_weights_memory=self.model_gpu_mem_usage,
            **self.memory_pool_info.to_dict(),
        )
        return memory_usage

    def _get_kv_cache_dtype(self):
        if self.server_args.kv_cache_dtype == "auto":
            kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            kv_cache_dtype = torch.float8_e5m2
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )
        return kv_cache_dtype

    def _init_memory_pool(
        self,
        max_num_reqs: int,
        max_total_num_tokens: int,
        init_req_to_token_only: bool = False,
        max_context_len: Optional[int] = None,
        max_alloc_num_tokens: Optional[int] = None
    ):
        tic = time.time()
        if max_context_len is None:
            max_context_len = self.model_config.context_len + 4
        memory_allocated_start = torch.cuda.memory_allocated()
        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs + 1,
            max_context_len=max_context_len,
            device=self.device,
            gpu_id=self.gpu_id,
            use_records=False,
            min_reserve_mem=self.min_reserve_mem,
        )
        mem_allocated_after_req_to_token_pool = torch.cuda.memory_allocated()
        req_to_token_pool_memory = (
            mem_allocated_after_req_to_token_pool - memory_allocated_start
        ) / (1 << 30)
        if init_req_to_token_only:
            logger.info(
                f"Create req_to_token_pool end. Time cost: {time.time() - tic:.4f}s. "
                f"req_to_token_pool memory={req_to_token_pool_memory:.2f} GB"
            )
            memory_pool_info = MemoryPoolInfo(
                req_to_token_pool_memory, req_to_token_pool_memory, 0
            )
            return memory_pool_info
        if (self.model_config.attention_arch == AttentionArch.MLA and not self.server_args.disable_mla):
            self.token_to_kv_pool = MLATokenToKVPool(
                max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                gpu_id=self.gpu_id,
                min_reserve_mem=self.min_reserve_mem,
            )
        elif self.server_args.enable_double_sparsity:
            self.token_to_kv_pool = DoubleSparseTokenToKVPool(
                max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(self.tp_size),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                gpu_id=self.gpu_id,
                heavy_channel_num=self.server_args.ds_heavy_channel_num,
                min_reserve_mem=self.min_reserve_mem,
            )
        else:
            # NOTE: 分离了初始分配和初始管理器
            self.token_to_kv_pool = MHATokenToKVPool(
                size=max_total_num_tokens,
                max_size=max_alloc_num_tokens if max_alloc_num_tokens is not None else max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(self.tp_size),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                gpu_id=self.gpu_id,
                model_name=self.model_name,
                enable_elastic_memory=self.server_args.enable_elastic_memory,
                min_reserve_mem=self.min_reserve_mem,
                enable_overlap=self.server_args.enable_overlap_schedule,
                use_kvcached_v0=self.server_args.use_kvcached_v0,
                enable_worker_pool=self.server_args.enable_worker_pool,
                shm=self.shm,
            )
        memory_allocated_end = torch.cuda.memory_allocated()
        memory_pool_memory = (memory_allocated_end - memory_allocated_start) / (1 << 30)
        token_to_kv_pool_memory = (
            memory_allocated_end - mem_allocated_after_req_to_token_pool
        ) / (1 << 30)
        logger.info(
            f"Create memory pool end. Time cost: {time.time() - tic:.4f}s. "
            f"total memory pool memory={memory_pool_memory:.2f} GB, "
            f"req_to_token_pool memory={req_to_token_pool_memory:.2f} GB, "
            f"token_to_kv_pool memory={token_to_kv_pool_memory:.2f} GB"
        )
        memory_pool_info = MemoryPoolInfo(
            memory_pool_memory, req_to_token_pool_memory, token_to_kv_pool_memory
        )
        return memory_pool_info

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        tic = time.time()
        if self.server_args.attention_backend == "flashinfer":
            self.attn_backend = FlashInferAttnBackend(self)
        elif self.server_args.attention_backend == "triton":
            assert self.sliding_window_size is None, (
                "Window attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            assert not self.model_config.is_encoder_decoder, (
                "Cross attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            if self.server_args.enable_double_sparsity:
                self.attn_backend = DoubleSparseAttnBackend(self)
            else:
                self.attn_backend = TritonAttnBackend(self)
        else:
            raise ValueError(
                f"Invalid attention backend: {self.server_args.attention_backend}"
            )
        logger.info(f"Init attention backend end. Time cost: {time.time() - tic:.4f}s")

    def init_double_sparsity_channel_config(self, selected_channel):

        selected_channel = "." + selected_channel + "_proj"
        self.sorted_channels = []
        # load channel config
        with open(self.server_args.ds_channel_config_path, "r") as f:
            channel_config = json.load(f)

        for i in range(self.model_config.num_hidden_layers):
            key = "model.layers." + str(i) + ".self_attn" + selected_channel
            self.sorted_channels.append(
                torch.tensor(channel_config[key])[
                    :, : self.server_args.ds_heavy_channel_num
                ]
                .contiguous()
                .cuda()
            )

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        self.cuda_graph_runner = None

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.disable_cuda_graph:
            return

        logger.info("Capture cuda graph begin. This can take up to several minutes.")
        self.cuda_graph_runner = CudaGraphRunner(self)

    def forward_decode(self, forward_batch: ForwardBatch):
        if self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch):
            return self.cuda_graph_runner.replay(forward_batch)

        forward_batch.positions = (forward_batch.seq_lens - 1).to(torch.int64)
        self.attn_backend.init_forward_metadata(forward_batch)
        return self.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

    def forward_extend(self, forward_batch: ForwardBatch):
        self.attn_backend.init_forward_metadata(forward_batch)
        if self.is_generation:
            return self.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
        else:
            # Only embedding models have get_embedding parameter
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                get_embedding=True,
            )

    def forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        """Forward pass through the model."""
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(forward_batch)
        elif forward_batch.forward_mode.is_extend():
            return self.forward_extend(forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

    def sample(
        self, logits_output: LogitsProcessorOutput, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """Sample next tokens from the logits."""
        # Put CPU-heavy tasks here. They will be overlapped with the forward pass.
        sampling_info = forward_batch.sampling_info
        sampling_info.update_regex_vocab_mask()
        sampling_info.update_penalties()
        logits = self.apply_logits_bias(logits_output.next_token_logits, sampling_info)

        # Sample the next tokens.
        next_token_ids = self.sampler(logits, sampling_info)
        return next_token_ids

    def apply_logits_bias(self, logits: torch.Tensor, sampling_info: SamplingBatchInfo):
        """Apply logits bias for sampling."""
        # Apply logit_bias
        if sampling_info.logit_bias is not None:
            logits.add_(sampling_info.logit_bias)

        # min-token, presence, frequency
        if sampling_info.linear_penalties is not None:
            logits.add_(sampling_info.linear_penalties)

        # repetition
        if sampling_info.scaling_penalties is not None:
            logits = torch.where(
                logits > 0,
                logits / sampling_info.scaling_penalties,
                logits * sampling_info.scaling_penalties,
            )

        # Apply regex vocab_mask
        if sampling_info.vocab_mask is not None:
            logits = logits.masked_fill(sampling_info.vocab_mask, float("-inf"))

        return logits

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        rope_scaling = getattr(self.model_config.hf_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        return rope_scaling.get("type", None) == "mrope"

    def cleanup(self):
        if hasattr(self, "shm") and self.shm is not None:
            try:
                self.shm.close()
                try:
                    self.shm.unlink()
                    logger.info(f"Successfully unlinked shared memory: {self.ipc_name}")
                except FileNotFoundError:
                    # Handle the case where shared memory has already been removed
                    logger.warning(f"Shared memory {self.ipc_name} already removed")
                except Exception as e:
                    logger.warning(
                        f"Error unlinking shared memory {self.ipc_name}: {e}"
                    )
            except Exception as e:
                logger.warning(f"Error closing shared memory: {e}")

        if hasattr(self, "token_to_kv_pool") and self.token_to_kv_pool is not None:
            try:
                self.token_to_kv_pool.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down token_to_kv_pool: {e}")

    def _signal_handler(self, signum, frame):
        """Handle signals to ensure cleanup before exit"""
        self.cleanup()
        # Re-raise the signal to allow the default handler to run
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def __del__(self):
        self.cleanup()


@lru_cache()
def import_model_classes():
    model_arch_name_to_cls = {}
    package_name = "sglang.srt.models"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}. " f"{e}")
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(
                    entry, list
                ):  # To support multiple model classes in one module
                    for tmp in entry:
                        assert (
                            tmp.__name__ not in model_arch_name_to_cls
                        ), f"Duplicated model implementation for {tmp.__name__}"
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert (
                        entry.__name__ not in model_arch_name_to_cls
                    ), f"Duplicated model implementation for {entry.__name__}"
                    model_arch_name_to_cls[entry.__name__] = entry

    return model_arch_name_to_cls


def load_model_cls_srt(model_arch: str) -> Optional[Type[nn.Module]]:
    model_arch_name_to_cls = import_model_classes()

    if model_arch not in model_arch_name_to_cls:
        raise ValueError(
            f"Unsupported architectures: {model_arch}. "
            f"Supported list: {list(model_arch_name_to_cls.keys())}"
        )
    return model_arch_name_to_cls[model_arch]


# Monkey patch model loader
setattr(ModelRegistry, "_try_load_model_cls", load_model_cls_srt)
setattr(ModelRegistry, "is_multimodal_model", is_multimodal_model)
setattr(ModelRegistry, "is_attention_free_model", is_attention_free_model)
setattr(ModelRegistry, "model_has_inner_state", model_has_inner_state)
setattr(ModelRegistry, "is_embedding_model", is_embedding_model)
