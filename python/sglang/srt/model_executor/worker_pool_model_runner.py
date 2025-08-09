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

import copy
import dataclasses
import gc
import getpass
import importlib
import importlib.resources
import json
import logging
import os
import pkgutil
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

from sglang.srt.model_executor.model_runner import BaseModelRunner

def create_empty_gpu_model_from_cpu_model(cpu_model, gpu_id):
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(cpu_model)
    model = pickle.loads(buf.getvalue())
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = torch.empty_like(state_dict[key], device=f"cuda:{gpu_id}")
    model.load_state_dict(state_dict, assign=True)
    return model

class WorkerPoolModelRunner(BaseModelRunner):
    """ModelRunner for worker pool mode."""

    def __init__(
        self,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int,
        worker_id: int,
        server_args: ServerArgs,  # default server args
        shared_cpu_models: Dict[
            str, List[nn.Module]
        ],  # model name -> list of different ranks of shared cpu models
        model_configs: Dict[str, ModelConfig],  # model name -> model config
        engine_id: str,
        input_queue: Optional[torch.multiprocessing.Queue] = None,
        output_queue: Optional[torch.multiprocessing.Queue] = None,
    ):
        self.worker_id = worker_id
        self.model_configs = model_configs
        self.ipc_name = f"ipc_{gpu_id}_{worker_id}_{getpass.getuser()}"
        super().__init__(
            mem_fraction_static=mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
            shared_cpu_models=shared_cpu_models,
            engine_id=engine_id,
            input_queue=input_queue,
            output_queue=output_queue,
        )
        self.model_gpu_mem_usage = 0
        self.is_generation = True
        self.sliding_window_size = None
        self.virtual_memory_size_gb = 25  # TODO: configurable

        # init memory pool
        if self.enable_elastic_memory:
            from kvcached import ops as kvcached_ops

            kvcached_ops.init_kvcached(
                virtual_mem_size_gb=self.virtual_memory_size_gb,
                reserve_virtual_mem=True,
            )
        else:
            raise NotImplementedError(
                "Static memory is not supported in worker pool mode"
            )
        self.max_num_reqs = self._get_max_num_reqs()
        self.max_total_num_tokens = self.server_args.max_total_tokens
        self.max_context_len = self._get_max_context_len()
        self.memory_pool_info = self._init_memory_pool(
            max_num_reqs=self.max_num_reqs,
            max_total_num_tokens=self.max_total_num_tokens,
            init_req_to_token_only=True,
            max_context_len=self.max_context_len,
        )

        self.init_cublas()
        self.init_cuda_graphs()

        self.is_active = False

    def _get_max_context_len(self):
        model_context_lens = [
            self.model_configs[model_name].context_len
            for model_name in self.model_configs
        ]
        max_context_len = max(model_context_lens) + 4
        return max_context_len

    def _get_max_num_reqs(self):
        return 512  # TODO: get from logs

    def _get_max_total_num_tokens(self):
        # max_num_token = int(self.virtual_memory_size_gb * (1 << 30) // self.cell_size)
        cell_size_per_layer = self.cell_size // self.model_config.num_hidden_layers // 2
        virtual_page_size = 2 * 1024 * 1024 # 2MB
        virtual_mem_size_per_layer = (
            self.virtual_memory_size_gb * (1 << 30)
            // self.model_config.num_hidden_layers
            // 2
            // virtual_page_size
            * virtual_page_size
        )
        max_num_token = virtual_mem_size_per_layer // cell_size_per_layer
        max_total_tokens = self.server_args.max_total_tokens
        if max_total_tokens is not None:
            if max_total_tokens > max_num_token:
                logger.warning(
                    f"max_total_tokens={max_total_tokens} in server_args is larger than the profiled value "
                    f"{max_num_token}. Use the profiled value instead."
                )
            max_num_token = min(max_num_token, max_total_tokens)
        if max_num_token <= 0:
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static or --max-memory-pool-size or --max-mem-usage."
            )
        return max_num_token

    def _set_model_params(self, model_name: str):
        self.model_config = self.model_configs[model_name]
        self.model_path = self.model_config.path
        self.model_name = model_name
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = self._get_kv_cache_dtype()
        self.cell_size = self._get_cell_size()
        self.max_total_num_tokens = self._get_max_total_num_tokens()
        self.cpu_model_ref = self._get_cpu_model_ref(model_name)

    def init_token_to_kv_pool(self):
        self.token_to_kv_pool = MHATokenToKVPool(
            size=self.max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(self.tp_size),
            head_dim=self.model_config.head_dim,
            layer_num=self.model_config.num_hidden_layers,
            device=self.device,
            gpu_id=self.gpu_id,
            model_name=self.model_name,
            enable_elastic_memory=self.enable_elastic_memory,
            min_reserve_mem=self.min_reserve_mem,
            enable_overlap=self.server_args.enable_overlap_schedule,
            use_kvcached_v0=self.server_args.use_kvcached_v0,
            enable_worker_pool=True,
            shm=self.shm,
        )

    def activate_async(self, check_mem=True):
        if check_mem:
            while (
                get_available_gpu_memory(self.device, self.gpu_id)
                - self.min_reserve_mem
                < self.model_gpu_mem_usage
            ):
                logger.info(
                    f"Waiting for enough memory to load the model.... Current available memory: {get_available_gpu_memory(self.device, self.gpu_id):.2f} GB, min reserve mem: {self.min_reserve_mem:.2f} GB, model memory usage: {self.model_gpu_mem_usage:.2f} GB"
                )
                time.sleep(0.1)

        def async_init_model():
            if self.tp_rank > 0:
                self._set_device()
            self.state_dict_host = TensorDict(self.cpu_model_ref.state_dict())

            buf = io.BytesIO()
            ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(self.cpu_model_ref)
            self.model = pickle.loads(buf.getvalue())

            self.init_token_to_kv_pool()
            self.init_attention_backend()

        def async_transfer_and_load():
            if self.tp_rank > 0:
                self._set_device()
            transfer_stream = torch.cuda.Stream(device=f"{self.device}:{self.gpu_id}")
            with torch.cuda.stream(transfer_stream):
                while (
                    not hasattr(self, "state_dict_host") or self.state_dict_host is None
                ):
                    time.sleep(0.01)
                state_dict_device = self.state_dict_host.to(
                    f"{self.device}:{self.gpu_id}",
                    non_blocking=True,
                )

                # Code protection, ensure model is loaded
                while not hasattr(self, "model") or self.model is None:
                    time.sleep(0.01)
                self.model.load_state_dict(state_dict_device, assign=True)

        def async_transfer_and_load_model_service():
            raise NotImplementedError("Model service is not supported in worker pool mode")
            self.input_queue.put((self.model_path, self.worker_id, self.gpu_id))
            self.model = self.output_queue.get()
            loading_time = self.output_queue.get()
            service_id = self.output_queue.get()
            logger.info(
                f"Load model from model service end. Time cost: {loading_time:.4f}s, service_id: {service_id}"
            )

        def async_init_others():
            self.init_token_to_kv_pool()
            self.init_attention_backend()

        import threading

        if self.use_model_service:
            transfer_func = async_transfer_and_load_model_service
            init_func = async_init_others
        else:
            transfer_func = async_transfer_and_load
            init_func = async_init_model

        transfer_thread = threading.Thread(target=async_transfer_and_load)
        model_thread = threading.Thread(target=async_init_model)
        model_thread.start()
        transfer_thread.start()
        model_thread.join()

    def activate(
        self,
        memory_pool_size: Optional[float] = None,
        gpu_id: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        if self.is_active:
            raise RuntimeError("Model runner is already active")
        assert model_name is not None, "model_name is required for worker pool"
        assert (
            gpu_id == self.gpu_id
        ), "gpu_id must be the same as the worker id, got {} and {}".format(
            gpu_id, self.gpu_id
        )
        tic = time.time()
        self._set_model_params(model_name)

        if self.server_args.async_loading:
            self.activate_async()
        else:
            self.load_gpu_model(use_model_service=self.use_model_service)
            self.init_token_to_kv_pool()
            self.init_attention_backend()

        self.is_active = True
        logger.info(
            f"Activate model {self.model_name} ({self.model_path}) time cost: {time.time() - tic:.4f}s."
        )

    def load_gpu_model(self, check_mem=True, use_model_service=False):
        # check whether the available memory is enough for the model
        memory_before_load = torch.cuda.memory_allocated() / (1 << 30)
        tic = time.time()
        if check_mem:
            while (
                get_available_gpu_memory(self.device, self.gpu_id)
                - self.min_reserve_mem
                < self.model_gpu_mem_usage
            ):
                logger.info(
                    f"Waiting for enough memory to load the model.... Current available memory: {get_available_gpu_memory(self.device, self.gpu_id):.2f} GB, min reserve mem: {self.min_reserve_mem:.2f} GB, model memory usage: {self.model_gpu_mem_usage:.2f} GB"
                )
                time.sleep(0.1)
        if use_model_service:
            t0 = time.perf_counter()
            model_key = self.model_path
            success = False
            if (model_key, self.tp_size) not in self.shared_cpu_models:
                raise RuntimeError(f"Model {(model_key, self.tp_size)} not found in shared cpu models, with keys: {self.shared_cpu_models.keys()}")
            cpu_model_ref = self.shared_cpu_models[(model_key, self.tp_size)][0]
            self.model = create_empty_gpu_model_from_cpu_model(cpu_model_ref, self.gpu_id)
            retry = 3
            while not success and retry > 0:
                self.input_queue.put((model_key, self.engine_id, self.gpu_id, self.model))
                timeout = 300
                try:
                    msg = self.output_queue.get(timeout=timeout)
                except Exception as e:
                    logger.error(f"Timeout. Failed to load model from model service. Error: {e}")
                    raise RuntimeError(f"Timeout. Failed to load model from model service. Error: {e}")
                if msg is None:
                    logger.error(f"Returned None. Failed to load model from model service. Error: {self.output_queue.get()}")
                    continue
                loading_time = self.output_queue.get()
                service_id = self.output_queue.get()
                success = True
                break
            if not success:
                raise RuntimeError(f"Retry failed. Failed to load model from model service. ")
            # else:
            #     self.model = model

            t1 = time.perf_counter()
            logger.info(
                f"Load model from model service end. Time cost: {t1 - t0:.4f}s, loading time: {loading_time:.4f}s, service_id: {service_id}"
            )
        else:
            buf = io.BytesIO()
            ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(self.cpu_model_ref)

            self.model = pickle.loads(buf.getvalue())
            self.state_dict_host = TensorDict(self.cpu_model_ref.state_dict())

            state_dict_device = self.state_dict_host.to(
                f"{self.device}:{self.gpu_id}",
            )
            self.model.load_state_dict(state_dict_device, assign=True)

        memory_after_load = torch.cuda.memory_allocated() / (1 << 30)
        logger.info(
            f"Load GPU model end. Time cost: {time.time() - tic:.4f}s. Current available memory: {get_available_gpu_memory(self.device, self.gpu_id):.2f} GB, model GPU memory usage: {memory_after_load - memory_before_load:.2f} GB"
        )
        return memory_after_load - memory_before_load

    def deactivate(self):
        """Deactivate this model runner to free up memory."""
        tic = time.time()
        if self.is_active:
            # Free memory pools
            self.token_to_kv_pool.release()

            # Delete GPU model
            self.delete_gpu_model()

            # Reset state
            self.attn_backend = None
            self.cuda_graph_runner = None

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            self.is_active = False
            logger.info(
                f"Deactivate time cost: {time.time() - tic:.4f}s. Current available memory: {get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
            )
        else:
            logger.warning(
                "Worker pool model runner is not active, while calling deactivate"
            )
