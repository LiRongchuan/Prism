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

import gc
import getpass
import io
import json
import logging
import os
import pickle
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.multiprocessing.queue import ForkingPickler
from vllm.config import DeviceConfig, LoadConfig
from vllm.config import ModelConfig as VllmModelConfig
from vllm.model_executor.model_loader import get_model

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.model_runner import BaseModelRunner, MemoryPoolInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    get_available_gpu_memory,
    is_generation_model,
    monkey_patch_vllm_dummy_weight_loader,
)

logger = logging.getLogger(__name__)


class SingleModelRunner(BaseModelRunner):
    """ModelRunner for single model mode (1:1 mapping)."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        shared_cpu_models: Dict[
            str, List[nn.Module]
        ],  # model name -> list of different ranks of shared cpu models
        engine_id: str,
        input_queue: Optional[torch.multiprocessing.Queue] = None,
        output_queue: Optional[torch.multiprocessing.Queue] = None,
    ):
        self.model_config = model_config
        self.model_name = model_config.name
        self.ipc_name = f"ipc_{gpu_id}_{self.model_name}_{getpass.getuser()}"
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
        # Check if shared CPU models are available for TP mode
        cpu_model_ref = None
        if self.tp_size > 1 and self.shared_cpu_models:
            # Only try to get CPU model ref if shared_cpu_models is not empty
            model_key = (self.model_config.path, self.tp_size)
            if model_key in self.shared_cpu_models:
                cpu_model_ref = self.shared_cpu_models[model_key][self.tp_rank]
        
        # Load the model and tokenizer
        if self.tp_size > 1:
            self.load_model(self.server_args.model_path)
        else:
            self.load_cpu_model(
                cpu_model_ref=cpu_model_ref, pin_memory=not self.use_model_service
            )

        if self.server_args.on:
            if not self.tp_size > 1:
                self.model_gpu_mem_usage = self.load_gpu_model(check_mem=False)
            else:
                self.model_gpu_mem_usage = self._get_profiled_model_gpu_mem_usage()
            self.memory_pool_info = self.init_memory_pool()
            if self.device == "cuda":
                self.init_cublas()
                self.init_attention_backend()
                self.init_cuda_graphs()
            else:
                self.cuda_graph_runner = None
                self.init_attention_backend()
        else:
            self.model_gpu_mem_usage = self._get_profiled_model_gpu_mem_usage()
            self.memory_pool_info = self.init_memory_pool(
                init_req_to_token_only=not self.enable_elastic_memory
            )
            if self.device == "cuda":
                self.init_cublas()
                self.init_cuda_graphs()
            else:
                self.cuda_graph_runner = None
            if self.tp_size > 1:
                self.delete_gpu_model()

    def _get_profiled_model_gpu_mem_usage(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        profiled_model_info_path = os.path.join(
            cur_dir, "../../multi_model/utils/model_info.json"
        )
        with open(profiled_model_info_path, "r") as f:
            profiled_model_info = json.load(f)
            model_path = self.model_config.path
            model_info = profiled_model_info.get(model_path, None)
            if model_info is None:
                logger.warning(f"Model {model_path} not found in profiled model info.")
                return 0
            else:
                return model_info["model_size"]

    def _get_max_num_reqs(self, max_total_num_tokens: int):
        max_num_reqs = self.server_args.max_running_requests
        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(max_total_num_tokens / self.model_config.context_len * 512),
                    2048,
                ),
                4096,
            )
        return max_num_reqs

    def load_model(self, model_path):
        # Prepare the vllm model config
        monkey_patch_vllm_dummy_weight_loader()
        self.load_config = LoadConfig(load_format=self.server_args.load_format)
        self.vllm_model_config = VllmModelConfig(
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
            self.vllm_model_config.hf_config.update(
                self.model_config.model_override_args
            )
        self.dtype = self.vllm_model_config.dtype

        tic = time.time()
        # Load the model
        self.model = get_model(
            model_config=self.vllm_model_config,
            load_config=self.load_config,
            device_config=DeviceConfig(device="cuda"),
            parallel_config=None,
            scheduler_config=None,
            lora_config=None,
            cache_config=None,
        )

        self.sliding_window_size = (
            self.model.get_attention_sliding_window_size()
            if hasattr(self.model, "get_attention_sliding_window_size")
            else None
        )   
        self.is_generation = is_generation_model(
            self.model_config.hf_config.architectures, self.server_args.is_embedding
        )
    
    def load_cpu_model(
        self, cpu_model_ref: Optional[List[nn.Module]] = None, pin_memory: bool = True
    ):

        # This can reduce thread conflicts and speed up weight loading.
        torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")
        # Prepare the vllm model config
        monkey_patch_vllm_dummy_weight_loader()
        self.load_config = LoadConfig(load_format=self.server_args.load_format)
        self.vllm_model_config = VllmModelConfig(
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
            self.vllm_model_config.hf_config.update(
                self.model_config.model_override_args
            )
        self.dtype = self.vllm_model_config.dtype
        if cpu_model_ref is None:
            if self.tp_size > 1:
                # Load the model
                self.cpu_model_ref = get_model(
                    model_config=self.vllm_model_config,
                    load_config=self.load_config,
                    device_config=DeviceConfig("cuda"),
                    parallel_config=None,
                    scheduler_config=None,
                    lora_config=None,
                    cache_config=None,
                )
            else:
                # Load the model
                self.cpu_model_ref = get_model(
                    model_config=self.vllm_model_config,
                    load_config=self.load_config,
                    device_config=DeviceConfig("cpu"),
                    parallel_config=None,
                    scheduler_config=None,
                    lora_config=None,
                    cache_config=None,
                )
            self.cpu_model_ref.make_empty_intermediate_tensors = (
                None  # make it pickable
            )
            self.cpu_model_ref.model.make_empty_intermediate_tensors = (
                None  # make it pickable
            )
        else:
            self.cpu_model_ref = cpu_model_ref
        
        # Only pin memory if we have a CPU model (either from shared memory or loaded on CPU)
        # In TP mode without shared memory, model is loaded on CUDA and cannot be pinned
        should_pin_memory = pin_memory and (
            cpu_model_ref is not None or  # Have shared CPU model
            (cpu_model_ref is None and self.tp_size == 1)  # Single GPU mode, loaded on CPU
        )
        
        if should_pin_memory:
            self.state_dict_host = TensorDict(self.cpu_model_ref.state_dict())
            self.state_dict_host.pin_memory()
        self.sliding_window_size = (
            self.cpu_model_ref.get_attention_sliding_window_size()
            if hasattr(self.cpu_model_ref, "get_attention_sliding_window_size")
            else None
        )
        self.is_generation = is_generation_model(
            self.model_config.hf_config.architectures, self.server_args.is_embedding
        )

    def activate(
        self,
        memory_pool_size: Optional[float] = None,
        gpu_id: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        memory_pinned = True
        restart_token_to_kv_pool = False
        if gpu_id is not None:
            if self.tp_size > 1:
                raise ValueError("Activating a model with tp > 1 is not supported yet.")
            original_gpu_id = self.gpu_id
            if original_gpu_id != gpu_id:
                self.gpu_id = gpu_id
                self._set_device()
                restart_token_to_kv_pool = True
        if self.server_args.async_loading:
            self.load_gpu_model_async(
                post_load_process=True,
                memory_pool_size=memory_pool_size,
                restart_token_to_kv_pool=restart_token_to_kv_pool,
                memory_pinned=memory_pinned,
                use_model_service=self.use_model_service,
            )
        else:
            if self.tp_size > 1:
                self.load_model(self.server_args.model_path)
            else:
                self.load_gpu_model(use_model_service=self.use_model_service)
            self.post_model_load_process(memory_pool_size, restart_token_to_kv_pool)
            self.init_attention_backend()

    def post_model_load_process(
        self, memory_pool_size: float, restart_token_to_kv_pool: bool
    ):
        if memory_pool_size is not None:
            self.max_total_num_tokens = self._get_max_total_num_tokens(memory_pool_size)
            self.max_num_reqs = self._get_max_num_reqs(self.max_total_num_tokens)
            logger.info(
                f"New memory pool size: {memory_pool_size:.2f} GB, "
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"max_num_reqs={self.max_num_reqs}"
            )

        if self.server_args.enable_elastic_memory:
            self.memory_pool_info = self._reinit_and_update_memory_pool(
                self.max_num_reqs, self.max_total_num_tokens, restart_token_to_kv_pool
            )
        else:
            self.memory_pool_info = self._init_memory_pool(
                self.max_num_reqs, self.max_total_num_tokens
            )

    def resize_memory_pool(self, new_memory_pool_size: Optional[float] = None):
        if self.server_args.enable_elastic_memory:
            if new_memory_pool_size is not None:
                self.max_total_num_tokens = self._get_max_total_num_tokens(
                    new_memory_pool_size
                )
                self.max_num_reqs = self._get_max_num_reqs(self.max_total_num_tokens)
                actual_memory = self.max_total_num_tokens * self.cell_size // (1 << 30)
                logger.info(
                    f"New memory pool size: {new_memory_pool_size:.2f} GB, "
                    f"max_total_num_tokens={self.max_total_num_tokens}, "
                    f"max_num_reqs={self.max_num_reqs}, "
                    f"actual target memory pool size={actual_memory:.2f} GB"
                )
            else:
                self.max_total_num_tokens = self.token_to_kv_pool.size
                self.max_num_reqs = self._get_max_num_reqs(self.max_total_num_tokens)
            success = self.token_to_kv_pool.update_size(self.max_total_num_tokens)
            return success
        else:
            raise ValueError("Only elastic memory is supported for resize_memory_pool")

    def load_gpu_model_async(
        self,
        check_mem=True,
        post_load_process=False,
        memory_pool_size=None,
        restart_token_to_kv_pool=None,
        memory_pinned=True,
    ):
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
            if not memory_pinned:
                # Need to obtain self.state_dict_host from model_ref, no need for pin_memory
                self.state_dict_host = TensorDict(self.cpu_model_ref.state_dict())

            buf = io.BytesIO()
            ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(self.cpu_model_ref)
            self.model = pickle.loads(buf.getvalue())

            if post_load_process:
                self.post_model_load_process(memory_pool_size, restart_token_to_kv_pool)
                self.init_attention_backend()

        def async_transfer_and_load():
            if self.tp_rank > 0:
                self._set_device()
            transfer_stream = torch.cuda.Stream(device=f"{self.device}:{self.gpu_id}")
            with torch.cuda.stream(transfer_stream):
                if not memory_pinned:
                    while (
                        not hasattr(self, "state_dict_host")
                        or self.state_dict_host is None
                    ):
                        time.sleep(0.01)
                    state_dict_device = self.state_dict_host.to(
                        f"{self.device}:{self.gpu_id}",
                        non_blocking=True,
                    )
                else:
                    state_dict_device = self.state_dict_host.to(
                        f"{self.device}:{self.gpu_id}",
                        non_blocking=True,
                        non_blocking_pin=True,
                        num_threads=4,
                    )
                # Code protection, ensure model is loaded
                while not hasattr(self, "model") or self.model is None:
                    time.sleep(0.01)
                self.model.load_state_dict(state_dict_device, assign=True)

        def async_transfer_and_load_model_service():
            self.input_queue.put(
                (self.server_args.model_path, self.worker_id, self.gpu_id)
            )
            self.model = self.output_queue.get()
            loading_time = self.output_queue.get()
            service_id = self.output_queue.get()
            logger.info(
                f"Load model from model service end. Time cost: {loading_time:.4f}s, service_id: {service_id}"
            )

        def async_init_others():
            if post_load_process:
                self.post_model_load_process(memory_pool_size, restart_token_to_kv_pool)
                self.init_attention_backend()

        import threading

        if self.use_model_service:
            transfer_func = async_transfer_and_load_model_service
            init_func = async_init_others
        else:
            transfer_func = async_transfer_and_load
            init_func = async_init_model

        model_thread = threading.Thread(target=init_func)
        transfer_thread = threading.Thread(target=transfer_func)
        model_thread.start()
        transfer_thread.start()
        model_thread.join()
        self.transfer_thread = transfer_thread
        self.model_thread = model_thread

    def _reinit_and_update_memory_pool(
        self,
        max_num_reqs: int,
        max_total_num_tokens: int,
        restart_token_to_kv_pool: bool = False,
    ):
        tic = time.time()
        memory_allocated_start = torch.cuda.memory_allocated()
        # initialize req_to_token_pool
        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs + 1,
            max_context_len=self.model_config.context_len + 4,
            device=self.device,
            gpu_id=self.gpu_id,
            use_records=False,
            min_reserve_mem=self.min_reserve_mem,
        )
        mem_allocated_after_req_to_token_pool = torch.cuda.memory_allocated()
        req_to_token_pool_memory = (
            mem_allocated_after_req_to_token_pool - memory_allocated_start
        ) / (1 << 30)
        logger.info(
            f"Req_to_token_pool is initialized. Time cost: {time.time() - tic:.4f}s. "
            f"req_to_token_pool memory={req_to_token_pool_memory:.2f} GB"
        )
        if not isinstance(self.token_to_kv_pool, MHATokenToKVPool):
            raise ValueError("Only MHATokenToKVPool is supported for elastic memory")

        kvcached_update_tic = time.time()
        if restart_token_to_kv_pool:
            success = self.token_to_kv_pool.kvcached_ops.shutdown_kvcached()
            self.token_to_kv_pool = MHATokenToKVPool(
                max_total_num_tokens,
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
                enable_worker_pool=False,
                shm=self.shm,
            )
        else:
            success = self.token_to_kv_pool.update_size(max_total_num_tokens)
        available_size = self.token_to_kv_pool.available_size()
        memory_pool_size = (
            available_size
            * self.token_to_kv_pool.cell_size
            * 2
            * self.token_to_kv_pool.layer_num
        ) // (1 << 30)
        logger.info(
            f"KVCached update size end. Time cost: {time.time() - kvcached_update_tic:.4f}s. "
            f"available size: {available_size}, memory pool size: {memory_pool_size:.2f} GB"
        )
        memory_allocated_end = torch.cuda.memory_allocated()
        memory_pool_memory = (memory_allocated_end - memory_allocated_start) / (1 << 30)
        token_to_kv_pool_memory = (
            memory_allocated_end - mem_allocated_after_req_to_token_pool
        ) / (1 << 30)

        logger.info(
            f"Update memory pool end. Time cost: {time.time() - tic:.4f}s. "
            f"memory pool memory={memory_pool_memory:.2f} GB, "
            f"req_to_token_pool memory={req_to_token_pool_memory:.2f} GB, "
            f"token_to_kv_pool memory={token_to_kv_pool_memory:.2f} GB"
        )
        memory_pool_info = MemoryPoolInfo(
            memory_pool_memory, req_to_token_pool_memory, token_to_kv_pool_memory
        )
        return memory_pool_info

    def init_memory_pool(self, init_req_to_token_only: bool = False):
        self.kv_cache_dtype = self._get_kv_cache_dtype()
        self.cell_size = self._get_cell_size()

        init_memory_pool_size = self._get_init_memory_pool_size()
        self.max_total_num_tokens = self._get_max_total_num_tokens(
            init_memory_pool_size
        )
        self.max_num_reqs = self._get_max_num_reqs(self.max_total_num_tokens)

        logger.info(
            f"Init memory pool size: {init_memory_pool_size:.2f} GB, "
            f"max_total_num_tokens: {self.max_total_num_tokens}, "
            f"max_num_reqs: {self.max_num_reqs}, "
            f"cell_size: {self.cell_size:.2f} bytes, {self.cell_size / 1024 ** 2:.2f} MB, "
            f"kv_cache_dtype: {self.kv_cache_dtype}"
        )
        memory_pool_info = self._init_memory_pool(
            self.max_num_reqs, self.max_total_num_tokens, init_req_to_token_only
        )
        return memory_pool_info

    def _get_init_memory_pool_size(self):
        from sglang.global_config import global_config

        max_memory_pool_size = self.server_args.max_memory_pool_size
        if max_memory_pool_size is None:
            max_memory_pool_size = self.max_mem_usage - self.model_gpu_mem_usage
        max_memory_pool_size -= global_config.flashinfer_workspace_size // (1 << 30)

        return max_memory_pool_size

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
            self.input_queue.put(
                (self.server_args.model_path, self.engine_id, self.gpu_id)
            )
            self.model = self.output_queue.get()
            loading_time = self.output_queue.get()
            service_id = self.output_queue.get()
            t1 = time.perf_counter()
            logger.info(
                f"Load model from model service end. Time cost: {t1 - t0:.4f}s, loading time: {loading_time:.4f}s, service_id: {service_id}"
            )
        else:
            buf = io.BytesIO()
            ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(self.cpu_model_ref)

            self.model = pickle.loads(buf.getvalue())

            state_dict_device = self.state_dict_host.to(
                f"{self.device}:{self.gpu_id}", non_blocking_pin=True, num_threads=4
            )
            self.model.load_state_dict(state_dict_device, assign=True)

        memory_after_load = torch.cuda.memory_allocated() / (1 << 30)
        logger.info(
            f"Load GPU model end. Time cost: {time.time() - tic:.4f}s. Current available memory: {get_available_gpu_memory(self.device, self.gpu_id):.2f} GB, model GPU memory usage: {memory_after_load - memory_before_load:.2f} GB"
        )
        return memory_after_load - memory_before_load

    def free_memory_pool(self):
        if hasattr(self, "req_to_token_pool") and self.req_to_token_pool is not None:
            self.req_to_token_pool.release()
        if hasattr(self, "token_to_kv_pool") and self.token_to_kv_pool is not None:
            self.token_to_kv_pool.release()

    def deactivate(self):
        tic = time.time()
        self.free_memory_pool()
        self.delete_gpu_model()
        self.attn_backend = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(
            f"Deactivate: Free memory pool and delete gpu model end. Time cost: {time.time() - tic:.4f}s. Current available memory: {get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
