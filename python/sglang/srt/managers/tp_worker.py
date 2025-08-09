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

"""A tensor parallel worker."""

import json
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.managers.io_struct import UpdateWeightReqInput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.single_model_runner import SingleModelRunner
from sglang.srt.model_executor.worker_pool_model_runner import WorkerPoolModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import broadcast_pyobj, is_multimodal_model, set_random_seed

logger = logging.getLogger(__name__)


class BaseTpModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        shared_cpu_models: Optional[Dict[str, List[nn.Module]]] = None,
        model_names_to_model_paths: Optional[Dict[str, str]] = None,
        engine_id: Optional[str] = None,
        input_queue: Optional[torch.multiprocessing.Queue] = None,
        output_queue: Optional[torch.multiprocessing.Queue] = None,
    ):
        # Parse args
        self.tp_rank = tp_rank
        self.server_args = server_args
        self.device = self.server_args.device
        self.model_names_to_model_paths = model_names_to_model_paths

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_rank,
            self.model_runner.tp_group.cpu_group,
        )[0]
        set_random_seed(self.random_seed)
        self.max_prefill_tokens = self.server_args.max_prefill_tokens

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            global_server_args_dict,
        )

    def get_model_config(self):
        return self.model_config

    def get_tokenizer(self):
        return self.tokenizer

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_tp_cpu_group(self):
        return self.model_runner.tp_group.cpu_group

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool,
        )

    def get_memory_usage(self):
        return self.model_runner.get_memory_usage()

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
        return logits_output, next_token_ids

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        embeddings = logits_output.embeddings
        return embeddings

    def update_weights(self, recv_req: UpdateWeightReqInput):
        success, message = self.model_runner.update_weights(
            recv_req.model_path, recv_req.load_format
        )
        return success, message

    def deactivate_model_runner(self):
        self.model_runner.deactivate()

    def update_memory_info(self):
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if self.server_args.max_running_requests is None
                else self.server_args.max_running_requests
            ),
            self.model_runner.req_to_token_pool.size,
        )
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

    def resize_memory_pool(self, new_memory_pool_size: Optional[float] = None):
        success = self.model_runner.resize_memory_pool(new_memory_pool_size)
        if success:
            self.update_memory_info()
        return success


class SingleModelTPWorker(BaseTpModelWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        shared_cpu_models: Optional[Dict[str, List[nn.Module]]] = None,
        model_names_to_model_paths: Optional[Dict[str, str]] = None,
        engine_id: Optional[str] = None,
        input_queue: Optional[torch.multiprocessing.Queue] = None,
        output_queue: Optional[torch.multiprocessing.Queue] = None,
    ):
        # init model config, model runner
        self.model_config = ModelConfig(
            server_args.model_name,
            server_args.model_path,
            server_args.trust_remote_code,
            context_length=server_args.context_length,
            model_override_args=json.loads(server_args.json_model_override_args),
        )

        self.model_name = self.model_config.name
        self.model_runner = SingleModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
            shared_cpu_models=shared_cpu_models,
            engine_id=engine_id,
            input_queue=input_queue,
            output_queue=output_queue,
        )

        # init tokenizer
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if is_multimodal_model(self.model_config.hf_config.architectures):
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )

        super().__init__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            nccl_port,
            shared_cpu_models,
            model_names_to_model_paths,
            engine_id,
            input_queue,
            output_queue,
        )

        self.update_memory_info()

    def activate_model_runner(self, memory_pool_size, gpu_id, model_name):
        self.model_runner.activate(
            memory_pool_size=memory_pool_size,
            gpu_id=gpu_id,
            model_name=model_name,
        )
        self.update_memory_info()


class WorkerPoolTPWorker(BaseTpModelWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        shared_cpu_models: Optional[Dict[str, List[nn.Module]]] = None,
        model_names_to_model_paths: Optional[Dict[str, str]] = None,
        engine_id: Optional[str] = None,
        input_queue: Optional[torch.multiprocessing.Queue] = None,
        output_queue: Optional[torch.multiprocessing.Queue] = None,
    ):
        self.model_configs = self._get_model_configs(
            server_args, model_names_to_model_paths
        )
        self.model_runner = WorkerPoolModelRunner(
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            nccl_port=nccl_port,
            worker_id=server_args.worker_id,
            server_args=server_args,
            shared_cpu_models=shared_cpu_models,
            model_configs=self.model_configs,
            engine_id=engine_id,
            input_queue=input_queue,
            output_queue=output_queue,
        )
        self.tokenizers = {
            model_name: get_tokenizer(
                model_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )
            for model_name, model_path in model_names_to_model_paths.items()
        }
        super().__init__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            nccl_port,
            shared_cpu_models,
            model_names_to_model_paths,
            engine_id,
            input_queue,
            output_queue,
        )

    def activate_model_runner(self, memory_pool_size, gpu_id, model_name):
        self.model_runner.activate(
            memory_pool_size=memory_pool_size,
            gpu_id=gpu_id,
            model_name=model_name,
        )
        self.model_config = self.model_configs[model_name]
        self.tokenizer = self.tokenizers[model_name]
        self.update_memory_info()

    def _get_model_configs(
        self, server_args: ServerArgs, model_names_to_model_paths: Dict[str, str]
    ):
        model_configs = {}
        for model_name, model_path in model_names_to_model_paths.items():
            model_configs[model_name] = ModelConfig(
                name=model_name,
                path=model_path,
                trust_remote_code=server_args.trust_remote_code,
                context_length=server_args.context_length,
                model_override_args=json.loads(server_args.json_model_override_args),
            )
        return model_configs
