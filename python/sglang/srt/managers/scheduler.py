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

"""A scheduler that manages a tensor parallel GPU worker."""


import atexit
import json
import logging
import os
import signal
import threading
import time
import warnings
from collections import deque
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import zmq

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.fsm_cache import FSMCache
from sglang.srt.constrained.jump_forward import JumpForwardCache
from sglang.srt.hf_transformers_utils import (
    get_context_length,
    get_processor,
    get_tokenizer,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActivateReqInput,
    ActivateReqOutput,
    BatchEmbeddingOut,
    BatchRetractDecodeReq,
    BatchRunReq,
    BatchTokenIDOut,
    DeactivateReqInput,
    DeactivateReqOutput,
    FlushCacheReq,
    GenerateReqInput,
    GetMemoryUsageReq,
    GetMemoryUsageReqOutput,
    GetMemPoolSizeReq,
    GetMemPoolSizeReqOutput,
    MemoryUsage,
    PreemptMode,
    ProfileReq,
    ResizeChunkInput,
    ResizeMemPoolReqInput,
    TokenizedGenerateReqInput,
    UpdateModelTput,
    UpdateQueueStats,
    UpdateWeightReqInput,
    UpdateWeightReqOutput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    BaseFinishReason,
    ImageInputs,
    Req,
    ScheduleBatch,
    global_server_args_dict,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.tp_worker import SingleModelTPWorker, WorkerPoolTPWorker
from sglang.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.redis_utils import RedisClient
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    broadcast_pyobj,
    change_logger_format,
    configure_logger,
    is_generation_model,
    is_multimodal_model,
    kill_parent_process,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import cleanup_zmq_ipc, get_exception_traceback

logger = logging.getLogger(__name__)

# Crash on warning if we are running CI tests
crash_on_warning = os.getenv("SGLANG_IS_IN_CI", "false") == "true"

# Test retract decode
test_retract = os.getenv("SGLANG_TEST_RETRACT", "false") == "true"


class Scheduler:
    """ TP-worker调度器 """
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        shared_cpu_models: Optional[Dict[str, List[nn.Module]]] = None,
        model_names_to_model_paths: Optional[Dict[str, str]] = None,
        engine_id: Optional[str] = None,
        input_queue: Optional[torch.multiprocessing.Queue] = None,
        output_queue: Optional[torch.multiprocessing.Queue] = None,
    ):
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.model_name = server_args.model_name
        self.instance_idx = getattr(server_args, "instance_idx", 0)        
        self.tp_size = server_args.tp_size
        self.schedule_policy = server_args.schedule_policy
        self.disable_regex_jump_forward = server_args.disable_regex_jump_forward
        self.lora_paths = server_args.lora_paths
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = server_args.enable_overlap_schedule
        self.enable_elastic_memory = server_args.enable_elastic_memory
        self.use_kvcached_v0 = server_args.use_kvcached_v0
        self.model_names_to_model_paths = model_names_to_model_paths
        self.enable_worker_pool = server_args.enable_worker_pool
        self.engine_id = engine_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        # 通信通道初始化
        if port_args.controller_ipc_name:
            context = zmq.Context(4)
        else:
            context = zmq.Context(3)
        self.ipc_files = set()
        if self.tp_rank == 0: # 仅在rank 0处理请求
            # 接收非生成请求
            self.recv_from_request_handler = context.socket(zmq.PULL)
            self.ipc_files.add(f"ipc://{port_args.scheduler_input_ipc_name}")
            self.recv_from_request_handler.bind(f"ipc://{port_args.scheduler_input_ipc_name}")
            # 接收GPU worker结果
            self.recv_from_gpu_scheduler = context.socket(zmq.PULL)
            worker_id = server_args.worker_id
            ipc_name = f"gpu_scheduler_{self.gpu_id}_to_worker_{worker_id}"
            self.ipc_files.add(f"ipc://{ipc_name}")
            self.recv_from_gpu_scheduler.bind(f"ipc://{ipc_name}")
            # 接收model scheduler信息
            self.recv_from_model_scheduler = context.socket(zmq.PULL)
            ipc_name = "model_scheduler_to_scheduler"
            self.ipc_files.add(f"ipc://{ipc_name}")
            self.recv_from_model_scheduler.bind(f"ipc://{ipc_name}")
            # 接收生成任务
            self.redis_client = RedisClient(server_args.redis_host, server_args.redis_port, server_args.redis_db)
            # 生成token -> Detokenizer
            self.send_to_detokenizer = context.socket(zmq.PUSH)
            self.send_to_detokenizer.connect(f"ipc://{port_args.detokenizer_ipc_name}")
            # 连接model scheduler
            if port_args.model_scheduler_ipc_name:
                self.send_to_model_scheduler = context.socket(zmq.PUSH)
                ipc_name = "scheduler_to_model_scheduler"
                self.ipc_files.add(f"ipc://{ipc_name}")
                self.send_to_model_scheduler.connect(f"ipc://{ipc_name}")
            else:
                self.send_to_model_scheduler = SimpleNamespace(send_pyobj=lambda x: None)
            # 连接global controller
            if port_args.controller_ipc_name:
                self.send_to_controller = context.socket(zmq.PUSH)
                self.send_to_controller.connect(f"ipc://{port_args.controller_ipc_name}")
            else:
                self.send_to_controller = SimpleNamespace(send_pyobj=lambda x: None)
        else:
            self.recv_from_request_handler = None
            self.recv_from_gpu_scheduler = None
            self.recv_from_model_scheduler = None
            self.redis_client = None
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_model_scheduler = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_controller = SimpleNamespace(send_pyobj=lambda x: None)
        # 终止自动清理
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        if not self.enable_worker_pool:
            self._init_tokenizer(server_args.model_name, server_args.model_path)
        self.is_generation = True
        # 启动TP worker
        if self.enable_overlap:
            TpWorkerClass = TpModelWorkerClient
        elif self.enable_worker_pool:
            TpWorkerClass = WorkerPoolTPWorker
        else:
            TpWorkerClass = SingleModelTPWorker
        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            nccl_port=port_args.nccl_port,
            shared_cpu_models=shared_cpu_models,
            model_names_to_model_paths=model_names_to_model_paths,
            engine_id=engine_id,
            input_queue=input_queue,
            output_queue=output_queue,
        )
        if not self.enable_worker_pool: # 获取运行限制
            (
                self.max_total_num_tokens,
                self.max_prefill_tokens,
                self.max_running_requests,
                self.max_req_len,
                self.max_req_input_len,
                self.random_seed,
                self.device,
                worker_global_server_args_dict,
            ) = self.tp_worker.get_worker_info()
        self.tp_cpu_group = self.tp_worker.get_tp_cpu_group()
        self.pad_input_ids_func = None
        set_random_seed(server_args.random_seed)
        
        # 初始化cache内存池
        self.first_time_activate = True
        if self.server_args.on and not self.enable_worker_pool:
            self.req_to_token_pool, self.token_to_kv_pool = (self.tp_worker.get_memory_pool())
            if (server_args.chunked_prefill_size is not None and server_args.disable_radix_cache):
                logger.info(f"Scheduler: use chunk cache")
                self.tree_cache = ChunkCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool=self.token_to_kv_pool,
                )
            else:
                logger.info(f"Scheduler: use radix cache")
                self.tree_cache = RadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool=self.token_to_kv_pool,
                    disable=server_args.disable_radix_cache,
                )
            self.tree_cache_metrics = {"total": 0, "hit": 0}
            self.policy = SchedulePolicy(self.schedule_policy, self.tree_cache)
            self.first_time_activate = False
            # Print debug info
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}"
            )
            
        # 运行状态
        self.batch_is_full = False
        self.waiting_queue: List[Req] = []
        self.running_batch: Optional[ScheduleBatch] = None
        self.cur_batch: Optional[ScheduleBatch] = None
        self.decode_forward_ct = 0
        self.stream_interval = server_args.stream_interval
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()
        self.total_input_len = 0
        self.num_received_requests = 0
        # chunked prefill设置
        self.chunked_prefill_size = server_args.chunked_prefill_size
        self.current_inflight_req: List[Req] = []
        self.is_mixed_chunk = self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        # 初始化约束解码状态缓存
        if not self.enable_worker_pool:
            self._init_fsm_cache(server_args.tokenizer_path)
        self.jump_forward_cache = JumpForwardCache()

        # token长度估计
        assert (server_args.schedule_conservativeness >= 0), "Invalid schedule_conservativeness"
        self.min_new_token_ratio = min(
            global_config.base_min_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.new_token_ratio = self.min_new_token_ratio
        self.new_token_ratio_decay = global_config.new_token_ratio_decay

        # Profiler
        if os.getenv("SGLANG_TORCH_PROFILER_DIR", "") == "":
            self.profiler = None
        else:
            self.torch_profiler_trace_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR")
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                self.torch_profiler_trace_dir,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
            )

        # 模型启动相关
        if self.server_args.on and not self.enable_worker_pool:
            self._activated = True # 模型初始状态
        else:
            self._activated = False
            self.tp_worker.deactivate_model_runner()
        self.waiting_queue_stash = [] # 重启模型等待队列

        # 吞吐量统计
        self.token_count = 0
        self.prefill_token_count = 0
        self.decode_token_count = 0
        self.last_tput_update_time = time.time()
        self.tput_update_thread = threading.Thread(target=self._send_token_tput_updates, daemon=True)
        self.tput_update_thread.start()
        
        # 服务质量统计
        self.variable_chunk = False
        self.ttft_slo_attainment = []
        self.tpot_slo_attainment = []

    def _init_fsm_cache(self, tokenizer_path):
        """约束解码状态机"""
        if not self.server_args.skip_tokenizer_init:
            self.regex_fsm_cache = FSMCache(
                tokenizer_path,
                {
                    "tokenizer_mode": self.server_args.tokenizer_mode,
                    "trust_remote_code": self.server_args.trust_remote_code,
                },
                skip_tokenizer_init=self.server_args.skip_tokenizer_init,
                constrained_json_whitespace_pattern=self.server_args.constrained_json_whitespace_pattern,
            )

    def _init_tokenizer(self, model_name, model_path):
        """加载tokenizer"""
        tokenizer_path = model_path
        self.model_config = ModelConfig(
            model_name,
            model_path,
            self.server_args.trust_remote_code,
            context_length=self.server_args.context_length,
            model_override_args=json.loads(self.server_args.json_model_override_args),
        )
        if self.server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if is_multimodal_model(self.model_config.hf_config.architectures):
                self.processor = get_processor(
                    tokenizer_path,
                    tokenizer_mode=self.server_args.tokenizer_mode,
                    trust_remote_code=self.server_args.trust_remote_code,
                )
                self.tokenizer = self.processor.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
            else:
                self.tokenizer = get_tokenizer(
                    tokenizer_path,
                    tokenizer_mode=self.server_args.tokenizer_mode,
                    trust_remote_code=self.server_args.trust_remote_code,
                )
        self.context_len = self.server_args.context_length or get_context_length(self.model_config.hf_config)

    def _send_token_tput_updates(self):
        """周期性向detokenizer发送吞吐量更新"""
        while True:
            try:
                time.sleep(1.0)
                current_time = time.time()
                elapsed_time = current_time - self.last_tput_update_time
                if elapsed_time > 0 and self._activated:
                    # 吞吐量计算
                    token_tput = (self.token_count / elapsed_time if self.token_count > 0 else 0.0)
                    prefill_token_tput = (
                        self.prefill_token_count / elapsed_time
                        if self.prefill_token_count > 0
                        else 0.0
                    )
                    decode_token_tput = (
                        self.decode_token_count / elapsed_time
                        if self.decode_token_count > 0
                        else 0.0
                    )

                    # 发送吞吐量更新
                    update = UpdateModelTput(
                        model_name=self.model_name,
                        instance_idx=self.instance_idx,
                        latest_token_tput=token_tput,
                        prefill_token_tput=prefill_token_tput,
                        decode_token_tput=decode_token_tput,
                        token_count=self.token_count,
                        prefill_token_count=self.prefill_token_count,
                        decode_token_count=self.decode_token_count,
                    )
                    self.send_to_detokenizer.send_pyobj(update)

                    # 重置统计数据
                    self.token_count = 0
                    self.prefill_token_count = 0
                    self.decode_token_count = 0
                    self.last_tput_update_time = current_time
            except Exception as e:
                logger.error(f"Error in token throughput update thread: {e}")

    @torch.inference_mode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        result_queue = deque()
        self.last_batch = None
        self.running_batch = None
        while True:
            recv_other_requests = self.recv_other_requests()
            recv_gpu_scheduler_requests = self.recv_gpu_scheduler_requests()
            non_gen_reqs = recv_other_requests + recv_gpu_scheduler_requests
            deactivate_req = self.process_input_other_requests(non_gen_reqs)
            if self._activated:
                recv_generation_requests = self.recv_generation_requests()
                self.process_input_gen_requests(recv_generation_requests)
                batch = self.get_next_batch_to_run()
                self.cur_batch = batch
                if batch:
                    result = self.run_batch(batch)
                    result_queue.append((batch.copy(), result))
                else:
                    self.batch_is_full = False
                if self.last_batch:
                    tmp_batch, tmp_result = result_queue.popleft()
                    self.process_batch_result(tmp_batch, tmp_result)
                elif batch is None:
                    time.sleep(0.001)
                    self.check_memory()
                    self.new_token_ratio = global_config.init_new_token_ratio
                self.last_batch = batch
                if deactivate_req:
                    self.handle_deactivate_request(deactivate_req, result_queue)
            else:
                time.sleep(0.001)

    @torch.inference_mode()
    def event_loop_normal(self):
        """调度器主循环"""
        self.last_batch = None
        while True:
            recv_other_requests = self.recv_other_requests()
            recv_model_scheduler_requests = self.recv_model_scheduler_requests()
            recv_gpu_scheduler_requests = self.recv_gpu_scheduler_requests()
            if len(recv_gpu_scheduler_requests) > 0:
                logger.info(f"Scheduler: Received {len(recv_gpu_scheduler_requests)} requests from gpu scheduler")
            non_gen_reqs = recv_other_requests + recv_model_scheduler_requests + recv_gpu_scheduler_requests
            deactivate_req = self.process_input_other_requests(non_gen_reqs)
            if deactivate_req: self.handle_deactivate_request(deactivate_req)
            if self._activated:
                recv_generation_requests = self.recv_generation_requests()
                if len(recv_generation_requests) > 0: logger.info(f"Received {len(recv_generation_requests)} generation requests")
                self.process_input_gen_requests(recv_generation_requests)
                batch = self.get_next_batch_to_run()
                if batch:
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)
                    # 多步decode
                    if batch.forward_mode.is_decode():
                        for _ in range(self.server_args.num_continuous_decode_steps - 1):
                            if self.running_batch is None: break
                            self.update_running_batch()
                            if self.running_batch is None: break
                            result = self.run_batch(batch)
                            self.process_batch_result(batch, result)
                else:
                    self.batch_is_full = False
                    time.sleep(0.001)
                    self.check_memory()
                    self.new_token_ratio = global_config.init_new_token_ratio
                self.last_batch = batch
                self.update_memory_usage()
            else:
                time.sleep(0.001)

    @torch.inference_mode()
    def _run_to_completion_overlap(self, result_queue: deque):
        batch = self.get_next_batch_to_run()
        if batch is not None:
            logger.info(f"Number of requests to run to completion: {batch.batch_size()}")
        while batch or len(self.waiting_queue) > 0:
            if batch:
                self.cur_batch = batch
                result = self.run_batch(batch)
                result_queue.append((batch.copy(), result))

            if self.last_batch:
                tmp_batch, tmp_result = result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)

            self.last_batch = batch
            batch = self.get_next_batch_to_run()

        if self.last_batch:
            tmp_batch, tmp_result = result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
            self.last_batch = None

    @torch.inference_mode()
    def _run_to_completion_normal(self):
        batch = self.get_next_batch_to_run()
        if batch is not None:
            logger.info(f"Number of requests to run to completion: {batch.batch_size()}")
        while batch or len(self.waiting_queue) > 0:
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)

                # Decode multiple steps to reduce the overhead
                if batch.forward_mode.is_decode():
                    for _ in range(self.server_args.num_continuous_decode_steps - 1):
                        if self.running_batch is None: break
                        self.update_running_batch()
                        if self.running_batch is None: break
                        result = self.run_batch(batch)
                        self.process_batch_result(batch, result)
            self.last_batch = batch
            batch = self.get_next_batch_to_run()

        self.last_batch = None

    def recv_generation_requests(self) -> List[GenerateReqInput]:
        """接收生成请求
        激进计算剩余prefill空间，添加prefill请求
        """
        recv_reqs: List[GenerateReqInput] = []
        if self.tp_rank == 0:
            # 最大input长度，采用默认值32768
            rem_input_tokens = self.max_prefill_tokens if self.max_prefill_tokens is not None else 32768
            # chunked prefill块长度
            # rem_chunk_tokens = self.chunked_prefill_size if self.chunked_prefill_size is not None else 8192
            # 计算剩余prefill数
            # approx_remain_prefill_tokens = min(self.rem_tokens, rem_input_tokens, rem_chunk_tokens)
            approx_remain_prefill_tokens = min(self.rem_tokens, rem_input_tokens)
            max_prefill_count = (int(approx_remain_prefill_tokens // self.avg_input_len) + 1)
            max_fetch_count = max_prefill_count - len(self.waiting_queue)
            if max_fetch_count > 0: # 最大接收数量导致waiting不一致
                recv_reqs = self.redis_client.recv_pyobj_non_block(
                    key=f"{self.server_args.backend_generate_request_key_prefix}:{self.model_name}",
                    count=max_fetch_count,
                )
        else:
            recv_reqs = []
        if self.tp_size != 1:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)
        return recv_reqs

    def recv_model_scheduler_requests(self) -> List[ResizeChunkInput]:
        """接收model scheduler调度请求"""
        if self.tp_rank == 0:
            recv_reqs = []
            while True:
                try:
                    recv_req: ResizeChunkInput = self.recv_from_model_scheduler.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                if recv_req.model_name == self.model_name:
                    recv_reqs.append(recv_req)
        else:
            recv_reqs = []
        if self.tp_size != 1:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)
        return recv_reqs[-1:]
    
    def recv_gpu_scheduler_requests(self) -> List:
        """接收GPU调度请求"""
        if self.tp_rank == 0:
            recv_reqs = []
            while True:
                try:
                    recv_req = self.recv_from_gpu_scheduler.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = []
        if self.tp_size != 1:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)
        return recv_reqs
    
    def recv_other_requests(self) -> List:
        """接收其他类型请求"""
        if self.tp_rank == 0:
            recv_reqs = []
            while True:
                try:
                    recv_req = self.recv_from_request_handler.recv_pyobj(zmq.NOBLOCK)
                    logger.info(f"Scheduler: Received request from request handler: {recv_req}")
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = []
        if self.tp_size != 1:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)
        return recv_reqs

    def process_input_gen_requests(self, recv_reqs: List[GenerateReqInput]):
        """批量tokenize及请求处理"""
        for recv_req in recv_reqs:
            assert isinstance(recv_req, GenerateReqInput)
            tokenized_req = self._tokenize(recv_req)
            self.handle_generate_request(tokenized_req)

    def process_input_other_requests(self, recv_reqs: List) -> Optional[DeactivateReqInput]:
        """处理控制请求
        停止时收到启动+停止时，返回停止请求
        """
        recv_deactivate_reqs = []
        recv_activate_reqs = []
        for recv_req in recv_reqs:
            if isinstance(recv_req, FlushCacheReq):
                self.flush_cache()
            elif isinstance(recv_req, AbortReq):
                self.abort_request(recv_req)
            elif isinstance(recv_req, UpdateWeightReqInput):
                success, message = self.update_weights(recv_req)
                self.send_to_detokenizer.send_pyobj(UpdateWeightReqOutput(success, message))
            elif isinstance(recv_req, ProfileReq):
                if recv_req == ProfileReq.START_PROFILE:
                    self.start_profile()
                else:
                    self.stop_profile()
            elif isinstance(recv_req, GetMemPoolSizeReq):
                rid = recv_req.rid
                self.send_to_detokenizer.send_pyobj(GetMemPoolSizeReqOutput(rid, self.max_total_num_tokens))
            elif isinstance(recv_req, GetMemoryUsageReq):
                rid = recv_req.rid
                memory_usage = self.get_memory_usage()
                self.send_to_detokenizer.send_pyobj(GetMemoryUsageReqOutput(rid, self.model_name, memory_usage))
            elif isinstance(recv_req, ActivateReqInput):
                recv_activate_reqs.append(recv_req)
            elif isinstance(recv_req, DeactivateReqInput):
                recv_deactivate_reqs.append(recv_req)
            elif isinstance(recv_req, ResizeMemPoolReqInput):
                self.resize_mem_pool(recv_req)
            elif isinstance(recv_req, ResizeChunkInput):
                logger.info(f"Received request: resize prefill chunk size to {recv_req.chunk_size}")
                self.chunked_prefill_size = recv_req.chunk_size
            else:
                logger.warning(f"Received invalid request: {recv_req}")
        deactivate_req = self.process_activate_deactivate_requests(recv_activate_reqs, recv_deactivate_reqs)
        return deactivate_req

    def process_activate_deactivate_requests(
        self,
        recv_activate_reqs: List[ActivateReqInput],
        recv_deactivate_reqs: List[DeactivateReqInput]
    ) -> Optional[DeactivateReqInput]:
        """模型启动/停止"""
        if len(recv_activate_reqs) > 1:
            logger.error(f"Received multiple activate requests: {recv_activate_reqs}")
        if len(recv_deactivate_reqs) > 1:
            logger.error(f"Received multiple deactivate requests: {recv_deactivate_reqs}")
        if not recv_activate_reqs and not recv_deactivate_reqs: return None
        activate_req = recv_activate_reqs[0] if recv_activate_reqs else None
        deactivate_req = recv_deactivate_reqs[0] if recv_deactivate_reqs else None
        # 单独启动或停止
        if not (activate_req and deactivate_req):
            if activate_req:
                logger.info("Processing activate request")
                self.handle_activate_request(activate_req)
            else:
                logger.info("Processing deactivate request")
                self.handle_deactivate_request(deactivate_req)
            return None
        logger.info("Processing both activate and deactivate requests")
        if self._activated: # 已启动，先停止后重启
            logger.info("Currently activated - deactivating first")
            self.handle_deactivate_request(deactivate_req)
            self.handle_activate_request(activate_req)
            return None
        else: # 未启动，启动，返回停止
            logger.info("Currently deactivated - activating first")
            self.handle_activate_request(activate_req)
            return deactivate_req

    def _tokenize(self, obj: GenerateReqInput):
        rid = obj.rid
        input_text = obj.text
        input_ids = self.tokenizer.encode(input_text) if obj.input_ids is None else obj.input_ids
        self._validate_input_length(input_ids)
        sampling_params = self._get_sampling_params(obj.sampling_params)
        assert obj.image_data is None, "Image inputs are not supported."
        tokenized_obj = TokenizedGenerateReqInput(
            rid,
            input_text,
            input_ids,
            None,
            sampling_params,
            obj.return_logprob,
            obj.logprob_start_len,
            obj.top_logprobs_num,
            obj.stream,
            obj.lora_path,
            obj.arrival_time,
            obj.slo,
        )
        return tokenized_obj

    def _get_sampling_params(self, sampling_params_data: dict) -> SamplingParams:
        sampling_params = SamplingParams(**sampling_params_data)
        if sampling_params.max_new_tokens != 0:
            sampling_params.normalize(self.tokenizer)
            sampling_params.verify()
        return sampling_params

    def _validate_input_length(self, input_ids: List[int]):
        """验证输入长度符合限制"""
        if len(input_ids) >= self.context_len:
            raise ValueError(
                f"The input ({len(input_ids)} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

    def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
        """对输入初始化请求对象"""
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            lora_path=recv_req.lora_path,
            arrival_time=recv_req.arrival_time,
            slo=recv_req.slo,
        )
        req.tokenizer = self.tokenizer
        req.stream = recv_req.stream
        req.return_logprob = recv_req.return_logprob
        req.logprob_start_len = recv_req.logprob_start_len
        req.top_logprobs_num = recv_req.top_logprobs_num
        # 默认只对输出token计算logprob
        if req.logprob_start_len == -1: req.logprob_start_len = len(recv_req.input_ids) - 1
        # 初始化 regex FSM
        if (req.sampling_params.json_schema is not None or req.sampling_params.regex is not None):
            if req.sampling_params.json_schema is not None:
                req.regex_fsm, computed_regex_string = self.regex_fsm_cache.query(("json", req.sampling_params.json_schema))
            elif req.sampling_params.regex is not None:
                req.regex_fsm, computed_regex_string = self.regex_fsm_cache.query(("regex", req.sampling_params.regex))
            if not self.disable_regex_jump_forward:
                req.jump_forward_map = self.jump_forward_cache.query(computed_regex_string)
        # 截断过长输入
        if len(req.origin_input_ids) > self.max_req_input_len:
            logger.warning("Request length is longer than the KV cache pool size or the max context length. Truncated!!!")
            req.origin_input_ids = req.origin_input_ids[:self.max_req_input_len]
        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )
        # 加入等待队列
        self.waiting_queue.append(req)
        self.num_received_requests += 1
        self.total_input_len += len(req.origin_input_ids)

    def print_decode_stats(self):
        used_tokens = self.max_total_num_tokens - self.rem_tokens
        throughput = self.num_generated_tokens / (time.time() - self.last_stats_tic)
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()
        num_running_reqs = len(self.running_batch.reqs) if self.running_batch else 0
        logger.info(
            f"Decode batch. "
            f"#running-req: {num_running_reqs}, "
            f"#token: {used_tokens}, "
            f"token usage: {used_tokens / self.max_total_num_tokens:.2f}, "
            f"gen throughput (token/s): {throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}"
        )

    def check_memory(self):
        # the checks does not make sense for the elastic case
        # comment out for now
        # available_size = (
        #     self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        # )
        # if available_size != self.max_total_num_tokens:
        #     warnings.warn(
        #         "Warning: "
        #         f"available_size={available_size}, max_total_num_tokens={self.max_total_num_tokens}\n"
        #         "KV cache pool leak detected!"
        #     )
        #     exit(1) if crash_on_warning else None
        # if len(self.req_to_token_pool.free_slots) != self.req_to_token_pool.size:
        #     warnings.warn(
        #         "Warning: "
        #         f"available req slots={len(self.req_to_token_pool.free_slots)}, "
        #         f"total slots={self.req_to_token_pool.size}\n"
        #         "Memory pool leak detected!"
        #     )
        #     exit(1) if crash_on_warning else None
        return

    def get_next_batch_to_run(self) -> ScheduleBatch:
        """获取下一运行batch
        Returns:
            ScheduleBatch: 下一运行批次
        """
        # 1. Inflight req：从last batch分离inflight batch，剩余running batch合并运行
        if self.last_batch and not self.last_batch.forward_mode.is_decode() and not self.last_batch.is_empty():
            if self.current_inflight_req:
                self.last_batch.filter_batch(current_inflight_req=self.current_inflight_req)
                for req in self.current_inflight_req:
                    self.tree_cache.cache_unfinished_req(req)
                    self.req_to_token_pool.free(req.req_pool_idx)
                self.batch_is_full = False
            if not self.last_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)
        # 2. Prefill req：内存充足，有请求等待时优先添加prefill，mix则将新batch与running batch合并运行
        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            self.send_to_controller.send_pyobj(
                BatchRunReq(
                    rids=[req.rid for req in new_batch.reqs],
                    model=self.model_config.name,
                    run_time=time.time(),
                    gpu_id=self.gpu_id,
                )
            )
            return new_batch
        # 3. Decode req：自回归运行decoding请求
        if self.running_batch is None: return
        before_bs = self.running_batch.batch_size()
        self.update_running_batch()
        if self.running_batch is None or before_bs != self.running_batch.batch_size(): self.batch_is_full = False
        return self.running_batch

    def _should_abort_req(self, req: Req) -> bool:
        if req.slo is not None and time.time() + 0.5 - req.arrival_time > req.slo:
            return True
        return False

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        # 无等待prefill请求
        if (self.batch_is_full or len(self.waiting_queue) == 0) and not self.current_inflight_req:
            return None
        # 运行请求数超限则停止添加
        running_bs = len(self.running_batch.reqs) if self.running_batch is not None else 0
        if running_bs >= self.max_running_requests:
            self.batch_is_full = True
            return None
        # 内存优先保障decode
        if self.running_batch is not None:
            reserved_for_decode = global_config.retract_decode_steps * self.running_batch.batch_size()
            if self.rem_tokens < reserved_for_decode: return None
        # 根据调度策略排序等待队列
        prefix_computed = self.policy.calc_priority(self.waiting_queue)
        # Prefill添加策略
        num_mixed_running = running_bs if self.is_mixed_chunk else 0
        adder = PrefillAdder(
            self.tree_cache,
            self.running_batch,
            self.new_token_ratio,
            self.rem_tokens,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            num_mixed_running,
            self.enable_elastic_memory,
        )
        # 更新chunk缓存
        for req in self.current_inflight_req: req.init_next_round_input(None if prefix_computed else self.tree_cache)
        self.current_inflight_req = adder.add_inflight_req(self.current_inflight_req)
        if self.lora_paths is not None:
            lora_set = set([req.lora_path for req in self.running_batch.reqs]) if self.running_batch is not None else set([])
        # 从等待队列添加请求至prefill batch
        abort_reqs = []
        for req in self.waiting_queue:
            # 抛弃策略：超过SLO则拒绝服务
            if self.server_args.abort_exceed_slos and self._should_abort_req(req):
                abort_reqs.append(req)
                continue
            if (
                self.lora_paths is not None
                and len(
                    lora_set
                    | set([req.lora_path for req in adder.can_run_list])
                    | set([req.lora_path])
                )
                > self.max_loras_per_batch
            ):
                self.batch_is_full = True
                break
            if running_bs + len(adder.can_run_list) >= self.max_running_requests: # 运行请求数超限则停止添加
                self.batch_is_full = True
                break
            # 添加至prefill batch
            req.init_next_round_input(None if prefix_computed else self.tree_cache)
            res = adder.add_one_req(req)
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.batch_is_full = True
                break
        # 移除遗弃请求
        if self.server_args.abort_exceed_slos:
            self.abort_exceed_slo_reqs(abort_reqs)
        # 更新等待队列
        can_run_list = adder.can_run_list
        if len(can_run_list) == 0: return None
        self.waiting_queue = [x for x in self.waiting_queue if x not in set(can_run_list)]
        for req in can_run_list: req.set_out_queue_time()
        # 对新增请求添加chunked prefill
        self.current_inflight_req.extend(adder.new_inflight_req)
        for req in self.current_inflight_req:
            req.is_inflight_req += 1
        # 打印统计数据
        if self.tp_rank == 0:
            if isinstance(self.tree_cache, RadixCache):
                self.tree_cache_metrics["total"] += (adder.log_input_tokens + adder.log_hit_tokens) / 10**9
                self.tree_cache_metrics["hit"] += (adder.log_hit_tokens) / 10**9
                tree_cache_hit_rate = (self.tree_cache_metrics["hit"] / self.tree_cache_metrics["total"])
            else:
                tree_cache_hit_rate = 0.0
            used_tokens = self.max_total_num_tokens - self.rem_tokens
            if num_mixed_running > 0:
                logger.info(
                    f"Prefill batch"
                    f"(mixed #running-req: {num_mixed_running}). "
                    f"#new-seq: {len(can_run_list)}, "
                    f"#new-token: {adder.log_input_tokens}, "
                    f"#cached-token: {adder.log_hit_tokens}, "
                    f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
                    f"token usage: {used_tokens / self.max_total_num_tokens:.2f}, "
                    f"#queue-req: {len(self.waiting_queue)}, "
                    f"#inflight-req: {len(self.current_inflight_req)}"
                )
            else:
                logger.info(
                    f"Prefill batch. "
                    f"#new-seq: {len(can_run_list)}, "
                    f"#new-token: {adder.log_input_tokens}, "
                    f"#cached-token: {adder.log_hit_tokens}, "
                    f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
                    f"token usage: {used_tokens / self.max_total_num_tokens:.2f}, "
                    f"#running-req: {running_bs}, "
                    f"#queue-req: {len(self.waiting_queue)}, "
                    f"#inflight-req: {len(self.current_inflight_req)}"
                )
        # 初始化batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
            self.model_config,
        )
        new_batch.prepare_for_extend()
        # Mixed chunked prefill：合并运行新batch与decoding batch
        if self.is_mixed_chunk and self.running_batch is not None:
            # self.running_batch.prepare_for_decode(self.enable_overlap)
            self.update_running_batch()
            if self.running_batch is not None:
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = None
        else:
            new_batch.decoding_reqs = None
        new_batch.decoding_reqs = None
        return new_batch

    def update_running_batch(self):
        """更新decoding batch信息
        过滤已完成请求
        检查剩余显存
        更新显存映射
        """
        global test_retract
        self.running_batch.filter_batch() # 过滤已完成请求
        if self.running_batch.is_empty():
            self.running_batch = None
            return
        # 检查显存剩余
        if not self.running_batch.check_decode_mem() or (test_retract and self.running_batch.batch_size() > 10):
            old_ratio = self.new_token_ratio
            retracted_reqs, new_token_ratio = self.running_batch.retract_decode() # 撤回请求以释放空间
            if new_token_ratio == 0.0: new_token_ratio = old_ratio
            # 撤回情况发送至controller
            self.send_to_controller.send_pyobj(
                BatchRetractDecodeReq(
                    rids=[req.rid for req in retracted_reqs],
                    len_output_ids=[len(req.output_ids) for req in retracted_reqs],
                    model=self.model_config.name,
                    retract_time=time.time(),
                )
            )
            self.new_token_ratio = new_token_ratio
            logger.info(
                "Decode out of memory happened. "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )
            self.waiting_queue.extend(retracted_reqs)
        else:
            self.new_token_ratio = max(self.new_token_ratio - self.new_token_ratio_decay, self.min_new_token_ratio)
        # 检查jump-forward
        if not self.disable_regex_jump_forward:
            jump_forward_reqs = self.running_batch.check_for_jump_forward(self.pad_input_ids_func)
            self.waiting_queue.extend(jump_forward_reqs)
            if self.running_batch.is_empty():
                self.running_batch = None
                return
        # 更新显存映射张量
        self.running_batch.prepare_for_decode(self.enable_overlap)

    def run_batch(self, batch: ScheduleBatch):
        """运行batch"""
        if self.is_generation: # 生成模型
            if batch.forward_mode.is_decode() or batch.extend_num_tokens != 0:
                model_worker_batch = batch.get_model_worker_batch()
                logits_output, next_token_ids = self.tp_worker.forward_batch_generation(model_worker_batch)
            else:
                logits_output = None
                next_token_ids = torch.full((batch.batch_size(),), self.tokenizer.eos_token_id if self.tokenizer is not None else 0)
            batch.output_ids = next_token_ids
            ret = logits_output, next_token_ids, model_worker_batch.bid
        else: # embedding/reward模型
            assert batch.extend_num_tokens != 0
            model_worker_batch = batch.get_model_worker_batch()
            embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
            ret = embeddings, model_worker_batch.bid
        return ret

    def process_batch_result(self, batch: ScheduleBatch, result):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result)
            if batch.is_empty():
                self.running_batch = None
        else:
            self.process_batch_result_prefill(batch, result)
        # 更新请求信息
        prefill_timestamps: Dict[str, float] = {}
        finish_timestamps: Dict[str, Tuple[float, int, float]] = {}
        for req in batch.reqs:
            if req.finish_timestamp is not None:
                finish_timestamps[req.rid] = (req.prefill_finish_timestamp, len(req.output_ids), req.finish_timestamp)
            elif req.prefill_finish_timestamp is not None:
                prefill_timestamps[req.rid] = req.prefill_finish_timestamp
        message = UpdateQueueStats(
            model_name=self.model_name,
            prefill_timestamps=prefill_timestamps,
            finish_timestamps=finish_timestamps,
            inflight_reqs=len(self.current_inflight_req),
            running_reqs=len(self.running_batch.reqs) if self.running_batch is not None else 0,
            chunk_size=self.chunked_prefill_size
        )
        self.send_to_model_scheduler.send_pyobj(message)
                

    def process_batch_result_prefill(self, batch: ScheduleBatch, result):
        if self.is_generation:
            logits_output, next_token_ids, bid = result
            if self.enable_overlap:
                logits_output, next_token_ids = self.tp_worker.resulve_batch_result(bid)
            else:
                # Move next_token_ids and logprobs to cpu
                if batch.return_logprob: # False
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs[
                            torch.arange(len(next_token_ids), device=f"{self.device}:{self.gpu_id}",),
                            next_token_ids,
                        ].tolist()
                    )
                    logits_output.input_token_logprobs = (logits_output.input_token_logprobs.tolist())
                    logits_output.normalized_prompt_logprobs = (logits_output.normalized_prompt_logprobs.tolist())
                next_token_ids = next_token_ids.tolist()

            # Check finish conditions
            logprob_pt = 0
            for i, req in enumerate(batch.reqs):
                # req.set_prefill_finish_time()
                if req.is_inflight_req > 0:
                    req.is_inflight_req -= 1
                else:
                    # Inflight reqs' prefill is not finished
                    if req.prefill_finish_timestamp is None: req.set_prefill_finish_time()
                    req.completion_tokens_wo_jump_forward += 1
                    req.output_ids.append(next_token_ids[i])
                    req.check_finished()
                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                        req.set_finish_time()
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        self.tree_cache.cache_unfinished_req(req)
                    if req.regex_fsm is not None:
                        req.regex_fsm_state = req.regex_fsm.get_next_state(req.regex_fsm_state, next_token_ids[i])
                    if req.return_logprob:
                        logprob_pt += self.add_logprob_return_values(i, req, logprob_pt, next_token_ids, logits_output)

            # XXX, MMY: Update prefill token count (to be checked if it is correct)
            prefill_tokens = 0
            for req in batch.reqs:
                prefill_tokens += req.extend_input_len
            self.prefill_token_count += prefill_tokens
            self.token_count += prefill_tokens
        else:  # embedding or reward model
            embeddings, bid = result
            embeddings = embeddings.tolist()
            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                req.embedding = embeddings[i]
                if req.is_inflight_req > 0:
                    req.is_inflight_req -= 1
                else:
                    # Inflight reqs' prefill is not finished
                    # dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                if req.finished():
                    self.tree_cache.cache_finished_req(req)
                    req.set_finish_time()
                else:
                    self.tree_cache.cache_unfinished_req(req)
        self.stream_output(batch.reqs)

    def process_batch_result_decode(self, batch: ScheduleBatch, result):
        logits_output, next_token_ids, bid = result
        self.num_generated_tokens += len(batch.reqs)

        # XXX, MMY: update decode token count basically same to self.num_generated_tokens.
        decode_tokens = len(batch.reqs)
        self.decode_token_count += decode_tokens
        self.token_count += decode_tokens

        if self.enable_overlap:
            logits_output, next_token_ids = self.tp_worker.resulve_batch_result(bid)
            next_token_logprobs = logits_output.next_token_logprobs
        else:
            # Move next_token_ids and logprobs to cpu
            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs[
                    torch.arange(
                        len(next_token_ids), device=f"{self.device}:{self.gpu_id}"
                    ),
                    next_token_ids,
                ].tolist()
            next_token_ids = next_token_ids.tolist()

        self.token_to_kv_pool.free_group_begin()

        # Check finish condition
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            req.append_decode_time()
            if self.server_args.enable_overlap_schedule and req.finished():
                self.token_to_kv_pool.free(batch.out_cache_loc[i : i + 1])
                continue

            req.completion_tokens_wo_jump_forward += 1
            req.output_ids.append(next_token_id)
            req.check_finished()

            if req.regex_fsm is not None:
                req.regex_fsm_state = req.regex_fsm.get_next_state(
                    req.regex_fsm_state, next_token_id
                )

            if req.finished():
                req.set_finish_time()
                self.tree_cache.cache_finished_req(req)

            if req.return_logprob:
                req.output_token_logprobs.append(
                    (next_token_logprobs[i], next_token_id)
                )
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs.append(logits_output.output_top_logprobs[i])

        self.stream_output(batch.reqs)

        self.token_to_kv_pool.free_group_end()

        self.decode_forward_ct = (self.decode_forward_ct + 1) % (1 << 30)
        if self.tp_rank == 0 and self.decode_forward_ct % 40 == 0:
            self.print_decode_stats()

    def add_logprob_return_values(
        self,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        req.output_token_logprobs.append(
            (output.next_token_logprobs[i], next_token_ids[i])
        )

        # If logprob_start_len > 0, then first logprob_start_len prompt tokens will be ignored.
        num_input_logprobs = req.extend_input_len - req.extend_logprob_start_len

        if req.normalized_prompt_logprob is None:
            req.normalized_prompt_logprob = output.normalized_prompt_logprobs[i]

        if req.input_token_logprobs is None:
            input_token_logprobs = output.input_token_logprobs[
                pt : pt + num_input_logprobs - 1 - req.last_update_decode_tokens
            ]
            input_token_ids = req.fill_ids[
                len(req.fill_ids)
                - num_input_logprobs
                + 1 : len(req.fill_ids)
                - req.last_update_decode_tokens
            ]
            req.input_token_logprobs = list(zip(input_token_logprobs, input_token_ids))

            if (
                req.logprob_start_len == 0
            ):  # The first token does not have logprob, pad it.
                req.input_token_logprobs = [
                    (None, req.fill_ids[0])
                ] + req.input_token_logprobs

        if req.last_update_decode_tokens != 0:
            # Some decode tokens are re-computed in an extend batch
            req.output_token_logprobs.extend(
                list(
                    zip(
                        output.input_token_logprobs[
                            pt
                            + num_input_logprobs
                            - 1
                            - req.last_update_decode_tokens : pt
                            + num_input_logprobs
                            - 1
                        ],
                        req.fill_ids[
                            len(req.fill_ids)
                            - req.last_update_decode_tokens : len(req.fill_ids)
                        ],
                    )
                )
            )

        if req.top_logprobs_num > 0:
            if req.input_top_logprobs is None:
                req.input_top_logprobs = output.input_top_logprobs[i]
                if req.logprob_start_len == 0:
                    req.input_top_logprobs = [None] + req.input_top_logprobs

            if req.last_update_decode_tokens != 0:
                req.output_top_logprobs.extend(
                    output.input_top_logprobs[i][-req.last_update_decode_tokens :]
                )
            req.output_top_logprobs.append(output.output_top_logprobs[i])

        return num_input_logprobs

    def stream_output(self, reqs: List[Req]):
        """Stream the output to detokenizer."""
        output_rids = []
        output_meta_info = []
        output_finished_reason: List[BaseFinishReason] = []
        if self.is_generation:
            output_vids = []
            decoded_texts = []
            output_read_ids = []
            output_read_offsets = []
            output_skip_special_tokens = []
            output_spaces_between_special_tokens = []
            output_no_stop_trim = []
        else:  # embedding or reward model
            output_embeddings = []

        is_stream_iter = self.decode_forward_ct % self.stream_interval == 0

        for req in reqs:
            if req.finished() or (
                req.stream and (is_stream_iter or len(req.output_ids) == 1)
            ):
                output_rids.append(req.rid)
                output_finished_reason.append(req.finished_reason)
                if self.is_generation:
                    output_vids.append(req.vid)
                    decoded_texts.append(req.decoded_text)
                    read_ids, read_offset = req.init_incremental_detokenize()
                    output_read_ids.append(read_ids)
                    output_read_offsets.append(read_offset)
                    output_skip_special_tokens.append(
                        req.sampling_params.skip_special_tokens
                    )
                    output_spaces_between_special_tokens.append(
                        req.sampling_params.spaces_between_special_tokens
                    )
                    output_no_stop_trim.append(req.sampling_params.no_stop_trim)

                    meta_info = {
                        "prompt_tokens": len(req.origin_input_ids),
                        "completion_tokens": len(req.output_ids),
                        "completion_tokens_wo_jump_forward": req.completion_tokens_wo_jump_forward,
                        "cached_tokens": req.cached_tokens,
                        "finish_reason": (
                            req.finished_reason.to_json()
                            if req.finished_reason is not None
                            else None
                        ),
                        "arrival_timestamp": req.arrival_time,
                        "out_queue_timestamp": req.out_queue_timestamp,
                        "prefill_finish_timestamp": req.prefill_finish_timestamp,
                        "finish_timestamp": req.finish_timestamp,
                        "decode_timestamps": req.decode_timestamps,
                    }
                    if req.return_logprob:
                        (
                            meta_info["input_token_logprobs"],
                            meta_info["output_token_logprobs"],
                            meta_info["input_top_logprobs"],
                            meta_info["output_top_logprobs"],
                            meta_info["normalized_prompt_logprob"],
                        ) = (
                            req.input_token_logprobs,
                            req.output_token_logprobs,
                            req.input_top_logprobs,
                            req.output_top_logprobs,
                            req.normalized_prompt_logprob,
                        )
                    output_meta_info.append(meta_info)
                else:  # embedding or reward model
                    output_embeddings.append(req.embedding)
                    meta_info = {
                        "prompt_tokens": len(req.origin_input_ids),
                    }
                    output_meta_info.append(meta_info)

        # Send to detokenizer
        if output_rids:
            if self.is_generation:
                self.send_to_detokenizer.send_pyobj(
                    BatchTokenIDOut(
                        output_rids,
                        output_vids,
                        decoded_texts,
                        output_read_ids,
                        output_read_offsets,
                        output_skip_special_tokens,
                        output_spaces_between_special_tokens,
                        output_meta_info,
                        output_finished_reason,
                        output_no_stop_trim,
                    )
                )
            else:  # embedding or reward model
                self.send_to_detokenizer.send_pyobj(
                    BatchEmbeddingOut(
                        output_rids,
                        output_embeddings,
                        output_meta_info,
                        output_finished_reason,
                    )
                )

    def update_memory_pool(self):
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        logger.info(
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
        )
        # 初始化内存池
        self.req_to_token_pool, self.token_to_kv_pool = self.tp_worker.get_memory_pool()
        if (self.server_args.chunked_prefill_size is not None and self.server_args.disable_radix_cache):
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
            )
        else:
            self.tree_cache = RadixCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                disable=self.server_args.disable_radix_cache,
            )
        self.tree_cache_metrics = {"total": 0, "hit": 0}
        self.policy = SchedulePolicy(self.schedule_policy, self.tree_cache)

    def get_memory_usage(self) -> MemoryUsage:
        return self.tp_worker.get_memory_usage()
    
    def update_memory_usage(self):
        memory_usage = self.get_memory_usage()
        memory_usage.token_to_kv_pool_memory = (1 - self.rem_tokens / self.max_total_num_tokens)
        self.send_to_controller.send_pyobj(GetMemoryUsageReqOutput("", self.model_name, memory_usage))

    def resize_mem_pool(self, recv_req: ResizeMemPoolReqInput) -> bool:
        """调整内存池大小
        当前占用>目标大小时调整失败
        """
        if_success = self.tp_worker.resize_memory_pool(recv_req.memory_pool_size)
        if if_success:
            worker_info = self.tp_worker.get_worker_info()
            self.max_total_num_tokens = worker_info[0]
            self.max_prefill_tokens = worker_info[1]
            self.max_running_requests = worker_info[2]
            self.max_req_len = worker_info[3]
            self.max_req_input_len = worker_info[4]
            logger.info(
                f"Memory pool resized successfully. "
                f"Target memory pool size={recv_req.memory_pool_size}. "
                f"New max_total_num_tokens={self.max_total_num_tokens}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}."
            )
        else:
            logger.warning(f"Can not resize memory pool because too many requests are running")
        return if_success

    def handle_activate_request(self, recv_req: ActivateReqInput):
        """处理模型启动请求"""
        # 启动前检查
        if self.tp_size > 1:
            gpu_id = self.gpu_id # 多卡未实现
        else:
            gpu_id = recv_req.gpu_id # 单卡可移动
        if self._activated: # 已启动
            if self.tp_rank == 0:
                logger.warning("Scheduler is already activated")
                activate_req_output = ActivateReqOutput(
                    rid=recv_req.rid,
                    gpu_id=gpu_id,
                    model_name=recv_req.model_name,
                    instance_idx=recv_req.instance_idx,
                    success=False, # 启动重复
                    memory_usage=self.get_memory_usage(),
                )
                self.send_to_detokenizer.send_pyobj(activate_req_output)
                self.redis_client.send_pyobj(
                    key=f"{self.server_args.engine_to_gpu_scheduler_key_prefix}:{self.gpu_id}",
                    obj=activate_req_output,
                )
            return
        logger.info(f"Scheduler receives the activate request with rid: {recv_req.rid}")
        # 禁止悬挂请求
        assert self.last_batch is None, "Last batch should be None"
        assert self.running_batch is None, "Running batch should be None"
        # 启动runner
        start_time = time.perf_counter()
        self.tp_worker.activate_model_runner(
            memory_pool_size=recv_req.memory_pool_size,
            gpu_id=gpu_id,
            model_name=recv_req.model_name,
        )
        if self.enable_worker_pool:
            self.model_name = recv_req.model_name
            self.model_config = self.tp_worker.get_model_config()
            self.tokenizer = self.tp_worker.get_tokenizer()
            self.context_len = self.server_args.context_length or get_context_length(self.model_config.hf_config)
            change_logger_format(prefix=f" GPU={recv_req.gpu_id} Worker {self.server_args.worker_id} ({self.model_name}) TP={self.tp_rank}")
        else: # 更新GPU和日志输出
            if recv_req.gpu_id is not None and self.tp_size == 1:
                self.gpu_id = recv_req.gpu_id
                change_logger_format(prefix=f" {self.model_name} GPU={recv_req.gpu_id} TP={self.tp_rank}")
        logger.info("Model runner activated")
        if self.first_time_activate:
            self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
            self.first_time_activate = False
        # 更新内存池
        self.update_memory_pool()
        logger.info("Memory pool updated")
        # 恢复等待队列
        self._restore_waiting_requests()
        logger.info("Waiting requests restored")
        # 启动完成
        self._activated = True
        logger.info(f"Scheduler activated. Activation takes {time.perf_counter() - start_time:.2f} s")
        if self.tp_size > 1:
            self.tp_worker.model_runner.tp_group.barrier()
        if self.tp_rank == 0:
            activate_req_output = ActivateReqOutput(
                rid=recv_req.rid,
                gpu_id=gpu_id,
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx,
                success=True,
                memory_usage=self.get_memory_usage(),
            )
            self.send_to_detokenizer.send_pyobj(activate_req_output) # 初始化detokenizer
            self.redis_client.send_pyobj(
                key=f"{self.server_args.engine_to_gpu_scheduler_key_prefix}:{self.gpu_id}",
                obj=activate_req_output,
            )

    def handle_deactivate_request(self, recv_req: DeactivateReqInput, result_queue: Optional[deque] = None):
        """处理模型停止请求"""
        if self.tp_size > 1:
            gpu_id = self.gpu_id # 多卡仅能在原地开关
        else:
            gpu_id = recv_req.gpu_id # 单卡可移动
        if not self._activated:
            if self.tp_rank == 0:
                logger.warning("Scheduler is already deactivated")
                deactivate_req_output = DeactivateReqOutput(
                    rid=recv_req.rid,
                    gpu_id=gpu_id,
                    model_name=recv_req.model_name,
                    instance_idx=recv_req.instance_idx,
                    success=False, # 停止重复
                    memory_usage=self.get_memory_usage(),
                )
                self.send_to_detokenizer.send_pyobj(deactivate_req_output)
                self.redis_client.send_pyobj(
                    key=f"{self.server_args.engine_to_gpu_scheduler_key_prefix}:{self.gpu_id}",
                    obj=deactivate_req_output,
                )
            return
        logger.info(f"Scheduler receives the deactivate requests with rid: {recv_req.rid}")
        start_time = time.perf_counter()
        logger.info(
            f"In handle deactivate request, waiting queue size: {len(self.waiting_queue)}. "
            f"Running batch size: {0 if self.running_batch is None else len(self.running_batch.reqs)}, "
            f"last batch size: {0 if self.last_batch is None else len(self.last_batch.reqs)}"
        )
        # 清空模型请求
        preempt = recv_req.preempt
        if recv_req.evict_waiting_requests or (preempt and recv_req.preempt_mode == PreemptMode.RECOMPUTE):
            # 清空等待队列，禁止悬挂
            self._evict_all_waiting_requests()
        # 根据抢占策略处理剩余请求
        if not preempt:
            if self.enable_overlap:
                self._run_to_completion_overlap(result_queue)
            else:
                self._run_to_completion_normal()
        else:
            preempt_mode = recv_req.preempt_mode
            if preempt_mode == PreemptMode.RETURN: # 完成剩余请求
                if self.enable_overlap:
                    self._complete_ongoing_requests_overlap(result_queue)
                else:
                    self._complete_ongoing_requests_normal()
            elif preempt_mode == PreemptMode.RECOMPUTE: # 撤回剩余请求
                if self.enable_overlap:
                    self._retract_running_batch_overlap(result_queue)
                else:
                    self._retract_running_batch_normal()
            else:
                raise NotImplementedError(f"Preempt mode {preempt_mode} is not supported")
        # 清空完成
        self._activated = False
        logger.info(f"Process ongoing requests with preempt ({preempt}) takes {time.perf_counter() - start_time:.4f} s")
        self.tp_worker.deactivate_model_runner()
        if self.tp_size > 1:
            self.tp_worker.model_runner.tp_group.barrier()
        if self.tp_rank == 0:
            deactivate_req_output = DeactivateReqOutput(
                rid=recv_req.rid,
                gpu_id=gpu_id,
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx,
                success=True,
                memory_usage=self.get_memory_usage(),
            )
            self.send_to_detokenizer.send_pyobj(deactivate_req_output)
            self.redis_client.send_pyobj(
                key=f"{self.server_args.engine_to_gpu_scheduler_key_prefix}:{self.gpu_id}",
                obj=deactivate_req_output,
            )
            logger.info(f"Total time taken for deactivation is {time.perf_counter() - start_time:.2f} s")

    def _convert_req_to_frontend_reqs(self, req: Req) -> GenerateReqInput:
        """对待清理请求生成传输对象"""
        sampling_params = {
            "max_new_tokens": req.sampling_params.max_new_tokens,
            "min_new_tokens": req.sampling_params.min_new_tokens,
            "stop": req.sampling_params.stop_strs,
            "stop_token_ids": (
                list(req.sampling_params.stop_token_ids)
                if req.sampling_params.stop_token_ids
                else None
            ),
            "temperature": req.sampling_params.temperature,
            "top_p": req.sampling_params.top_p,
            "top_k": req.sampling_params.top_k,
            "min_p": req.sampling_params.min_p,
            "frequency_penalty": req.sampling_params.frequency_penalty,
            "presence_penalty": req.sampling_params.presence_penalty,
            "repetition_penalty": req.sampling_params.repetition_penalty,
            "ignore_eos": req.sampling_params.ignore_eos,
            "skip_special_tokens": req.sampling_params.skip_special_tokens,
            "spaces_between_special_tokens": req.sampling_params.spaces_between_special_tokens,
            "regex": req.sampling_params.regex,
            "n": req.sampling_params.n,
            "json_schema": req.sampling_params.json_schema,
            "no_stop_trim": req.sampling_params.no_stop_trim,
        }
        generate_req = GenerateReqInput(
            rid=req.rid,
            text=req.origin_input_text if hasattr(req, "origin_input_text") else None,
            input_ids=req.origin_input_ids,
            sampling_params=sampling_params,
            return_logprob=req.return_logprob,
            logprob_start_len=req.logprob_start_len,
            top_logprobs_num=req.top_logprobs_num,
            stream=req.stream,
            lora_path=req.lora_path,
            arrival_time=req.arrival_time,
            slo=req.slo,
            image_data=None,
            model=self.model_name,
            prompt_len=len(req.origin_input_ids),
            output_len=512,
        )
        return generate_req

    def _evict_all_waiting_requests(self):
        if not self.waiting_queue:
            return
        try:
            # 将等待请求发送回前端
            for req in self.waiting_queue:
                self.redis_client.send_pyobj(
                    key=f"{self.server_args.frontend_generate_request_key_prefix}:{self.model_name}",
                    obj=self._convert_req_to_frontend_reqs(req),
                )
            logger.info(f"Evicted {len(self.waiting_queue)} requests back to Redis.")
        except Exception as e:
            logger.error(f"Error evicting requests to Redis: {e}")
        self.waiting_queue.clear()

    def _restore_waiting_requests(self):
        """恢复等待队列"""
        self.waiting_queue.extend(self.waiting_queue_stash)
        self.waiting_queue_stash = []

    def _complete_ongoing_requests_overlap(self, result_queue: deque):
        batch = self.get_next_batch_to_run()
        if batch is not None:
            logger.info(f"Number of requests to directly finish: {batch.batch_size()}")

        while batch or len(self.waiting_queue) > 0:
            if batch:
                for req in batch.reqs:
                    req.finished_reason = FINISH_LENGTH(len(req.output_ids))
                    req.set_finish_time()
                    if self.use_kvcached_v0:
                        try:
                            self.tree_cache.cache_finished_req(req)
                        except Exception as e:
                            logger.error(f"Error in cache_finished_req: {e}")
                self.stream_output(batch.reqs)

            if self.last_batch:
                tmp_batch, tmp_result = result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            batch = self.get_next_batch_to_run()
        self.last_batch = None
        self.batch_is_full = False

        assert self.running_batch is None, "Running batch should be None"

    def _complete_ongoing_requests_normal(self):
        batch = self.get_next_batch_to_run()
        if batch is not None:
            logger.info(f"Number of requests to directly finish: {batch.batch_size()}")

        while batch or len(self.waiting_queue) > 0:
            if batch:
                for req in batch.reqs:
                    req.finished_reason = FINISH_LENGTH(len(req.output_ids))
                    req.set_finish_time()
                    if self.use_kvcached_v0:
                        try:
                            self.tree_cache.cache_finished_req(req)
                        except Exception as e:
                            logger.error(f"Error in cache_finished_req: {e}")
                self.stream_output(batch.reqs)
            batch = self.get_next_batch_to_run()
        self.last_batch = None
        self.batch_is_full = False

        assert self.running_batch is None, "Running batch should be None"

    def _retract_running_batch_overlap(self, result_queue: deque):
        if self.last_batch:
            tmp_batch, tmp_result = result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        self._retract_running_batch_normal()

    def _retract_running_batch_normal(self):
        # merge the prefill batch into the running batch
        if (
            self.last_batch
            and not self.last_batch.forward_mode.is_decode()
            and not self.last_batch.is_empty()
        ):
            if self.current_inflight_req:
                self.last_batch.filter_batch(
                    current_inflight_req=self.current_inflight_req
                )
                self.tree_cache.cache_unfinished_req(self.current_inflight_req)
                # Inflight request keeps its rid but will get a new req_pool_idx.
                self.req_to_token_pool.free(self.current_inflight_req.req_pool_idx)
                self.batch_is_full = False
            if not self.last_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)
        logger.info(
            f"Retract running batch, running batch size: {0 if self.running_batch is None else len(self.running_batch.reqs)}, last batch size: {0 if self.last_batch is None else len(self.last_batch.reqs)}"
        )
        # retract running batch
        rids = []
        len_output_ids = []
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                req.prefix_indices = []
                req.last_node = None
                req.extend_input_len = 0
                req.last_update_decode_tokens = 0
                req.logprob_start_len = 10**9
                rids.append(req.rid)
                len_output_ids.append(len(req.output_ids))
                self.waiting_queue.append(req)
                if self.use_kvcached_v0:
                    try:
                        self.tree_cache.cache_finished_req(req)
                    except Exception as e:
                        logger.error(f"Error in cache_finished_req: {e}")

        self.send_to_controller.send_pyobj(
            BatchRetractDecodeReq(
                rids=rids,
                len_output_ids=len_output_ids,
                model=self.model_config.name,
                retract_time=time.time(),
            )
        )
        self.batch_is_full = False
        self.running_batch = None
        self.last_batch = None

    def flush_cache(self) -> bool:
        """清空并刷新缓存
        需要无运行中队列
        """
        if len(self.waiting_queue) == 0 and (self.running_batch is None or len(self.running_batch.reqs) == 0):
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool.clear()
            self.tree_cache_metrics = {"total": 0, "hit": 0}
            self.regex_fsm_cache.reset()
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            return True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
            )
            return False

    def abort_request(self, recv_req: AbortReq):
        # Delete requests in the waiting queue
        to_del = None
        for i, req in enumerate(self.waiting_queue):
            if req.rid == recv_req.rid:
                to_del = i
                break

        if to_del is not None:
            del self.waiting_queue[to_del]

        # Delete requests in the running batch
        if self.running_batch:
            for req in self.running_batch.reqs:
                if req.rid == recv_req.rid and not req.finished():
                    req.finished_reason = FINISH_ABORT()
                    self.tree_cache.cache_finished_req(req)
                    break

    def abort_exceed_slo_reqs(self, abort_reqs: List[Req]):
        for req in abort_reqs:
            self.waiting_queue.remove(req)
            req.finished_reason = FINISH_ABORT()
            req.set_finish_time()
        self.stream_output(abort_reqs)

    def update_weights(self, recv_req: UpdateWeightReqInput):
        """In-place update of the weights."""
        success, message = self.tp_worker.update_weights(recv_req)
        if success:
            flash_cache_success = self.flush_cache()
            assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return success, message

    def start_profile(self) -> None:
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self) -> None:
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()
        self.profiler.export_chrome_trace(
            self.torch_profiler_trace_dir + "/" + str(time.time()) + ".trace.json.gz"
        )
        logger.info("Profiler is done")

    @property
    def avg_input_len(self):
        if self.num_received_requests <= 1: return 1024
        return self.total_input_len / self.num_received_requests
    
    @property
    def rem_tokens(self):
        return self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()

    def _signal_handler(self, signum, frame):
        """Handle signals to ensure cleanup before exit"""
        self.cleanup()
        # Re-raise the signal to allow the default handler to run
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def cleanup(self):
        """Clean up all IPC sockets and files when exiting"""
        try:
            # Only clean up sockets for tp_rank 0 (which owns them)
            if self.tp_rank == 0:
                # Prepare sockets dictionary
                zmq_sockets = {}

                if hasattr(self, "recv_from_request_handler") and isinstance(
                    self.recv_from_request_handler, zmq.Socket
                ):
                    zmq_sockets["recv_from_request_handler"] = (
                        self.recv_from_request_handler
                    )

                if hasattr(self, "send_to_detokenizer") and isinstance(
                    self.send_to_detokenizer, zmq.Socket
                ):
                    zmq_sockets["send_to_detokenizer"] = self.send_to_detokenizer

                if hasattr(self, "send_to_controller") and isinstance(
                    self.send_to_controller, zmq.Socket
                ):
                    zmq_sockets["send_to_controller"] = self.send_to_controller

                # Clean up using utility function
                cleanup_zmq_ipc(
                    zmq_sockets=zmq_sockets,
                    ipc_files=getattr(self, "ipc_files", set()),
                    component_name="Scheduler",
                    gpu_id=self.gpu_id,
                    rank=self.tp_rank,
                )

                # Close redis client
                if hasattr(self, "redis_client") and self.redis_client is not None:
                    self.redis_client.close()

        except Exception as e:
            logger.error(
                f"Error during Scheduler cleanup for GPU {self.gpu_id} TP rank {self.tp_rank}: {e}"
            )

    def __del__(self):
        """Ensure cleanup when the object is garbage collected"""
        self.cleanup()


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    shared_cpu_models: Optional[Dict[str, List[nn.Module]]] = None,
    model_names_to_model_paths: Optional[Dict[str, str]] = None,
    engine_id: Optional[str] = None,
    input_queue: Optional[torch.multiprocessing.Queue] = None,
    output_queue: Optional[torch.multiprocessing.Queue] = None,
):
    if dp_rank is None:
        if server_args.enable_worker_pool:
            configure_logger(
                server_args,
                prefix=f" GPU={gpu_id} Worker={server_args.worker_id} TP{tp_rank}",
            )
        else:
            configure_logger(
                server_args,
                prefix=f" {server_args.model_name} GPU={gpu_id} TP{tp_rank}",
            )
    else:
        configure_logger(
            server_args,
            prefix=f" {server_args.model_name} GPU={gpu_id} DP{dp_rank} TP{tp_rank}",
        )

    suppress_other_loggers()

    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            dp_rank,
            shared_cpu_models,
            model_names_to_model_paths,
            engine_id,
            input_queue,
            output_queue,
        )
        mem_usage = scheduler.get_memory_usage()
        pipe_writer.send(mem_usage)
        # pipe_writer.send("ready")
        if server_args.enable_overlap_schedule:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()
