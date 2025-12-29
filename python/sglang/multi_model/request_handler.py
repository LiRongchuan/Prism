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

"""TokenizerManager is a process that tokenizes the text."""

import asyncio
import atexit
import dataclasses
import glob
import json
import logging
import os
import signal
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import fastapi
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks

from sglang.multi_model.multi_model_server_args import MultiModelServerArgs
from sglang.srt.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_processor,
    get_tokenizer,
)
from sglang.srt.managers.image_processor import (
    get_dummy_image_processor,
    get_image_processor,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActivateReqInput,
    ActivateReqOutput,
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    DeactivateReqInput,
    DeactivateReqOutput,
    EmbeddingReqInput,
    FinishReq,
    FlushCacheReq,
    GenerateReqInput,
    GetMemoryUsageReq,
    GetMemoryUsageReqOutput,
    GetMemPoolSizeReq,
    GetMemPoolSizeReqOutput,
    MemoryUsage,
    ProfileReq,
    ResizeMemPoolReqInput,
    RewardReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    TokenizedRewardReqInput,
    UpdateModelTput,
    UpdateWeightReqInput,
    UpdateWeightReqOutput,
)
from sglang.srt.redis_utils import AsyncRedisClient
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs
from sglang.srt.utils import is_generation_model, is_multimodal_model
from sglang.utils import cleanup_zmq_ipc, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: List
    finished: bool
    event: asyncio.Event


class RequestHandler:
    """RequestHandler is send requests to the redis generation queue and get response from the redis result queue."""

    def __init__(
        self,
        server_args: MultiModelServerArgs,
        port_args_dict: Dict[str, PortArgs],
        num_engines: int,
        ipc_name: str,
        gpu_id_to_model_instance: Dict[int, Dict[str, int]],
        controller_ipc_name: Optional[str] = None,
    ):
        self.server_args = server_args
        # Init inter-process communication
        num_detokenizer = 1
        num_controller = 1 if controller_ipc_name is not None else 0
        context = zmq.asyncio.Context(io_threads=num_engines + num_detokenizer + num_controller)

        # Store IPC file paths for cleanup
        self.ipc_files = set()
        self.ipc_files.add(f"ipc://{ipc_name}")

        # receive from all the detokenizers
        self.recv_from_detokenizer = context.socket(zmq.PULL)
        self.recv_from_detokenizer.bind(f"ipc://{ipc_name}")

        # for sending generation requests to scheduler
        self.redis_client = AsyncRedisClient(server_args.redis_host, server_args.redis_port, server_args.redis_db)

        # send requests other than generation, e.g. start/stop profiling, abort requests, control memory pool size, activate/deactivate models
        self.send_to_scheduler_dict = defaultdict(
            list
        )  # key: model_name, value: zmq sockets
        for model_name, port_args_list in port_args_dict.items():
            for port_args in port_args_list:
                send_to_scheduler = context.socket(zmq.PUSH)
                send_to_scheduler.connect(f"ipc://{port_args.scheduler_input_ipc_name}")
                self.send_to_scheduler_dict[model_name].append(send_to_scheduler)

        if controller_ipc_name is not None:
            # send requests to the controller
            self.send_to_controller = context.socket(zmq.PUSH)
            self.send_to_controller.connect(f"ipc://{controller_ipc_name}")
        else:
            self.send_to_controller = None

        self.gpu_id_to_model_instance = gpu_id_to_model_instance
        self.send_to_gpu_scheduler_dict: Dict[int, zmq.Socket] = {}
        if server_args.enable_gpu_scheduler:
            for gpu_id in gpu_id_to_model_instance.keys():
                ipc_name = f"request_handler_to_gpu_scheduler_{gpu_id}"
                self.ipc_files.add(f"ipc://{ipc_name}")
                self.send_to_gpu_scheduler_dict[gpu_id] = context.socket(zmq.PUSH)
                self.send_to_gpu_scheduler_dict[gpu_id].connect(f"ipc://{ipc_name}")
        # elif server_args.enable_model_scheduler:
        #     ipc_name = f"request_handler_to_model_scheduler"
        #     self.ipc_files.add(f"ipc://{ipc_name}")
        #     self.send_to_model_scheduler = context.socket(zmq.PUSH)
        #     self.send_to_model_scheduler.connect(f"ipc://{ipc_name}")
        # Register cleanup function
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        self.is_generation = True
        # Store states
        self.to_create_loop = True
        self.rid_to_state: Dict[str, ReqState] = {}

    def _signal_handler(self, signum, frame):
        """Handle signals to ensure cleanup before exit"""
        self.cleanup()
        # Re-raise the signal to allow the default handler to run
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def cleanup(self):
        """Clean up all IPC sockets and files when exiting"""
        try:
            # Prepare sockets dictionary
            zmq_sockets = {
                "recv_from_detokenizer": getattr(self, "recv_from_detokenizer", None),
                "send_to_controller": getattr(self, "send_to_controller", None),
            }

            # Add scheduler sockets
            if hasattr(self, "send_to_scheduler_dict"):
                for model_name, sockets in self.send_to_scheduler_dict.items():
                    for i, socket in enumerate(sockets):
                        zmq_sockets[f"send_to_scheduler_{model_name}_{i}"] = socket

            # Add GPU scheduler sockets
            if hasattr(self, "send_to_gpu_scheduler_dict"):
                for gpu_id, socket in self.send_to_gpu_scheduler_dict.items():
                    zmq_sockets[f"send_to_gpu_scheduler_{gpu_id}"] = socket
                    
            # if hasattr(self, "send_to_model_scheduler"):
            #     zmq_sockets["send_to_model_scheduler"] = self.send_to_model_scheduler
                
            # Clean up using utility function
            cleanup_zmq_ipc(
                zmq_sockets=zmq_sockets,
                ipc_files=getattr(self, "ipc_files", set()),
                component_name="RequestHandler",
            )

            # Clean up any leftover IPC files that match our pattern
            for pattern in ["request_handler_to_gpu_scheduler_*"]:
                for file_path in glob.glob(pattern):
                    try:
                        if os.path.exists(file_path):
                            os.unlink(file_path)
                            logger.info(f"Removed leftover IPC file: {file_path}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove leftover IPC file {file_path}: {e}"
                        )

            # Close redis client
            if hasattr(self, "redis_client"):
                asyncio.run(self.redis_client.close())

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Ensure cleanup when the object is garbage collected"""
        self.cleanup()

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput, RewardReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        if self.to_create_loop:
            self.create_handle_loop()

        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        obj.post_init()
        is_single = obj.is_single
        if is_single:
            async for response in self._handle_single_request(obj, request):
                yield response
        else:
            async for response in self._handle_batch_request(obj, request):
                yield response

    async def _send_single_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput, RewardReqInput],
        index: Optional[int] = None,
        input_id_index: Optional[int] = None,
        is_cache_for_prefill: Optional[bool] = False,
    ):
        logger.info(f"Sending single request for {obj.model}")
        single_request_obj = None
        if not is_cache_for_prefill:  # The normal case with a single prompt
            if index is None:
                single_request_obj = obj
            else:
                rid = obj.rid[index]
                if hasattr(obj, "conv"):
                    # reward model
                    raise ValueError("Reward model is not supported.")
                elif obj.input_ids is None:
                    input_text = obj.text[input_id_index]
                    input_ids = None
                else:
                    input_text = (obj.text[input_id_index] if obj.text is not None else None)
                    input_ids = obj.input_ids[input_id_index]

                sampling_params = obj.sampling_params[index]
                if self.is_generation:
                    image_inputs = await self.image_processor.process_images_async(
                        obj.image_data[index], input_text or input_ids, obj
                    )
                    if image_inputs and "input_ids" in image_inputs:
                        input_ids = image_inputs["input_ids"]
                    return_logprob = obj.return_logprob[index]
                    logprob_start_len = obj.logprob_start_len[index]
                    top_logprobs_num = obj.top_logprobs_num[index]

        else:  # A prefill request to cache the common prompt for parallel sampling
            assert self.is_generation
            if obj.text is not None:
                if isinstance(obj.text, list):
                    input_text = obj.text[input_id_index]
                    rid = obj.rid[index]
                else:
                    input_text = obj.text
                    rid = obj.rid[0]
                if obj.input_ids is not None:
                    input_ids = obj.input_ids
                    if isinstance(obj.input_ids, list) and isinstance(obj.input_ids[0], list):
                        # when obj["input_ids"] is List[List[int]]
                        input_ids = obj.input_ids[input_id_index]
                        rid = obj.rid[index]
                    else:
                        input_ids = obj.input_ids
                        rid = obj.rid[0]
                else:
                    input_ids = None
            else:
                input_text = None
                if isinstance(obj.input_ids, list) and isinstance(obj.input_ids[0], list):
                    # when obj["input_ids"] is List[List[int]]
                    input_ids = obj.input_ids[input_id_index]
                    rid = obj.rid[index]
                else:
                    input_ids = obj.input_ids
                    rid = obj.rid[0]

            sampling_params = obj.sampling_params[0]
            sampling_params.max_new_tokens = 0
            image_inputs = await self.image_processor.process_images_async(obj.image_data[0], input_text or input_ids, obj)
            if image_inputs and "input_ids" in image_inputs:
                input_ids = image_inputs["input_ids"]
            return_logprob = obj.return_logprob[0]
            logprob_start_len = obj.logprob_start_len[0]
            top_logprobs_num = obj.top_logprobs_num[0]

        if single_request_obj is None:
            lora_path = (
                obj.lora_path[input_id_index]
                if isinstance(obj.lora_path, list)
                else obj.lora_path
            )
            single_request_obj = GenerateReqInput(
                rid=rid,
                text=input_text,
                input_ids=input_ids,
                image_data=image_inputs,
                sampling_params=sampling_params,
                return_logprob=return_logprob,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=top_logprobs_num,
                stream=obj.stream,
                lora_path=lora_path,
                model=obj.model,
                slo=obj.slo,
                slo_ttft=obj.slo_ttft,
                slo_tpot=obj.slo_tpot,
                prompt_len=len(input_text)
            )

        assert self.is_generation, "Only generation models are supported."

        # Send request to the generation queue with model name in the key
        model = obj.model
        try:
            # 向scheduler发送
            # enable-gpu-scheduler：前端 -> gpu_scheduler -> 后端
            # enable-model-scheduler：前端 -> model_scheduler -> 后端
            # 否则前端等于后端
            await self.redis_client.send_pyobj(
                key=f"{self.server_args.frontend_generate_request_key_prefix}:{model}",
                obj=single_request_obj,
            )
        except asyncio.exceptions.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error when sending request to {model}: {e}")
            raise e

        # send request to the controller to update the model queue info
        if self.send_to_controller:
            self.send_to_controller.send_pyobj(single_request_obj)
        return single_request_obj.rid, single_request_obj.input_ids

    async def _handle_single_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput, RewardReqInput],
        request: Optional[fastapi.Request] = None,
        index: Optional[int] = None,
        input_id_index: Optional[int] = None,
        is_cache_for_prefill: Optional[bool] = False,
    ):
        try:
            rid, input_ids = await self._send_single_request(
                obj,
                index,
                input_id_index=input_id_index,
                is_cache_for_prefill=is_cache_for_prefill,
            )
        except Exception as e:
            logger.error(f"Error when handling single request for {obj.model}: {e}")
            raise e

        # Recv results
        event = asyncio.Event()
        state = ReqState([], False, event)
        self.rid_to_state[rid] = state

        if not is_cache_for_prefill:
            async for response in self._wait_for_response(state, obj, rid, request):
                yield response
        else:
            assert self.is_generation
            await self._wait_for_cache_prefill_response(state, obj, rid, request)
            yield input_ids

    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput, RewardReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        batch_size = obj.batch_size
        if self.is_generation:
            parallel_sample_num = obj.parallel_sample_num

            if parallel_sample_num != 1:
                # Send prefill requests to cache the common prefix
                parallel_sample_num += 1
                input_id_result = [] if obj.input_ids is None else None
                for i in range(batch_size):
                    async for input_id in self._handle_single_request(
                        obj,
                        request,
                        index=i,
                        input_id_index=i,
                        is_cache_for_prefill=True,
                    ):
                        if input_id_result is not None:
                            input_id_result.append(input_id)
                if input_id_result is not None:
                    obj.input_ids = input_id_result
        else:
            parallel_sample_num = 1

        # First send out all requests
        generators = []
        for i in range(batch_size):
            for j in range(parallel_sample_num):
                if j == 0 and parallel_sample_num != 1:
                    continue
                index = i * parallel_sample_num + j
                if parallel_sample_num != 1:
                    # Here when using parallel sampling we should consider prefill stage so the index is :  j + i * (parallel_sample_num-1) + batch_size - 1
                    index += batch_size - 1 - i

                rid, _ = await self._send_single_request(
                    obj, index, input_id_index=i, is_cache_for_prefill=False
                )

                event = asyncio.Event()
                state = ReqState([], False, event)
                self.rid_to_state[rid] = state

                generators.append(
                    self._wait_for_response(
                        state,
                        obj,
                        rid,
                        request,
                        index=index,
                        response_index=len(generators),
                    )
                )

        # Then process the responses based on streaming option
        is_stream = hasattr(obj, "stream") and obj.stream

        tasks = [asyncio.create_task(gen.__anext__()) for gen in generators]
        output_list = [None] * len(tasks)

        # Fetch results
        while tasks:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                cur_index = tasks.index(task)

                try:
                    result = task.result()

                    if is_stream:
                        yield result
                    else:
                        output_list[result["index"]] = result

                    tasks[cur_index] = asyncio.create_task(
                        generators[cur_index].__anext__()
                    )
                except StopAsyncIteration:
                    del generators[cur_index]
                    del tasks[cur_index]

        if not is_stream:
            yield output_list

    async def _wait_for_response(
        self,
        state: ReqState,
        obj: Union[GenerateReqInput, EmbeddingReqInput, RewardReqInput],
        rid: str,
        request: Optional[fastapi.Request] = None,
        index: Optional[int] = None,
        response_index: int = 0,
    ):
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    for rid in [obj.rid] if obj.is_single else obj.rid:
                        self.abort_request(rid, obj.model)
                    raise ValueError(f"Abort request {rid}")
                continue

            if self.is_generation:
                out = self.convert_logprob_style(
                    state.out_list[-1],
                    obj.return_logprob if index is None else obj.return_logprob[index],
                    (
                        obj.top_logprobs_num
                        if index is None
                        else obj.top_logprobs_num[index]
                    ),
                    obj.return_text_in_logprobs,
                )
            else:  # isinstance(obj, (EmbeddingReqInput, RewardReqInput))
                out = state.out_list[-1]

            out["index"] = response_index

            # Log requests
            if self.server_args.log_requests and state.finished:
                logger.info(f"in={obj}, out={out}")

            state.out_list = []
            if state.finished:
                del self.rid_to_state[rid]
                finish_obj = FinishReq(
                    rid=rid,
                    model=obj.model,
                    finish_time=time.time(),
                    is_warmup=obj.is_warmup,
                )
                if self.send_to_controller:
                    self.send_to_controller.send_pyobj(finish_obj)
                yield out
                break

            state.event.clear()
            yield out

    async def _wait_for_cache_prefill_response(
        self,
        state: ReqState,
        obj: GenerateReqInput,
        rid: str,
        request: Optional[fastapi.Request] = None,
    ):
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
                break
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    for rid in obj.rid:
                        self.abort_request(rid, obj.model)
                    raise ValueError(f"Abort request {rid}")
                continue

        assert state.finished
        del self.rid_to_state[rid]

    def flush_cache(self, req: FlushCacheReq):
        model_name = req.model_name
        instance_idx = req.instance_idx
        send_to_scheduler = self.send_to_scheduler_dict[model_name][instance_idx]
        send_to_scheduler.send_pyobj(req)

    def abort_request(self, rid: str, model_name: str):
        if rid not in self.rid_to_state:
            return
        del self.rid_to_state[rid]
        req = AbortReq(rid)
        send_to_schedulers = self.send_to_scheduler_dict[model_name]
        for send_to_schedulers in send_to_schedulers:
            send_to_schedulers.send_pyobj(req)

    def start_profile(self):
        req = ProfileReq.START_PROFILE
        for send_to_schedulers in self.send_to_scheduler_dict.values():
            for send_to_scheduler in send_to_schedulers:
                send_to_scheduler.send_pyobj(req)

    def stop_profile(self):
        req = ProfileReq.STOP_PROFILE
        for send_to_schedulers in self.send_to_scheduler_dict.values():
            for send_to_scheduler in send_to_schedulers:
                send_to_scheduler.send_pyobj(req)

    def _send_req_to_scheduler(
        self,
        req: Union[ActivateReqInput, DeactivateReqInput, ResizeMemPoolReqInput, GetMemoryUsageReq],
        model_name: Optional[str] = None,
    ):
        # send request to the corresponding scheduler
        if model_name is None:
            model_name = req.model_name
        instance_idx = req.instance_idx
        send_to_scheduler = self.send_to_scheduler_dict[model_name][instance_idx]
        send_to_scheduler.send_pyobj(req)

    async def get_memory_pool_size(self, req: GetMemPoolSizeReq):
        if self.to_create_loop:
            self.create_handle_loop()

        model_name = req.model_name
        instance_idx = req.instance_idx
        send_to_scheduler = self.send_to_scheduler_dict[model_name][instance_idx]
        send_to_scheduler.send_pyobj(req)
        self.mem_pool_size = asyncio.Future()
        return await self.mem_pool_size

    def resize_mem_pool(self, req: ResizeMemPoolReqInput):
        self._send_req_to_scheduler(req)

    async def _send_req_and_wait_for_response(
        self, req: Union[ActivateReqInput, DeactivateReqInput, GetMemoryUsageReq]
    ):
        # send request to the corresponding scheduler
        self._send_req_to_scheduler(req)

        # await for the response
        rid = req.rid
        event = asyncio.Event()
        state = ReqState([], False, event)
        self.rid_to_state[rid] = state

        # wait for the response
        try:
            await state.event.wait()
            success = state.finished
            out = state.out_list[-1]
        except:
            logger.error(f"Error when sending request {req} and waiting for response: {get_exception_traceback()}")
            raise
        finally:
            if rid in self.rid_to_state:
                del self.rid_to_state[rid]
        return success, out

    async def get_memory_usage(self, req: GetMemoryUsageReq) -> MemoryUsage:
        success, out = await self._send_req_and_wait_for_response(req)
        assert success
        return out

    async def deactivate(self, req: DeactivateReqInput):
        if not self.server_args.enable_worker_pool:
            model_name = req.model_name
        else:
            model_name = req.gpu_id

        if self.server_args.enable_gpu_scheduler:
            gpu_id = None
            for gpu, model_instances in self.gpu_id_to_model_instance.items():
                if model_name in model_instances:
                    gpu_id = gpu
                    break
            if gpu_id is not None:
                send_to_gpu_scheduler = self.send_to_gpu_scheduler_dict[gpu_id]
                req.gpu_id = gpu_id  # TODO: this step should better be from controller
                send_to_gpu_scheduler.send_pyobj(req)
                # remove model from gpu_id_to_model_instance
                if (
                    gpu_id in self.gpu_id_to_model_instance
                    and model_name in self.gpu_id_to_model_instance[gpu_id]
                ):
                    del self.gpu_id_to_model_instance[gpu_id][model_name]
                    logger.info(f"Model {model_name} removed from GPU {gpu_id} mapping")

                self._send_req_to_scheduler(req, model_name)
            else:
                logger.info(f"Cannot find GPU ID for model {model_name}")
            success, memory_usage = True, None  # compatible with the old version
        else:
            success, memory_usage = await self._send_req_and_wait_for_response(req)
        # delete model from gpu_id_to_model_instance
        for gpu_id, model_instances in self.gpu_id_to_model_instance.items():
            if model_name in model_instances:
                del model_instances[model_name]
                logger.info(f"Model {model_name} removed from GPU {gpu_id} mapping")
                break
        return success, memory_usage

    async def activate(self, req: ActivateReqInput):
        if not self.server_args.enable_worker_pool:
            model_name = req.model_name
        else:
            model_name = req.gpu_id

        if self.server_args.enable_gpu_scheduler:
            gpu_id = req.gpu_id
            send_to_gpu_scheduler = self.send_to_gpu_scheduler_dict[gpu_id]
            send_to_gpu_scheduler.send_pyobj(req)

            # add model to gpu_id_to_model_instance
            instance_idx = req.instance_idx

            if gpu_id not in self.gpu_id_to_model_instance:
                self.gpu_id_to_model_instance[gpu_id] = {}

            self.gpu_id_to_model_instance[gpu_id][model_name] = instance_idx
            logger.info(f"Model {model_name} added to GPU {gpu_id} mapping")

            self._send_req_to_scheduler(req, model_name)
            success, memory_usage = True, None  # compatible with the old version
        else:
            if self.to_create_loop:
                self.create_handle_loop()
            success, memory_usage = await self._send_req_and_wait_for_response(req)
        # add to gpu_id_to_model_instance
        instance_idx = req.instance_idx
        self.gpu_id_to_model_instance[req.gpu_id][model_name] = instance_idx
        logger.info(f"Model {model_name} added to GPU {req.gpu_id} mapping")
        return success, memory_usage

    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(3)
            model_name = obj.model
            if obj.is_single:
                self.abort_request(obj.rid, model_name)
            else:
                for rid in obj.rid:
                    self.abort_request(rid, model_name)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    def create_handle_loop(self):
        if not self.to_create_loop:
            return

        self.to_create_loop = False
        loop = asyncio.get_event_loop()
        loop.create_task(self.handle_loop())

    async def handle_loop(self):
        """The event loop that handles requests"""

        while True:
            recv_obj: Union[
                BatchStrOut,
                BatchEmbeddingOut,
                BatchTokenIDOut,
                UpdateWeightReqOutput,
                GetMemPoolSizeReqOutput,
                ActivateReqOutput,
                DeactivateReqOutput,
                GetMemoryUsageReqOutput,
                UpdateModelTput,
            ] = await self.recv_from_detokenizer.recv_pyobj()

            if isinstance(recv_obj, UpdateWeightReqOutput):
                self.model_update_result.set_result(recv_obj)
                continue
            elif isinstance(recv_obj, GetMemPoolSizeReqOutput):
                self.mem_pool_size.set_result(recv_obj)
                continue
            elif isinstance(recv_obj, UpdateModelTput):
                self.send_to_controller.send_pyobj(recv_obj)
                continue
            elif (
                isinstance(recv_obj, GetMemoryUsageReqOutput)
                or isinstance(recv_obj, ActivateReqOutput)
                or isinstance(recv_obj, DeactivateReqOutput)
            ):
                rid = recv_obj.rid
                state = self.rid_to_state.get(rid, None)
                memory_usage = recv_obj.memory_usage

                if state is None:
                    logger.error(
                        f"State not found for received {type(recv_obj)} with rid {rid}"
                    )
                    continue
                state.out_list.append(memory_usage)
                state.finished = (
                    recv_obj.success
                    if isinstance(recv_obj, ActivateReqOutput)
                    or isinstance(recv_obj, DeactivateReqOutput)
                    else True
                )
                state.event.set()
                del self.rid_to_state[rid]
                continue

            assert isinstance(
                recv_obj, (BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut)
            ), f"Unexpected obj received: {type(recv_obj)}"

            for i, rid in enumerate(recv_obj.rids):
                state = self.rid_to_state.get(rid, None)
                if state is None:
                    continue

                recv_obj.meta_info[i]["id"] = rid
                if isinstance(recv_obj, BatchStrOut):
                    out_dict = {
                        "text": recv_obj.output_strs[i],
                        "meta_info": recv_obj.meta_info[i],
                    }
                elif isinstance(recv_obj, BatchTokenIDOut):
                    read_start = 0 if i == 0 else recv_obj.read_offsets[i - 1]
                    out_dict = {
                        "token_ids": recv_obj.decode_ids[
                            read_start : recv_obj.read_offsets[i]
                        ],
                        "meta_info": recv_obj.meta_info[i],
                    }

                else:
                    assert isinstance(recv_obj, BatchEmbeddingOut)
                    out_dict = {
                        "embedding": recv_obj.embeddings[i],
                        "meta_info": recv_obj.meta_info[i],
                    }
                state.out_list.append(out_dict)
                state.finished = recv_obj.finished_reason[i] is not None
                state.event.set()

    def convert_logprob_style(
        self,
        ret: dict,
        return_logprob: bool,
        top_logprobs_num: int,
        return_text_in_logprobs: bool,
    ):
        if return_logprob:
            raise ValueError("return_logprob is not supported.")
            ret["meta_info"]["input_token_logprobs"] = self.detokenize_logprob_tokens(
                ret["meta_info"]["input_token_logprobs"], return_text_in_logprobs
            )
            ret["meta_info"]["output_token_logprobs"] = self.detokenize_logprob_tokens(
                ret["meta_info"]["output_token_logprobs"], return_text_in_logprobs
            )

            if top_logprobs_num > 0:
                ret["meta_info"]["input_top_logprobs"] = (
                    self.detokenize_top_logprobs_tokens(
                        ret["meta_info"]["input_top_logprobs"],
                        return_text_in_logprobs,
                    )
                )
                ret["meta_info"]["output_top_logprobs"] = (
                    self.detokenize_top_logprobs_tokens(
                        ret["meta_info"]["output_top_logprobs"], return_text_in_logprobs
                    )
                )
        return ret

    def detokenize_logprob_tokens(
        self, token_logprobs: List[Tuple[float, int]], decode_to_text: bool
    ):
        # TODO(lianmin): This should run on DetokenizerManager
        if not decode_to_text:
            return [(logprob, token_id, None) for logprob, token_id in token_logprobs]

        assert self.tokenizer is not None
        token_ids = [tid for _, tid in token_logprobs]
        token_texts = self.tokenizer.batch_decode(token_ids)
        return [
            (logprob, token_id, token_text)
            for (logprob, token_id), token_text, in zip(token_logprobs, token_texts)
        ]

    def detokenize_top_logprobs_tokens(self, top_logprobs, decode_to_text: bool):
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        for i, token_top_logprobs in enumerate(top_logprobs):
            if token_top_logprobs:
                top_logprobs[i] = self.detokenize_logprob_tokens(
                    token_top_logprobs, decode_to_text
                )
        return top_logprobs
