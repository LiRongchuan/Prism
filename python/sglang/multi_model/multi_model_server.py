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

"""
The entry point of inference server.
SRT = SGLang Runtime.
"""

import asyncio
import atexit
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time
from http import HTTPStatus
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import orjson
import torch

from sglang.utils import get_exception_traceback

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse, Response, StreamingResponse
from uvicorn.config import LOGGING_CONFIG

from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.multi_model.model_sevice import ModelService
from sglang.multi_model.multi_model_server_args import MultiModelServerArgs
from sglang.multi_model.request_handler import RequestHandler
from sglang.multi_model.request_handler_worker_pool import RequestHandlerWorkerPool

from sglang.multi_model.scheduling.controller_global import run_controller_process
from sglang.multi_model.scheduling.gpu.gpu_scheduler import run_gpu_scheduler_process
from sglang.multi_model.utils.load_cpu_model import (
    init_torch_distributed_tp_1,
    load_shared_cpu_model,
)
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import (
    ActivateReqInput,
    DeactivateReqInput,
    EmbeddingReqInput,
    FlushCacheReq,
    GenerateReqInput,
    GetMemPoolSizeReq,
    MemoryUsage,
    ResizeMemPoolReqInput,
    RewardReqInput,
    UpdateWeightReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.redis_utils import RedisClient
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    add_api_key_middleware,
    assert_pkg_version,
    configure_logger,
    is_port_available,
    kill_child_process,
    maybe_set_triton_cache_manager,
    prepare_model_and_tokenizer,
    set_ulimit,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


app = FastAPI()
request_handler = None
model_names = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """Check the health of the inference server by generating one token."""
    gri = GenerateReqInput(
        text="s", sampling_params={"max_new_tokens": 1, "temperature": 0.7}
    )
    try:
        async for _ in request_handler.generate_request(gri, request):
            break
        return Response(status_code=200)
    except Exception as e:
        logger.exception(e)
        return Response(status_code=503)


@app.get("/get_model_names")
async def get_model_names():
    """Get the model names."""
    return list(model_names)


@app.get("/flush_cache")
async def flush_cache(req: FlushCacheReq):
    """Flush the radix cache."""
    request_handler.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200,
    )


@app.get("/start_profile")
@app.post("/start_profile")
async def start_profile():
    """Start profiling."""
    request_handler.start_profile()
    return Response(
        content="Start profiling.\n",
        status_code=200,
    )


@app.get("/stop_profile")
@app.post("/stop_profile")
async def stop_profile():
    """Stop profiling."""
    request_handler.stop_profile()
    return Response(
        content="Stop profiling. This will take some time.\n",
        status_code=200,
    )


@app.api_route("/get_memory_pool_size", methods=["GET", "POST"])
async def get_memory_pool_size(obj: GetMemPoolSizeReq):
    """Get the memory pool size in number of tokens"""
    try:
        ret = await request_handler.get_memory_pool_size(obj)
        return ret.size
    except Exception as e:
        logger.error(f"Error: {get_exception_traceback()}")
        return JSONResponse(
            {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
        )


@app.api_route("/deactivate", methods=["GET", "POST"])
async def deactivate(obj: DeactivateReqInput):
    """Deactivate a model."""
    tic = time.time()
    try:
        success, memory_usage = await request_handler.deactivate(obj)
        logger.info(f"[Server] Deactivate time cost: {time.time() - tic:.4f}s")
        return ORJSONResponse(
            {
                "success": success,
                "message": "Model deactivated successfully",
                "memory_usage": memory_usage,
            }
        )
    except Exception as e:
        logger.error(f"Error: {get_exception_traceback()}")
        return ORJSONResponse(
            {"success": False, "error": {"message": str(e), "type": type(e).__name__}},
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.api_route("/activate", methods=["GET", "POST"])
async def activate(obj: ActivateReqInput):
    """Activate a model."""
    try:
        success, memory_usage = await request_handler.activate(obj)
        return ORJSONResponse(
            {
                "success": success,
                "message": "Model activated successfully",
                "memory_usage": memory_usage,
            }
        )
    except Exception as e:
        return ORJSONResponse(
            {"success": False, "error": {"message": str(e), "type": type(e).__name__}},
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.api_route("/resize_mem_pool", methods=["GET", "POST"])
async def resize_mem_pool(obj: ResizeMemPoolReqInput):
    """Resize the memory pool."""
    request_handler.resize_mem_pool(obj)
    return Response(status_code=200)


# fastapi implicitly converts json in the request to obj (dataclass)
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in request_handler.generate_request(obj, request):
                    yield b"data: " + orjson.dumps(
                        out, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
            except ValueError as e:
                logger.error(f"Error: {get_exception_traceback()}")
                out = {"error": {"message": str(e)}}
                yield b"data: " + orjson.dumps(
                    out, option=orjson.OPT_NON_STR_KEYS
                ) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=request_handler.create_abort_task(obj),
        )
    else:
        try:
            ret = await request_handler.generate_request(obj, request).__anext__()
            return ret
        except ValueError as e:
            return ORJSONResponse(
                {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
            )


app.post("/generate")(generate_request)
app.put("/generate")(generate_request)


@dataclasses.dataclass
class EngineInfo:
    port_args: PortArgs
    model_path: str
    # scheduler_procs: List[mp.Process]
    gpu_ids: List[int]
    model_name: str
    instance_idx: int
    memory_usage: MemoryUsage
    init_memory_pool_size: float
    on: bool = True


def launch_engine(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_ids: Optional[List[int]] = None,
    instance_idx: Optional[int] = 0,
    shared_cpu_models: Optional[Dict[Tuple[str, int], List[Any]]] = None,
    model_names_to_model_paths: Optional[Dict[str, str]] = None,
    engine_id: Optional[str] = None,
    input_queue=None,
    output_queue=None,
):
    """
    Launch the Scheduler in a subprocess, and the Detokenizer Manager in another subprocess.
    """
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        scheduler_procs = []
        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        assert len(tp_rank_range) == len(gpu_ids)
        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = gpu_ids[tp_rank % tp_size_per_node]
            proc = mp.Process(
                target=run_scheduler_process,
                args=(
                    server_args,
                    port_args,
                    gpu_id,
                    tp_rank,
                    None,
                    writer,
                    shared_cpu_models,
                    model_names_to_model_paths,
                    engine_id,
                    input_queue,
                    output_queue,
                ),
            )
            proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)

        if server_args.node_rank >= 1:
            # For other nodes, they do not need to run tokenizer or detokenizer,
            # so they can just wait here.
            while True:
                pass
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
            model_names_to_model_paths,
        ),
    )
    detoken_proc.start()

    # Wait for model to finish loading
    for i in range(len(scheduler_pipe_readers)):
        memory_usage = scheduler_pipe_readers[i].recv()

    if server_args.enable_worker_pool:
        logger.info(
            f"Worker {server_args.worker_id} of GPU {gpu_ids[0]} loaded in process {scheduler_procs[i].pid}"
        )
    else:
        logger.info(
            f"Model {server_args.model_name} instance {instance_idx} loaded in process {scheduler_procs[i].pid}"
        )
    return EngineInfo(
        port_args=port_args,
        # scheduler_procs=scheduler_procs,
        gpu_ids=gpu_ids,
        model_name=server_args.model_name,
        model_path=server_args.model_path,
        instance_idx=instance_idx,
        memory_usage=memory_usage,
        on=server_args.on,
        init_memory_pool_size=server_args.max_memory_pool_size,
    )


def launch_request_handler(
    server_args: MultiModelServerArgs,
    port_args_dict: Dict[str, PortArgs],
    request_handler_ipc_name: str,
    num_engines: int,
    gpu_id_to_model_instance: Dict[int, Dict[str, int]],
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
    controller_ipc_name: Optional[str] = None,
):
    """
    Launch request_handler for sending requests and receive responses.
    The request_handler runs in the same process (the main process) as the HTTP server.
    """
    # check whether the redis server is running
    try:
        redis_client = RedisClient(
            server_args.redis_host, server_args.redis_port, server_args.redis_db
        )
        redis_client.client.ping()
    except Exception as e:
        logger.error(
            f"Redis server is not running at {server_args.redis_host}:{server_args.redis_port}. Please start the redis server first."
        )
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(get_exception_traceback())
        kill_child_process(os.getpid(), including_parent=False)

    # clear the redis queue
    redis_client.clear_queue()

    global request_handler
    if not server_args.enable_worker_pool:
        request_handler = RequestHandler(
            server_args,
            port_args_dict,
            num_engines=num_engines,
            ipc_name=request_handler_ipc_name,
            gpu_id_to_model_instance=gpu_id_to_model_instance,
            controller_ipc_name=controller_ipc_name,
        )
    else:
        request_handler = RequestHandlerWorkerPool(
            server_args,
            port_args_dict,
            num_engines=num_engines,
            num_gpus=server_args.num_gpus,
            ipc_name=request_handler_ipc_name,
            controller_ipc_name=controller_ipc_name,
        )


def launch_controller(
    multi_model_server_args: MultiModelServerArgs,
    recv_from_request_handler_ipc_name: str,
    recv_from_schedulers_ipc_name: str,
    engine_info_dict: Dict[str, List[EngineInfo]],
    model_names_to_model_paths: Dict[str, str],
    init_placements: Optional[Dict[str, List[int]]] = None,
):
    controller_proc = mp.Process(
        target=run_controller_process,
        args=(
            multi_model_server_args,
            recv_from_request_handler_ipc_name,
            recv_from_schedulers_ipc_name,
            engine_info_dict,
            model_names_to_model_paths,
            init_placements,
        ),
    )
    controller_proc.start()
    print("Controller process started.")


def launch_gpu_scheduler_process(
    multi_model_server_args: MultiModelServerArgs,
    model_names_to_model_paths: Dict[str, str],
    engine_info_dict: Dict[str, List[EngineInfo]],
    gpu_id: Tuple[int],
    init_model_names: List[str],
):
    reader, writer = mp.Pipe(duplex=False)
    gpu_scheduler_proc = mp.Process(
        target=run_gpu_scheduler_process,
        args=(
            multi_model_server_args,
            engine_info_dict,
            model_names_to_model_paths,
            gpu_id,
            init_model_names,
            writer,
        ),
    )
    gpu_scheduler_proc.start()
    reader.recv()
    logger.info(f"GPU scheduler process started for GPU {gpu_id}")


def launch_http_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
):
    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Send a warmup request
    t = threading.Thread(
        target=_wait_and_warmup, args=(server_args, pipe_finish_writer, os.getpid())
    )
    t.start()

    try:
        # Listen for HTTP requests
        LOGGING_CONFIG["formatters"]["default"][
            "fmt"
        ] = "[%(asctime)s] %(levelprefix)s %(message)s"
        LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
        LOGGING_CONFIG["formatters"]["access"][
            "fmt"
        ] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
        LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        t.join()


def load_shared_cpu_models(
    multi_model_server_args: MultiModelServerArgs,
):
    """
    Load shared CPU models.
    """
    print("Initializing torch distributed...")
    init_torch_distributed_tp_1(device="cpu")
    print("Torch distributed initialized.")
    model_configs = multi_model_server_args.model_configs
    path_to_shared_cpu_models = {}
    # Each model is identified with (model_path, tp_size) tuple
    model_ids = set(
        (model_config.model_path, model_config.tp_size)
        for model_config in model_configs
    )
    model_server_args = [
        ServerArgs(model_path=model_path, tp_size=tp_size)
        for model_path, tp_size in model_ids
    ]
    max_tp_size = max(args.tp_size for args in model_server_args)
    if multi_model_server_args.enable_cpu_share_memory:
        print(f"Loading {len(model_ids)} shared models to cpu...")
        tic = time.time()
        # NOTE: Load models sequentially to avoid TP size/rank error setting across multi-threads
        max_workers = len(model_ids) if max_tp_size == 1 or len(model_ids) == 1 else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            shared_cpu_models = list(
                executor.map(load_shared_cpu_model, model_server_args)
            )
        toc = time.time()
        print(f"Shared models loaded in {toc - tic:.2f} seconds.")
        path_to_shared_cpu_models = dict(zip(model_ids, shared_cpu_models))
    return path_to_shared_cpu_models



def launch_model_service(
    multi_model_server_args: MultiModelServerArgs,
    path_to_shared_cpu_models: Dict[str, torch.nn.Module],
    engine_ids: List[str],
):
    """
    Launch model service.
    """
    cpu_model_dict = {}
    for model_path, tp_size in path_to_shared_cpu_models.keys():
        cpu_model_dict[model_path] = path_to_shared_cpu_models[(model_path, tp_size)][0]
    num_devices = torch.cuda.device_count()
    gpu_ids = list(range(num_devices))
    input_queue = torch.multiprocessing.Queue()
    output_queues = {
        engine_id: torch.multiprocessing.Queue() for engine_id in engine_ids
    }
    max_loading_threads = min(4, num_devices)
    num_shards = 1
    num_model_service_workers = multi_model_server_args.num_model_service_workers
    num_model_service_workers = 1
    from sglang.multi_model.model_sevice import run_model_service
    for service_worker_id in range(num_model_service_workers):
        p = torch.multiprocessing.Process(
            target=run_model_service,
            args=(
                multi_model_server_args,
                cpu_model_dict,
                input_queue,
                output_queues,
                max_loading_threads,
                gpu_ids,
                num_shards,
                service_worker_id,
            ),
        )
        p.start()
        for output_queue in output_queues.values():
            output_queue.get()
        print(f"Model service worker {service_worker_id} started.")
    return input_queue, output_queues


def launch_model_engines(
    multi_model_server_args: MultiModelServerArgs,
    path_to_shared_cpu_models: Dict[Tuple[str, int], List[Any]],
    model_names_to_model_paths: Dict[str, str],
    request_handler_ipc_name: str,
    schedulers_to_controller_ipc_name: Optional[str] = None,
    input_queue: Optional[torch.multiprocessing.Queue] = None,
    output_queues: Optional[Dict[str, torch.multiprocessing.Queue]] = None,
):
    """
    Launch model engines in parallel.
    """
    model_configs = multi_model_server_args.model_configs
    engine_info_dict = defaultdict(list)  # model_name -> list of EngineInfo
    port_args_dict = defaultdict(list)  # model_name -> list of PortArgs
    gpu_id_to_model_instance = defaultdict(dict)  # gpu_id -> model_name -> instance_idx
    num_engines = 0

    def launch_engine_wrapper(args):
        """Wrapper function to unpack arguments for launch_engine"""
        (
            server_args,
            port_args,
            gpu_ids,
            instance_idx,
            path_to_shared_cpu_models,
            model_names_to_model_paths,
            engine_id,
            input_queue,
            output_queue,
        ) = args
        return (
            server_args.model_name,
            launch_engine(
                server_args=server_args,
                port_args=port_args,
                gpu_ids=gpu_ids,
                instance_idx=instance_idx,
                shared_cpu_models=path_to_shared_cpu_models,
                model_names_to_model_paths=model_names_to_model_paths,
                engine_id=engine_id,
                input_queue=input_queue,
                output_queue=output_queue,
            ),
        )

    start_time = time.perf_counter()
    # Prepare all engine configurations
    engine_launch_args = []
    start_port = multi_model_server_args.port
    for model_config in model_configs:
        instance_configs = model_config.get_instance_configs()
        for i, instance_config in enumerate(instance_configs):
            server_args = ServerArgs.from_multi_model_server_args(
                multi_model_server_args=multi_model_server_args,
                instance_config=instance_config,
            )
            gpu_ids = instance_config.gpu_ids
            print(
                f"Preparing engine for {server_args.model_name} ({server_args.model_path}) on GPU {gpu_ids} with initialize status on={instance_config.on}"
            )

            shared_cpu_model = None
            if multi_model_server_args.enable_cpu_share_memory:
                if (
                    server_args.model_path,
                    server_args.tp_size,
                ) not in path_to_shared_cpu_models:
                    raise RuntimeError(
                        f"Shared model for {server_args.model_name} ({server_args.model_path}) with tp_size={server_args.tp_size} is not loaded."
                    )
                shared_cpu_model = path_to_shared_cpu_models[
                    (server_args.model_path, server_args.tp_size)
                ]
            # Allocate ports for inter-process communications
            port_args = PortArgs.init_with_request_handler_ipc_name(
                start_port, request_handler_ipc_name, schedulers_to_controller_ipc_name
            )
            start_port = port_args.nccl_port
            engine_id = f"{model_config.model_name}_{i}"
            engine_launch_args.append(
                (
                    server_args,
                    port_args,
                    gpu_ids,
                    i,
                    path_to_shared_cpu_models,
                    model_names_to_model_paths,
                    engine_id,
                    (
                        input_queue
                        if multi_model_server_args.enable_model_service
                        else None
                    ),
                    (
                        output_queues[engine_id]
                        if multi_model_server_args.enable_model_service
                        else None
                    ),
                )
            )
    # Launch engines in parallel with small delays
    all_results = []
    with ThreadPoolExecutor(max_workers=len(engine_launch_args)) as executor:
        all_results = list(executor.map(launch_engine_wrapper, engine_launch_args))

    # Process results
    for model_name, engine_info in all_results:
        engine_info_dict[model_name].append(engine_info)
        port_args_dict[model_name].append(engine_info.port_args)
        num_engines += 1
        gpu_ids = engine_info.gpu_ids
        assert len(gpu_ids) > 0
        # NOTE(ke): For TP case, attach model instance to rank0 gpu_id
        gpu_id = gpu_ids[0]
        gpu_id_to_model_instance[gpu_id][model_name] = engine_info.instance_idx

    logger.info(
        f"All {num_engines} engines prepared in {time.perf_counter() - start_time:.2f} seconds."
    )
    return engine_info_dict, port_args_dict, gpu_id_to_model_instance, num_engines, None


def launch_worker_pool_engines(
    multi_model_server_args: MultiModelServerArgs,
    path_to_shared_cpu_models: Dict[Tuple[str, int], List[Any]],
    model_names_to_model_paths: Dict[str, str],
    request_handler_ipc_name: str,
    schedulers_to_controller_ipc_name: Optional[str] = None,
    input_queue: Optional[torch.multiprocessing.Queue] = None,
    output_queues: Optional[Dict[str, torch.multiprocessing.Queue]] = None,
):
    """
    Launch worker pool engines in parallel.
    """
    model_configs = multi_model_server_args.model_configs
    engine_info_dict = defaultdict(list)  # gpu_id -> list of EngineInfo
    port_args_dict = defaultdict(list)  # gpu_id -> list of PortArgs
    gpu_id_to_model_instance = defaultdict(dict)  # gpu_id -> model_name -> instance_idx
    num_engines = 0

    def launch_engine_wrapper(args):
        """Wrapper function to unpack arguments for launch_engine"""
        (
            server_args,
            port_args,
            gpu_ids,
            worker_id,
            path_to_shared_cpu_models,
            model_names_to_model_paths,
            engine_id,
            input_queue,
            output_queue,
        ) = args
        return (
            gpu_ids,
            launch_engine(
                server_args=server_args,
                port_args=port_args,
                gpu_ids=gpu_ids,
                instance_idx=worker_id,
                shared_cpu_models=path_to_shared_cpu_models,
                model_names_to_model_paths=model_names_to_model_paths,
                engine_id=engine_id,
                input_queue=input_queue,
                output_queue=output_queue,
            ),
        )

    # Get initial placements
    init_placements = defaultdict(list)  # GPU ID -> list of model names
    for model_config in model_configs:
        model_name = model_config.model_name
        instance_configs = model_config.get_instance_configs()
        for i, instance_config in enumerate(instance_configs):
            gpu_id = instance_config.gpu_ids[0]
            on = instance_config.on
            if on:
                init_placements[gpu_id].append(model_name)

    start_time = time.perf_counter()
    # Prepare all engine configurations
    engine_launch_args = []
    start_port = multi_model_server_args.port
    workers_per_gpu = multi_model_server_args.workers_per_gpu
    num_gpus = multi_model_server_args.num_gpus
    for gpu_id in range(num_gpus):
        for worker_id in range(workers_per_gpu):
            engine_id = f"{gpu_id}_{worker_id}"
            server_args = ServerArgs.from_multi_model_server_args(
                multi_model_server_args=multi_model_server_args,
                worker_id=worker_id,
            )
            # Allocate ports for inter-process communications
            port_args = PortArgs.init_with_request_handler_ipc_name(
                start_port, request_handler_ipc_name, schedulers_to_controller_ipc_name
            )
            start_port = port_args.nccl_port
            engine_launch_args.append(
                (
                    server_args,
                    port_args,
                    [gpu_id],
                    worker_id,
                    path_to_shared_cpu_models,
                    model_names_to_model_paths,
                    engine_id,
                    (
                        input_queue
                        if multi_model_server_args.enable_model_service
                        else None
                    ),
                    (
                        output_queues[engine_id]
                        if multi_model_server_args.enable_model_service
                        else None
                    ),
                )
            )

    # Launch engines in parallel with small delays
    all_results = []
    with ThreadPoolExecutor(max_workers=len(engine_launch_args)) as executor:
        all_results = list(executor.map(launch_engine_wrapper, engine_launch_args))

    # Process results
    for gpu_ids, engine_info in all_results:
        gpu_id = gpu_ids[0]
        engine_info_dict[gpu_id].append(engine_info)
        port_args_dict[gpu_id].append(engine_info.port_args)
        num_engines += 1
        gpu_ids = engine_info.gpu_ids
        assert len(gpu_ids) > 0
        gpu_id = gpu_ids[0]
        gpu_id_to_model_instance[gpu_id][gpu_id] = engine_info.instance_idx

    logger.info(
        f"All {num_engines} engines prepared in {time.perf_counter() - start_time:.2f} seconds."
    )

    return (
        engine_info_dict,
        port_args_dict,
        gpu_id_to_model_instance,
        num_engines,
        init_placements,
    )


def launch_multi_model_server(
    multi_model_server_args: MultiModelServerArgs,
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
):
    """
    Launch SRT (SGLang Runtime) Server

    The SRT server consists of an HTTP server, a request_handler and the SRT engines. Each SRT engine corresponds to one model resides on one GPU.

    1. HTTP server: A FastAPI server.
    2. Request handler: Send generation requests to the Redis queue, and recieve the results back. Route other requests to the cooresponding engines.
    3. Each instance of each model is one SRT engine, which includes:
        1. Scheduler (subprocess): Receive requests from the request_handler and the generation queue, tokenizes and schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        2. Detokenizer Manager (subprocess): Detokenizes the output tokens and sends the result back to the Request Handler.

    Note:
    1. The HTTP server and Request Hander both run in the main process.
    2. Inter-process communication is done through ICP (each process uses a different port) via the ZMQ library.
    3. For generation requests, the request handler sends the request to the Redis queue, and the scheduler processes pick up the requests from the queue. After generation, the results are sends back to the request handler by the detokenizer manager through ICP.
    """
    torch.multiprocessing.set_start_method("spawn")

    configure_logger(multi_model_server_args)
    # check whether the port are available
    if not is_port_available(multi_model_server_args.port):
        raise RuntimeError(
            f"Port {multi_model_server_args.port} is not available. Please choose another port."
        )

    request_handler_ipc_name = tempfile.NamedTemporaryFile(delete=False).name
    request_handler_to_controller_ipc_name = tempfile.NamedTemporaryFile(
        delete=False
    ).name
    if multi_model_server_args.enable_controller:
        schedulers_to_controller_ipc_name = tempfile.NamedTemporaryFile(
            delete=False
        ).name
    else:
        schedulers_to_controller_ipc_name = None

    # get model_names_to_model_paths
    model_configs = multi_model_server_args.model_configs
    model_names_to_model_paths = {
        model_config.model_name: model_config.model_path
        for model_config in model_configs
    }
    global model_names
    model_names = list(model_names_to_model_paths.keys())

    path_to_shared_cpu_models = load_shared_cpu_models(multi_model_server_args)

    if multi_model_server_args.enable_worker_pool:

        # launch model service
        if multi_model_server_args.enable_model_service:
            engine_ids = []
            workers_per_gpu = multi_model_server_args.workers_per_gpu
            num_gpus = multi_model_server_args.num_gpus
            for gpu_id in range(num_gpus):
                for worker_id in range(workers_per_gpu):
                    engine_ids.append(f"{gpu_id}_{worker_id}")
            input_queue, output_queues = launch_model_service(
                multi_model_server_args, path_to_shared_cpu_models, engine_ids
            )
        else:
            input_queue = None
            output_queues = None

        # launch worker pool engines
        (
            engine_info_dict,
            port_args_dict,
            gpu_id_to_model_instance,
            num_engines,
            init_placements,
        ) = launch_worker_pool_engines(
            multi_model_server_args,
            path_to_shared_cpu_models,
            model_names_to_model_paths,
            request_handler_ipc_name,
            schedulers_to_controller_ipc_name,
            input_queue,
            output_queues,
        )
    else:
        # launch model service
        if multi_model_server_args.enable_model_service:
            engine_ids = []
            for model_config in model_configs:
                instance_configs = model_config.get_instance_configs()
                for i, _ in enumerate(instance_configs):
                    engine_ids.append(f"{model_config.model_name}_{i}")
            input_queue, output_queues = launch_model_service(
                multi_model_server_args, path_to_shared_cpu_models, engine_ids
            )
        else:
            input_queue = None
            output_queues = None

        # launch model engines
        (
            engine_info_dict,
            port_args_dict,
            gpu_id_to_model_instance,
            num_engines,
            init_placements,
        ) = launch_model_engines(
            multi_model_server_args,
            path_to_shared_cpu_models,
            model_names_to_model_paths,
            request_handler_ipc_name,
            schedulers_to_controller_ipc_name,
            input_queue,
            output_queues,
        )

    # Launch the controller
    if multi_model_server_args.enable_controller:
        launch_controller(
            multi_model_server_args,
            recv_from_request_handler_ipc_name=request_handler_to_controller_ipc_name,
            recv_from_schedulers_ipc_name=schedulers_to_controller_ipc_name,
            engine_info_dict=engine_info_dict,
            model_names_to_model_paths=model_names_to_model_paths,
            init_placements=init_placements,
        )

    # Launch the GPU schedulers
    if multi_model_server_args.enable_gpu_scheduler:
        for gpu_id in gpu_id_to_model_instance.keys():
            launch_gpu_scheduler_process(
                multi_model_server_args,
                model_names_to_model_paths,
                engine_info_dict,
                gpu_id,
                init_placements[gpu_id],
            )

    # Launch the request handler
    launch_request_handler(
        multi_model_server_args,
        port_args_dict,
        request_handler_ipc_name,
        num_engines,
        gpu_id_to_model_instance,
        pipe_finish_writer,
        controller_ipc_name=request_handler_to_controller_ipc_name,
    )

    # assert that request_handler_ipc_name is not deleted
    assert os.path.exists(request_handler_ipc_name)

    # Add api key authorization
    if multi_model_server_args.api_key:
        add_api_key_middleware(app, multi_model_server_args.api_key)

    threads = []
    url = multi_model_server_args.url()
    # if enable_worker_pool, send activate and warmup requests to the gpu workers
    if multi_model_server_args.enable_worker_pool:
        for gpu_id, model_names in init_placements.items():
            for model_name in model_names:
                # _wait_and_activate(model_name, gpu_id)

                t = threading.Thread(
                    target=_wait_and_warmup,
                    args=(
                        url,
                        model_name,
                        pipe_finish_writer,
                        os.getpid(),
                    ),
                )
                t.start()
                threads.append(t)
    else:
        # Send warmup requests to models that are ready
        for model_name, engine_infos in engine_info_dict.items():
            for engine_info in engine_infos:
                if engine_info.on:
                    t = threading.Thread(
                        target=_wait_and_warmup,
                        args=(
                            url,
                            model_name,
                            pipe_finish_writer,
                            os.getpid(),
                        ),
                    )
                    t.start()
                    threads.append(t)

    try:
        # Listen for HTTP requests
        LOGGING_CONFIG["formatters"]["default"][
            "fmt"
        ] = "[%(asctime)s] %(levelprefix)s %(message)s"
        LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
        LOGGING_CONFIG["formatters"]["access"][
            "fmt"
        ] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
        LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
        uvicorn.run(
            app,
            host=multi_model_server_args.host,
            port=multi_model_server_args.port,
            log_level=multi_model_server_args.log_level_http
            or multi_model_server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    except Exception as e:
        # logger.error(f"Error in HTTP server: {get_exception_traceback()}")
        kill_child_process(os.getpid(), including_parent=False)
    finally:
        for t in threads:
            t.join()


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer",
            "0.1.6",
            "Please uninstall the old version and "
            "reinstall the latest version by following the instructions "
            "at https://docs.flashinfer.ai/installation.html.",
        )

    mp.set_start_method("spawn", force=True)


def _wait_and_activate(model_name, gpu_id):
    logger.info(f"Activating gpu {gpu_id} for model {model_name}...")
    activate_req = ActivateReqInput(
        model_name=model_name,
        gpu_id=gpu_id,
        instance_idx=0,
    )
    try:
        asyncio.run(request_handler.activate(activate_req))
    except Exception:
        last_traceback = get_exception_traceback()
        logger.error(f"Initialization failed. activate error: {last_traceback}")
        return


def _wait_and_warmup(url, model_name, pipe_finish_writer, pid):
    headers = {}
    logger.info(f"Waiting for the server to be ready for model {model_name}...")

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(
                f"{url}/get_model_names",
                timeout=5,
                headers=headers,
            )
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_child_process(pid, including_parent=False)
        return

    # Send a warmup request
    request_name = "/generate"
    max_new_tokens = 8
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
        "model": model_name,
        "is_warmup": True,
    }
    json_data["text"] = "The capital city of France is"

    try:
        res = requests.post(
            url + request_name,
            json=json_data,
            headers=headers,
            timeout=30000,
        )
        assert res.status_code == 200, f"{res}"
    except Exception:
        last_traceback = get_exception_traceback()
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_child_process(pid, including_parent=False)
        return

    # logger.info(f"{res.json()=}")

    logger.info(f"The server for {model_name} is fired up and ready to roll!")
    if pipe_finish_writer is not None:
        pipe_finish_writer.send("ready")
