import argparse
import asyncio
import csv
import json
import os
import random
import resource
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from trace import Request, TraceConfig, generate_synthetic_reqs
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import tqdm
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

global args


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    latency_server: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    tpot: float = 0.0
    wait_time: float = 0.0

    # Time info for plotting
    arrival_time: float = 0.0
    out_queue_time: float = 0.0
    prefill_finish_time: float = 0.0
    finish_time: float = 0.0
    decode_timestamps: List[float] = field(default_factory=list)

    prompt_len: int = 0
    error: str = ""
    output_len: int = 0
    slo: float = 0.0
    model: str = ""

    slo_ttft: float = 0.0
    slo_tpot: float = 0.0


async def send_generate_request(
    server: str,
    req: Request,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = server + "/generate"

    headers = {"User-Agent": "Benchmark Client"}
    sampling_params = {
        "ignore_eos": True,
        "max_new_tokens": int(req.output_len),
    }
    pload = {
        "text": req.prompt,
        "sampling_params": sampling_params,
        "rid": req.req_id,
        "model": req.model,
        "slo": req.slo,
        "slo_ttft": req.slo_ttft,
        "slo_tpot": req.slo_tpot,
        "prompt_len": req.prompt_len,
        "output_len": req.output_len,
    }

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        output.prompt_len = req.prompt_len
        output.slo = req.slo
        output.slo_ttft = req.slo_ttft
        output.slo_tpot = req.slo_tpot
        output.model = req.model

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=pload, headers=headers
            ) as response:
                if response.status == 200:
                    success = True
                    reason = None
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8")
                        latency = time.perf_counter() - st

                        data = json.loads(chunk)
                        if data["text"]:
                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)

                            most_recent_timestamp = timestamp
                        if data["meta_info"]:
                            finish_reason = data["meta_info"]["finish_reason"]["type"]
                            if finish_reason == "abort":
                                print(
                                    f"Benchmark: request {req.req_id} was aborted due to exceed slo."
                                )
                                success = False
                                reason = "Exceed SLO"
                            arrival_time = data["meta_info"]["arrival_timestamp"]
                            out_queue_time = data["meta_info"]["out_queue_timestamp"]
                            prefill_finish_time = data["meta_info"][
                                "prefill_finish_timestamp"
                            ]
                            decode_timestamps = data["meta_info"]["decode_timestamps"]
                            finish_time = data["meta_info"]["finish_timestamp"]

                            output.arrival_time = arrival_time
                            output.out_queue_time = out_queue_time
                            output.prefill_finish_time = prefill_finish_time
                            output.finish_time = finish_time
                            output.decode_timestamps = decode_timestamps

                            output.ttft = prefill_finish_time - arrival_time
                            output.tpot = (finish_time - prefill_finish_time) / (
                                req.output_len if req.output_len and req.output_len > 1 else 256 - 1
                            )
                            output.latency_server = finish_time - arrival_time
                            output.wait_time = out_queue_time - arrival_time

                            output.itl = [
                                decode_timestamps[i + 1] - decode_timestamps[i]
                                for i in range(len(decode_timestamps) - 1)
                            ]

                    if success:
                        output.latency = latency
                        output.success = True
                        output.output_len = req.output_len
                    else:
                        output.error = reason
                        output.success = False
                    # print(
                    #     f"Req_id {req.req_id}, Req.model: {req.model}, Success: {output.success}, Latency: {output.latency:.2f}s, Output len: {output.output_len}"
                    # )
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            output.success = False
            if isinstance(e, aiohttp.ServerDisconnectedError):
                output.error = "Server disconnected"
            else:
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
            print(f"Error in sending generate request {req.req_id} model {req.model}, error: {output.error}")

    if pbar:
        pbar.update(1)
    return output


async def send_request(
    server: str,
    req_name: str,
):
    api_url = server + "/" + req_name

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.get(api_url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return None
        except Exception:
            exec_info = sys.exc_info()
            print("Error:", "".join(traceback.format_exception(*exec_info)))
            raise


async def send_deactivate_request(
    model_name: str,
    server: str,
    instance_idx: Optional[int] = 0,
    preempt: bool = False,
) -> None:
    # await asyncio.sleep(0.1)
    preempt_mode = "RECOMPUTE"
    evict_waiting_requests = False
    deactivate_url = server + "/deactivate"
    pload = {
        "model_name": model_name,
        "instance_idx": instance_idx,
        "evict_waiting_requests": evict_waiting_requests,
        "preempt": preempt,
        "preempt_mode": preempt_mode,
    }
    print(f"deactivate request: {pload}")
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(deactivate_url, json=pload) as response:
            response_json = await response.json()
            print(
                f"deactivate request success: {response_json.get('success', None)}. Content: {response_json}"
            )


async def send_activate_request(
    model_name: str,
    server: str,
    instance_idx: Optional[int] = 0,
    memory_pool_size: Optional[int] = None,
) -> None:
    # await asyncio.sleep(1)
    pload = {
        "model_name": model_name,
        "instance_idx": instance_idx,
        "memory_pool_size": memory_pool_size,
    }
    activate_url = server + "/activate"
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(activate_url, json=pload) as response:
            response_json = await response.json()
            print(
                f"activate request for model {model_name} with memory pool size {memory_pool_size} success: {response_json.get('success', None)}. Content: {response_json}"
            )


async def send_resize_mem_pool_request(
    server: str, model_name: str, memory_pool_size: Optional[int] = None
):
    resize_url = server + "/resize_mem_pool"
    pload = {"model_name": model_name, "memory_pool_size": memory_pool_size}
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(resize_url, json=pload) as response:
            await response.text()
            print(f"send resize mem pool request success")


async def async_request_profile(api_url: str) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        try:
            async with session.post(url=api_url) as response:
                if response.status == 200:
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


async def run_swap_loop(
    queue: asyncio.Queue, server: str, preempt: bool, target_memory_pool_size: int
):
    """
    Swap two models in the queue. One swap includes deactivating model 1 and activating model 2. Different swaps run sequentially.
    Queue format: (model_1, model_2)
        * model_1: model to deactivate
        * model_2: model to activate
    """

    while True:
        try:
            element = await queue.get()
            if element == "stop":
                break
            model_1, model_2 = element
            print(f"Deactivating {model_1} and activating {model_2}")
            results = []
            if model_1 is not None:
                results.append(
                    asyncio.create_task(
                        send_deactivate_request(model_1, server, preempt=preempt)
                    )
                )
            if model_2 is not None:
                results.append(
                    asyncio.create_task(
                        send_activate_request(
                            model_2, server, memory_pool_size=target_memory_pool_size
                        )
                    )
                )
            await asyncio.gather(*results)
        except Exception as e:
            print(f"Error in swap loop: {e}")


async def run_activate_deactivate_loop(
    queue: asyncio.Queue,
    server: str,
    preempt: bool,
    target_memory_pool_size: int,
    enable_elastic_memory: bool,
):
    """
    Activate or deactivate a model in the queue. Different activate deactivate runs sequentially.
    Queue format: (if_activate, model_name, model_to_resize, tasks)
        * if_activate: bool, whether to activate or deactivate
        * model_name: str, model to activate or deactivate
        * models_to_resize: List[str], models to resize memory pool in each activate deactivate run
        * tasks: List[asyncio.Task], tasks to wait for
    """
    while True:
        try:
            element = await queue.get()
            if element == "stop":
                break
            if_activate, model_name, models_to_resize, tasks = element
            results = []
            if if_activate:
                print(f"Activating model {model_name}")
                if enable_elastic_memory:
                    for model_to_resize in models_to_resize:
                        print(f"Resizing memory pool for model {model_to_resize}")
                        results.append(
                            asyncio.create_task(
                                send_resize_mem_pool_request(
                                    server, model_to_resize, target_memory_pool_size
                                )
                            )
                        )
                results.append(
                    asyncio.create_task(
                        send_activate_request(
                            model_name, server, memory_pool_size=target_memory_pool_size
                        )
                    )
                )
            else:
                try:
                    await asyncio.gather(*tasks)
                except Exception as e:
                    print(f"Error in activate deactivate loop: {e}")
                print(f"Deactivating model {model_name}")
                print(f"finished tasks are {[task.get_name() for task in tasks]}")
                results.append(
                    asyncio.create_task(
                        send_deactivate_request(model_name, server, preempt=preempt)
                    )
                )
                if enable_elastic_memory:
                    for model_to_resize in models_to_resize:
                        print(f"Resizing memory pool for model {model_to_resize}")
                        results.append(
                            asyncio.create_task(
                                send_resize_mem_pool_request(
                                    server, model_to_resize, memory_pool_size=None
                                )
                            )
                        )
            try:
                await asyncio.gather(*results)
            except Exception as e:
                exec_info = sys.exc_info()
                print("Error:", "".join(traceback.format_exception(*exec_info)))
            if if_activate:
                await asyncio.sleep(1)
        except Exception as e:
            exec_info = sys.exc_info()
            print("Error:", "".join(traceback.format_exception(*exec_info)))


async def run_generation_adaptive(
    input_requests: List[Request],
    server: str,
    dynamic_models: List[str],
    persistent_models: List[str],
    enable_elastic_memory: bool = False,
    target_memory_pool_size: int = 16,
    preempt: bool = False,
    pbar: Optional[tqdm] = None,
    debug: bool = False,
):
    IDLE_THRESHOLD = 2
    tasks: List[asyncio.Task] = []
    results = [None] * len(input_requests)
    control_tasks = []
    queue = asyncio.Queue()
    control_tasks.append(
        asyncio.create_task(
            run_activate_deactivate_loop(
                queue, server, preempt, target_memory_pool_size, enable_elastic_memory
            )
        )
    )
    last_req_time = {model: None for model in dynamic_models}
    model_status = {model: "inactive" for model in dynamic_models}
    num_activations = 0
    num_deactivations = 0

    start = time.perf_counter()
    for i, req in enumerate(input_requests):

        # send generation requests
        sleep_time = max(0, start + req.arrival_time - time.perf_counter())
        await asyncio.sleep(sleep_time)

        task = asyncio.create_task(
            send_generate_request(server, req, pbar=pbar),
            name=f"req_{req.req_id}_{req.model}",
        )
        task._idx = i
        if debug:
            print(
                f"Sending request {req.req_id} for model {req.model} at time {time.perf_counter() - start:.2f}"
            )

        tasks.append(task)

        # activate model if its first request comes in
        if req.model in dynamic_models:
            if last_req_time[req.model] is None:
                await queue.put((True, req.model, persistent_models, None))
                model_status[req.model] = "active"
                num_activations += 1
            last_req_time[req.model] = time.perf_counter()

        # deactivate model if it doesn't have any requests for IDLE_THRESHOLD seconds
        for model in dynamic_models:
            if (
                model_status[model] == "active"
                and last_req_time[model] is not None
                and time.perf_counter() - last_req_time[model] > IDLE_THRESHOLD
            ):
                await queue.put((False, model, persistent_models, []))
                model_status[model] = "inactive"
                last_req_time[model] = None
                num_deactivations += 1
    while True:
        done, pending = await asyncio.wait(tasks, timeout=60)
        for task in done:
            results[task._idx] = task.result()
        if len(pending) == 0:
            await queue.put("stop")
            break
        for task in pending:
            print(f"Waiting for task {task.get_name()}")
        tasks = list(pending)

    await asyncio.gather(*control_tasks)
    print(f"Finished sending requests, model status: {model_status}")

    return results, (num_activations, num_deactivations)


async def run_generation_sequential_swapping(
    input_requests: List[Request],
    server: str,
    target_memory_pool_size: int = 16,
    preempt: bool = False,
    pbar: Optional[tqdm] = None,
    debug: bool = False,
):
    """
    Assume the requests of different models come in sequentially.
    Only swap models when all pending requests for the current model are completed.
    """
    tasks: List[asyncio.Task] = []
    control_tasks = []
    outputs = []
    completed_tasks = set()
    num_swaps = 0
    current_model = None
    pending_tasks_by_model = {}
    
    task_to_index: Dict[asyncio.Task, int] = {}
    results = [None] * len(input_requests)

    swap_queue = asyncio.Queue()
    control_tasks.append(
        asyncio.create_task(
            run_swap_loop(swap_queue, server, preempt, target_memory_pool_size)
        )
    )

    start_time = time.perf_counter()
    for i, req in enumerate(input_requests):
        sleep_time = max(0, start_time + req.arrival_time - time.perf_counter())
        await asyncio.sleep(sleep_time)

        if req.model != current_model:
            # Wait for all pending tasks of the current model to complete before swapping
            if (
                current_model in pending_tasks_by_model
                and pending_tasks_by_model[current_model]
            ):
                if debug:
                    print(
                        f"Waiting for {len(pending_tasks_by_model[current_model])} pending tasks of model {current_model} to complete before swapping"
                    )
                pending_model_tasks = pending_tasks_by_model[current_model]
                done, _ = await asyncio.wait(pending_model_tasks)
                for task in done:
                    if task not in completed_tasks:
                        result = task.result()
                        results[task_to_index[task]] = result
                        completed_tasks.add(task)
                pending_tasks_by_model[current_model] = []

            # Now safe to swap models
            await swap_queue.put((current_model, req.model))
            current_model = req.model
            num_swaps += 1

        task = asyncio.create_task(
            send_generate_request(server, req, pbar=pbar),
            name=f"req_{req.req_id}_{req.model}",
        )
        if debug:
            print(
                f"Sending request {req.req_id} for model {req.model} at time {time.perf_counter() - start_time:.2f}"
            )
        tasks.append(task)
        task_to_index[task] = i

        # Track this task by model
        if req.model not in pending_tasks_by_model:
            pending_tasks_by_model[req.model] = []
        pending_tasks_by_model[req.model].append(task)

    while tasks:
        done, pending = await asyncio.wait(tasks, timeout=60)
        for task in done:
            if task not in completed_tasks:
                result = task.result()
                results[task_to_index[task]] = result
                completed_tasks.add(task)
            for model, model_tasks in list(pending_tasks_by_model.items()):
                if task in model_tasks:
                    model_tasks.remove(task)
        if not pending:
            break
        for task in pending:
            print(f"Waiting for task {task.get_name()}")
        tasks = list(pending)

    await swap_queue.put("stop")
    await asyncio.gather(*control_tasks)

    outputs = [r for r in results if r is not None]

    if len(outputs) != len(input_requests):
        print(
            f"WARNING: Number of outputs ({len(outputs)}) does not match number of requests ({len(input_requests)})"
        )

    return results, num_swaps


async def run_generation_one_queue(
    input_requests: List[Request],
    server: str,
    dynamic_models: List[str],
    persistent_models: List[str],
    target_memory_pool_size: int = 16,
    preempt: bool = False,
    pbar: Optional[tqdm] = None,
    debug: bool = False,
):
    IDLE_THRESHOLD = 2
    MAX_RUN_TIME = 10
    tasks: List[asyncio.Task] = []
    results = [None] * len(input_requests)
    control_tasks: List[asyncio.Task] = []
    swap_queue = asyncio.Queue()
    control_tasks.append(
        asyncio.create_task(
            run_swap_loop(swap_queue, server, preempt, target_memory_pool_size)
        )
    )

    start = time.perf_counter()
    num_swaps = 0
    current_model = persistent_models[0]
    model_status = {model: "inactive" for model in dynamic_models}
    last_request_time = {model: None for model in dynamic_models}
    last_activation_time = {model: None for model in persistent_models}

    for i, req in enumerate(input_requests):

        # send generation requests
        sleep_time = max(0, start + req.arrival_time - time.perf_counter())

        await asyncio.sleep(sleep_time)
        task = asyncio.create_task(
            send_generate_request(server, req, pbar=pbar),
            name=f"req_{req.req_id}_{req.model}",
        )
        task._idx = i
        if debug:
            print(
                f"Sending request {req.req_id} for model {req.model} at time {time.perf_counter() - start:.2f}"
            )

        tasks.append(task)

        # swap model with current model if it's the first request of a dynamic model
        if req.model in dynamic_models:
            if (
                model_status[req.model] == "inactive"
                and current_model not in dynamic_models
            ):
                await swap_queue.put((current_model, req.model))
                model_status[req.model] = "active"
                last_activation_time[current_model] = None
                current_model = req.model
                num_swaps += 1
            last_request_time[req.model] = time.perf_counter()

        # swap model with persistent model if it doesn't have any requests for IDLE_THRESHOLD seconds
        for model in dynamic_models:
            if (
                model_status[model] == "active"
                and last_request_time[model] is not None
                and time.perf_counter() - last_request_time[model] > IDLE_THRESHOLD
            ):
                assert (
                    current_model == model
                ), f"current_model: {current_model}, model: {model}"
                await swap_queue.put((model, persistent_models[0]))
                model_status[model] = "inactive"
                last_request_time[model] = None
                current_model = persistent_models[0]
                num_swaps += 1

        # swap model with current model if current model has been running for MAX_RUN_TIME seconds
        if len(persistent_models) == 2:
            if (
                current_model in persistent_models
                and last_activation_time[current_model] is not None
                and time.perf_counter() - last_activation_time[current_model]
                > MAX_RUN_TIME
            ):
                model_to_swap = (
                    persistent_models[0]
                    if current_model == persistent_models[1]
                    else persistent_models[1]
                )
                await swap_queue.put((current_model, model_to_swap))
                last_activation_time[current_model] = None
                current_model = model_to_swap
                num_swaps += 1

    while True:
        done, pending = await asyncio.wait(tasks, timeout=60)
        for task in done:
            results[task._idx] = task.result()
        if len(pending) == 0:
            await swap_queue.put("stop")
            break
        for task in pending:
            print(f"Waiting for task {task.get_name()}")
        tasks = list(pending)

    await asyncio.gather(*control_tasks)
    return results, num_swaps


async def run_generation_basic(
    input_requests: List[Request],
    server: str,
    pbar: Optional[tqdm] = None,
    debug: bool = False,
):
    tasks: List[asyncio.Task] = []
    results = [None] * len(input_requests)
    start = time.perf_counter()
    for i, req in enumerate(input_requests):
        sleep_time = max(0, start + req.arrival_time - time.perf_counter())
        print(f"Request {req.req_id} arrives at {time.perf_counter()}, sleeping for {sleep_time} seconds")
        await asyncio.sleep(sleep_time)
        task = asyncio.create_task(
            send_generate_request(server, req, pbar=pbar),
            name=f"req_{req.req_id}_{req.model}",
        )
        task._idx = i
        if debug:
            print(
                f"Sending request {req.req_id} for model {req.model} at time {time.perf_counter() - start:.2f}"
            )
        tasks.append(task)

    while True:
        done, pending = await asyncio.wait(tasks, timeout=60)
        for task in done:
            results[task._idx] = task.result()
        if len(pending) == 0:
            break
        for task in pending:
            print(f"Waiting for task {task.get_name()}")
        tasks = list(pending)
    return results


async def run_tp_mode(
    args: argparse.Namespace,
    trace_config: TraceConfig,
    requests: List[Request]
):
    """Run TP benchmark mode"""
    server = args.base_url or f"http://{args.host}:{args.port}"
    
    print(f"Starting TP benchmark: {args.exp_name}")
    print(f"Generated {len(requests)} requests")
    
    # Run benchmark  
    start_time = time.perf_counter()
    outputs = await run_generation_basic(
        requests, server, 
        pbar=None if args.disable_tqdm else tqdm(total=len(requests)),
        debug=args.debug
    )
    duration = time.perf_counter() - start_time
    
    # Calculate basic metrics (TP mode version)
    successful = [o for o in outputs if o.success]
    
    if not successful:
        print("No successful requests!")
        return
        
    total_input = sum(r.prompt_len for r in requests)
    total_output = sum(o.output_len for o in successful)
    
    ttfts = [o.ttft for o in successful if o.ttft > 0]
    tpots = [o.tpot for o in successful if o.tpot > 0]
    latencies = [o.latency for o in successful]
    
    # Print results
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Completed requests: {len(successful)}")
    print(f"Aborted requests: {len(outputs) - len(successful)}")
    print(f"Request throughput: {len(successful) / duration:.2f} req/s")
    print(f"Input throughput: {total_input / duration:.2f} tokens/s")
    print(f"Output throughput: {total_output / duration:.2f} tokens/s")
    
    if ttfts:
        print(f"Mean TTFT: {np.mean(ttfts) * 1000:.1f} ms")
        print(f"P99 TTFT: {np.percentile(ttfts, 99) * 1000:.1f} ms")
    
    if tpots:
        print(f"Mean TPOT: {np.mean(tpots) * 1000:.1f} ms")
        print(f"P99 TPOT: {np.percentile(tpots, 99) * 1000:.1f} ms")
        
    if latencies:
        print(f"Mean Latency: {np.mean(latencies) * 1000:.1f} ms")
        print(f"P99 Latency: {np.percentile(latencies, 99) * 1000:.1f} ms")
    
    print("=" * 60)
    
    # Save results (TP mode format)
    os.makedirs(args.results_path, exist_ok=True)
    os.makedirs(args.request_path, exist_ok=True)
    
    # Save metrics
    if trace_config.e2e_benchmark:
        metrics_file = (
            f"{args.exp_name}_e2e_{args.num_gpus}gpu_{trace_config.time_scale}x_"
            f"{trace_config.replication}rep.json"
        )
        requests_file = (
            f"{args.exp_name}_e2e_{args.num_gpus}gpu_{trace_config.time_scale}x_"
            f"{trace_config.replication}rep_output_requests.json"
        )
    else:
        metrics_file = f"{args.exp_name}_{args.num_gpus}gpu.json"
        requests_file = f"{args.exp_name}_{args.num_gpus}gpu_output_requests.json"
    
    # Save detailed metrics
    with open(os.path.join(args.results_path, metrics_file), "w") as f:
        result = {
            "timestamp": datetime.now().isoformat(),
            "exp_name": args.exp_name,
            "num_gpus": args.num_gpus,
            "completed": len(successful),
            "aborted": len(outputs) - len(successful),
            "request_throughput": len(successful) / duration,
            "input_throughput": total_input / duration,
            "output_throughput": total_output / duration,
        }
        if ttfts:
            result.update({
                "mean_ttft_ms": np.mean(ttfts) * 1000,
                "p99_ttft_ms": np.percentile(ttfts, 99) * 1000,
            })
        if tpots:
            result.update({
                "mean_tpot_ms": np.mean(tpots) * 1000,
                "p99_tpot_ms": np.percentile(tpots, 99) * 1000,
            })
        json.dump(result, f, indent=2)
    
    # Save request outputs
    with open(os.path.join(args.request_path, requests_file), "w") as f:
        output_data = []
        for output in outputs:
            output_data.append({
                "success": output.success,
                "latency": output.latency,
                "ttft": output.ttft,
                "tpot": output.tpot,
                "output_len": output.output_len,
                "model": output.model,
                "error": output.error,
            })
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {args.results_path}")
    print(f"Requests saved to {args.request_path}")


def get_all_metric_file_name(trace_config, benchmark_mode: str, num_gpus: int):
    if benchmark_mode == "e2e":
        filename = f"{benchmark_mode}_num_gpus-{num_gpus}_time_scale-{trace_config.time_scale}_replication-{trace_config.replication}"
    elif benchmark_mode == "micro":
        filename = f"{benchmark_mode}_time_scale-{trace_config.time_scale}_replication-{trace_config.replication}"
    else:
        filename = f"{benchmark_mode}_num_models-{len(trace_config.model_paths)}_req_rate-{trace_config.req_rate}_duration-{trace_config.duration}_alpha-{trace_config.alpha}_cv-{trace_config.cv}_slo-{trace_config.slo}"
    filename_all_metrics = f"{filename}_all.jsonl"
    return filename_all_metrics


def get_key_metric_file_name(trace_config, benchmark_mode: str, num_gpus: int):
    if benchmark_mode == "e2e":
        filename = f"{benchmark_mode}_rank-4_model_70B_replication-{trace_config.replication}"
        # filename = f"{benchmark_mode}_num_gpus-{num_gpus}"
    elif benchmark_mode == "micro":
        filename = f"{benchmark_mode}"
    else:
        filename = f"{benchmark_mode}_num_models-{len(trace_config.model_paths)}_req_rate-{trace_config.req_rate}_duration-{trace_config.duration}_alpha-{trace_config.alpha}_cv-{trace_config.cv}_slo-{trace_config.slo}"
    filename_key_metrics = f"{filename}_key_metrics.tsv"
    return filename_key_metrics


def get_output_request_file_name(
    exp_name, trace_config, benchmark_mode: str, num_gpus: int
):
    if benchmark_mode == "e2e":
        filename = f"{benchmark_mode}_{exp_name}_num_gpus-{num_gpus}_time_scale-{trace_config.time_scale}_replication-{trace_config.replication}.json"
    elif benchmark_mode == "micro":
        filename = f"{benchmark_mode}_{exp_name}_time_scale-{trace_config.time_scale}_replication-{trace_config.replication}.json"
    else:
        filename = f"{benchmark_mode}_{exp_name}_num_models-{len(trace_config.model_paths)}_req_rate-{trace_config.req_rate}_duration-{trace_config.duration}_alpha-{trace_config.alpha}_cv-{trace_config.cv}_slo-{trace_config.slo}.json"
    return filename


async def benchmark(
    args: argparse.Namespace,
    input_requests: List[Request],
    server: str,
    trace_config: TraceConfig,
) -> None:

    # test_run(test_request=input_requests[0], server=server)

    pbar = None if args.disable_tqdm else tqdm(total=len(input_requests))

    if args.profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(api_url=server + "/start_profile")
        if profile_output.success:
            print("Profiler started")

    benchmark_start_time = time.perf_counter()

    if args.policy is None:
        outputs = await run_generation_basic(input_requests, server, pbar)
        num_swaps = 0
    else:
        if args.policy == "adaptive":
            outputs, num_swaps = await run_generation_adaptive(
                input_requests=input_requests,
                server=server,
                dynamic_models=args.dynamic_models,
                persistent_models=args.persistent_models,
                enable_elastic_memory=args.enable_elastic_memory,
                target_memory_pool_size=args.memory_pool_size,
                preempt=args.preempt,
                pbar=pbar,
                debug=args.debug,
            )
        elif args.policy == "one-queue":
            outputs, num_swaps = await run_generation_one_queue(
                input_requests=input_requests,
                server=server,
                dynamic_models=args.dynamic_models,
                persistent_models=args.persistent_models,
                target_memory_pool_size=args.memory_pool_size,
                preempt=args.preempt,
                pbar=pbar,
                debug=args.debug,
            )
        elif args.policy == "sequential-swapping":
            outputs, num_swaps = await run_generation_sequential_swapping(
                input_requests=input_requests,
                server=server,
                target_memory_pool_size=args.memory_pool_size,
                preempt=args.preempt,
                pbar=pbar,
                debug=args.debug,
            )
        else:
            raise ValueError(f"Invalid policy: {args.policy}")

    benchmark_duration = time.perf_counter() - benchmark_start_time
    # Stop profiler
    if args.profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=server + "/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    if trace_config.micro_benchmark:
        benchmark_mode = "micro"
    elif trace_config.e2e_benchmark:
        benchmark_mode = "e2e"
    else:
        benchmark_mode = "synthetic"

    metrics, model_to_metrics = get_benchmark_metrics(
        input_requests, outputs, benchmark_duration
    )
    if metrics is None:
        return None

    print_metrics(
        exp_name=args.exp_name,
        trace_config=trace_config,
        metrics=metrics,
        model_to_metrics=model_to_metrics,
        num_swaps=num_swaps,
        benchmark_duration=benchmark_duration,
        num_gpus=args.num_gpus,
        benchmark_mode=benchmark_mode,
    )
    save_results(
        exp_name=args.exp_name,
        trace_config=trace_config,
        metrics=metrics,
        model_to_metrics=model_to_metrics,
        num_swaps=num_swaps,
        benchmark_duration=benchmark_duration,
        save_path=args.results_path,
        outputs=outputs,
        benchmark_mode=benchmark_mode,
        num_gpus=args.num_gpus,
        save_length=not args.hyper_trace,
    )
    save_output_requests(
        exp_name=args.exp_name,
        trace_config=trace_config,
        outputs=outputs,
        save_path=args.request_path,
        benchmark_mode=benchmark_mode,
        num_gpus=args.num_gpus,
    )
    return


@dataclass
class BenchmarkMetrics:
    completed: int
    aborted: int
    total_input: int
    total_output: int
    average_input_len: float
    average_output_len: float
    request_throughput: float
    input_throughput: float
    output_throughput: float
    input_output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    p95_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    p95_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    p95_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    p95_e2e_latency_ms: float
    mean_e2e_latency_server_ms: float
    median_e2e_latency_server_ms: float
    p99_e2e_latency_server_ms: float
    p95_e2e_latency_server_ms: float
    mean_wait_time_ms: float
    median_wait_time_ms: float
    p99_wait_time_ms: float
    p95_wait_time_ms: float
    average_attainment: float = 0.0
    average_attainment_ttft: float = 0.0
    average_attainment_tpot: float = 0.0


def calculate_metrics(
    input_requests: List[Request],
    outputs: List[RequestFuncOutput],
    dur_s: float,
) -> BenchmarkMetrics:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    aborted = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    wait_times: List[float] = []
    e2e_latencies_server: List[float] = []
    attainment_slo: List[int] = []
    attainment_ttft: List[int] = []
    attainment_tpot: List[int] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            # if output_len > 1:
            #     tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            tpots.append(outputs[i].tpot)
            wait_times.append(outputs[i].wait_time)

            e2e_latencies.append(outputs[i].latency)
            e2e_latencies_server.append(outputs[i].latency_server)

            if outputs[i].slo is not None:
                if outputs[i].slo_ttft is not None and outputs[i].slo_tpot is not None:
                    print(f"model: {outputs[i].model}, arrival_time: {outputs[i].arrival_time}, outputs[i].ttft: {outputs[i].ttft}, outputs[i].slo_ttft: {outputs[i].slo_ttft}, outputs[i].tpot: {outputs[i].tpot}, outputs[i].slo_tpot: {outputs[i].slo_tpot}")
                    # change from latency to ttft slo
                    attainment_ttft.append(1 if outputs[i].ttft < outputs[i].slo_ttft else 0)
                    attainment_tpot.append(1 if outputs[i].tpot < outputs[i].slo_tpot else 0)
                else:
                    attainment_slo.append(1 if outputs[i].latency < outputs[i].slo else 0)
            else:
                attainment_ttft.append(1)
                attainment_tpot.append(1)
                attainment_slo.append(1)
            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)
            attainment_ttft.append(0)
            attainment_tpot.append(0)
            attainment_slo.append(0)
            aborted += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
        return None
    metrics = BenchmarkMetrics(
        completed=completed,
        aborted=aborted,
        total_input=total_input,
        total_output=sum(output_lens),
        average_input_len=total_input / completed,
        average_output_len=np.mean(output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        input_output_throughput=(total_input + sum(output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        p95_ttft_ms=np.percentile(ttfts or 0, 95) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        p95_tpot_ms=np.percentile(tpots or 0, 95) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        p95_itl_ms=np.percentile(itls or 0, 95) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        p95_e2e_latency_ms=np.percentile(e2e_latencies, 95) * 1000,
        mean_e2e_latency_server_ms=np.mean(e2e_latencies_server) * 1000,
        median_e2e_latency_server_ms=np.median(e2e_latencies_server) * 1000,
        p99_e2e_latency_server_ms=np.percentile(e2e_latencies_server, 99) * 1000,
        p95_e2e_latency_server_ms=np.percentile(e2e_latencies_server, 95) * 1000,
        mean_wait_time_ms=np.mean(wait_times) * 1000,
        median_wait_time_ms=np.median(wait_times) * 1000,
        p99_wait_time_ms=np.percentile(wait_times, 99) * 1000,
        p95_wait_time_ms=np.percentile(wait_times, 95) * 1000,
        average_attainment_ttft=np.mean(attainment_ttft) if len(attainment_ttft) > 0 else 0,
        average_attainment_tpot=np.mean(attainment_tpot) if len(attainment_tpot) > 0 else 0,
    )

    return metrics


def get_benchmark_metrics(
    input_requests: List[Request], outputs: List[RequestFuncOutput], dur_s: float
) -> BenchmarkMetrics:
    all_model_metrics = calculate_metrics(input_requests, outputs, dur_s)

    if all_model_metrics is None:
        print(f"Error running benchmark for request rate: {trace_config.req_rate}")
        print("-" * 30)
        return None, None

    model_to_input_requests = {}
    model_to_outputs = {}
    for i, req in enumerate(input_requests):
        if req.model not in model_to_input_requests:
            model_to_input_requests[req.model] = []
            model_to_outputs[req.model] = []
        model_to_input_requests[req.model].append(req)
        model_to_outputs[req.model].append(outputs[i])

    model_to_metrics = {}
    for model, reqs in model_to_input_requests.items():
        model_outputs = model_to_outputs[model]
        model_metrics = calculate_metrics(reqs, model_outputs, dur_s)
        model_to_metrics[model] = model_metrics
    return all_model_metrics, model_to_metrics


def print_metrics(
    exp_name: str,
    trace_config: TraceConfig,
    metrics: BenchmarkMetrics,
    model_to_metrics: Dict[str, BenchmarkMetrics],
    num_swaps: Union[int, Tuple[int, int]],
    benchmark_duration: float,
    benchmark_mode: str,
    num_gpus: int,
):
    request_rate = trace_config.req_rate
    slo = trace_config.slo
    duration = trace_config.duration
    input_range = trace_config.input_range
    output_range = trace_config.output_range
    alpha = trace_config.alpha
    cv = trace_config.cv
    on_off_model_percentage = trace_config.on_off_model_percentage
    on_off_cycle_len = trace_config.on_off_cycle_len

    print(
        "\n{s:{c}^{n}}".format(
            s=f" {benchmark_mode.capitalize()} Benchmark Result ", n=50, c="="
        )
    )
    print("{:<40} {:<10}".format("Experiment:", exp_name))
    print("{:<40} {:<10}".format("Num models:", len(model_to_metrics)))
    # print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    # print("{:<40} {:<10}".format("Alpha:", alpha))
    # print("{:<40} {:<10}".format("CV:", cv))
    print("{:<40} {:<10}".format("SLO:", slo))
    print("{:<40} {:<10}".format("Num GPUs:", num_gpus))
    # print("{:<40} {:<10.2f}".format("On-Off ratio:", on_off_model_percentage))
    # print("{:<40} {:<10}".format("On-Off cycle len:", on_off_cycle_len))
    # print("{:<40} {:<10}".format("Input range:", f"{input_range[0]}-{input_range[1]}"))
    # print(
    #     "{:<40} {:<10}".format("Output range:", f"{output_range[0]}-{output_range[1]}")
    # )

    if isinstance(num_swaps, int):
        if num_swaps != 0:
            print("{:<40} {:<10}".format("Num swaps:", num_swaps))
    elif isinstance(num_swaps, tuple):
        num_activations, num_deactivations = num_swaps
        print("{:<40} {:<10}".format("Num activations:", num_activations))
        print("{:<40} {:<10}".format("Num deactivations:", num_deactivations))
    print("{:<40} {:<10.2f}".format("Average Attainment:", metrics.average_attainment))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Aborted requests:", metrics.aborted))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10.2f}".format("Average input len:", metrics.average_input_len))
    print("{:<40} {:<10.2f}".format("Average output len:", metrics.average_output_len))
    # print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    # print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{s:{c}^{n}}".format(s="Throughput", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input + Output token throughput (tok/s):",
            metrics.input_output_throughput,
        )
    )
    print("{s:{c}^{n}}".format(s="E2E Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    print(
        "{:<40} {:<10.2f}".format("P99 E2E Latency (ms):", metrics.p99_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format("P95 E2E Latency (ms):", metrics.p95_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Mean Server E2E Latency (ms):", metrics.mean_e2e_latency_server_ms
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median Server E2E Latency (ms):", metrics.median_e2e_latency_server_ms
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{:<40} {:<10.2f}".format("P95 TTFT (ms):", metrics.p95_ttft_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{:<40} {:<10.2f}".format("P95 TPOT (ms):", metrics.p95_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("{:<40} {:<10.2f}".format("P95 ITL (ms):", metrics.p95_itl_ms))
    print("{s:{c}^{n}}".format(s="Wait Time", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean Wait Time (ms):", metrics.mean_wait_time_ms))
    print(
        "{:<40} {:<10.2f}".format("Median Wait Time (ms):", metrics.median_wait_time_ms)
    )
    print("{:<40} {:<10.2f}".format("P99 Wait Time (ms):", metrics.p99_wait_time_ms))
    print("{:<40} {:<10.2f}".format("P95 Wait Time (ms):", metrics.p95_wait_time_ms))

    num_models = len(model_to_metrics)
    if num_models > 1:  # print per model metrics
        print("{s:{c}^{n}}".format(s="Each Model Metrics", n=50, c="-"))
        for model, model_metrics in model_to_metrics.items():
            print("{s:{c}^{n}}".format(s=f"Model: {model}", n=50, c="*"))
            print(
                "{:<40} {:<10}".format("Successful requests:", model_metrics.completed)
            )
            print("{:<40} {:<10}".format("Aborted requests:", model_metrics.aborted))
            print(
                "{:<40} {:<10.2f}".format(
                    "Average Attainment:", model_metrics.average_attainment
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Average Attainment TTFT:", model_metrics.average_attainment_ttft
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Average Attainment TPOT:", model_metrics.average_attainment_tpot
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Request throughput (req/s):", model_metrics.request_throughput
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Input token throughput (tok/s):", model_metrics.input_throughput
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Output token throughput (tok/s):", model_metrics.output_throughput
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Input + Output token throughput (tok/s):",
                    model_metrics.input_output_throughput,
                )
            )

            print(
                "{:<40} {:<10.2f}".format(
                    "Mean E2E Latency (ms):", model_metrics.mean_e2e_latency_ms
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Median E2E Latency (ms):", model_metrics.median_e2e_latency_ms
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "P99 E2E Latency (ms):", model_metrics.p99_e2e_latency_ms
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "P95 E2E Latency (ms):", model_metrics.p95_e2e_latency_ms
                )
            )
            # ttft
            print(
                "{:<40} {:<10.2f}".format("Mean TTFT (ms):", model_metrics.mean_ttft_ms)
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Median TTFT (ms):", model_metrics.median_ttft_ms
                )
            )
            print(
                "{:<40} {:<10.2f}".format("P99 TTFT (ms):", model_metrics.p99_ttft_ms)
            )
            print(
                "{:<40} {:<10.2f}".format("P95 TTFT (ms):", model_metrics.p95_ttft_ms)
            )
            # tpot
            print(
                "{:<40} {:<10.2f}".format("Mean TPOT (ms):", model_metrics.mean_tpot_ms)
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Median TPOT (ms):", model_metrics.median_tpot_ms
                )
            )
            print(
                "{:<40} {:<10.2f}".format("P99 TPOT (ms):", model_metrics.p99_tpot_ms)
            )
            print(
                "{:<40} {:<10.2f}".format("P95 TPOT (ms):", model_metrics.p95_tpot_ms)
            )
            # itl
            print(
                "{:<40} {:<10.2f}".format("Mean ITL (ms):", model_metrics.mean_itl_ms)
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Median ITL (ms):", model_metrics.median_itl_ms
                )
            )
            print("{:<40} {:<10.2f}".format("P99 ITL (ms):", model_metrics.p99_itl_ms))
            print("{:<40} {:<10.2f}".format("P95 ITL (ms):", model_metrics.p95_itl_ms))
            # wait time
            print(
                "{:<40} {:<10.2f}".format(
                    "Mean Wait Time (ms):", model_metrics.mean_wait_time_ms
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Median Wait Time (ms):", model_metrics.median_wait_time_ms
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "P99 Wait Time (ms):", model_metrics.p99_wait_time_ms
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "P95 Wait Time (ms):", model_metrics.p95_wait_time_ms
                )
            )

    print("=" * 50)


def save_results(
    exp_name: str,
    trace_config: TraceConfig,
    metrics: BenchmarkMetrics,
    model_to_metrics: Dict[str, BenchmarkMetrics],
    num_swaps: int,
    benchmark_duration: float,
    save_path: str,
    outputs: List[RequestFuncOutput],
    benchmark_mode: str,
    num_gpus: int,
    save_length: bool = True,
):
    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            # "dataset_name": args.dataset_name,
            "exp_name": exp_name,
            "num_models": len(trace_config.model_paths),
            # "request_rate": trace_config.req_rate,
            # "alpha": trace_config.alpha,
            # "cv": trace_config.cv,
            # "num_swaps": num_swaps,
            # "request_duration": trace_config.duration,
            "benchmark_duration": benchmark_duration,
            # "input_range_min": trace_config.input_range[0],
            # "input_range_max": trace_config.input_range[1],
            # "output_range_min": trace_config.output_range[0],
            # "output_range_max": trace_config.output_range[1],
            "average_input_tokens": metrics.total_input / metrics.completed,
            "average_output_tokens": metrics.total_output / metrics.completed,
            "request_throughput": metrics.request_throughput,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
            "p95_e2e_latency_ms": metrics.p95_e2e_latency_ms,
            "mean_e2e_latency_server_ms": metrics.mean_e2e_latency_server_ms,
            "p99_e2e_latency_server_ms": metrics.p99_e2e_latency_server_ms,
            "p95_e2e_latency_server_ms": metrics.p95_e2e_latency_server_ms,
            "average_attainment": metrics.average_attainment,
            "median_ttft_ms": metrics.median_ttft_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "completed": metrics.completed,
            "aborted": metrics.aborted,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "input_output_throughput": metrics.input_output_throughput,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "std_ttft_ms": metrics.std_ttft_ms,
            "p99_ttft_ms": metrics.p99_ttft_ms,
            "p95_ttft_ms": metrics.p95_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
            "std_tpot_ms": metrics.std_tpot_ms,
            "p99_tpot_ms": metrics.p99_tpot_ms,
            "p95_tpot_ms": metrics.p95_tpot_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "std_itl_ms": metrics.std_itl_ms,
            "p99_itl_ms": metrics.p99_itl_ms,
            "p95_itl_ms": metrics.p95_itl_ms,
            "ttfts": [output.ttft for output in outputs],
            "itls": [output.itl for output in outputs],
            "errors": [output.error for output in outputs],
            "average_attainment_ttft": metrics.average_attainment_ttft,
            "average_attainment_tpot": metrics.average_attainment_tpot,
        }

        if save_length:
            result["input_lens"] = [output.prompt_len for output in outputs]
            result["output_lens"] = [output.output_len for output in outputs]

        for model, model_metrics in model_to_metrics.items():
            result[model] = {
                "completed": model_metrics.completed,
                "aborted": model_metrics.aborted,
                "request_throughput": model_metrics.request_throughput,
                "input_throughput": model_metrics.input_throughput,
                "output_throughput": model_metrics.output_throughput,
                "input_output_throughput": model_metrics.input_output_throughput,
                "mean_e2e_latency_ms": model_metrics.mean_e2e_latency_ms,
                "median_e2e_latency_ms": model_metrics.median_e2e_latency_ms,
                "p99_e2e_latency_ms": model_metrics.p99_e2e_latency_ms,
                "p95_e2e_latency_ms": model_metrics.p95_e2e_latency_ms,
                "average_attainment": model_metrics.average_attainment,
                "mean_ttft_ms": model_metrics.mean_ttft_ms,
                "median_ttft_ms": model_metrics.median_ttft_ms,
                "p99_ttft_ms": model_metrics.p99_ttft_ms,
                "p95_ttft_ms": model_metrics.p95_ttft_ms,
                "mean_tpot_ms": model_metrics.mean_tpot_ms,
                "median_tpot_ms": model_metrics.median_tpot_ms,
                "p99_tpot_ms": model_metrics.p99_tpot_ms,
                "p95_tpot_ms": model_metrics.p95_tpot_ms,
                "mean_itl_ms": model_metrics.mean_itl_ms,
                "median_itl_ms": model_metrics.median_itl_ms,
                "p99_itl_ms": model_metrics.p99_itl_ms,
                "p95_itl_ms": model_metrics.p95_itl_ms,
            }
            if save_length:
                result[model]["input_lens"] = [
                    output.prompt_len for output in outputs if output.model == model
                ]
                result[model]["output_lens"] = [
                    output.output_len for output in outputs if output.model == model
                ]

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        output_file_name_all = get_all_metric_file_name(
            trace_config, benchmark_mode, num_gpus
        )
        output_all = os.path.join(save_path, output_file_name_all)

        # save all metrics to jsonl
        with open(output_all, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"All results saved to {output_all}")

        output_key = os.path.join(
            save_path, get_key_metric_file_name(trace_config, benchmark_mode, num_gpus)
        )
        # Make sure parent directory of output_key exists
        os.makedirs(os.path.dirname(output_key), exist_ok=True)

        # save key metrics to tsv
        key_metrics_columns = [
            "exp_name",
            "time_scale",
            "replication",
            "Mean E2E Latency (s)",
            "P99 E2E Latency (s)",
            "P95 E2E Latency (s)",
            "Request Tput (req/s)",
            "Output Token Tput (tok/s)",
            "Input+Output token Tput (tok/s)",
            "Mean TTFT (s)",
            "P99 TTFT (s)",
            "P95 TTFT (s)",
            "Mean TPOT (ms)",
            "P99 TPOT (ms)",
            "P95 TPOT (ms)",
            "Mean ITL (ms)",
            "P99 ITL (ms)",
            "P95 ITL (ms)",
            "SLO Attainment",
            "Median E2E Latency (ms)",
            "Median TTFT (s)",
            "Median TPOT (ms)",
            "Median ITL (ms)",
            "Average Attainment TTFT",
            "Average Attainment TPOT",
        ]
        # if file does not exist, create it
        if not os.path.isfile(output_key):
            with open(output_key, "w") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(key_metrics_columns)

        with open(output_key, "a") as f:
            writer = csv.writer(f, delimiter="\t")
            key_metrics = [
                result["exp_name"],
                trace_config.time_scale,
                trace_config.replication,
                result["mean_e2e_latency_ms"] / 1000,
                result["p99_e2e_latency_ms"] / 1000,
                result["p95_e2e_latency_ms"] / 1000,
                result["request_throughput"],
                result["output_throughput"],
                result["input_output_throughput"],
                result["mean_ttft_ms"] / 1000,
                result["p99_ttft_ms"] / 1000,
                result["p95_ttft_ms"] / 1000,
                result["mean_tpot_ms"],
                result["p99_tpot_ms"],
                result["p95_tpot_ms"],
                result["mean_itl_ms"],
                result["p99_itl_ms"],
                result["p95_itl_ms"],
                result["average_attainment"],
                result["median_e2e_latency_ms"],
                result["median_ttft_ms"] / 1000,
                result["median_tpot_ms"],
                result["median_itl_ms"],
                result["average_attainment_ttft"],
                result["average_attainment_tpot"],
            ]
            writer.writerow(key_metrics)
        print(f"Key metrics saved to {output_key}")
    else:
        print(f"Error running benchmark for request rate: {trace_config.req_rate}")
        print("-" * 30)


def save_output_requests(
    exp_name, trace_config, outputs, save_path, benchmark_mode, num_gpus: int = 1
):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    output_file_name = get_output_request_file_name(
        exp_name, trace_config, benchmark_mode, num_gpus
    )
    output_file = os.path.join(save_path, output_file_name)
    # Make sure parent directory of output_file exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    serializable_outputs = []
    for req_output in outputs:
        serializable_output = {
            "success": req_output.success,
            "latency": req_output.latency,
            "latency_server": req_output.latency_server,
            "ttft": req_output.ttft,
            "itl": req_output.itl,
            "tpot": req_output.tpot,
            "wait_time": req_output.wait_time,
            "arrival_time": req_output.arrival_time,
            "out_queue_time": req_output.out_queue_time,
            "prefill_finish_time": req_output.prefill_finish_time,
            "finish_time": req_output.finish_time,
            "decode_timestamps": req_output.decode_timestamps,
            "prompt_len": req_output.prompt_len,
            "error": req_output.error,
            "output_len": req_output.output_len,
            "slo": req_output.slo,
            "model": req_output.model,
        }
        serializable_outputs.append(serializable_output)

    with open(output_file, "a") as f:
        f.write(json.dumps(serializable_outputs) + "\n")
    print(f"Output requests saved to {output_file}")


async def test_run(test_request: Request, server: str):
    print("Starting initial single prompt test run...")
    print(f"Sending test request to server: {test_request.model}")
    test_output = await send_generate_request(server, test_request)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")


async def benchmark_with_timeout(
    args, input_requests, server, trace_config, timeout=60 * 60
):
    try:
        return await asyncio.wait_for(
            benchmark(
                args=args,
                input_requests=input_requests,
                server=server,
                trace_config=trace_config,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        print("Benchmark timed out. Please increase the timeout.")
        return "failed"


def run_benchmark(
    args_: argparse.Namespace, trace_config, requests=None, timeout=60 * 60
):
    global args
    args = args_
    server = args.base_url or f"http://{args.host}:{args.port}"

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if requests is None:
        requests = generate_synthetic_reqs(trace_config)

    if args.debug:
        print("num requests:", len(requests))
        for req in requests[:10]:
            print(
                f"req {req.req_id}: model {req.model}, arrival time {req.arrival_time:.2f}, input len {req.prompt_len}, output len {req.output_len}"
            )
            print(req)
    # benchmark with 20 minutes timeout
    results = asyncio.run(
        benchmark_with_timeout(args, requests, server, trace_config, timeout)
    )

    return results


def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument("--num-models", "-n", type=int, default=4)
    parser.add_argument(
        "--model-paths",
        "-m",
        type=str,
        nargs="+",
        help="The paths of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--base-url", type=str, default=None)

    parser.add_argument("--dataset", type=str, help="Path to the dataset.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--results-path", type=str, default="benchmark-results")
    parser.add_argument("--request-path", type=str, default="output-requests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="collocate",
        help="Name of the experiment. It will be used as a key in the output json.",
    )
    parser.add_argument("--disable-tqdm", action="store_true")

    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        choices=["adaptive", "one-queue", "sequential-swapping"],
        help="The policy to swap models. Options: adaptive, one-queue, sequential-swapping",
    )

    parser.add_argument(
        "--dynamic-models",
        type=str,
        nargs="+",
        help="Models that will be dynamically loaded and unloaded.",
    )
    parser.add_argument(
        "--persistent-models",
        type=str,
        nargs="+",
        help="Models that will be persistently loaded.",
    )
    parser.add_argument(
        "--preempt",
        action="store_true",
        help="Whether to preempt the requests on model deactivation.",
    )
    parser.add_argument("--memory-pool-size", type=float, default=16)
    parser.add_argument("--enable-elastic-memory", action="store_true")
    parser.add_argument("--req-rate", type=int, default=20)
    parser.add_argument("--real-trace", type=str, default=None)
    parser.add_argument("--csv-trace", type=str, default=None, help="Path to CSV trace file")
    parser.add_argument("--micro-benchmark", action="store_true")
    parser.add_argument("--e2e-benchmark", action="store_true")
    parser.add_argument("--hyper-trace", action="store_true")
    parser.add_argument("--hyper-trace-selected-models", nargs='+', type=str, default=None),
    parser.add_argument("--hyper-trace-per-model-ttft-slo-scale", nargs='+', type=int, default=None),
    parser.add_argument("--hyper-trace-per-model-tpot-slo-scale", nargs='+', type=int, default=None),
    parser.add_argument("--gpu-scheduler-benchmark", action="store_true"),
    parser.add_argument("--uniform-trace", action="store_true"),
    parser.add_argument("--two-phase-trace", action="store_true"),
    parser.add_argument("--time-scale", type=float, default=1)
    parser.add_argument("--replication", type=int, default=1)
    parser.add_argument("--model-lst", type=str, default="model_lst.yml")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ttft-slo-scale", type=float, default=5)
    parser.add_argument("--tpot-slo-scale", type=float, default=5)
    parser.add_argument("--mmy-debug", action="store_true")
    args = parser.parse_args()

    if args.model_paths is None:
        model_paths = [f"model_{i+1}" for i in range(args.num_models)]
        # model_paths = [f"model_{i}" for i in range(args.num_models, 0, -1)]
    else:
        model_paths = args.model_paths

    if args.real_trace:
        print(f"Real trace file: {args.real_trace}")
        from trace import RealWorldTrace

        # If model_paths were provided as arguments, use them directly
        # Otherwise, load from yaml file (original behavior)
        if args.model_paths is not None:
            # Use the model_paths passed as arguments (TP mode)
            config = TraceConfig(
                ttft_slo_scale=args.ttft_slo_scale,
                tpot_slo_scale=args.tpot_slo_scale,
                model_paths=args.model_paths,
                slo=100,
                micro_benchmark=args.micro_benchmark,
                e2e_benchmark=args.e2e_benchmark,
                time_scale=args.time_scale,
                replication=args.replication,
            )
            
            # Use the TP e2e benchmark method (from trace_1.py)
            requests = RealWorldTrace(
                pkl_file_path=args.real_trace
            ).generate_e2e_benchmark_reqs(config, num_models=args.num_models)
        else:
            # Use yaml file and 18m method (original behavior)
            import yaml

            with open(args.model_lst, "r") as file:
                model_list = yaml.safe_load(file)["model"]

            config = TraceConfig(
                ttft_slo_scale=args.ttft_slo_scale,
                tpot_slo_scale=args.tpot_slo_scale,
                model_paths=model_list,
                slo=100,
                micro_benchmark=args.micro_benchmark,
                e2e_benchmark=args.e2e_benchmark,
                time_scale=args.time_scale,
                replication=args.replication,
            )
            
            requests = RealWorldTrace(
                pkl_file_path=args.real_trace
            ).generate_e2e_benchmark_reqs_18m(config, num_models=args.num_models)
        
        # Check if this is TP mode (model_paths provided directly)
        if args.model_paths is not None:
            # TP mode - run directly
            asyncio.run(run_tp_mode(args, config, requests))
        else:
            # Complex mode - use original benchmark logic
            run_benchmark(args, config, requests)
    else:
        # Handle synthetic trace case (no real-trace file provided)
        print("Using synthetic trace generation")
        
        # Create TraceConfig for synthetic trace
        config = TraceConfig(
            req_rate=args.req_rate,
            duration=60,  # default duration
            input_range=(256, 512),  # default input range
            output_range=(256, 512),  # default output range  
            model_paths=model_paths,
            seed=args.seed,
            alpha=2.1,  # default alpha for power law distribution
            cv=1,  # default coefficient of variation
            slo=10,  # default SLO
            micro_benchmark=args.micro_benchmark,
            e2e_benchmark=args.e2e_benchmark,
            time_scale=args.time_scale,
            replication=args.replication,
            ttft_slo_scale=args.ttft_slo_scale,
            tpot_slo_scale=args.tpot_slo_scale,
        )
        
        # Run benchmark with synthetic trace
        run_benchmark(args, config)
