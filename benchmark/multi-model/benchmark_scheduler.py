import argparse
import asyncio
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
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import tqdm
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

global args


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    extra_request_body: Dict[str, Any]


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0
    slo: float = 0.0
    model: str = ""


async def send_generate_request(
    backend: str,
    server: str,
    req: Request,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = server + "/generate"

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "srt":
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
            "prompt_len": req.prompt_len,
            "output_len": req.output_len,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        output.prompt_len = req.prompt_len
        output.slo = req.slo
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
                                    f"Aborted request {req.req_id} due to exceed slo."
                                )
                                success = False
                                reason = "Exceed SLO"

                    if success:
                        output.latency = latency
                        output.success = True
                        output.output_len = req.output_len
                    else:
                        output.error = reason
                        output.success = False
                    print(
                        f"Req_id {req.req_id}, Req.model: {req.model}, Success: {output.success}, Latency: {output.latency:.2f}s, Output len: {output.output_len}"
                    )
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            print(f"Error in send_generate_request {req.req_id} model {req.model}: {e}")
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            print(
                f"Error in sending generate request {req.req_id} model {req.model}: {output.error}"
            )

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


async def deactivate_model(
    model_name: str,
    server: str,
    instance_idx: Optional[int] = 0,
    preempt: bool = False,
) -> None:
    await asyncio.sleep(0.1)
    preempt_mode = "RETURN"
    evict_waiting_requests = False
    deactivate_url = server + "/deactivate"
    pload = {
        "model_name": model_name,
        "instance_idx": instance_idx,
        "evict_waiting_requests": evict_waiting_requests,
        "preempt": preempt,
        "preempt_mode": preempt_mode,
    }
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(deactivate_url, json=pload) as response:
            response_json = await response.json()
            print(
                f"deactivate request success: {response_json.get('success', None)}. Content: {response_json}"
            )


async def activate_model(
    model_name: str,
    server: str,
    instance_idx: Optional[int] = 0,
    memory_pool_size: Optional[int] = None,
) -> None:
    await asyncio.sleep(1)
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
                f"activate request success: {response_json.get('success', None)}. Content: {response_json}"
            )


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


async def run_swap_loop(queue: asyncio.Queue, server: str, preempt: bool):

    while True:
        element = await queue.get()
        if element == "stop":
            break
        model_1, model_2 = element
        print(f"Swapping {model_1} and {model_2}")
        results = []
        if model_1 is not None:
            results.append(
                asyncio.create_task(deactivate_model(model_1, server, preempt=preempt))
            )
        if model_2 is not None:
            results.append(asyncio.create_task(activate_model(model_2, server)))
        await asyncio.gather(*results)
        # await asyncio.sleep(3)


async def benchmark(
    mode: str,
    backend: str,
    input_requests: List[Request],
    server: str,
    trace_config: TraceConfig,
    debug: bool = False,
    disable_tqdm: bool = False,
    send_swap_requests: bool = False,
    preempt: bool = False,
) -> None:
    request_rate = trace_config.req_rate
    slo = trace_config.slo
    duration = trace_config.duration
    input_range = trace_config.input_range
    output_range = trace_config.output_range
    alpha = trace_config.alpha
    cv = trace_config.cv
    on_off_model_percentage = trace_config.on_off_model_percentage
    on_off_cycle_len = trace_config.on_off_cycle_len

    # print("Starting initial single prompt test run...")
    # test_req = input_requests[0]
    # print(f"Sending test request to server: {test_req.model}")
    # test_output = await send_generate_request(backend, server, test_req)
    # if not test_output.success:
    #     raise ValueError(
    #         "Initial test run failed - Please make sure benchmark arguments "
    #         f"are correctly specified. Error: {test_output.error}"
    #     )
    # else:
    #     print("Initial test run completed. Starting main benchmark run...")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    start = time.time()
    tasks: List[asyncio.Task] = []
    control_tasks: List[asyncio.Task] = []

    if send_swap_requests:
        swap_queue = asyncio.Queue()
        control_tasks.append(
            asyncio.create_task(run_swap_loop(swap_queue, server, preempt))
        )
    else:
        swap_queue = None

    # tmp: for testing deactivation
    counter = 1
    num_swaps = 0
    model_last_req = None
    last_model = None
    current_model = None
    for req in input_requests:
        sleep_time = max(0, start + req.arrival_time - time.time())
        if send_swap_requests:
            if req.model != current_model:
                await swap_queue.put((current_model, req.model))
                last_model = current_model
                current_model = req.model
                num_swaps += 1
        await asyncio.sleep(sleep_time)
        if model_last_req is None:
            model_last_req = req.model
        if debug:
            print(
                f"Req {req.req_id} for model {req.model} waited {sleep_time:.2f} before sending to server."
            )
        task = asyncio.create_task(
            send_generate_request(backend, server, req, pbar=pbar)
        )
        tasks.append(task)

    outputs = []
    while True:
        done, pending = await asyncio.wait(tasks, timeout=20)
        for task in done:
            outputs.append(task.result())
        if send_swap_requests:
            if len(pending) == 0:
                await swap_queue.put("stop")
                break
            else:
                print("Waiting for requests to finish")
                tasks = list(pending)
                # try to swap models
                # num_swaps += 1
                # await swap_queue.put((current_model, last_model))
                # current_model, last_model = last_model, current_model
        else:
            if len(pending) == 0:
                break
            else:
                print("Waiting for requests to finish")
                tasks = list(pending)

    await asyncio.gather(*control_tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    # get_num_swaps = send_request(server, "get_num_swaps")
    # get_num_swaps_result = await get_num_swaps
    # assert get_num_swaps_result is not None
    # assert "num_swaps" in get_num_swaps_result
    # num_swaps = get_num_swaps_result["num_swaps"]

    metrics = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
    )

    # split requests according to model
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
        model_metrics = calculate_metrics(
            input_requests=reqs,
            outputs=model_outputs,
            dur_s=benchmark_duration,
        )
        model_to_metrics[model] = model_metrics

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Mode:", mode))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print("{:<40} {:<10}".format("Alpha:", alpha))
    print("{:<40} {:<10}".format("CV:", cv))
    print("{:<40} {:<10}".format("SLO:", slo))
    print("{:<40} {:<10}".format("Trace duration:", duration))
    print("{:<40} {:<10}".format("On-Off ratio:", on_off_model_percentage))
    print("{:<40} {:<10}".format("On-Off cycle len:", on_off_cycle_len))

    if num_swaps != 0:
        print("{:<40} {:<10}".format("Num swaps:", num_swaps))
    print("{:<40} {:<10.2f}".format("Average Attainment:", metrics.average_attainment))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Aborted requests:", metrics.aborted))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
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
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
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

    num_models = len(model_to_metrics)
    if num_models > 1:  # print per model metrics
        print("{s:{c}^{n}}".format(s="Each Model Metrics", n=50, c="-"))
        for model, model_metrics in model_to_metrics.items():
            print(f"*** Model: {model} ***")
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
    # print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    # print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    # print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    # print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    # print(
    #     "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    # )
    # print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    # print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    # print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    # print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    # print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    # print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    # print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            # "dataset_name": args.dataset_name,
            "mode": mode,
            "request_rate": request_rate,
            "alpha": alpha,
            "cv": cv,
            "num_swaps": num_swaps,
            "request_duration": duration,
            "benchmark_duration": benchmark_duration,
            "average_input_tokens": metrics.total_input / metrics.completed,
            "average_output_tokens": metrics.total_output / metrics.completed,
            "request_throughput": metrics.request_throughput,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
            "average_attainment": metrics.average_attainment,
            # "median_ttft_ms": metrics.median_ttft_ms,
            # "median_itl_ms": metrics.median_itl_ms,
            "completed": metrics.completed,
            "aborted": metrics.aborted,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "input_output_throughput": metrics.input_output_throughput,
            # "mean_ttft_ms": metrics.mean_ttft_ms,
            # "std_ttft_ms": metrics.std_ttft_ms,
            "p99_ttft_ms": metrics.p99_ttft_ms,
            # "mean_tpot_ms": metrics.mean_tpot_ms,
            # "median_tpot_ms": metrics.median_tpot_ms,
            # "std_tpot_ms": metrics.std_tpot_ms,
            # "p99_tpot_ms": metrics.p99_tpot_ms,
            # "mean_itl_ms": metrics.mean_itl_ms,
            # "median_itl_ms": metrics.median_itl_ms,
            # "std_itl_ms": metrics.std_itl_ms,
            # "p99_itl_ms": metrics.p99_itl_ms,
            # "input_lens": [output.prompt_len for output in outputs],
            # "output_lens": output_lens,
            # "ttfts": [output.ttft for output in outputs],
            # "itls": [output.itl for output in outputs],
            # "errors": [output.error for output in outputs],
        }
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
                "average_attainment": model_metrics.average_attainment,
            }
        return result
    else:
        print(f"Error running benchmark for request rate: {request_rate}")
        print("-" * 30)
        return None


@dataclass
class BenchmarkMetrics:
    completed: int
    aborted: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    input_output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    average_attainment: float = 0.0


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
    attainment: List[int] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)

            e2e_latencies.append(outputs[i].latency)
            if outputs[i].slo is not None:
                attainment.append(1 if outputs[i].latency < outputs[i].slo else 0)
            else:
                attainment.append(1)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)
            attainment.append(0)
            aborted += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        aborted=aborted,
        total_input=total_input,
        total_output=sum(output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        input_output_throughput=(total_input + sum(output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        average_attainment=np.mean(attainment),
    )

    return metrics


def run_benchmark_with_requests(args_: argparse.Namespace, requests, trace_config):
    global args
    args = args_
    server = args.base_url or f"http://{args.host}:{args.port}"

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(f"Request sent to {req.model}")

    # benchmark
    results = asyncio.run(
        benchmark(
            mode=args.mode,
            backend=args.backend,
            input_requests=requests,
            server=server,
            trace_config=trace_config,
            debug=args.debug,
            disable_tqdm=args.disable_tqdm,
            send_swap_requests=args.send_swap_requests,
            preempt=args.preempt,
        )
    )
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    output_file_name = get_output_file_name(trace_config)
    output = os.path.join(args.results_path, output_file_name)

    with open(output, "a") as f:
        f.write(json.dumps(results) + "\n")

    return results


def run_benchmark(args_: argparse.Namespace, trace_config):
    global args
    args = args_
    server = args.base_url or f"http://{args.host}:{args.port}"

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    requests = generate_synthetic_reqs(trace_config)

    if args.debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)

    # benchmark
    results = asyncio.run(
        benchmark(
            mode=args.mode,
            backend=args.backend,
            input_requests=requests,
            server=server,
            trace_config=trace_config,
            debug=args.debug,
            disable_tqdm=args.disable_tqdm,
            send_swap_requests=args.send_swap_requests,
            preempt=args.preempt,
        )
    )
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    output_file_name = get_output_file_name(trace_config)
    output = os.path.join(args.results_path, output_file_name)

    with open(output, "a") as f:
        f.write(json.dumps(results) + "\n")

    if os.path.exists("stats.log"):
        import figplot

        from sglang.multi_model.scheduling.stats import ScheduleStats

        with open("stats.log", "r") as f:
            # Ensure the file is a JSON array
            json_list = json.load(f)
            stats = [ScheduleStats.from_dict(item) for item in json_list]
            figplot.plot_sim_stats(
                requests, trace_config, stats, f"mod_availability.pdf"
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


def get_output_file_name(trace_config):
    now = datetime.now().strftime("%m%d")
    # filename = f"{now}_on-off-ratio_{trace_config.on_off_model_percentage}_duration-{trace_config.duration}_req_rate-{trace_config.req_rate}_alpha-{trace_config.alpha}_cv-{trace_config.cv}_slo-{trace_config.slo}.json"
    filename = f"{now}_on-off-ratio_{trace_config.on_off_model_percentage}_duration-{trace_config.duration}_alpha-{trace_config.alpha}_cv-{trace_config.cv}_slo-{trace_config.slo}.json"
    return filename


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
    parser.add_argument(
        "--backend",
        type=str,
        default="srt",
        choices=["srt"],
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--base-url", type=str, default=None)

    parser.add_argument("--dataset", type=str, help="Path to the dataset.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--results-path", type=str, default="benchmark-results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="collocate",
        help="Mode of the benchmark. It will be used as a key in the output json.",
    )
    parser.add_argument("--send-swap-requests", action="store_true")
    parser.add_argument("--preempt", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")

    args = parser.parse_args()
    print(f"run with preempt: {args.preempt}")

    if args.model_paths is None:
        model_paths = [f"model_{i+1}" for i in range(args.num_models)]
        # model_paths = [f"model_{i}" for i in range(args.num_models, 0, -1)]
    else:
        model_paths = args.model_paths

    alpha = 2.1
    req_rate = 40
    cv = 1
    seed = 42
    # input_range = [768, 769]
    # output_range = [768, 769]
    input_range = [256, 512]
    output_range = [256, 512]
    duration = 120
    slo = 20
    on_off_model_percentage = 0.5
    on_off_cycle_len = 30

    trace_config = TraceConfig(
        req_rate=req_rate,
        duration=duration,
        input_range=input_range,
        output_range=output_range,
        model_paths=model_paths,
        seed=seed,
        alpha=alpha,
        cv=cv,
        slo=slo,
        sequential=False,
        on_off_cycle_len=on_off_cycle_len,
        off_ratio=0.9,
        on_off_model_percentage=on_off_model_percentage,
    )

    run_benchmark(args, trace_config)
