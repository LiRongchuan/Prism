import io
import os
import pickle
import time
from typing import Dict, List

import torch
from torch.multiprocessing.queue import ForkingPickler
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
from vllm.model_executor.model_loader import get_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.utils import monkey_patch_vllm_dummy_weight_loader

# os.environ["TRANSFORMERS_OFFLINE"] = "1"
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.queues import Empty

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
import logging
logger = logging.getLogger(__name__)

def monkey_patch_transformers_llama_rotary_embedding():
    # monkey patch transformers's modeling_llama.py's rotary_embedding.py's forward method
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    original_forward = LlamaRotaryEmbedding.forward

    def forward_new(self, *args, **kwargs):
        self.inv_freq = self.inv_freq.to(args[0].device)
        return original_forward(self, *args, **kwargs)

    LlamaRotaryEmbedding.forward = forward_new


def _dummy_init_distributed():
    def port_in_use(port):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    port = 23456
    while port_in_use(port):
        port += 1
    dist_init_method = f"tcp://127.0.0.1:{port}"
    backend = "gloo"
    init_distributed_environment(
        backend=backend,
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=dist_init_method,
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    monkey_patch_vllm_dummy_weight_loader()


def load_model(
    share_mem=True,
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    use_transformers=False,
):
    if use_transformers:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model_config = VllmModelConfig(
            model=model_path,
            tokenizer=None,
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="auto",
            seed=0,
        )
        load_config = LoadConfig(load_format="auto")
        t0 = time.perf_counter()
        model = get_model(
            model_config=model_config,
            load_config=load_config,
            device_config=DeviceConfig("cpu"),
            parallel_config=None,
            scheduler_config=None,
            lora_config=None,
            cache_config=None,
        )
        t1 = time.perf_counter()
        print(f"Time taken to load model: {t1 - t0:.4f} seconds")
        model.make_empty_intermediate_tensors = None
        model.model.make_empty_intermediate_tensors = None
    if share_mem:
        model.share_memory()
    return model


def create_empty_gpu_model(model_serialized, gpu_id):
    model = pickle.loads(model_serialized)
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = torch.empty_like(state_dict[key], device=f"cuda:{gpu_id}")
    model.load_state_dict(state_dict, assign=True)
    return model


def create_empty_gpu_model_from_cpu_model(cpu_model, gpu_id):
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(cpu_model)
    model = pickle.loads(buf.getvalue())
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = torch.empty_like(state_dict[key], device=f"cuda:{gpu_id}")
    model.load_state_dict(state_dict, assign=True)
    return model


def gpu_transfer(
    gpu_tensor, cpu_tensor, broker_gpu_id, start, end, stream, target_gpu_id
):
    with torch.cuda.stream(stream):
        if broker_gpu_id == target_gpu_id:
            gpu_tensor[start:end].copy_(cpu_tensor[start:end])
        else:
            gpu_tensor[start:end].copy_(
                cpu_tensor[start:end].to(f"cuda:{broker_gpu_id}", non_blocking=True)
            )


def multi_thread_copy_model_to_gpu(
    cpu_state_dict,
    gpu_state_dict,
    target_gpu_id,
    executor,
    num_gpus,
    max_threads,
    streams,
    num_shards,
    shard_id,
):
    futures = []
    for key in gpu_state_dict.keys():
        gpu_tensor = gpu_state_dict[key]
        cpu_tensor = cpu_state_dict[key]
        size = gpu_tensor.shape[0]
        start_shard = shard_id * size // num_shards
        end_shard = (shard_id + 1) * size // num_shards
        local_gpu_tensor = gpu_tensor[start_shard:end_shard]
        local_cpu_tensor = cpu_tensor[start_shard:end_shard]
        local_size = local_gpu_tensor.shape[0]
        assert local_size % max_threads == 0
        for broker_id in range(max_threads):
            start = broker_id * local_size // max_threads
            end = (broker_id + 1) * local_size // max_threads
            broker_gpu_id = (broker_id + target_gpu_id + 1) % num_gpus
            future = executor.submit(
                gpu_transfer,
                local_gpu_tensor,
                local_cpu_tensor,
                broker_gpu_id,
                start,
                end,
                streams[broker_id],
                target_gpu_id,
            )
            futures.append(future)
    return futures


def multi_process_worker(
    model_dict, input_queue, output_queue, max_threads, gpu_ids, num_shards, shard_id
):
    executor = ThreadPoolExecutor(max_workers=max_threads)
    streams = {i: torch.cuda.Stream(device=f"cuda:{i}") for i in gpu_ids}
    output_queue.put("ready")
    num_gpus = len(gpu_ids)
    while True:
        model_key, gpu_model, target_gpu_id = input_queue.get()
        futures = multi_thread_copy_model_to_gpu(
            model_dict[model_key].state_dict(),
            gpu_model.state_dict(),
            target_gpu_id,
            executor,
            num_gpus,
            max_threads,
            streams,
            num_shards,
            shard_id,
        )
        for future in futures:
            future.result()
        output_queue.put("done")


class ModelService:
    def __init__(
        self,
        model_dict: Dict[str, torch.nn.Module],
        input_queue: torch.multiprocessing.Queue,
        output_queue: Dict[str, torch.multiprocessing.Queue],
        max_threads: int,
        gpu_ids: List[int],
        num_shards: int = 1,
        service_id: int = 0,
    ):
        self.model_dict = model_dict
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.max_threads = max_threads
        self.num_shards = num_shards
        self.gpu_ids = gpu_ids
        self.service_id = service_id
        if num_shards > 1:
            self.worker_input_queues = [
                torch.multiprocessing.Queue() for i in range(self.num_shards)
            ]
            self.worker_output_queues = [
                torch.multiprocessing.Queue() for i in range(self.num_shards)
            ]
            self.workers = [
                torch.multiprocessing.Process(
                    target=multi_process_worker,
                    args=(
                        self.model_dict,
                        self.worker_input_queues[i],
                        self.worker_output_queues[i],
                        self.max_threads,
                        self.gpu_ids,
                        self.num_shards,
                        i,
                    ),
                )
                for i in range(self.num_shards)
            ]
            for worker in self.workers:
                worker.start()
            for worker_output_queue in self.worker_output_queues:
                worker_output_queue.get()
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_threads)
            self.streams = {i: torch.cuda.Stream(device=f"cuda:{i}") for i in gpu_ids}

    def run(self):
        for output_queue in self.output_queue.values():
            output_queue.put("ready")
        while True:
            try:
                model_key, engine_id, target_gpu_id, gpu_model = self.input_queue.get(timeout=5)
            except Empty as e:
                logging.info(f"Model service {self.service_id} input queue is empty, waiting for next request")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            logging.info(f"Model key: {model_key}, engine id: {engine_id}, target gpu id: {target_gpu_id}")
            try:
                # gpu_model = create_empty_gpu_model(self.model_dict_serialized[model_key], target_gpu_id)
                t0 = time.perf_counter()
                # gpu_model = create_empty_gpu_model_from_cpu_model(
                #     self.model_dict[model_key], target_gpu_id
                # )
                if self.num_shards > 1:
                    for i in range(self.num_shards):
                        self.worker_input_queues[i].put(
                            (model_key, gpu_model, target_gpu_id)
                        )

                    self.output_queue[engine_id].put(gpu_model)
                    for i in range(self.num_shards):
                        self.worker_output_queues[i].get()
                    t1 = time.perf_counter()
                    self.output_queue[engine_id].put(t1 - t0)
                    self.output_queue[engine_id].put(self.service_id)
                else:
                    futures = multi_thread_copy_model_to_gpu(
                        self.model_dict[model_key].state_dict(),
                        gpu_model.state_dict(),
                        target_gpu_id,
                        self.executor,
                        len(self.gpu_ids),
                        self.max_threads,
                        self.streams,
                        1,
                        0,
                    )
                    self.output_queue[engine_id].put("success")
                    for future in futures:
                        future.result()
                    t1 = time.perf_counter()
                    logging.info(f"Time taken to copy model to GPU {target_gpu_id}, service {self.service_id}, engine {engine_id}, model {model_key}: {t1 - t0:.4f} seconds")
                    self.output_queue[engine_id].put(t1 - t0)
                    self.output_queue[engine_id].put(self.service_id)
                del gpu_model
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                logging.error(f"Model service {self.service_id} met error, when loading model {model_key} to GPU {target_gpu_id} for engine {engine_id} on service {self.service_id}, error: {e}")
                self.output_queue[engine_id].put(None)
                self.output_queue[engine_id].put(f"Error: {e}")
                continue

    def __del__(self):
        if self.num_shards > 1:
            for worker in self.workers:
                worker.kill()

            for worker in self.workers:
                worker.join()




def run_model_service(
    multi_model_server_args,
    cpu_model_dict: Dict[str, torch.nn.Module],
    input_queue: torch.multiprocessing.Queue,
    output_queues: Dict[str, torch.multiprocessing.Queue],
    max_threads: int,
    gpu_ids: List[int],
    num_shards: int,
    instance: int,
):
    from sglang.srt.utils import configure_logger
    configure_logger(
        multi_model_server_args,
        prefix=f" Model_Service_{instance}",
        log_file_suffix="model_service",
    )
    model_service = ModelService(
        cpu_model_dict, input_queue, output_queues, max_threads, gpu_ids, num_shards
    )
    model_service.run()

# def run_model_service(
#     model_dict, input_queue, output_queue, max_threads, gpu_ids, num_shards, service_id
# ):
#     _dummy_init_distributed()
#     model_service = ModelService(
#         model_dict,
#         input_queue,
#         output_queue,
#         max_threads,
#         gpu_ids,
#         num_shards,
#         service_id,
#     )
#     model_service.run()


def get_gpu_model(input_queue, output_queues, engine_id, target_gpu_id, model_key):
    t0 = time.perf_counter()
    input_queue.put((model_key, engine_id, target_gpu_id))
    gpu_model = output_queues[engine_id].get()
    loading_time = output_queues[engine_id].get()
    service_id = output_queues[engine_id].get()
    t1 = time.perf_counter()
    # print(f"\033[92mTime taken to get the model from the input queue: {t1 - t0:.4f} seconds")
    print(
        f"\033[92mTime taken to get the model from the input queue: {t1 - t0:.4f} seconds, loading time: {loading_time:.4f} seconds, service_id: {service_id}\033[0m"
    )
    return gpu_model


def create_model_dict(models, use_transformers=False):
    model_dict = {}
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = [
            executor.submit(
                load_model, model_path=model_path, use_transformers=use_transformers
            )
            for model_path in models
        ]
        for model_path, future in zip(models, futures):
            model_dict[model_path] = future.result()
    return model_dict


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    # models = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]
    models = ["meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]
    input_queue = torch.multiprocessing.Queue()
    workers_per_gpu = 2
    num_gpus = 2
    output_queues = {}
    engine_ids = []
    for gpu_id in range(num_gpus):
        for worker_id in range(workers_per_gpu):
            engine_id = f"{gpu_id}_{worker_id}"
            output_queues[engine_id] = torch.multiprocessing.Queue()
            engine_ids.append(engine_id)
    max_threads = 2
    gpu_ids = [i for i in range(2)]
    num_shards = 1
    use_transformers = True
    model_dict = create_model_dict(models, use_transformers)
    # run model service in a separate process
    num_services = 2
    for service_id in range(num_services):
        p = torch.multiprocessing.Process(
            target=run_model_service,
            args=(
                model_dict,
                input_queue,
                output_queues,
                max_threads,
                gpu_ids,
                num_shards,
                service_id,
            ),
        )
        p.start()
        time.sleep(1)
        for output_queue in output_queues.values():
            ready_signal = output_queue.get()
            assert ready_signal == "ready"
        print(f"\033[92mService {service_id} is ready\033[0m")
    if use_transformers:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        monkey_patch_transformers_llama_rotary_embedding()
    while True:
        input_str = input("input: worker_id,worker_id2,worker_id3,... (q to quit):\n")
        if input_str == "q":
            break
        if input_str == "gc":
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # worker_ids = input_str.strip().split(",")
        try:
            load_engine_ids = [int(i) for i in input_str.strip().split(",")]
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")
            continue
        print(load_engine_ids)
        gpu_models = []
        target_gpu_ids = []
        for i, engine_id in enumerate(engine_ids):
            target_gpu_id = int(engine_id.split("_")[0])
            if i in load_engine_ids:
                model_key = models[i % len(models)]
                print(
                    f"\033[92mGetting model {model_key} for engine {engine_id}\033[0m"
                )
                gpu_model = get_gpu_model(
                    input_queue, output_queues, engine_id, target_gpu_id, model_key
                )
                gpu_models.append(gpu_model)
                target_gpu_ids.append(target_gpu_id)
        if use_transformers:
            time.sleep(1)
            for gpu_model, target_gpu_id in zip(gpu_models, target_gpu_ids):
                for i in range(2):
                    print(
                        f"\033[92mWorker {worker_id} on GPU {target_gpu_id} is generating...\033[0m"
                    )
                    prompt = "Hello, how are you?"
                    inputs = tokenizer(prompt, return_tensors="pt").to(
                        f"cuda:{target_gpu_id}"
                    )
                    result = gpu_model.generate(**inputs)
                    # print the result
                    print(
                        "\033[94m"
                        + tokenizer.decode(result[0], skip_special_tokens=True)
                        + "\033[0m"
                    )
                    print(
                        f"\033[92mWorker {worker_id} on GPU {target_gpu_id} has finished generating.\033[0m"
                    )
        del gpu_models
        gc.collect()
        torch.cuda.empty_cache()
    p.kill()
    p.join()
