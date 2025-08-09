import copy
import json
import logging
import os
import signal
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiohttp
import torch
import zmq

from sglang.multi_model.scheduling.policy.simple_global import SimpleGlobalPolicy
from sglang.multi_model.multi_model_server_args import MultiModelServerArgs
from sglang.multi_model.scheduling.action import BaseAction
from sglang.multi_model.scheduling.constants import (
    AIOHTTP_TIMEOUT_SECONDS,
    UPDATE_QUEUE_INTERVAL,
)
from sglang.multi_model.scheduling.gpu.request_queue import RequestQueue
from sglang.multi_model.scheduling.model_queue_tracker import ModelQueueTracker
from sglang.multi_model.scheduling.policy import *
from sglang.multi_model.scheduling.state import ModelInstanceState, ModelState
from sglang.multi_model.scheduling.stats import ScheduleStats
from sglang.srt.managers.io_struct import (
    BatchRetractDecodeReq,
    BatchRunReq,
    FinishReq,
    GenerateReqInput,
    MemoryUsage,
    UpdateModelTput,
)
from sglang.srt.utils import configure_logger, kill_parent_process, get_available_gpu_memory
from sglang.utils import get_exception_traceback

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT_SECONDS)

logger = logging.getLogger(__name__)


class GlobalController:
    """Global controller for multi-model scheduling and resource management."""
    
    def __init__(
        self,
        server_args: MultiModelServerArgs,
        recv_from_request_handler_ipc_name,
        recv_from_schedulers_ipc_name,
        engine_info_dict,
        model_names_to_model_paths,
        init_placements,
    ):
        self.server_args = server_args
        context = zmq.Context(2)
        self.recv_from_request_handler = context.socket(zmq.PULL)
        self.recv_from_request_handler.bind(
            f"ipc://{recv_from_request_handler_ipc_name}"
        )

        self.recv_from_schedulers = context.socket(zmq.PULL)
        self.recv_from_schedulers.bind(f"ipc://{recv_from_schedulers_ipc_name}")

        self._shutdown_event = threading.Event()

        self.models = model_names_to_model_paths.keys()
        self.model_names_to_model_paths = model_names_to_model_paths

        # Information that helps the policy make scheduling decisions
        self.model_queues = {
            model_name: ModelQueueTracker(model_name) for model_name in self.models
        }
        self.model_queues_lock = threading.Lock()

        self.enable_worker_pool = self.server_args.enable_worker_pool

        # Compute the number of GPUs dynamically using a set
        if not self.enable_worker_pool:
            self.model_instance_state_dict = self._init_model_instance_state_dict(
                engine_info_dict, self.enable_worker_pool, init_placements
            )
            models: List[ModelInstanceState] = sum(
                self.model_instance_state_dict.values(), []
            )
            # NOTE(ke): For TP case, only consider rank0 state
            gpu_ids = set([mod.gpu_ids[0] for mod in models])
            self.gpu_ids = list(gpu_ids)
        else:
            self.gpu_ids = list(range(self.server_args.num_gpus))
        logger.info(f"All assigned GPU ids: {self.gpu_ids}")

        current_device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        # Convert to GB and compute the maximum usable memory
        gpu_mem = total_memory / (1 << 30)
        logger.info(f"Total single GPU memory: {gpu_mem:.2f} GB")
        
        start_mem_check = time.time()
        gpu_available_memory = {}
        for gpu_id in range(len(self.gpu_ids)):
            gpu_available_memory[gpu_id] = get_available_gpu_memory("cuda", gpu_id)
            logger.info(f"GPU {gpu_id}, memory: {gpu_available_memory[gpu_id]:.2f} GB")
        end_mem_check = time.time()
        logger.info(f"Time for checking GPU memory: {end_mem_check - start_mem_check:.4f}s")

        # Load model weight configuration
        model_weight_config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../utils/model_info.json"
        )
        with open(model_weight_config, "r") as file:
            model_weights_info = json.load(file)
            logger.info("Model weight information loaded successfully")

        # Map model names to their weight information
        model_config_file = self.server_args.model_config_file
        self.model_weights_info_after_renamed = {}
        with open(model_config_file, "r") as file:
            model_config = json.load(file)
            for model_entry in model_config:
                model_name = model_entry["model_name"]
                model_path = model_entry["model_path"]
                assert (
                    model_path in model_weights_info
                ), f"Model path '{model_path}' not found in model_weights_info"
                self.model_weights_info_after_renamed[model_name] = model_weights_info[
                    model_path
                ]

        self.model_instance_state_dict = self._init_model_instance_state_dict(
            engine_info_dict, self.enable_worker_pool, init_placements
        )

        # Initialize scheduling policy
        if self.server_args.policy == "simple-global":
            self.policy = SimpleGlobalPolicy(
                num_gpus=len(self.gpu_ids),
                gpu_mem=gpu_mem,
                model_weights_info=self.model_weights_info_after_renamed,
                workers_per_gpu=self.server_args.workers_per_gpu,
            )
        else:
            raise ValueError(f"Unknown policy: {self.server_args.policy}")

        logger.info(f"Using policy: {self.policy.__class__.__name__}")

        self._start_run_policy = (
            threading.Event()
        )  # waiting for the first generate request
        # background thread to update the queue info
        self.queue_update_thread = threading.Thread(target=self.run_queue_update_loop)
        self.queue_update_thread.start()

        # Start background thread to collect statistics and make scheduling decisions
        logger.info("Starting schedule statistic thread")
        self.schedule_statistic_thread = threading.Thread(
            target=self.run_schedule_statistics_loop
        )
        self.schedule_statistic_thread.start()

    def run_queue_update_loop(self):
        """Background loop to update queue information from incoming requests."""
        while not self._shutdown_event.is_set():
            try:
                recv_reqs = self.recv_requests()
                if len(recv_reqs) > 0:
                    with self.model_queues_lock:
                        self.handle_requests(recv_reqs)
                time.sleep(UPDATE_QUEUE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in queue update loop: {get_exception_traceback()}")

    def recv_requests(self):
        """Receive requests from both request handler and schedulers."""
        recv_reqs = []
        
        # Receive from request handler
        while True:
            try:
                recv_req = self.recv_from_request_handler.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)

        # Receive from schedulers
        while True:
            try:
                recv_req = self.recv_from_schedulers.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)
        return recv_reqs

    def handle_requests(
        self,
        recv_reqs: List[
            Union[
                GenerateReqInput,
                FinishReq,
                BatchRunReq,
                BatchRetractDecodeReq,
                UpdateModelTput,
            ]
        ],
    ):
        """Process incoming requests and update model queue states."""
        model_names = set()
        for recv_req in recv_reqs:
            if isinstance(recv_req, GenerateReqInput):
                if not recv_req.is_warmup and not self._start_run_policy.is_set():
                    self._start_run_policy.set()
                try:
                    self.model_queues[recv_req.model].enqueue_req(recv_req)
                    model_names.add(recv_req.model)
                except Exception as e:
                    logger.error(f"Error enqueuing request: {recv_req}, recv_req.model: {recv_req.model}, self.model_queues.keys(): {self.model_queues.keys()}")
            elif isinstance(recv_req, BatchRunReq):
                self.model_queues[recv_req.model].start_running_reqs(recv_req)
                model_names.add(recv_req.model)
            elif isinstance(recv_req, BatchRetractDecodeReq):
                self.model_queues[recv_req.model].preempt_reqs(recv_req)
                model_names.add(recv_req.model)
            elif isinstance(recv_req, FinishReq):
                if not recv_req.is_warmup:
                    self.model_queues[recv_req.model].finish_req(recv_req)
                    model_names.add(recv_req.model)
            elif isinstance(recv_req, UpdateModelTput):
                # Update the token throughput for the model
                model_name = recv_req.model_name
                assert (
                    model_name in self.model_queues
                ), f"Model {model_name} not found in model_queues"
                self.model_queues[model_name].latest_token_tput = (
                    recv_req.latest_token_tput
                )
                self.model_queues[model_name].prefill_token_tput = (
                    recv_req.prefill_token_tput
                )
                self.model_queues[model_name].decode_token_tput = (
                    recv_req.decode_token_tput
                )
            else:
                raise ValueError(f"Unknown request type: {type(recv_req)}")

        # Log queue information for models with new requests
        for model_name in model_names:
            model_queue = self.model_queues[model_name]
            logger.info(f"{model_queue}")
    
    def _get_gpu_to_active_instances(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ) -> Dict[int, List[ModelInstanceState]]:
        """Create a helper data structure that maps GPU IDs to their active model instances."""
        gpu_to_active_instances: Dict[int, List[ModelInstanceState]] = dict()
        for gpu_id in self.gpu_ids:
            gpu_to_active_instances[gpu_id] = []

        for model_name, instances in model_instance_state_dict.items():
            model_active_instances = {}  # GPU ID -> list of active instances
            for instance in instances:
                if instance.state == ModelState.ACTIVE:
                    for gpu_id in instance.gpu_ids:
                        if gpu_id not in model_active_instances:
                            model_active_instances[gpu_id] = []
                        model_active_instances[gpu_id].append(instance)

            assert len(model_active_instances) <= 1
            
            if model_active_instances:
                gpu_id = list(model_active_instances.keys())[0]
                gpu_to_active_instances[gpu_id].extend(model_active_instances[gpu_id])

        return gpu_to_active_instances

    def run_scheduling_loop(self):
        """Main scheduling loop that generates and executes scheduling actions."""
        logger.info("Waiting for the first generate request to start model scheduling")
        self._start_run_policy.wait()
        SCHEDULE_INTERVAL = 5
        logger.info(f"Schedule interval: {SCHEDULE_INTERVAL} seconds")

        logger.info("Starting model scheduling")
        while not self._shutdown_event.is_set():
            with self.model_queues_lock:
                # Deep copy is not thread-safe, so we do it within the lock
                model_queues_cpy = copy.deepcopy(self.model_queues)

            actions = self.policy.gen_actions(
                model_queues_cpy, self.model_instance_state_dict
            )
            if len(actions) > 0:
                exec_start = time.time()
                self.execute_actions(actions)
                exec_finish = time.time()
                if (exec_finish - exec_start) < SCHEDULE_INTERVAL:
                    time.sleep(SCHEDULE_INTERVAL - (exec_finish - exec_start))
            else:
                time.sleep(5)

    def execute_actions(self, actions: List[BaseAction]):
        """Execute a list of scheduling actions in parallel."""
        tic = time.time()
        threads = ThreadPoolExecutor(max_workers=len(actions))
        futures = [
            threads.submit(
                action.execute, self.server_args.url(), self.model_instance_state_dict
            )
            for action in actions
        ]

        # Wait for all actions to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error executing action: {get_exception_traceback()}")

        threads.shutdown()
        toc = time.time()
        logger.info(
            f"Executed {len(actions)} actions in {toc - tic:.2f} seconds. Actions: {actions}"
        )
        self.print_model_instance_state_dict()

    def _init_model_instance_state_dict(
        self, engine_info_dict, enable_worker_pool, init_placements
    ):
        """Initialize model instance state dictionary based on engine information."""
        if not enable_worker_pool:
            model_instance_state_dict = defaultdict(list)
            for model_name, engine_info_list in engine_info_dict.items():
                for engine_info in engine_info_list:
                    instance_state = ModelInstanceState(
                        model_name=model_name,
                        model_path=engine_info.model_path,
                        instance_idx=engine_info.instance_idx,
                        gpu_ids=engine_info.gpu_ids,
                        memory_usage=engine_info.memory_usage,
                        init_memory_pool_size=engine_info.init_memory_pool_size,
                        state=(
                            ModelState.ACTIVE if engine_info.on else ModelState.INACTIVE
                        ),
                    )
                    model_instance_state_dict[model_name].append(instance_state)
            return model_instance_state_dict
        else:
            model_instance_state_dict = defaultdict(list)
            
            # Initialize instances based on initial placements
            for gpu_id, model_names in init_placements.items():
                for model_name in model_names:
                    model_weights_memory = self.model_weights_info_after_renamed[
                        model_name
                    ]["model_size"]
                    for idx in self.gpu_ids:
                        instance_state = ModelInstanceState(
                            model_name=model_name,
                            model_path=self.model_names_to_model_paths[model_name],
                            instance_idx=idx,
                            gpu_ids=[idx],
                            memory_usage=MemoryUsage(
                                total_used_memory=model_weights_memory,
                                model_weights_memory=model_weights_memory,
                                memory_pool_memory=0,
                                req_to_token_pool_memory=0,
                                token_to_kv_pool_memory=0,
                            ),
                            init_memory_pool_size=0,
                            state=ModelState.ACTIVE if idx == gpu_id else ModelState.INACTIVE,
                        )
                        model_instance_state_dict[model_name].append(instance_state)

            # Add inactive models not in initial placements
            for model_name in self.models:
                if model_name not in model_instance_state_dict:
                    model_weights_memory = self.model_weights_info_after_renamed[
                        model_name
                    ]["model_size"]
                    for gpu_id in self.gpu_ids:
                        instance_state = ModelInstanceState(
                            model_name=model_name,
                            model_path=self.model_names_to_model_paths[model_name],
                            instance_idx=gpu_id,
                            gpu_ids=[gpu_id],
                            memory_usage=MemoryUsage(
                                total_used_memory=0,
                                model_weights_memory=model_weights_memory,
                                memory_pool_memory=0,
                                req_to_token_pool_memory=0,
                                token_to_kv_pool_memory=0,
                            ),
                            init_memory_pool_size=0,
                            state=ModelState.INACTIVE,
                        )
                        model_instance_state_dict[model_name].append(instance_state)

            return model_instance_state_dict

    def run_schedule_statistics_loop(self):
        """Background loop to collect and log scheduling statistics."""
        current_time = datetime.now().isoformat()
        output_file = os.path.join(os.getcwd(), "benchmark", "multi-model", f"stats_{current_time}.log")
        if os.path.exists(output_file):
            os.remove(output_file)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        logger.info(f"Schedule statistics will be logged to {output_file}")

        # Load model weight information for statistics
        logger.info("Loading model weight information for statistics")
        model_weight_config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../utils/model_info.json"
        )
        with open(model_weight_config, "r") as file:
            model_weights_info = json.load(file)
            logger.info(f"Loaded model weights from {model_weight_config}")
        
        # Map model weights using configuration
        logger.info("Loading model config and mapping weights")
        model_config_file = self.server_args.model_config_file
        model_weights_info_after_renamed = {}
        with open(model_config_file, "r") as file:
            model_config = json.load(file)
            for model_entry in model_config:
                model_name = model_entry["model_name"]
                model_path = model_entry["model_path"]
                model_weights_info_after_renamed[model_name] = model_weights_info[model_path]
            logger.info(f"Mapped {len(model_weights_info_after_renamed)} models to their weights")

        # Set GPU total memory (hardcoded for now)
        gpu_total_memory = {}
        for gpu_id in self.gpu_ids:
            gpu_total_memory[gpu_id] = 80  # 80GB
        logger.info(f"Set GPU total memory: {gpu_total_memory}")

        while not self._shutdown_event.is_set():
            mod_on_gpu: Dict[str, List[int]] = dict()
            mod_priority: Dict[str, float] = dict()
            mod_q_waiting_len: Dict[str, int] = dict()
            mod_req_per_sec: Dict[str, int] = dict()
            mod_running_req_len: Dict[str, int] = dict()
            mod_total_req_len: Dict[str, int] = dict()

            current_time: datetime = datetime.now().isoformat()

            # GPU-level tracking
            gpu_active_models: Dict[int, List[str]] = {gpu_id: [] for gpu_id in self.gpu_ids}
            gpu_requests: Dict[int, int] = {gpu_id: 0 for gpu_id in self.gpu_ids}
            gpu_active_weights: Dict[int, float] = {gpu_id: 0.0 for gpu_id in self.gpu_ids}
            gpu_memory_per_request: Dict[int, float] = {gpu_id: 0.0 for gpu_id in self.gpu_ids}

            models: List[ModelInstanceState] = sum(
                self.model_instance_state_dict.values(), []
            )
            
            for mod in models:
                m_name = mod.model_name
                gpu_id = mod.gpu_ids[0]
                
                if mod.state == ModelState.ACTIVE:
                    mod_on_gpu[m_name] = gpu_id
                    if m_name not in gpu_active_models[gpu_id]:
                        gpu_active_models[gpu_id].append(m_name)
                    
                    gpu_active_weights[gpu_id] += model_weights_info_after_renamed[m_name]["model_size"]
                    logger.debug(f"Model {m_name} is active on GPU {gpu_id}, weight added: {model_weights_info_after_renamed[m_name]['model_size']}")
                else:
                    mod_on_gpu[m_name] = [-1]

                # Collect priority and queue statistics for each model
                mod_priority[m_name] = mod.priority
                mod_q_waiting_len[m_name] = len(self.model_queues[m_name].waiting_reqs)
                mod_running_req_len[m_name] = len(self.model_queues[m_name].running_reqs)
                mod_total_req_len[m_name] = mod_running_req_len[m_name] + mod_q_waiting_len[m_name]

                # Calculate GPU request distribution
                if isinstance(mod_on_gpu[m_name], list): 
                    gpu_id = mod_on_gpu[m_name][0]
                else:
                    gpu_id = mod_on_gpu[m_name]

                if gpu_id >= 0:
                    gpu_requests[gpu_id] += mod_total_req_len[m_name]

            # Calculate memory per request for each GPU
            for gpu_id in self.gpu_ids:
                if gpu_active_weights[gpu_id] > 0 and gpu_requests[gpu_id] > 0:
                    # Available memory = Total memory - Active model weights
                    available_memory = gpu_total_memory[gpu_id] - gpu_active_weights[gpu_id]
                    if available_memory > 0:
                        gpu_memory_per_request[gpu_id] = available_memory / gpu_requests[gpu_id]
                    else:
                        gpu_memory_per_request[gpu_id] = 0.0
                    logger.debug(f"GPU {gpu_id}: active weight={gpu_active_weights[gpu_id]}, requests={gpu_requests[gpu_id]}, memory per request={gpu_memory_per_request[gpu_id]}")
                else:
                    gpu_memory_per_request[gpu_id] = gpu_total_memory[gpu_id] - gpu_active_weights[gpu_id]

            # Create and log statistics
            stats = ScheduleStats(
                time=current_time,
                mod_req_per_sec=mod_req_per_sec,
                mod_on_gpu=mod_on_gpu,
                mod_priority=mod_priority,
                mod_q_waiting_len=mod_q_waiting_len,
                mod_total_req_len=mod_total_req_len,
                gpu_metrics={
                    "gpu_active_models": {str(k): v for k, v in gpu_active_models.items()},
                    "gpu_active_weights": {str(k): v for k, v in gpu_active_weights.items()},
                    "gpu_requests": {str(k): v for k, v in gpu_requests.items()},
                    "gpu_memory_per_request": {str(k): v for k, v in gpu_memory_per_request.items()},
                    "gpu_total_memory": {str(k): v for k, v in gpu_total_memory.items()}
                }                   
            )
            
            # Write statistics to file
            data = []
            if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
                with open(output_file, "r") as f:
                    data = json.load(f)
            data.append(asdict(stats))
            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)

            time.sleep(1)

    def print_model_instance_state_dict(self):
        """Print the current state of all model instances."""
        for model_name, instance_states in self.model_instance_state_dict.items():
            model_path = instance_states[0].model_path
            instance_states_str = [str(instance_state) for instance_state in instance_states]
            logger.info(f"{model_name} ({model_path}): {', '.join(instance_states_str)}")

    def shutdown(self):
        """Shutdown the controller and all background threads."""
        self._shutdown_event.set()
        self.queue_update_thread.join()
        self.schedule_statistic_thread.join()
        self.recv_from_request_handler.close()


def run_controller_process(
    server_args: MultiModelServerArgs,
    recv_from_request_handler_ipc_name,
    recv_from_schedulers_ipc_name,
    engine_info_dict,
    model_names_to_model_paths,
    init_placements,
):
    """Run the global controller process."""
    controller = None
    configure_logger(
        server_args, prefix=" GlobalController", log_file_suffix="global_controller"
    )

    try:
        controller = GlobalController(
            server_args,
            recv_from_request_handler_ipc_name,
            recv_from_schedulers_ipc_name,
            engine_info_dict,
            model_names_to_model_paths,
            init_placements,
        )

        def signal_handler(signum, frame):
            logger.info("Received signal to shutdown")
            if controller:
                controller.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        controller.run_scheduling_loop()

    except Exception as e:
        msg = get_exception_traceback()
        logger.error(msg)
        if controller:
            controller.shutdown()
        kill_parent_process()
