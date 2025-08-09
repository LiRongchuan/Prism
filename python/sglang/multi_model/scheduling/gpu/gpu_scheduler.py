import atexit
import logging
import os
import signal
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Dict, List, Optional, Union

import multiprocessing as mp

import zmq

from sglang.multi_model.multi_model_server_args import MultiModelServerArgs
from sglang.multi_model.scheduling.gpu.request_queue import RequestQueue
from sglang.multi_model.scheduling.gpu.resource_manager import ResourceManager
from sglang.multi_model.scheduling.gpu.worker_pool import WorkerPool
from sglang.multi_model.utils.get_memory_pool_size import get_model_path_to_cell_size
from sglang.srt.managers.io_struct import (
    ActivateReqInput,
    ActivateReqOutput,
    DeactivateReqInput,
    DeactivateReqOutput,
    GenerateReqInput,
)
from sglang.srt.redis_utils import RedisClient
from sglang.srt.utils import configure_logger, kill_parent_process
from sglang.utils import cleanup_zmq_ipc, get_exception_traceback

logger = logging.getLogger(__name__)


class GPUScheduler:
    """
    GPU scheduler responsible for managing model scheduling and request processing on a single GPU.
    Main functions include:
    - Receiving and processing activation/deactivation requests
    - Managing request queues and priority scheduling
    - Performing admission control and resource management
    """

    def __init__(
        self,
        multi_model_server_args: MultiModelServerArgs,
        engine_info_dict,
        model_names_to_model_paths,
        gpu_id,
        init_model_names: List[str],
    ):
        self._model_states = {}
        self.redis_client = RedisClient(
            multi_model_server_args.redis_host,
            multi_model_server_args.redis_port,
            multi_model_server_args.redis_db,
        )

        self.server_args = multi_model_server_args
        self._init_model_name_to_paths(model_names_to_model_paths)
        model_name_to_cell_size = self._get_model_name_to_cell_size()
        self.queue = RequestQueue(model_name_to_cell_size)

        self.gpu_id = gpu_id

        # Create ZMQ context
        num_io_threads = (
            1
            if not self.server_args.enable_worker_pool
            else 1 + self.server_args.workers_per_gpu
        )
        self.context = zmq.Context(io_threads=num_io_threads)
        self.recv_from_request_handler_ipc_name = (
            f"request_handler_to_gpu_scheduler_{gpu_id}"
        )
        self.recv_from_request_handler = self.context.socket(zmq.PULL)
        self.recv_from_request_handler.bind(
            f"ipc://{self.recv_from_request_handler_ipc_name}"
        )

        self._init_model_states(engine_info_dict)
        logger.info(f"Model states init: {self._model_states}")

        self._maybe_init_worker_pool()
        self.resource_manager = ResourceManager(
            gpu_id,
            multi_model_server_args.mem_fraction_static,
            self._get_active_or_activating_model_names(),
            engine_info_dict,
            multi_model_server_args.enable_worker_pool,
            model_names_to_model_paths,
            num_workers=multi_model_server_args.workers_per_gpu,
        )
        self._shutdown_event = Event()
        self._receiver_thread = None
        if self.server_args.enable_worker_pool:
            self._init_models(init_model_names)

    def _init_models(self, init_model_names: List[str]):
        """Initialize specified models."""
        for model_name in init_model_names:
            self._set_model_state(model_name, "activating")
            activate_req = ActivateReqInput(
                model_name=model_name,
                instance_idx=self.gpu_id,
                gpu_id=self.gpu_id,
            )
            self.worker_pool.handle_activate_model(activate_req)

        while True:
            self._recv_from_engine()
            if all(
                self._model_states[model_name] == "activated"
                for model_name in init_model_names
            ):
                break
            else:
                logger.info(f"Waiting for models to be activated: {self._model_states}")
                time.sleep(0.01)

    def _maybe_init_worker_pool(self):
        """Initialize worker pool if enabled."""
        if self.server_args.enable_worker_pool:
            self.worker_pool = WorkerPool(
                self.server_args.workers_per_gpu, self.gpu_id, self.context
            )
        else:
            self.worker_pool = None

    def run_scheduling_loop(self):
        """
        Run scheduling loop with main steps:
        1. Receive activate/deactivate requests from the controller
           1.1 If deactivate request, pop requests from frontend queue to backend queue, adjust model status dict
        2. Pull requests from active models from frontend queue, calculate request priority, add to priority queue
        3. Pop requests from priority queue, estimate memory usage and current available memory, send to backend queue
        """
        self._receiver_thread = Thread(target=self._recv_requests_loop, daemon=True)
        self._receiver_thread.start()

        try:
            while not self._shutdown_event.is_set():
                available_kv_cache_memory = (
                    self.resource_manager.get_available_kv_cache_memory()
                )

                # Get backend queue lengths for each active model
                active_models = self._get_active_or_activating_model_names()
                model_backend_queue_lens = {
                    model_name: self.redis_client.get_queue_length(
                        f"{self.server_args.backend_generate_request_key_prefix}:{model_name}"
                    )
                    for model_name in active_models
                }

                # Pop requests from priority queue with admission control
                reqs_can_be_admitted = self.queue.admission_control(
                    available_resources=available_kv_cache_memory,
                    model_backend_queue_lens=model_backend_queue_lens,
                    model_states=self._model_states,
                    allow_sending_when_activating=True,
                )

                for model_name, reqs in reqs_can_be_admitted.items():
                    logger.info(f"Admitted {len(reqs)} requests for {model_name}")
                self._send_to_backend_queue(reqs_can_be_admitted)

                time.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in scheduling loop: {get_exception_traceback()}")
            self.shutdown()

    def _get_active_or_activating_model_names(self):
        """Get list of active or activating model names."""
        ret = []
        for m, s in self._model_states.items():
            if s in ("activating", "activated", "deactivating"):
                ret.append(m)
        return ret

    def _recv_requests_loop(self):
        """Background thread that receives requests and adds them to priority queue."""
        while not self._shutdown_event.is_set():
            try:
                reqs = self._recv_from_frontend_queue()
                self.queue.add_requests(reqs)
                self._recv_from_request_handler()
                self._recv_from_engine()
            except Exception as e:
                if self._shutdown_event.is_set():
                    break
                logger.error(f"Receiver error: {get_exception_traceback()}")
                raise e

    def _recv_from_frontend_queue(self):
        """Receive requests from frontend queue."""
        recv_reqs = []
        models_can_recv = [
            m
            for m, state in self._model_states.items()
            if state in ("activating", "activated")
        ]

        for model_name in models_can_recv:
            reqs = self.redis_client.pop_all(
                key=f"{self.server_args.frontend_generate_request_key_prefix}:{model_name}",
            )
            recv_reqs.extend(reqs)
        return recv_reqs

    def _recv_from_engine(self):
        """Receive activation/deactivation result messages from engine."""
        messages = self.redis_client.pop_all(
            key=f"{self.server_args.engine_to_gpu_scheduler_key_prefix}:{self.gpu_id}",
        )
        for message in messages:
            # Process activate, deactivate and resize memory pool request outputs
            if isinstance(message, ActivateReqOutput):
                model_name = message.model_name
                instance_idx = message.instance_idx
                success = message.success
                logger.info(
                    f"Model {model_name} (instance {instance_idx}) activation "
                    f"{'succeeded' if success else 'failed'} on GPU {self.gpu_id}"
                )
                if success:
                    self._set_model_state(model_name, "activated")
                # Can record additional activation completion status here, such as updating model state dictionaries

            elif isinstance(message, DeactivateReqOutput):
                model_name = message.model_name
                instance_idx = message.instance_idx
                success = message.success
                logger.info(
                    f"Model {model_name} (instance {instance_idx}) deactivation "
                    f"{'succeeded' if success else 'failed'} on GPU {self.gpu_id}"
                )
                if success:
                    self._set_model_state(model_name, "deactivated")
                # Can record additional deactivation completion status here, such as cleaning up resources
            else:
                logger.info(f"Received unsupported message type: {type(message)}")

        return messages

    def _recv_from_request_handler(self):
        """Receive activate/deactivate requests from request handler."""
        try:
            messages = []
            while True:
                try:
                    message = self.recv_from_request_handler.recv_pyobj(zmq.NOBLOCK)
                    messages.append(message)
                except zmq.ZMQError:
                    break

            for message in messages:
                # Process activate, deactivate and resize memory pool requests
                if isinstance(message, DeactivateReqInput):
                    model_name = message.model_name
                    logger.info(f"Received deactivate request for model {model_name}")
                    self._set_model_state(model_name, "deactivating")
                    logger.info(
                        f"Received deactivate request for {model_name}, "
                        f"model_states: {self._model_states}"
                    )
                    if self.server_args.enable_worker_pool:
                        self.worker_pool.handle_deactivate_model(message)
                    # Get all waiting requests in the queue for this model and send them back to controller
                    waiting_reqs = self.queue.pop_model_requests(model_name)
                    if waiting_reqs:
                        self._send_waiting_reqs_to_frontend_queue(
                            model_name, waiting_reqs
                        )
                    # Pop all backend queue to frontend queue
                    backend_reqs = self.redis_client.pop_all(
                        key=f"{self.server_args.backend_generate_request_key_prefix}:{model_name}"
                    )
                    if backend_reqs:
                        for req in backend_reqs:
                            self.redis_client.send_pyobj(
                                key=f"{self.server_args.frontend_generate_request_key_prefix}:{model_name}",
                                obj=req,
                            )
                    
                elif isinstance(message, ActivateReqInput):
                    logger.info(
                        f"Received activate request for model {message.model_name}"
                    )
                    # No special handling needed here, only take action when tokenizer sends activation result
                    self._set_model_state(message.model_name, "activating")
                    logger.info(
                        f"Received activate request for {message.model_name}, "
                        f"model_states: {self._model_states}"
                    )
                    if self.server_args.enable_worker_pool:
                        result = self.worker_pool.handle_activate_model(message)
                        if not result:
                            logger.info(f"Failed to activate model {message.model_name}")
                            self._set_model_state(message.model_name, "deactivated")
            return messages
        except Exception as e:
            logger.error(
                f"Error receiving messages from request handler: {get_exception_traceback()}"
            )
            return []

    def _set_model_state(self, model_name: str, new_state: str):
        """Set model state and update resource manager."""
        old_state = self._model_states.get(model_name, "deactivated")
        self._model_states[model_name] = new_state

        states_need_resource = ("activating", "activated", "deactivating")
        old_need = old_state in states_need_resource
        new_need = new_state in states_need_resource

        if (not old_need) and new_need:
            self.resource_manager.add_active_model(model_name)
        elif old_need and (not new_need):
            self.resource_manager.remove_active_model(model_name)
        if new_state == "activated":
            self.queue.clear_activating_usage(model_name)

    def _send_waiting_reqs_to_frontend_queue(self, model_name, reqs):
        """Send waiting requests back to the frontend queue."""
        try:
            logger.info(
                f"Sending {len(reqs)} queued requests of model {model_name} "
                f"back to frontend queue"
            )

            for req in reqs:
                # Consider adding waiting request information to the request object
                self.redis_client.send_pyobj(
                    key=f"{self.server_args.frontend_generate_request_key_prefix}:{model_name}",
                    obj=req,
                )
        except Exception as e:
            logger.error(
                f"Error sending preempted requests: {get_exception_traceback()}"
            )

    def _send_to_backend_queue(self, reqs: Dict[str, List[GenerateReqInput]]):
        """Send requests to backend queue."""
        for model_name, reqs in reqs.items():
            for req in reqs:
                self.redis_client.send_pyobj(
                    key=f"{self.server_args.backend_generate_request_key_prefix}:{model_name}",
                    obj=req,
                )

    def _init_model_states(self, engine_info_dict: Dict[str, List]):
        """
        Scan engine_info_dict and mark models with on=True as activated.
        Other situations can be handled as needed. For example, on=False can be regarded as deactivated.
        """
        if not self.server_args.enable_worker_pool:
            for model_name, engine_info_list in engine_info_dict.items():
                self._model_states[model_name] = "deactivated"
                for engine_info in engine_info_list:
                    # NOTE(ke): For TP case, use rank0 gpu_id as engine_gpu_id
                    engine_gpu_id = engine_info.gpu_ids[0]
                    if engine_gpu_id == self.gpu_id and engine_info.on:
                        # Assume default instance is 0
                        self._model_states[model_name] = "activating"
                        break
        else:
            for model_name in self._model_name_to_paths.keys():
                self._model_states[model_name] = "deactivated"

    def _init_model_name_to_paths(self, model_names_to_model_paths: Dict[str, str]):
        """Initialize model name to paths mapping."""
        self._model_name_to_paths = model_names_to_model_paths

    def _get_model_name_to_cell_size(self):
        """Get mapping from model name to cell size."""
        model_paths = list(set(self._model_name_to_paths.values()))
        cell_sizes = get_model_path_to_cell_size(model_paths)
        return {
            model_name: cell_sizes[model_path]
            for model_name, model_path in self._model_name_to_paths.items()
        }

    def shutdown(self):
        """Shutdown the scheduler."""
        self._shutdown_event.set()
        try:
            if hasattr(self, "redis_client") and self.redis_client is not None:
                try:
                    self.redis_client.close()
                except Exception as e:
                    logger.warning(f"Error closing Redis client during shutdown: {e}")
        except Exception as e:
            logger.warning(f"Error during redis_client shutdown: {e}")

        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=2)
        self.cleanup()

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """Clean up all IPC sockets and files."""
        try:
            # Prepare sockets dictionary
            zmq_sockets = {}

            if hasattr(self, "recv_from_request_handler"):
                zmq_sockets["recv_from_request_handler"] = (
                    self.recv_from_request_handler
                )

            # Get IPC file paths
            ipc_files = set()
            if hasattr(self, "recv_from_request_handler_ipc_name"):
                ipc_files.add(f"ipc://{self.recv_from_request_handler_ipc_name}")

            # Clean up using utility function
            cleanup_zmq_ipc(
                zmq_sockets=zmq_sockets,
                ipc_files=ipc_files,
                component_name="GPUScheduler",
                gpu_id=self.gpu_id,
            )

            # Cleanup worker pool if it exists
            if hasattr(self, "worker_pool") and self.worker_pool is not None:
                try:
                    self.worker_pool.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up worker pool: {e}")

            # Close ZMQ context
            if hasattr(self, "context"):
                try:
                    self.context.term()
                except Exception as e:
                    logger.warning(f"Error terminating ZMQ context: {e}")

        except Exception as e:
            logger.error(f"Error during cleanup for GPU {self.gpu_id}: {e}")


def run_gpu_scheduler_process(
    multi_model_server_args: MultiModelServerArgs,
    engine_info_dict,
    model_names_to_model_paths,
    gpu_id,
    init_model_names: List[str],
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
):
    """Run GPU scheduler process."""
    gpu_scheduler = None
    configure_logger(
        multi_model_server_args,
        prefix=f" GPU_Scheduler_{gpu_id}",
        log_file_suffix="gpu_scheduler",
    )
    logger.info(f"starting GPU scheduler for GPU {gpu_id}")

    try:
        gpu_scheduler = GPUScheduler(
            multi_model_server_args,
            engine_info_dict,
            model_names_to_model_paths,
            gpu_id,
            init_model_names,
        )
        pipe_finish_writer.send("ready")

        def signal_handler(signum, frame):
            logger.info("Received signal to shutdown")
            if gpu_scheduler:
                gpu_scheduler.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        gpu_scheduler.run_scheduling_loop()

    except Exception as e:
        msg = get_exception_traceback()
        logger.error(msg)
        if gpu_scheduler:
            gpu_scheduler.shutdown()
        kill_parent_process()
