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
    GenerateReqInput
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
        model_names_to_model_paths: Dict[str, str],
        gpu_id: int,
        init_model_names: List[str],
    ):
        self.gpu_id = gpu_id
        self.server_args = multi_model_server_args
        # 初始化模型信息
        self._model_states = {} # 模型启动状态
        self._init_model_states(engine_info_dict)
        logger.info(f"Model states init: {self._model_states}")
        self._model_name_to_paths = model_names_to_model_paths
        model_name_to_cell_size = self._get_model_name_to_cell_size()
        self.queue = RequestQueue(model_name_to_cell_size) # 全部模型请求队列
        # 初始化Redis服务器
        self.redis_client = RedisClient(
            multi_model_server_args.redis_host,
            multi_model_server_args.redis_port,
            multi_model_server_args.redis_db,
        )
        self._shutdown_event = Event()
        self._receiver_thread = None
        # 创建ZMQ上下文
        num_io_threads = (
            1
            if not self.server_args.enable_worker_pool
            else 1 + self.server_args.workers_per_gpu
        )
        self.context = zmq.Context(io_threads=num_io_threads)
        self.recv_from_request_handler_ipc_name = f"request_handler_to_gpu_scheduler_{gpu_id}"
        self.recv_from_request_handler = self.context.socket(zmq.PULL)
        self.recv_from_request_handler.bind(f"ipc://{self.recv_from_request_handler_ipc_name}")
        self.recv_from_scheduler_ipc_name = f"scheduler_to_gpu_scheduler_{gpu_id}"
        self.recv_from_scheduler = self.context.socket(zmq.PULL)
        self.recv_from_scheduler.bind(f"ipc://{self.recv_from_scheduler_ipc_name}")
        # 创建资源管理器        
        self.resource_manager = ResourceManager(
            gpu_id,
            multi_model_server_args.mem_fraction_static,
            self._get_active_or_activating_model_names(),
            engine_info_dict,
            multi_model_server_args.enable_worker_pool,
            model_names_to_model_paths,
            num_workers=multi_model_server_args.workers_per_gpu,
        )
        # 初始化worker pool
        self._init_worker_pool()
        if self.server_args.enable_worker_pool:
            self.worker_pool = WorkerPool(self.server_args.workers_per_gpu, self.gpu_id, self.context)
            self._init_models(init_model_names)
        else: self.worker_pool = None

    def _init_model_states(self, engine_info_dict: Dict[str, List]):
        """初始化模型启动状态
        TP时仅在rank0 GPU维护
        多实例时有实例活跃则记为激活
        """
        if not self.server_args.enable_worker_pool:
            for model_name, engine_info_list in engine_info_dict.items():
                self._model_states[model_name] = "deactivated"
                for engine_info in engine_info_list:
                    engine_gpu_id = engine_info.gpu_ids[0]
                    if engine_gpu_id == self.gpu_id and engine_info.on:
                        self._model_states[model_name] = "activating"
                        break
        else:
            for model_name in self._model_name_to_paths.keys():
                self._model_states[model_name] = "deactivated"

    def _get_model_name_to_cell_size(self) -> Dict[str, int]:
        """获取model name -> cell size字典"""
        model_paths = list(set(self._model_name_to_paths.values()))
        cell_sizes = get_model_path_to_cell_size(model_paths)
        return {model_name: cell_sizes[model_path]for model_name, model_path in self._model_name_to_paths.items()}

    def _get_active_or_activating_model_names(self):
        """获取需求资源（activating, activated, deactivating）状态模型"""
        ret = []
        for m, s in self._model_states.items():
            if s in ("activating", "activated", "deactivating"):
                ret.append(m)
        return ret

    def _init_worker_pool(self):
        """创建worker pool"""
        if self.server_args.enable_worker_pool:
            self.worker_pool = WorkerPool(self.server_args.workers_per_gpu, self.gpu_id, self.context)
        else:
            self.worker_pool = None

    def _init_models(self, init_model_names: List[str]):
        """对worker pool有效，初始化指定模型"""
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
            if all(self._model_states[model_name] == "activated" for model_name in init_model_names):
                break
            else:
                logger.info(f"Waiting for models to be activated: {self._model_states}")
                time.sleep(0.01)
                
    def _set_model_state(self, model_name: str, new_state: str):
        """更新模型状态及资源管理器"""
        old_state = self._model_states.get(model_name, "deactivated")
        self._model_states[model_name] = new_state
        states_need_resource = ("activating", "activated", "deactivating")
        old_need = old_state in states_need_resource
        new_need = new_state in states_need_resource
        if (not old_need) and new_need: # 非活->活跃：添加资源跟踪
            self.resource_manager.add_active_model(model_name)
        elif old_need and (not new_need): # 活跃->非活：移除资源跟踪
            self.resource_manager.remove_active_model(model_name)
        if new_state == "activated": # 清空待分配显存记录
            self.queue.clear_activating_usage(model_name)

    def _recv_from_engine(self):
        """从Engine接收activate/deactivate结果，更新模型状态"""
        messages = self.redis_client.pop_all(
            key=f"{self.server_args.engine_to_gpu_scheduler_key_prefix}:{self.gpu_id}",
        )
        for message in messages:
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
            else:
                logger.info(f"Received unsupported message type: {type(message)}")
        return messages

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
                available_kv_cache_memory = self.resource_manager.get_available_kv_cache_memory()
                # 获取后端运行请求队列长度
                active_models = self._get_active_or_activating_model_names()
                model_backend_queue_lens = {
                    model_name: self.redis_client.get_queue_length(
                        f"{self.server_args.backend_generate_request_key_prefix}:{model_name}"
                    ) for model_name in active_models
                }
                # 准入控制：根据优先级
                reqs_can_be_admitted = self.queue.admission_control(
                    model_states=self._model_states,
                    available_resources=available_kv_cache_memory,
                    model_backend_queue_lens=model_backend_queue_lens,
                    allow_sending_when_activating=True,
                )
                for model_name, reqs in reqs_can_be_admitted.items():
                    logger.info(f"✔ Admitted {len(reqs)} requests for {model_name}")
                self._send_to_backend_queue(reqs_can_be_admitted)
                time.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in scheduling loop: {get_exception_traceback()}")
            self.shutdown()

    def _recv_requests_loop(self):
        """处理模型生成/启动/关闭请求"""
        while not self._shutdown_event.is_set():
            try:
                reqs = self._recv_from_frontend_queue() # 接收生成请求
                self.queue.add_requests(reqs) # 加入等待队列
                self._recv_from_request_handler() # 接收开关请求
                self._recv_from_engine() # 接收开关结果
            except Exception as e:
                if self._shutdown_event.is_set(): break
                logger.error(f"Receiver error: {get_exception_traceback()}")
                raise e

    def _recv_from_frontend_queue(self):
        """从前端获取请求队列
        仅对活跃请求获取
        """
        recv_reqs = []
        models_can_recv = [
            m for m, state in self._model_states.items()
            if state in ("activating", "activated") # 可接收请求的状态
        ]
        for model_name in models_can_recv:
            reqs = self.redis_client.pop_all(
                key=f"{self.server_args.frontend_generate_request_key_prefix}:{model_name}"
            )
            recv_reqs.extend(reqs)
        return recv_reqs

    def _recv_from_request_handler(self):
        """从request handler接收并处理模型开关请求"""
        try:
            messages = []
            while True:
                try:
                    message = self.recv_from_request_handler.recv_pyobj(zmq.NOBLOCK)
                    messages.append(message)
                except zmq.ZMQError:
                    break
            for message in messages:
                if isinstance(message, ActivateReqInput): # 启动模型
                    logger.info(f"Received activate request for model {message.model_name}")
                    self._set_model_state(message.model_name, "activating")
                    logger.info("Received activate request for {message.model_name}, model_states: {self._model_states}")
                    if self.server_args.enable_worker_pool:
                        result = self.worker_pool.handle_activate_model(message)
                        if not result:
                            logger.info(f"Failed to activate model {message.model_name}")
                            self._set_model_state(message.model_name, "deactivated")
                elif isinstance(message, DeactivateReqInput): # 关闭模型
                    model_name = message.model_name
                    logger.info(f"Received deactivate request for model {model_name}")
                    self._set_model_state(model_name, "deactivating")
                    logger.info(
                        f"Received deactivate request for {model_name}, "
                        f"model_states: {self._model_states}"
                    )
                    if self.server_args.enable_worker_pool:
                        self.worker_pool.handle_deactivate_model(message)
                    # 从队列弹出待关闭模型全部等待请求，发送回前端
                    waiting_reqs = self.queue.pop_model_requests(model_name)
                    if waiting_reqs:
                        self._send_waiting_reqs_to_frontend_queue(model_name, waiting_reqs)
                    # 从后端撤回待关闭模型全部运行请求，发送回前端
                    backend_reqs = self.redis_client.pop_all(key=f"{self.server_args.backend_generate_request_key_prefix}:{model_name}")
                    if backend_reqs:
                        for req in backend_reqs:
                            self.redis_client.send_pyobj(
                                key=f"{self.server_args.frontend_generate_request_key_prefix}:{model_name}",
                                obj=req,
                            )
            return messages
        except Exception as e:
            logger.error(f"Error receiving messages from request handler: {get_exception_traceback()}")
            return []

    def _send_waiting_reqs_to_frontend_queue(self, model_name, reqs):
        """将等待请求发送回前端"""
        try:
            logger.info(f"Sending {len(reqs)} queued requests of model {model_name} back to frontend queue")
            for req in reqs:
                self.redis_client.send_pyobj(key=f"{self.server_args.frontend_generate_request_key_prefix}:{model_name}", obj=req)
        except Exception as e:
            logger.error(f"Error sending preempted requests: {get_exception_traceback()}")

    def _send_to_backend_queue(self, reqs: Dict[str, List[GenerateReqInput]]):
        """向后端发送生成请求"""
        for model_name, reqs in reqs.items():
            for req in reqs:
                self.redis_client.send_pyobj(key=f"{self.server_args.backend_generate_request_key_prefix}:{model_name}", obj=req)

    def shutdown(self):
        """关停gpu scheduler"""
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

    def cleanup(self):
        """Clean up all IPC sockets and files."""
        try:
            # Prepare sockets dictionary
            zmq_sockets = {}
            if hasattr(self, "recv_from_request_handler"):
                zmq_sockets["recv_from_request_handler"] = self.recv_from_request_handler
            if hasattr(self, "recv_from_scheduler"):
                zmq_sockets["recv_from_scheduler"] = self.recv_from_scheduler
            # Get IPC file paths
            ipc_files = set()
            if hasattr(self, "recv_from_request_handler_ipc_name"):
                ipc_files.add(f"ipc://{self.recv_from_request_handler_ipc_name}")
            if hasattr(self, "recv_from_scheduler_ipc_name"):
                ipc_files.add(f"ipc://{self.recv_from_scheduler_ipc_name}")
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
            
    def __del__(self):
        self.cleanup()


def run_gpu_scheduler_process(
    multi_model_server_args: MultiModelServerArgs,
    engine_info_dict,
    model_names_to_model_paths,
    gpu_id,
    init_model_names: List[str] = None,
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
