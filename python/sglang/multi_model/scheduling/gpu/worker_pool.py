import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import zmq

from sglang.srt.managers.io_struct import (
    ActivateReqInput,
    DeactivateReqInput,
    DeactivateReqOutput,
)
from sglang.utils import cleanup_zmq_ipc

logger = logging.getLogger(__name__)


@dataclass
class Worker:
    """Worker configuration class."""
    worker_id: int
    gpu_id: int
    model_name: Optional[str] = None

    def __post_init__(self):
        self.ipc_name = f"gpu_scheduler_{self.gpu_id}_to_worker_{self.worker_id}"


class WorkerPool:
    """
    Worker pool manager for a single GPU.
    Manages worker allocation, release, and model activation/deactivation operations.
    """

    def __init__(self, num_workers: int, gpu_id: int, zmq_context: zmq.Context):
        """
        Initialize worker pool.
        
        Args:
            num_workers: Total number of workers in the pool
            gpu_id: GPU device ID
            zmq_context: ZMQ context
        """
        self.num_workers = num_workers
        self.gpu_id = gpu_id
        self._model_to_worker: Dict[str, int] = {}
        self._free_workers: Deque[int] = deque()
        self._worker_to_ipc_name: Dict[int, zmq.Socket] = {}
        self.zmq_context = zmq_context
        self._init_workers()

    def _init_workers(self):
        """Initialize all workers."""
        for i in range(self.num_workers):
            worker = Worker(i, self.gpu_id)
            self._free_workers.append(worker.worker_id)
            self._worker_to_ipc_name[i] = self.zmq_context.socket(zmq.PUSH)
            self._worker_to_ipc_name[i].connect(f"ipc://{worker.ipc_name}")

    def get_idle_worker(self):
        """
        Get an idle worker.
        
        Returns:
            worker_id: Idle worker ID, or None if no idle worker available
        """
        if len(self._free_workers) == 0:
            return None
        return self._free_workers[0]

    def assign_worker(self, worker_id: int, model_name: str):
        """
        Assign a worker to a model.
        
        Args:
            worker_id: Worker ID
            model_name: Model name
        """
        assert worker_id in self._free_workers, f"Worker {worker_id} is not free"
        self._model_to_worker[model_name] = worker_id
        self._free_workers.remove(worker_id)

    def release_worker(self, model_name: str):
        """
        Release a worker from a model.
        
        Args:
            model_name: Model name
            
        Returns:
            worker_id: Released worker ID
        """
        assert (
            model_name in self._model_to_worker
        ), f"Model {model_name} is not served by any worker"
        worker_id = self._model_to_worker.pop(model_name)
        self._free_workers.append(worker_id)
        return worker_id

    def handle_activate_model(self, req: ActivateReqInput):
        """
        Handle model activation request.
        
        Args:
            req: Activation request input
            
        Returns:
            bool: Whether activation succeeded
        """
        model_name = req.model_name

        if req.gpu_id is not None:
            assert (
                req.gpu_id == self.gpu_id
            ), f"GPU ID mismatch: {req.gpu_id} != {self.gpu_id}"

        # Check if the model is already served by a worker
        if model_name in self._model_to_worker:
            logger.info(f"Model {model_name} is already served by a worker")
            return False

        worker_id = self.get_idle_worker()
        if worker_id is None:
            logger.error(f"No idle worker found for GPU {self.gpu_id}")
            return False
            
        self.assign_worker(worker_id, model_name)
        logger.info(
            f"Assign worker {worker_id} to {model_name}. "
            f"Current model_to_worker: {self._model_to_worker}"
        )
        self._worker_to_ipc_name[worker_id].send_pyobj(req)
        return True

    def handle_deactivate_model(self, req: DeactivateReqInput):
        """
        Handle model deactivation request.
        
        Args:
            req: Deactivation request input
            
        Returns:
            bool: Whether deactivation succeeded
        """
        model_name = req.model_name

        if req.gpu_id is not None:
            assert (
                req.gpu_id == self.gpu_id
            ), f"GPU ID mismatch: {req.gpu_id} != {self.gpu_id}"

        if model_name not in self._model_to_worker:
            logger.error(f"Model {model_name} is not served by any workers")
            return False
            
        worker_id = self.release_worker(model_name)
        logger.info(
            f"Release worker {worker_id} from model {model_name}. "
            f"Current model_to_worker: {self._model_to_worker}"
        )
        self._worker_to_ipc_name[worker_id].send_pyobj(req)
        return True

    def cleanup(self):
        """Clean up all ZMQ sockets and IPC files."""
        try:
            # Prepare sockets dictionary
            zmq_sockets = {}

            # Add worker sockets
            for worker_id, socket in self._worker_to_ipc_name.items():
                zmq_sockets[f"worker_{worker_id}"] = socket

            # Get IPC file paths
            ipc_files = set()
            for i in range(self.num_workers):
                worker = Worker(i, self.gpu_id)
                ipc_files.add(f"ipc://{worker.ipc_name}")

            # Clean up using utility function
            cleanup_zmq_ipc(
                zmq_sockets=zmq_sockets,
                ipc_files=ipc_files,
                component_name="WorkerPool",
                gpu_id=self.gpu_id,
            )

        except Exception as e:
            logger.error(f"Error during WorkerPool cleanup for GPU {self.gpu_id}: {e}")
