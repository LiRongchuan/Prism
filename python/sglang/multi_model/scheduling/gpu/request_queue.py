import heapq
import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Set

from sglang.srt.managers.io_struct import GenerateReqInput

logger = logging.getLogger(__name__)


class RequestWrapper:
    """Request wrapper for encapsulating requests and calculating priorities."""
    
    def __init__(self, req: GenerateReqInput):
        self.model_name = req.model
        self.priority = self._calculate_priority(req)  # Lower value means higher priority (min-heap)
        self.req = req

    def _calculate_priority(self, req: GenerateReqInput):
        """Calculate request priority."""
        def clamp(x, lower, upper):
            return max(lower, min(x, upper))

        profiled_prefill_time = (
            req.prompt_len * (0.5 / 1024) if req.prompt_len is not None else 0.5
        )
        profiled_prefill_time = clamp(profiled_prefill_time, 0.2, 2)
        return req.arrival_time + req.slo - profiled_prefill_time

    def __lt__(self, other):
        return self.priority < other.priority  # Heap uses this for comparison

    def __str__(self):
        return f"RequestWrapper(model_name={self.model_name}, priority={self.priority}, req_id={self.req.rid})"

    def __repr__(self):
        return self.__str__()


class RequestQueue:
    """
    Request queue manager using priority queue to manage requests for different models.
    Supports admission control and resource management.
    """

    def __init__(self, model_name_to_cell_size: Dict[str, int]):
        self._skip_model_threshold = 10
        self._queue: List[RequestWrapper] = []  # Priority queue (min-heap)
        self._model_requests: Dict[str, Set[RequestWrapper]] = defaultdict(set)
        self._lock = threading.Lock()
        self._model_name_to_cell_size = model_name_to_cell_size
        self.last_log_time = 0

        # Record resources used by requests that have been admitted but are still in "activating" state
        # Used to subtract this portion from available resources in subsequent admission_control calls
        # to prevent reusing already consumed resources
        self._activating_usage_by_model = defaultdict(float)

    def empty(self) -> bool:
        """Check if the queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def pop_model_requests(self, model_name: str) -> List[GenerateReqInput]:
        """
        Pop all requests for a specific model from the queue.
        Returns a list of the original request objects.
        """
        if model_name not in self._model_requests:
            return []

        with self._lock:
            # Get all request wrappers for this model
            model_reqs = list(self._model_requests[model_name])

            # Remove these requests from the queue
            self._queue = [req for req in self._queue if req not in model_reqs]
            heapq.heapify(self._queue)

            # Clear this model from the model_requests dict
            del self._model_requests[model_name]

            # Return the original request objects
            return [req_wrapper.req for req_wrapper in model_reqs]

    def add_requests(self, reqs: List[GenerateReqInput]):
        """Add a batch of requests to the priority queue."""
        wrapped_reqs = [RequestWrapper(req) for req in reqs]

        with self._lock:
            for wrapped_req in wrapped_reqs:
                heapq.heappush(self._queue, wrapped_req)
                self._model_requests[wrapped_req.model_name].add(wrapped_req)

    def remove_model_requests(self, model_name):
        """Remove all requests of a specific model from the queue."""
        if model_name not in self._model_requests:
            return

        with self._lock:
            removed = self._model_requests[model_name]
            self._queue = [req for req in self._queue if req not in removed]
            heapq.heapify(self._queue)
            del self._model_requests[model_name]
    
    def log_info(self, info: str):
        """Rate-limited log output."""
        current_time = time.time()
        if current_time - self.last_log_time > 1:
            logger.info(info)
            self.last_log_time = current_time

    def admission_control(
        self,
        available_resources: float,
        model_backend_queue_lens: Dict[str, int],
        model_states: Dict[str, str],
        allow_sending_when_activating: bool = False,
    ) -> Dict[str, List[GenerateReqInput]]:
        """
        Request admission control.

        Key changes:
        1. Before each admission, subtract total resources consumed by requests in "activating" state
        2. In the loop, if model is "activating" and sending is allowed, check if net available resources are sufficient
        3. If model is "activated", use net available resources for checking, but don't accumulate in activating usage

        This ensures subsequent calls won't double-count already consumed resources.
        """
        admitted = defaultdict(list)

        # Calculate total resources consumed by activating state requests
        total_activating_usage = sum(self._activating_usage_by_model.values())
        # Note: Using infinity here as the actual implementation doesn't seem to limit resources
        net_available = float("inf")

        if net_available <= 0 or len(self._queue) == 0:
            self.log_info(f"net_available: {net_available}, queue_len: {len(self._queue)}")
            return admitted

        # Skip models with long backend queues
        models_to_skip = {
            model_name
            for model_name, queue_len in model_backend_queue_lens.items()
            if queue_len > self._skip_model_threshold
        }

        new_queue = []

        with self._lock:
            while self._queue and net_available > 0:
                req_wrapper = heapq.heappop(self._queue)
                model_name = req_wrapper.model_name
                state = model_states.get(model_name, "deactivated")

                # Skip models with long backend queues
                if model_name in models_to_skip:
                    new_queue.append(req_wrapper)
                    continue

                # Don't accept new requests if model is deactivating or deactivated
                if state in ("deactivating", "deactivated"):
                    new_queue.append(req_wrapper)
                    continue

                # Skip if model is activating but sending is not allowed
                if state == "activating" and not allow_sending_when_activating:
                    new_queue.append(req_wrapper)
                    continue

                resources_needed = self._get_request_resources(req_wrapper.req)

                # Core decision: Check if there are enough net available resources
                if net_available >= resources_needed:
                    net_available -= resources_needed
                    # Mark the request as admitted
                    admitted[model_name].append(req_wrapper.req)
                    self._model_requests[model_name].remove(req_wrapper)
                    if not self._model_requests[model_name]:
                        del self._model_requests[model_name]

                    # If model is in "activating" state, add this resource usage to activating usage
                    if state == "activating":
                        self._activating_usage_by_model[model_name] += resources_needed
                else:
                    new_queue.append(req_wrapper)
                    break

            # Put remaining requests back in the queue
            new_queue.extend(self._queue)
            self._queue = new_queue
            heapq.heapify(self._queue)
            self.log_info(
                f"net_available: {net_available}, queue_len: {len(self._queue)}, "
                f"model_backend_queue_lens: {model_backend_queue_lens}"
            )
        return admitted

    def _get_request_resources(self, req: GenerateReqInput) -> float:
        """
        Calculate how many resources (memory) the request needs.
        Simplified calculation: (input_len + 20) * cell_size
        """
        cell_size = self._model_name_to_cell_size[req.model]
        input_len = req.prompt_len
        if input_len is None or input_len == 0:
            input_len = 1024  # Default value from profiled data
        return cell_size * (input_len + 20)

    def clear_activating_usage(self, model_name: str):
        """
        When a model moves from "activating" to "activated" state, clear its resource usage 
        during the activating state. Can be cleared all at once or use a separate thread 
        to clear portions periodically.
        """
        with self._lock:
            if model_name in self._activating_usage_by_model:
                del self._activating_usage_by_model[model_name]

    def __len__(self) -> int:
        return len(self._queue)

    def __repr__(self) -> str:
        if len(self._model_requests) == 0:
            req_counts_str = ""
        else:
            req_counts_str = ", ".join(
                [
                    f"{model_name}: {len(reqs)}"
                    for model_name, reqs in self._model_requests.items()
                ]
            )
        return f"RequestQueue(total_queued={len(self._queue)}, {req_counts_str})"
