import dataclasses
import logging
import time
from collections import deque
from datetime import timedelta
from enum import Enum
from typing import List, Optional

from sglang.srt.managers.io_struct import (
    BatchRetractDecodeReq,
    BatchRunReq,
    FinishReq,
    GenerateReqInput,
)

logger = logging.getLogger(__name__)


# for future req statistics analysis
class ReqState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMTED = "preempted"
    FINISHED = "finished"


@dataclasses.dataclass
class Req:
    rid: str
    arrival_time: Optional[float] = None
    slo: Optional[float] = None
    model: Optional[str] = None
    prompt_len: Optional[int] = None
    output_len: Optional[int] = None

    # running
    start_running_time: Optional[float] = None
    gpu_id: Optional[int] = None

    # preempted
    preempted_time: Optional[float] = None
    generated_output_len: Optional[int] = None

    finish_time: Optional[float] = None
    _left_exec_time: Optional[float] = None
    remaining_time_budget: Optional[float] = None
    last_exec_tstamp: float = float(
        "-inf"
    )  # last time when _left_exec_time was updated

    state: ReqState = ReqState.WAITING
    is_warmup: bool = False

    @property
    def left_exec_time(self):
        if self._left_exec_time is None:
            # if a request's output length is None, we use default value 512
            # TODO: change to a more accurate way of estimating the left execution time
            if self.output_len is None:
                self._left_exec_time = 512 * 0.02
            else:
                self._left_exec_time = self.output_len * 0.02
        return self._left_exec_time

    @left_exec_time.setter
    def left_exec_time(self, new_value):
        self._left_exec_time = new_value

    def update_wait_info(
        self,
        arrival_time: float,
        slo: float,
        model: str,
        prompt_len: int,
        output_len: int,
        is_warmup: bool,
    ):
        if self.state == ReqState.WAITING:
            logger.warning(
                f"Request {self.rid} (arrival_time: {arrival_time}) is already in waiting state. New arrival time: {arrival_time}"
            )
        self.arrival_time = arrival_time
        self.slo = slo
        self.model = model
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.is_warmup = is_warmup

    def update_running_info(self, start_time: float, gpu_id: int):
        self.start_running_time = start_time
        self.gpu_id = gpu_id

    def update_preempt_info(self, retract_time: float, output_len: int):
        self.preempted_time = retract_time
        self.generated_output_len = output_len

    def update_finish_info(self, finish_time: float):
        self.state = ReqState.FINISHED
        self.finish_time = finish_time

    def set_state(self, state: ReqState):
        self.state = state


class ModelQueueTracker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.received_reqs: dict[str, Req] = {}  # save all received requests.
        self.waiting_reqs: deque[Req] = deque()
        self.running_reqs: List[Req] = []

        self._last_arrival_time = float("-inf")
        self._last_finished_time = float("-inf")
        self.start_time = time.time()

        # XXX, MMY: Throughput tracking
        self.latest_token_tput = 0.0  # PD total throughput
        self.prefill_token_tput = 0.0  # Prefill throughput
        self.decode_token_tput = 0.0  # Decode throughput

    def _get_req_from_queue(self, queue: deque[Req], rid: str) -> Optional[Req]:
        """Helper method to find and remove a request from a queue."""
        req = next((r for r in queue if r.rid == rid), None)
        if req is not None:
            queue.remove(req)
        return req

    def _update_req_in_queue(self, queue: deque[Req], req: Req):
        """Helper method to update a request in a queue."""
        for i, r in enumerate(queue):
            if r.rid == req.rid:
                queue[i] = req
                break

    def enqueue_req(self, generate_req: GenerateReqInput) -> None:
        if (
            generate_req.rid not in self.received_reqs
            or self.received_reqs[generate_req.rid].state == ReqState.FINISHED
        ):
            # Handle new request
            req = Req(
                rid=generate_req.rid,
                arrival_time=generate_req.arrival_time,
                slo=generate_req.slo,
                model=self.model_name,
                prompt_len=generate_req.prompt_len,
                output_len=generate_req.output_len,
                is_warmup=generate_req.is_warmup,
            )
            self.received_reqs[generate_req.rid] = req
            if not generate_req.is_warmup:
                self.waiting_reqs.append(req)
            
            # we should not update the last arrival time for warmup requests
            if generate_req.is_warmup is False: 
                self._last_arrival_time = max(
                    self._last_arrival_time, generate_req.arrival_time
                )
        else:
            # existed request
            req = self.received_reqs[generate_req.rid]
            req.update_wait_info(
                generate_req.arrival_time,
                generate_req.slo,
                self.model_name,
                generate_req.prompt_len,
                generate_req.output_len,
                generate_req.is_warmup,
            )
            self.received_reqs[generate_req.rid] = req

            if req.state == ReqState.RUNNING:
                self._update_req_in_queue(self.running_reqs, req)

            # we should not update the last arrival time for warmup requests
            if req.is_warmup is False:
                self._last_arrival_time = max(
                    self._last_arrival_time, generate_req.arrival_time
                )
                
            if req.is_warmup:
                state = req.state
                if state == ReqState.RUNNING:
                    self._get_req_from_queue(self.running_reqs, req.rid)

    def start_running_reqs(self, batch_run_req: BatchRunReq):
        for rid in batch_run_req.rids:
            req = self.received_reqs.get(rid)

            if req is None:
                # Handle new request
                req = Req(rid=rid)
                req.update_running_info(batch_run_req.run_time, batch_run_req.gpu_id)
                req.set_state(ReqState.RUNNING)
                self.running_reqs.append(req)
                self.received_reqs[rid] = req
                continue

            if req.is_warmup:
                continue

            req.update_running_info(batch_run_req.run_time, batch_run_req.gpu_id)

            if (req.state == ReqState.WAITING) or (
                req.state == ReqState.PREEMTED
                and req.start_running_time > req.preempted_time
            ):
                waiting_req = self._get_req_from_queue(self.waiting_reqs, rid)
                if waiting_req:
                    waiting_req.set_state(ReqState.RUNNING)
                    self.running_reqs.append(waiting_req)
                else:
                    raise ValueError(f"Request {rid} not found in waiting queue")

            elif req.state == ReqState.PREEMTED:  # preempted before running
                self._update_req_in_queue(self.waiting_reqs, req)
            elif req.state == ReqState.RUNNING:
                self._update_req_in_queue(self.running_reqs, req)

            self.received_reqs[rid] = req

    def preempt_reqs(self, batch_retract_decode_req: BatchRetractDecodeReq):
        for rid, len_output_id in zip(
            batch_retract_decode_req.rids, batch_retract_decode_req.len_output_ids
        ):
            req = self.received_reqs.get(rid)
            if req is None:
                raise ValueError(
                    f"Received preempt request {rid} for model {self.model_name} not found in received requests"
                )

            if req.is_warmup:
                continue

            req.update_preempt_info(
                batch_retract_decode_req.retract_time, len_output_id
            )
            if (
                req.state == ReqState.RUNNING
                and req.start_running_time <= req.preempted_time
            ):
                running_req = self._get_req_from_queue(self.running_reqs, rid)
                if running_req:
                    running_req.set_state(ReqState.PREEMTED)
                    self.waiting_reqs.appendleft(running_req)
                else:
                    raise ValueError(f"Request {rid} not found in running queue")
            elif req.state == ReqState.RUNNING:
                self._update_req_in_queue(self.running_reqs, req)
            elif req.state == ReqState.WAITING:
                req.set_state(ReqState.PREEMTED)
                self._update_req_in_queue(self.waiting_reqs, req)

            self.received_reqs[rid] = req

    def finish_req(self, finish_req: FinishReq):
        rid = finish_req.rid
        req = self.received_reqs.get(rid)
        if req is None:
            # Handle new request
            req = Req(rid=rid)
            req.update_finish_info(finish_req.finish_time)
            self.received_reqs[rid] = req
        else:
            if req.is_warmup:
                return
            if req.state == ReqState.RUNNING:
                req = self._get_req_from_queue(self.running_reqs, rid)
            elif req.state == ReqState.WAITING or req.state == ReqState.PREEMTED:
                req = self._get_req_from_queue(self.waiting_reqs, rid)

            req.update_finish_info(finish_req.finish_time)
            self.received_reqs[rid] = req

        self._last_finished_time = finish_req.finish_time
        return

    def get_last_finished_time(self) -> float:
        return self._last_finished_time

    def get_last_arrival_time(self) -> float:
        return self._last_arrival_time

    def get_earliest_arrival_time(self) -> float:
        if self.waiting_reqs:
            if self.waiting_reqs[0].state == ReqState.WAITING:
                return self.waiting_reqs[0].arrival_time
            elif self.waiting_reqs[0].state == ReqState.PREEMTED:
                return self.waiting_reqs[0].preempted_time
            else:
                logger.warning(
                    f"Request {self.waiting_reqs[0].rid} in waiting queue is in {self.waiting_reqs[0].state} state"
                )
        return float("inf")

    def get_last_arrival_time(self) -> float:
        return self._last_arrival_time

    def get_num_waiting_reqs(self) -> int:
        return len(self.waiting_reqs)

    def get_num_running_reqs(self) -> int:
        return len(self.running_reqs)

    def get_num_unfinished_reqs(self) -> int:
        return self.get_num_waiting_reqs() + self.get_num_running_reqs()

    def __str__(self):
        last_finish_time_ref = (
            timedelta(seconds=self.get_last_finished_time() - self.start_time)
            if self.get_last_finished_time() != float("-inf")
            else "N/A"
        )
        earliest_arrival_time_ref = (
            timedelta(seconds=self.get_earliest_arrival_time() - self.start_time)
            if self.get_earliest_arrival_time() != float("inf")
            else "N/A"
        )
        last_arrival_time_ref = (
            timedelta(seconds=self.get_last_arrival_time() - self.start_time)
            if self.get_last_arrival_time() != float("-inf")
            else "N/A"
        )
        return f"{self.model_name}: num_unfinished_reqs: {self.get_num_unfinished_reqs()}, num_waiting_reqs: {self.get_num_waiting_reqs()}, num_running_reqs: {self.get_num_running_reqs()}, last_arrival_time_ref: {last_arrival_time_ref}, earliest_arrived_req_in_queue: {earliest_arrival_time_ref}, last_finish_time_ref: {last_finish_time_ref}"
