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

"""DetokenizerManager is a process that detokenizes the token ids."""

import atexit
import dataclasses
import logging
import os
import signal
from collections import OrderedDict
from typing import Dict, List, Union

import zmq

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.io_struct import (
    ActivateReqInput,
    ActivateReqOutput,
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    DeactivateReqInput,
    DeactivateReqOutput,
    GetMemPoolSizeReqOutput,
    UpdateModelTput,
    UpdateWeightReqOutput,
)
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_STR, FINISH_MATCHED_TOKEN
from sglang.srt.redis_utils import RedisClient
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, kill_parent_process
from sglang.utils import cleanup_zmq_ipc, find_printable_text, get_exception_traceback

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""

    vid: int
    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int


class DetokenizerManager:
    """DetokenizerManager is a process that detokenizes the token ids."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        model_names_to_model_paths: Dict[str, str],
    ):
        # Init inter-process communication
        context = zmq.Context(1)

        # Store IPC file paths for cleanup
        self.ipc_files = set()
        self.ipc_files.add(f"ipc://{port_args.detokenizer_ipc_name}")

        self.recv_from_scheduler = context.socket(zmq.PULL)
        self.recv_from_scheduler.bind(f"ipc://{port_args.detokenizer_ipc_name}")

        self.send_to_request_handler = context.socket(zmq.PUSH)
        self.send_to_request_handler.connect(
            f"ipc://{port_args.request_handler_ipc_name}"
        )
        self.server_args = server_args
        self.model_names_to_model_paths = model_names_to_model_paths

        if server_args.skip_tokenizer_init:
            self.tokenizer = None
            self.tokenizers = None
        else:
            self.tokenizers = {
                model_name: get_tokenizer(
                    model_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
                for model_name, model_path in model_names_to_model_paths.items()
            }
            if not server_args.enable_worker_pool:
                self.tokenizer = self.tokenizers[server_args.model_name]
            else:
                self.tokenizer = None

        self.decode_status = LimitedCapacityDict()

        # Register cleanup function
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle signals to ensure cleanup before exit"""
        self.cleanup()
        # Re-raise the signal to allow the default handler to run
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def cleanup(self):
        """Clean up all IPC sockets and files when exiting"""
        try:
            # Prepare sockets dictionary
            zmq_sockets = {
                "recv_from_scheduler": getattr(self, "recv_from_scheduler", None),
                "send_to_request_handler": getattr(
                    self, "send_to_request_handler", None
                ),
            }

            # Clean up using utility function
            cleanup_zmq_ipc(
                zmq_sockets=zmq_sockets,
                ipc_files=getattr(self, "ipc_files", set()),
                component_name="DetokenizerManager",
            )

        except Exception as e:
            logger.error(f"Error during DetokenizerManager cleanup: {e}")

    def __del__(self):
        """Ensure cleanup when the object is garbage collected"""
        self.cleanup()

    def _reinit_tokenizer(self, model_name):
        if self.server_args.enable_worker_pool:
            new_tokenizer = self.tokenizers[model_name]
            self.tokenizer = new_tokenizer

    def trim_eos(self, output: Union[str, List[int]], finished_reason, no_stop_trim):
        if no_stop_trim:
            return output

        # Trim stop str. TODO(lmzheng): handle the case where multiple stop strs are hit
        if isinstance(finished_reason, FINISH_MATCHED_STR) and isinstance(output, str):
            pos = output.find(finished_reason.matched)
            return output[:pos] if pos != -1 else output
        if isinstance(finished_reason, FINISH_MATCHED_TOKEN) and isinstance(
            output, list
        ):
            assert len(output) > 0
            return output[:-1]
        return output

    def event_loop(self):
        """The event loop that handles requests"""

        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            if (
                isinstance(recv_obj, GetMemPoolSizeReqOutput)
                or isinstance(recv_obj, DeactivateReqOutput)
                or isinstance(recv_obj, UpdateModelTput)
            ):
                # Send to request handler
                self.send_to_request_handler.send_pyobj(recv_obj)
                continue
            elif isinstance(recv_obj, ActivateReqOutput):
                logger.info(
                    f"Detokenizer: Received activate request for {recv_obj.model_name}"
                )
                self._reinit_tokenizer(recv_obj.model_name)
                self.send_to_request_handler.send_pyobj(recv_obj)
                continue
            elif self.tokenizer is None:
                # If the tokenizer is skipped, no detokenization is needed
                self.send_to_request_handler.send_pyobj(recv_obj)
                continue

            if self.tokenizer is None:
                self.send_to_request_handler.send_pyobj(recv_obj)

            bs = len(recv_obj.rids)

            # Initialize decode status
            read_ids, surr_ids = [], []
            for i in range(bs):
                rid = recv_obj.rids[i]
                vid = recv_obj.vids[i]
                if rid not in self.decode_status or self.decode_status[rid].vid != vid:
                    s = DecodeStatus(
                        vid=vid,
                        decoded_text=recv_obj.decoded_texts[i],
                        decode_ids=recv_obj.decode_ids[i],
                        surr_offset=0,
                        read_offset=recv_obj.read_offsets[i],
                    )
                    self.decode_status[rid] = s
                else:
                    s = self.decode_status[rid]
                    s.decode_ids = recv_obj.decode_ids[i]

                read_ids.append(
                    self.trim_eos(
                        s.decode_ids[s.surr_offset :],
                        recv_obj.finished_reason[i],
                        recv_obj.no_stop_trim[i],
                    )
                )
                surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

            # TODO(lmzheng): handle skip_special_tokens/spaces_between_special_tokens per request
            surr_texts = self.tokenizer.batch_decode(
                surr_ids,
                skip_special_tokens=recv_obj.skip_special_tokens[0],
                spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
            )
            read_texts = self.tokenizer.batch_decode(
                read_ids,
                skip_special_tokens=recv_obj.skip_special_tokens[0],
                spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
            )

            # Incremental decoding
            output_strs = []
            for i in range(bs):
                s = self.decode_status[recv_obj.rids[i]]
                new_text = read_texts[i][len(surr_texts[i]) :]
                if recv_obj.finished_reason[i] is None:
                    # Streaming chunk: update the decode status
                    if len(new_text) > 0 and not new_text.endswith("�"):
                        s.decoded_text = s.decoded_text + new_text
                        s.surr_offset = s.read_offset
                        s.read_offset = len(s.decode_ids)
                        new_text = ""
                    else:
                        new_text = find_printable_text(new_text)

                output_strs.append(
                    self.trim_eos(
                        s.decoded_text + new_text,
                        recv_obj.finished_reason[i],
                        recv_obj.no_stop_trim[i],
                    )
                )

            pyobj = BatchStrOut(
                rids=recv_obj.rids,
                output_strs=output_strs,
                meta_info=recv_obj.meta_info,
                finished_reason=recv_obj.finished_reason,
            )

            self.send_to_request_handler.send_pyobj(pyobj)


class LimitedCapacityDict(OrderedDict):
    def __init__(self, capacity=1 << 15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)
        # Set the new item
        super().__setitem__(key, value)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    model_names_to_model_paths: Dict[str, str],
):
    configure_logger(server_args)

    try:
        manager = DetokenizerManager(server_args, port_args, model_names_to_model_paths)
        manager.event_loop()
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()
