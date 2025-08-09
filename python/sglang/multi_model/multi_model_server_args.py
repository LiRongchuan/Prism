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

"""The arguments of the server."""

import argparse
import dataclasses
import json
import logging
import random
import tempfile
import uuid
from typing import List, Optional

from sglang.srt.utils import is_flashinfer_available, is_ipv6, is_port_available

logger = logging.getLogger(__name__)


from dataclasses import dataclass, field
from typing import List, Optional


@dataclasses.dataclass
class Placement:
    gpu_ids: List[int]  # for TP > 1
    on: bool = True
    max_memory_pool_size: Optional[float] = None


@dataclasses.dataclass
class ModelConfig:
    model_name: str  # A unique name for the each LLM.
    model_path: str
    tokenizer_path: Optional[str] = None
    tp_size: int = 1
    init_placements: List[Placement] = dataclasses.field(
        default_factory=list
    )  # Initial placements for each instance

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

    def get_instance_configs(self) -> List["InstanceConfig"]:
        return [
            InstanceConfig(
                model_name=self.model_name,
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path,
                gpu_ids=placement["gpu_ids"],
                tp_size=self.tp_size,
                on=placement.get("on", True),
                max_memory_pool_size=placement.get("max_memory_pool_size", None),
            )
            for placement in self.init_placements
        ]


@dataclasses.dataclass
class InstanceConfig:
    model_name: str
    model_path: str
    tokenizer_path: Optional[str] = None
    gpu_ids: List[int] = field(default_factory=list)
    tp_size: int = 1
    on: bool = True
    max_memory_pool_size: Optional[float] = None


def load_model_configs(file_path: str) -> List[ModelConfig]:
    with open(file_path, "r") as f:
        config_data = json.load(f)
        model_configs = [ModelConfig(**model) for model in config_data]
    return model_configs


@dataclasses.dataclass
class MultiModelServerArgs:
    # Model and tokenizer
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    tokenizer_path: Optional[str] = None
    model_config_file: Optional[str] = None
    model_configs: Optional[List[ModelConfig]] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    trust_remote_code: bool = True
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    device: str = "cuda"
    served_model_name: Optional[str] = None
    chat_template: Optional[str] = None
    is_embedding: bool = False

    # Port
    host: str = "127.0.0.1"
    port: int = 30000

    # Worker pool
    enable_worker_pool: bool = False
    workers_per_gpu: int = 1  # Number of workers per GPU
    num_gpus: int = 1  # Number of GPUs to use

    # Memory and scheduling
    mem_fraction_static: Optional[float] = None
    max_running_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_mem_usage: Optional[float] = None
    max_memory_pool_size: Optional[float] = None
    chunked_prefill_size: int = 8192
    max_prefill_tokens: int = 16384
    schedule_policy: str = "lpm"
    schedule_conservativeness: float = 1.0

    # Other runtime options
    tp_size: int = 1
    stream_interval: int = 1
    random_seed: Optional[int] = None
    constrained_json_whitespace_pattern: Optional[str] = None

    # Logging
    log_level: str = "info"
    log_level_http: Optional[str] = None
    log_requests: bool = False
    show_time_cost: bool = False
    log_file: Optional[str] = None

    # Other
    api_key: Optional[str] = None
    file_storage_pth: str = "SGLang_storage"
    enable_cache_report: bool = False

    # Data parallelism
    dp_size: int = 1
    load_balance_method: str = "round_robin"

    # Distributed args
    dist_init_addr: Optional[str] = None
    nnodes: int = 1
    node_rank: int = 0

    # Model override args in JSON
    json_model_override_args: str = "{}"

    # redis args for multi-model serving
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    enable_controller: bool = False
    enable_gpu_scheduler: bool = False
    policy: str = "simple-global"
    queue_id: str = str(uuid.uuid4().hex)

    # Async loading
    async_loading: bool = False

    # Double Sparsity
    enable_double_sparsity: bool = False
    ds_channel_config_path: str = None
    ds_heavy_channel_num: int = 32
    ds_heavy_token_num: int = 256
    ds_heavy_channel_type: str = "qk"
    ds_sparse_decode_threshold: int = 4096

    # LoRA
    lora_paths: Optional[List[str]] = None
    max_loras_per_batch: int = 8

    # Kernel backend
    attention_backend: Optional[str] = None
    sampling_backend: Optional[str] = None

    # Optimization/debug options
    disable_flashinfer: bool = False
    disable_flashinfer_sampling: bool = False
    disable_radix_cache: bool = False
    disable_regex_jump_forward: bool = False
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    disable_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    disable_mla: bool = False
    disable_penalizer: bool = False
    disable_nan_detection: bool = False
    enable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_torch_compile: bool = False
    max_torch_compile_bs: int = 32
    torchao_config: str = ""
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    num_continuous_decode_steps: int = 1
    enable_elastic_memory: bool = False
    use_kvcached_v0: bool = False
    enable_cpu_share_memory: bool = False
    enable_model_service: bool = False
    num_model_service_workers: int = 1
    abort_exceed_slos: bool = False

    def __post_init__(self):
        if self.model_path is not None:
            # set missing default values, for the single model case
            if self.tokenizer_path is None:
                self.tokenizer_path = self.model_path
            if self.served_model_name is None:
                self.served_model_name = self.model_path
            if self.model_name is None:
                self.model_name = self.model_path

        # Mem fraction depends on the tensor parallelism size
        if self.mem_fraction_static is None:
            if self.tp_size >= 16:
                self.mem_fraction_static = 0.79
            elif self.tp_size >= 8:
                self.mem_fraction_static = 0.83
            elif self.tp_size >= 4:
                self.mem_fraction_static = 0.85
            elif self.tp_size >= 2:
                self.mem_fraction_static = 0.87
            else:
                self.mem_fraction_static = 0.88

        if self.model_configs is None:
            if self.model_config_file:
                self.model_configs = load_model_configs(self.model_config_file)
            elif self.model_path:
                model_config = ModelConfig(
                    model_name=self.model_name,
                    model_path=self.model_path,
                    tokenizer_path=self.tokenizer_path,
                    init_placements=[
                        {
                            "gpu_ids": [0],
                            "on": True,
                        }
                    ],
                )
                self.model_configs = [model_config]
            else:
                raise ValueError(
                    "model_config_file or model_path is required when model_configs is not provided"
                )
            print(self.model_configs)

        if self.chunked_prefill_size <= 0:
            # Disable chunked prefill
            self.chunked_prefill_size = None

        if self.random_seed is None:
            self.random_seed = random.randint(0, 1 << 30)

        # Deprecation warnings
        if self.disable_flashinfer:
            logger.warning(
                "The option '--disable-flashinfer' will be deprecated in the next release. "
                "Please use '--attention-backend triton' instead."
            )
            self.attention_backend = "triton"
        if self.disable_flashinfer_sampling:
            logger.warning(
                "The option '--disable-flashinfer-sampling' will be deprecated in the next release. "
                "Please use '--sampling-backend pytorch' instead. "
            )
            self.sampling_backend = "pytorch"

        if not is_flashinfer_available():
            self.attention_backend = "triton"
            self.sampling_backend = "pytorch"

        # Default kernel backends
        if self.attention_backend is None:
            self.attention_backend = "flashinfer"

        if self.sampling_backend is None:
            self.sampling_backend = "flashinfer"

        if self.enable_overlap_schedule:
            logger.warning(
                "Overlap scheduler mode is enabled. This is an experimental feature. "
                "Sampling penalizer (e.g., frequency and repetition penalty), constrained decoding (e.g., regex, JSON), "
                "and embedding APIs are not supported and will lead to wrong results. "
                "The NaN detection is also disabled."
            )
            self.disable_penalizer = True
            self.disable_nan_detection = True

            # Model-specific patches
            # if "Alibaba-NLP/gte-Qwen2-1.5B-instruct" == self.model_path:
            #     logger.info(
            #         "Not sure why, the tokenizer will add an additional token at the end of the prompt when trust_remote_mode=True"
            #     )
            #     self.trust_remote_code = False

            # if "gemma-2" in self.model_path.lower():
            #     logger.info("When using sliding window in gemma-2, turn on flashinfer.")
            self.attention_backend = "flashinfer"

        if self.queue_id is None:
            self.queue_id = str(uuid.uuid4().hex)
        if self.enable_gpu_scheduler:
            self.frontend_generate_request_key_prefix = (
                f"frontend_generate_request_{self.queue_id}"
            )
            self.backend_generate_request_key_prefix = (
                f"backend_generate_request_{self.queue_id}"
            )
            self.engine_to_gpu_scheduler_key_prefix = (
                f"engine_to_gpu_scheduler_{self.queue_id}"
            )
        else:
            self.frontend_generate_request_key_prefix = (
                f"generate_request_{self.queue_id}"
            )
            self.backend_generate_request_key_prefix = (
                f"generate_request_{self.queue_id}"
            )
            self.engine_to_gpu_scheduler_key_prefix = (
                f"engine_to_gpu_scheduler_{self.queue_id}"
            )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            help="The name of the model. This is an unique identifier for the model in multi-model serving.",
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=MultiModelServerArgs.tokenizer_path,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--model-config-file",
            type=str,
            help="The path of the model config file.",
            # required=True,
        )
        parser.add_argument(
            "--model-configs",
            type=json.loads,
            help="The model configs in JSON format.",
        )
        parser.add_argument(
            "--host",
            type=str,
            default=MultiModelServerArgs.host,
            help="The host of the server.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=MultiModelServerArgs.port,
            help="The port of the server.",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=MultiModelServerArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer if available, and 'slow' will "
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--skip-tokenizer-init",
            action="store_true",
            help="If set, skip init tokenizer and pass input_ids in generate request",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default=MultiModelServerArgs.load_format,
            choices=["auto", "pt", "safetensors", "npcache", "dummy"],
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=MultiModelServerArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="Data type for model weights and activations.\n\n"
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            "BF16 precision for BF16 models.\n"
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.',
        )
        parser.add_argument(
            "--kv-cache-dtype",
            type=str,
            default=MultiModelServerArgs.kv_cache_dtype,
            choices=["auto", "fp8_e5m2"],
            help='Data type for kv cache storage. "auto" will use model data type. "fp8_e5m2" is supported for CUDA 11.8+.',
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=MultiModelServerArgs.quantization,
            choices=[
                "awq",
                "fp8",
                "gptq",
                "marlin",
                "gptq_marlin",
                "awq_marlin",
                "bitsandbytes",
            ],
            help="The quantization method.",
        )
        parser.add_argument(
            "--context-length",
            type=int,
            default=MultiModelServerArgs.context_length,
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cuda", "xpu"],
            help="The device type.",
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=MultiModelServerArgs.served_model_name,
            help="Override the model name returned by the v1/models endpoint in OpenAI API server.",
        )
        parser.add_argument(
            "--chat-template",
            type=str,
            default=MultiModelServerArgs.chat_template,
            help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
        )
        parser.add_argument(
            "--is-embedding",
            action="store_true",
            help="Whether to use a CausalLM as an embedding model.",
        )
        parser.add_argument(
            "--mem-fraction-static",
            type=float,
            default=MultiModelServerArgs.mem_fraction_static,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--max-mem-usage",
            type=float,
            default=MultiModelServerArgs.max_mem_usage,
            # default=66.93, # A100 80GB
            help="The maximum memory that model weights and KV cache can use. If not specified, it will be automatically calculated based on the mem-fraction-static. This option is mainly used for elastic memory allocation. Please specify this option for the collocation case.",
        )
        parser.add_argument(
            "--max-running-requests",
            type=int,
            default=MultiModelServerArgs.max_running_requests,
            help="The maximum number of running requests.",
        )
        parser.add_argument(
            "--max-total-tokens",
            type=int,
            default=MultiModelServerArgs.max_total_tokens,
            help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. "
            "This option is typically used for development and debugging purposes.",
        )
        parser.add_argument(
            "--max-memory-pool-size",
            type=float,
            default=MultiModelServerArgs.max_memory_pool_size,
            help="The maximum memory that the memory pool can use. If not specified, it will be automatically calculated based on the mem-fraction-static.",
        )
        parser.add_argument(
            "--chunked-prefill-size",
            type=int,
            default=MultiModelServerArgs.chunked_prefill_size,
            help="The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill",
        )
        parser.add_argument(
            "--max-prefill-tokens",
            type=int,
            default=MultiModelServerArgs.max_prefill_tokens,
            help="The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length.",
        )
        parser.add_argument(
            "--schedule-policy",
            type=str,
            default=MultiModelServerArgs.schedule_policy,
            choices=["lpm", "random", "fcfs", "dfs-weight"],
            help="The scheduling policy of the requests.",
        )
        parser.add_argument(
            "--schedule-conservativeness",
            type=float,
            default=MultiModelServerArgs.schedule_conservativeness,
            help="How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=MultiModelServerArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--stream-interval",
            type=int,
            default=MultiModelServerArgs.stream_interval,
            help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=MultiModelServerArgs.random_seed,
            help="The random seed.",
        )
        parser.add_argument(
            "--constrained-json-whitespace-pattern",
            type=str,
            default=MultiModelServerArgs.constrained_json_whitespace_pattern,
            help=r"Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [\n\t ]*",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default=MultiModelServerArgs.log_level,
            help="The logging level of all loggers.",
        )
        parser.add_argument(
            "--log-level-http",
            type=str,
            default=MultiModelServerArgs.log_level_http,
            help="The logging level of HTTP server. If not set, reuse --log-level by default.",
        )
        parser.add_argument(
            "--log-file",
            type=str,
            default=MultiModelServerArgs.log_file,
            help="The filename for HTTP server logs.",
        )
        parser.add_argument(
            "--log-requests",
            action="store_true",
            help="Log the inputs and outputs of all requests.",
        )
        parser.add_argument(
            "--show-time-cost",
            action="store_true",
            help="Show time cost of custom marks.",
        )
        parser.add_argument(
            "--api-key",
            type=str,
            default=MultiModelServerArgs.api_key,
            help="Set API key of the server. It is also used in the OpenAI API compatible server.",
        )
        parser.add_argument(
            "--file-storage-pth",
            type=str,
            default=MultiModelServerArgs.file_storage_pth,
            help="The path of the file storage in backend.",
        )
        parser.add_argument(
            "--enable-cache-report",
            action="store_true",
            help="Return number of cached tokens in usage.prompt_tokens_details for each openai request.",
        )

        # Data parallelism
        parser.add_argument(
            "--data-parallel-size",
            "--dp-size",
            type=int,
            default=MultiModelServerArgs.dp_size,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--load-balance-method",
            type=str,
            default=MultiModelServerArgs.load_balance_method,
            help="The load balancing strategy for data parallelism.",
            choices=[
                "round_robin",
                "shortest_queue",
            ],
        )
        parser.add_argument(
            "--enable-worker-pool",
            action="store_true",
            help="Enable worker pool for multi-model serving.",
        )
        parser.add_argument(
            "--workers-per-gpu",
            type=int,
            default=MultiModelServerArgs.workers_per_gpu,
            help="The number of workers per GPU for multi-model serving.",
        )
        parser.add_argument(
            "--num-gpus",
            type=int,
            default=MultiModelServerArgs.num_gpus,
            help="Number of GPUs to use for worker pool.",
        )

        # Multi-node distributed serving args
        parser.add_argument(
            "--dist-init-addr",
            "--nccl-init-addr",  # For backward compatbility. This will be removed in the future.
            type=str,
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
        )
        parser.add_argument(
            "--nnodes",
            type=int,
            default=MultiModelServerArgs.nnodes,
            help="The number of nodes.",
        )
        parser.add_argument(
            "--node-rank",
            type=int,
            default=MultiModelServerArgs.node_rank,
            help="The node rank.",
        )

        # Model override args
        parser.add_argument(
            "--json-model-override-args",
            type=str,
            help="A dictionary in JSON string format used to override default model configurations.",
            default=MultiModelServerArgs.json_model_override_args,
        )

        # Double Sparsity
        parser.add_argument(
            "--enable-double-sparsity",
            action="store_true",
            help="Enable double sparsity attention",
        )
        parser.add_argument(
            "--ds-channel-config-path",
            type=str,
            default=MultiModelServerArgs.ds_channel_config_path,
            help="The path of the double sparsity channel config",
        )
        parser.add_argument(
            "--ds-heavy-channel-num",
            type=int,
            default=MultiModelServerArgs.ds_heavy_channel_num,
            help="The number of heavy channels in double sparsity attention",
        )
        parser.add_argument(
            "--ds-heavy-token-num",
            type=int,
            default=MultiModelServerArgs.ds_heavy_token_num,
            help="The number of heavy tokens in double sparsity attention",
        )
        parser.add_argument(
            "--ds-heavy-channel-type",
            type=str,
            default=MultiModelServerArgs.ds_heavy_channel_type,
            help="The type of heavy channels in double sparsity attention",
        )
        parser.add_argument(
            "--ds-sparse-decode-threshold",
            type=int,
            default=MultiModelServerArgs.ds_sparse_decode_threshold,
            help="The type of heavy channels in double sparsity attention",
        )

        # LoRA
        parser.add_argument(
            "--lora-paths",
            type=str,
            nargs="*",
            default=None,
            # action=LoRAPathAction,
            help="The list of LoRA adapters. You can provide a list of either path in str or renamed path in the format {name}={path}",
        )
        parser.add_argument(
            "--max-loras-per-batch",
            type=int,
            default=8,
            help="Maximum number of adapters for a running batch, include base-only request",
        )

        # Kernel backend
        parser.add_argument(
            "--attention-backend",
            type=str,
            choices=["flashinfer", "triton"],
            default=MultiModelServerArgs.attention_backend,
            help="Choose the kernels for attention layers.",
        )
        parser.add_argument(
            "--sampling-backend",
            type=str,
            choices=["flashinfer", "pytorch"],
            default=MultiModelServerArgs.sampling_backend,
            help="Choose the kernels for sampling layers.",
        )

        # Optimization/debug options
        parser.add_argument(
            "--disable-flashinfer",
            action="store_true",
            help="Disable flashinfer attention kernels. This option will be deprecated in the next release. Please use '--attention-backend triton' instead.",
        )
        parser.add_argument(
            "--disable-flashinfer-sampling",
            action="store_true",
            help="Disable flashinfer sampling kernels. This option will be deprecated in the next release. Please use '--sampling-backend pytorch' instead.",
        )
        parser.add_argument(
            "--disable-radix-cache",
            action="store_true",
            help="Disable RadixAttention for prefix caching.",
        )
        parser.add_argument(
            "--disable-regex-jump-forward",
            action="store_true",
            help="Disable regex jump-forward.",
        )
        parser.add_argument(
            "--enable-elastic-memory",
            action="store_true",
            help="Enable elastic memory with kvcached in LLM engine.",
        )
        parser.add_argument(
            "--use-kvcached-v0",
            action="store_true",
            help="Use kvcached v0 for elastic memory.",
        )
        parser.add_argument(
            "--enable-cpu-share-memory",
            action="store_true",
            help="Enable CPU share memory for multi-model serving.",
        )
        parser.add_argument(
            "--enable-model-service",
            action="store_true",
            help="Enable model service for multi-model serving.",
        )
        parser.add_argument(
            "--num-model-service-workers",
            type=int,
            default=MultiModelServerArgs.num_model_service_workers,
            help="The number of model service workers.",
        )
        parser.add_argument(
            "--abort-exceed-slos",
            action="store_true",
            help="Abort requests that exceed SLOs.",
        )
        parser.add_argument(
            "--enable-controller",
            action="store_true",
            help="Enable controller for multi-model serving.",
        )
        parser.add_argument(
            "--enable-gpu-scheduler",
            action="store_true",
            help="Enable GPU scheduler for multi-model serving.",
        )
        parser.add_argument(
            "--disable-cuda-graph",
            action="store_true",
            help="Disable cuda graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph-padding",
            action="store_true",
            help="Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
        )
        parser.add_argument(
            "--disable-disk-cache",
            action="store_true",
            help="Disable disk cache to avoid possible crashes related to file system or high concurrency.",
        )
        parser.add_argument(
            "--disable-custom-all-reduce",
            action="store_true",
            default=False,
            help="Disable the custom all-reduce kernel and fall back to NCCL.",
        )
        parser.add_argument(
            "--disable-mla",
            action="store_true",
            help="Disable Multi-head Latent Attention (MLA) for DeepSeek-V2.",
        )
        parser.add_argument(
            "--disable-penalizer",
            action="store_true",
            help="Disable the logit penalizers (e.g., frequency and repetition penalty) for better performance if they are not used in any requests.",
        )
        parser.add_argument(
            "--disable-nan-detection",
            action="store_true",
            help="Disable the NaN detection for better performance.",
        )
        parser.add_argument(
            "--enable-overlap-schedule",
            action="store_true",
            help="Overlap the CPU scheduler with GPU model worker. Experimental feature.",
        )
        parser.add_argument(
            "--enable-mixed-chunk",
            action="store_true",
            help="Enabling mixing prefill and decode in a batch when using chunked prefill.",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile. Experimental feature.",
        )
        parser.add_argument(
            "--max-torch-compile-bs",
            type=int,
            default=MultiModelServerArgs.max_torch_compile_bs,
            help="Set the maximum batch size when using torch compile.",
        )
        parser.add_argument(
            "--torchao-config",
            type=str,
            default=MultiModelServerArgs.torchao_config,
            help="Optimize the model with torchao. Experimental feature. Current choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo",
        )
        parser.add_argument(
            "--enable-p2p-check",
            action="store_true",
            help="Enable P2P check for GPU access, otherwise the p2p access is allowed by default.",
        )
        parser.add_argument(
            "--triton-attention-reduce-in-fp32",
            action="store_true",
            help="Cast the intermidiate attention results to fp32 to avoid possible crashes related to fp16."
            "This only affects Triton attention kernels.",
        )
        parser.add_argument(
            "--num-continuous-decode-steps",
            type=int,
            default=MultiModelServerArgs.num_continuous_decode_steps,
            help="Run multiple continuous decoding steps to reduce scheduling overhead. "
            "This can potentially increase throughput but may also increase time-to-first-token latency. "
            "The default value is 1, meaning only run one decoding step at a time.",
        )
        parser.add_argument(
            "--redis-host",
            type=str,
            default=MultiModelServerArgs.redis_host,
            help="The host of the Redis server for multi-model serving.",
        )
        parser.add_argument(
            "--redis-port",
            type=int,
            default=MultiModelServerArgs.redis_port,
            help="The port of the Redis server for multi-model serving.",
        )
        parser.add_argument(
            "--redis-db",
            type=int,
            default=MultiModelServerArgs.redis_db,
            help="The db of the Redis server for multi-model serving.",
        )
        parser.add_argument(
            "--async-loading",
            action="store_true",
            help="Async loading model when activating.",
        )
        parser.add_argument(
            "--policy",
            type=str,
            choices=[
                "simple-global",
            ],
            default=MultiModelServerArgs.policy,
            help="The policy for multi-model scheduling.",
        )
        parser.add_argument(
            "--queue-id",
            type=str,
            default=MultiModelServerArgs.queue_id,
            help="The queue id for multi-model scheduling.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.tp_size = args.tensor_parallel_size
        args.dp_size = args.data_parallel_size
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def url(self):
        if is_ipv6(self.host):
            return f"http://[{self.host}]:{self.port}"
        else:
            return f"http://{self.host}:{self.port}"

    def check_server_args(self):
        assert (
            self.tp_size % self.nnodes == 0
        ), "tp_size must be divisible by number of nodes"
        assert not (
            self.dp_size > 1 and self.nnodes != 1
        ), "multi-node data parallel is not supported"
        assert (
            self.max_loras_per_batch > 0
            # FIXME
            and (self.lora_paths is None or self.disable_cuda_graph)
            and (self.lora_paths is None or self.disable_radix_cache)
        ), "compatibility of lora and cuda graph and radix attention is in progress"

        if isinstance(self.lora_paths, list):
            lora_paths = self.lora_paths
            self.lora_paths = {}
            for lora_path in lora_paths:
                if "=" in lora_path:
                    name, path = lora_path.split("=", 1)
                    self.lora_paths[name] = path
                else:
                    self.lora_paths[lora_path] = lora_path


def prepare_server_args(argv: List[str]) -> MultiModelServerArgs:
    """
    Prepare the server arguments from the command line arguments.

    Args:
        args: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The server arguments.
    """
    parser = argparse.ArgumentParser()
    MultiModelServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    server_args = MultiModelServerArgs.from_cli_args(raw_args)
    return server_args
