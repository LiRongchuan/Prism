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

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a a request to its token locations.
BaseTokenToKVPool maps a token location to its KV cache data.
"""

import getpass
import logging
import time
from typing import List, Optional, Tuple, Union

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)

PHYSICAL_MEM_CHECK_FREQ = 0.01


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        gpu_id: int,
        use_records: bool,
        min_reserve_mem: float,
    ):
        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.gpu_id = gpu_id
        self._init_req_to_token(min_reserve_mem)
        self.free_slots = list(range(size))
        self.write_records = []
        self.use_records = use_records

        if self.use_records:
            self.write = self.write_with_records
        else:
            self.write = self.write_without_records

    def _init_req_to_token(self, min_reserve_mem: float):
        required_bytes = self.size * self.max_context_len * 4
        required_mem = required_bytes / 1024**3

        # min_reserve_mem is GB
        # wait until the memory is enough
        tic = time.time()
        while (
            get_available_gpu_memory(self.device, self.gpu_id) - min_reserve_mem
            < required_mem
        ):
            logger.info(
                f"Waiting for enough memory to initialize the request to token pool.... Current available memory: {get_available_gpu_memory(self.device, self.gpu_id):.2f} GB, min reserve mem: {min_reserve_mem:.2f} GB, required memory for req_to_token pool: {required_mem:.2f} GB"
            )
            time.sleep(0.1)

        start_create = time.time()
        self.req_to_token = torch.zeros(
            (self.size, self.max_context_len),
            dtype=torch.int32,
            device=f"{self.device}:{self.gpu_id}",
        )
        logger.info(
            f"Initialize req_to_token pool end. Takes {time.time() - start_create:.4f}s. Wait time: {start_create - tic:.4f}s. Init time: {time.time() - start_create:.4f}s"
        )

    def write(self, indices, values):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            try:
                self.free_slots.extend(free_index)
            except TypeError:
                raise TypeError(
                    f"free_index must be an int or a list of ints, but got {type(free_index)}. free_index: {free_index}"
                )

    def clear(self):
        self.free_slots = list(range(self.size))
        self.write_records = []

    def write_without_records(self, indices, values):
        self.req_to_token[indices] = values

    def write_with_records(self, indices, values):
        self.req_to_token[indices] = values
        self.write_records.append((indices, values))

    def get_write_records(self):
        ret = self.write_records
        self.write_records = []
        return ret

    def apply_write_records(self, write_records: List[Tuple]):
        for indices, values in write_records:
            self.req_to_token[indices] = values

    def release(self):
        self.req_to_token.untyped_storage().resize_(0)
        self.req_to_token = None


class BaseTokenToKVPool:
    """A memory pool that maps a token location to its kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        gpu_id: int,
        min_reserve_mem: float,
    ):
        self.size = size
        self.dtype = dtype
        if dtype == torch.float8_e5m2:
            # NOTE: Store as torch.uint8 because Tensor index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device
        self.gpu_id = gpu_id
        self.min_reserve_mem = min_reserve_mem

        self.free_slots = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index.to(f"{self.device}:{self.gpu_id}", non_blocking=True)

    def free(self, free_index: torch.Tensor):
        if self.is_not_in_free_group:
            self.free_slots = torch.concat((self.free_slots, free_index.cpu()))
        else:
            self.free_group.append(free_index)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.concat(self.free_group))

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_slots = torch.arange(1, self.size + 1, dtype=torch.int32)
        self.is_in_free_group = False
        self.free_group = []

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def release(self):
        raise NotImplementedError()


class MHATokenToKVPool(BaseTokenToKVPool):

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        gpu_id: int,
        model_name: str,
        enable_elastic_memory: bool = False,
        min_reserve_mem: float = 0.0,
        enable_overlap: bool = False,
        use_kvcached_v0: bool = False,
        enable_worker_pool: bool = False,
        shm=None,
    ):
        super().__init__(
            size,
            dtype=dtype,
            device=device,
            gpu_id=gpu_id,
            min_reserve_mem=min_reserve_mem,
        )
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.enable_elastic_memory = enable_elastic_memory
        self.last_available_size = None
        self.enable_overlap = enable_overlap
        self.use_kvcached_v0 = use_kvcached_v0
        self.model_name = model_name
        self.enable_worker_pool = enable_worker_pool
        self.shm = shm

        if enable_elastic_memory:
            try:
                from kvcached import ops as kvcached_ops
                from kvcached.slab_allocator import KVCacheManager

                self.kvcached_ops = kvcached_ops
                self.kv_cache_manager = KVCacheManager

                if not enable_worker_pool:
                    self.kvcached_ops.init_kvcached(self.gpu_id)

                self.init_kv_allocator()
            except ImportError as e:
                raise ImportError(
                    "kvcached package is required for elastic memory. Please install it first."
                ) from e
        else:
            self._init_kv_cache(self.min_reserve_mem)

    def init_kv_allocator(self):
        k_buffer, v_buffer = self.kvcached_ops.sgl_alloc_kv_cache(
            self.size,
            self.head_num,
            self.head_dim,
            self.dtype,
            f"{self.device}:{self.gpu_id}",
            self.layer_num,
        )
        self.k_buffer = k_buffer
        self.v_buffer = v_buffer
        self.cell_size = self.head_num * self.head_dim * self.dtype.itemsize
        if self.use_kvcached_v0:
            self.kv_allocator = self.kv_cache_manager(
                self.size,
                1,
                self.cell_size,
                num_layers=self.layer_num,
                shm=self.shm,
            )
            logger.info("Elastic memory: kv_cache_manager_v0 initialized")
        else:
            self.kv_allocator = self.kv_cache_manager(
                self.size,
                1,
                self.cell_size,
                num_layers=self.layer_num,
                enable_overlap=self.enable_overlap,
                ipc_name=self.ipc_name,
            )
            logger.info("Elastic memory: kv_cache_manager initialized")

    def _init_kv_cache(self, min_reserve_mem: float):
        tic = time.time()
        cell_size = (
            self.head_num
            * self.head_dim
            * self.layer_num
            * 2
            * torch._utils._element_size(self.dtype)
        )
        required_bytes = cell_size * self.size
        required_mem = required_bytes / 1024**3

        # wait until the memory is enough, leave some room to allow required mem to be allocated
        while (
            get_available_gpu_memory(self.device, self.gpu_id)
            < required_mem + min_reserve_mem
        ):
            logger.info(
                f"Waiting for enough memory to initialize the kv cache.... Current available memory: {get_available_gpu_memory(self.device, self.gpu_id):.2f} GB, min reserve mem: {min_reserve_mem:.2f} GB, required memory for kv cache: {required_mem:.2f} GB"
            )
            time.sleep(0.1)

        start_create = time.time()
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.k_buffer = [
            torch.empty(
                (self.size + 1, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=f"{self.device}:{self.gpu_id}",
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.empty(
                (self.size + 1, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=f"{self.device}:{self.gpu_id}",
            )
            for _ in range(self.layer_num)
        ]
        logger.info(
            f"Initialize token_to_kv pool end. Takes {time.time() - start_create:.4f}s. Wait time: {start_create - tic:.4f}s. Init time: {time.time() - start_create:.4f}s"
        )

    def alloc(self, need_size: int):
        if self.enable_elastic_memory:
            indices = self.kv_allocator.alloc(need_size)
            if self.use_kvcached_v0:
                if isinstance(indices, list):
                    indices = torch.tensor(
                        indices,
                        dtype=torch.int32,
                        device=f"{self.device}:{self.gpu_id}",
                    )
                elif indices is not None:
                    indices = indices.to(
                        f"{self.device}:{self.gpu_id}", non_blocking=True
                    )
                return indices
            else:
                return (
                    indices.to(f"{self.device}:{self.gpu_id}", non_blocking=True)
                    if indices is not None
                    else None
                )
        else:
            return super().alloc(need_size)

    def free(self, free_index: torch.Tensor):
        if self.enable_elastic_memory:
            if self.use_kvcached_v0:
                return self.kv_allocator.free(free_index.cpu().numpy())
            else:
                return self.kv_allocator.free(free_index.cpu())
        else:
            return super().free(free_index)

    def _physical_free_size(self, min_reserve_mem_size: int) -> int:
        avail_phy_mem_size, _ = torch.cuda.mem_get_info()
        avail_phy_mem_size -= min_reserve_mem_size * (1 << 30)
        from kvcached.slab_allocator import PAGE_SIZE

        avail_phy_pages = avail_phy_mem_size // PAGE_SIZE
        # Each layer needs to reserve K and V tensors.
        avail_phy_blocks = (avail_phy_pages // self.layer_num // 2) * (
            PAGE_SIZE // self.kv_allocator.block_mem_size
        )
        return avail_phy_blocks

    def available_size(self):
        if self.enable_elastic_memory:
            if self.use_kvcached_v0:
                return self.kv_allocator.available_size()

            if self.last_available_size is None:
                free_size = self._physical_free_size(0.5)
                check_time = time.perf_counter()
                self.last_available_size = (free_size, check_time)
            elif (
                time.perf_counter() - self.last_available_size[1]
                > PHYSICAL_MEM_CHECK_FREQ
            ):
                free_size = self._physical_free_size(0.5)
                self.last_available_size = (free_size, time.perf_counter())
            else:
                free_size, _ = self.last_available_size
            return min(free_size, self.kv_allocator.available_size())
        return super().available_size()

    def update_size(self, new_size: int):
        if self.enable_elastic_memory:
            return self.kv_allocator.resize(new_size)
        else:
            raise ValueError("Only elastic memory is supported for update_size")

    def try_to_reserve(self, need_size: int):
        if self.enable_elastic_memory:
            return self.kv_allocator.try_to_reserve(need_size)
        return True

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.k_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
            self.v_buffer[layer_id][loc] = cache_v.view(self.store_dtype)
        else:
            self.k_buffer[layer_id][loc] = cache_k
            self.v_buffer[layer_id][loc] = cache_v

    def release(self):
        if not self.enable_elastic_memory:
            self.k_buffer = None
            self.v_buffer = None
        else:
            if self.use_kvcached_v0:
                self.kv_allocator.trim()
                if self.enable_worker_pool:
                    del self.kv_allocator
                    self.kvcached_ops.free_kv_cached_tensors()
            else:
                self.kv_allocator.clear()

    def shutdown(self):
        if self.enable_elastic_memory:
            self.kvcached_ops.shutdown_kvcached()
            del self.kv_allocator
            self.k_buffer = None
            self.v_buffer = None


# This compiled version is slower in the unit test
# python3 -m unittest test_bench_serving.TestBenchServing.test_offline_throughput_non_stream_small_batch_size
@torch.compile(dynamic=True)
def copy_two_array(loc, dst_1, src_1, dst_2, src_2, dtype, store_dtype):
    dst_1[loc] = src_1.to(dtype).view(store_dtype)
    dst_2[loc] = src_2.to(dtype).view(store_dtype)


class MLATokenToKVPool(BaseTokenToKVPool):

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
    ):
        super().__init__(size, dtype, device)

        self.kv_lora_rank = kv_lora_rank
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.kv_buffer = [
            torch.empty(
                (size + 1, 1, kv_lora_rank + qk_rope_head_dim),
                dtype=self.store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id].view(self.dtype)
        return self.kv_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id][..., : self.kv_lora_rank].view(self.dtype)
        return self.kv_buffer[layer_id][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
        else:
            self.kv_buffer[layer_id][loc] = cache_k

    def release(self):
        self.kv_buffer = None


class DoubleSparseTokenToKVPool(BaseTokenToKVPool):

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
    ):
        super().__init__(size, dtype, device)

        # [size, head_num, head_dim] for each layer
        self.k_buffer = [
            torch.empty((size + 1, head_num, head_dim), dtype=dtype, device=device)
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.empty((size + 1, head_num, head_dim), dtype=dtype, device=device)
            for _ in range(layer_num)
        ]

        # [size, head_num, heavy_channel_num] for each layer
        self.label_buffer = [
            torch.empty(
                (size + 1, head_num, heavy_channel_num), dtype=dtype, device=device
            )
            for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id][loc] = cache_k
        self.v_buffer[layer_id][loc] = cache_v
        self.label_buffer[layer_id][loc] = cache_label

    def release(self):
        self.k_buffer = None
        self.v_buffer = None
        self.label_buffer = None
