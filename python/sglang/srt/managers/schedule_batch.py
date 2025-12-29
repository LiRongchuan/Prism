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
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
"""


import time
import torch
import logging
import dataclasses
from sglang.srt.constrained import RegexGuide
from sglang.srt.server_args import ServerArgs
from sglang.global_config import global_config
from typing import List, Optional, Tuple, Union
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.constrained.jump_forward import JumpForwardMap
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.mem_cache.memory_pool import (
    PHYSICAL_MEM_CHECK_FREQ,
    BaseTokenToKVPool,
    ReqToTokenPool,
)
INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# Put some global args for easy access
global_server_args_dict = {
    "attention_backend": ServerArgs.attention_backend,
    "sampling_backend": ServerArgs.sampling_backend,
    "triton_attention_reduce_in_fp32": ServerArgs.triton_attention_reduce_in_fp32,
    "disable_mla": ServerArgs.disable_mla,
    "torchao_config": ServerArgs.torchao_config,
    "disable_nan_detection": ServerArgs.disable_nan_detection,
}

logger = logging.getLogger(__name__)


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_LENGTH(BaseFinishReason):
    """ 长度超限终止输出 """
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }
        
        
class FINISH_MATCHED_TOKEN(BaseFinishReason):
    """ EOS终止输出 """
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    """ 指定字符终止 """
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_ABORT(BaseFinishReason):
    """ 错误终止输出 """
    def __init__(self):
        super().__init__(is_error=True)

    def to_json(self):
        return {
            "type": "abort",
        }


@dataclasses.dataclass
class ImageInputs:
    """The image related inputs."""
    pixel_values: torch.Tensor
    image_hashes: Optional[list] = None
    image_sizes: Optional[list] = None
    image_offsets: Optional[list] = None
    pad_values: Optional[list] = None
    modalities: Optional[list] = None
    num_image_tokens: Optional[int] = None
    image_embeds: Optional[List[torch.Tensor]] = None
    aspect_ratio_ids: Optional[List[torch.Tensor]] = None
    aspect_ratio_mask: Optional[List[torch.Tensor]] = None
    # QWen2-VL related
    image_grid_thws: List[Tuple[int, int, int]] = None

    @staticmethod
    def from_dict(obj, vocab_size):
        # 使用image hash生成伪token，用于前缀匹配
        ret = ImageInputs(
            pixel_values=obj["pixel_values"],
            image_hashes=hash(tuple(obj["image_hashes"])),
        )
        image_hash = ret.image_hashes
        ret.pad_values = [
            (image_hash) % vocab_size,
            (image_hash >> 16) % vocab_size,
            (image_hash >> 32) % vocab_size,
            (image_hash >> 64) % vocab_size,
        ]
        optional_args = [
            "image_sizes",
            "modalities",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "image_grid_thws",
        ]
        for arg in optional_args:
            if arg in obj:
                setattr(ret, arg, obj[arg])
        return ret


class Req:
    """ 跟踪请求生命周期 """
    
    def __init__(
        self,
        rid: str,
        origin_input_text: str, # 初始字符串
        origin_input_ids: Tuple[int], # 初始token列表
        sampling_params: SamplingParams,
        lora_path: Optional[str] = None,
        arrival_time: Optional[float] = None,
        slo: Optional[float] = None,
    ):
        # 生命周期信息
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids_unpadded = origin_input_ids # 后续需要对image padding
        self.origin_input_ids = origin_input_ids
        self.output_ids = []  # Decode每阶段的输出token id
        self.fill_ids = None  # fill_ids = origin_input_ids + output_ids，用于前缀匹配
        self.sampling_params = sampling_params
        self.lora_path = lora_path
        self.arrival_time = arrival_time
        self.slo = slo
        self.out_queue_timestamp = None # 调度器dequeue时间戳
        self.prefill_finish_timestamp = None # Prefill阶段完成时间戳
        self.decode_timestamps = [] # Decode每阶段完成时间戳，包含finish_timestamp
        self.finish_timestamp = None # 全阶段完成时间戳
        
        # 内存池信息
        self.req_pool_idx = None
        self.prefix_indices = [] #
        self.extend_input_len = 0
        self.last_node = None
        self.is_inflight_req = 0
        self.cached_tokens = 0
        
        # 请求检查
        self.tokenizer = None
        self.finished_reason = None

        # 流式输出解码策略
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.stream = False
        self.vid = 0  # version id to sync decode status with in detokenizer_manager
        self.decoded_text = ""
        self.read_offset = None # 已读偏移量，新文本开始
        self.surr_offset = None # 周围偏移量，确保detokenize包含足够上下文
        
        # Logprobs (arguments) 用于日志输出token概率信息，对齐OpenAI API
        self.return_logprob = False
        self.logprob_start_len = 0
        self.top_logprobs_num = 0
        # Logprobs (internal values)
        self.last_update_decode_tokens = 0
        self.extend_logprob_start_len = 0 # extend部分中，需要计算输出概率的起始位置
        # Logprobs (return value)
        self.normalized_prompt_logprob = None
        self.input_token_logprobs = None
        self.input_top_logprobs = None
        self.output_token_logprobs = []
        self.output_top_logprobs = []
        
        # 正则输出约束，指定字符串token用跳跃输出补全
        self.regex_fsm: RegexGuide = None
        self.regex_fsm_state: int = 0
        self.jump_forward_map: JumpForwardMap = None
        
        # 其他功能字段
        self.embedding = None # 用于embedding请求
        self.image_inputs: Optional[ImageInputs] = None # 图像输入
        self.mrope_position_delta = [] # Qwen2-VL专用，M-ROPE位置偏移
        self.completion_tokens_wo_jump_forward = 0 # 推理Decode token数，不含跳跃输出

    def finished(self) -> bool:
        return self.finished_reason is not None

    def init_next_round_input(self, tree_cache: Optional[BasePrefixCache] = None):
        """
        匹配并获取token前缀缓存地址
        Chunked Prefill阶段
            |--------------- fill ids ---------------|
            | xxxxxxxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
            | ---------- ^ ----------- ^ ----------- |
            |-- prefix --|-- extend1 --|-- extend2 --|
            |   cached   |   prefill   |    idle     |
            |--------- prefix ---------|-- extend1 --|
            |          cached          |   prefill   |
            |---------------- prefix ----------------|
        Decode阶段
            用fill ids前 n-1 个token作为前缀，复用缓存
            第 n-1 个token为新输入，需要计算
            输出第 n+1 个token
        """
        self.fill_ids = self.origin_input_ids + self.output_ids
        if tree_cache is not None:
            # chunked cache不进行匹配，直接获取对应长度的token缓存地址
            self.prefix_indices, self.last_node = tree_cache.match_prefix(
                rid=self.rid,
                key=self.adjust_max_prefix_ids() # fill_ids[:-1]
            )
        self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)

    def adjust_max_prefix_ids(self):
        """不输出logprob时，fill_id除最后一个token外，全部用于匹配"""
        self.fill_ids = self.origin_input_ids + self.output_ids
        cached_len = len(self.fill_ids)
        max_prefix_len = cached_len - 1
        if self.return_logprob:
            if self.normalized_prompt_logprob is None:
                max_prefix_len = min(max_prefix_len, cached_len - 2)
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        max_prefix_len = max(max_prefix_len, 0)
        return self.fill_ids[:max_prefix_len]

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        """计算流式输出增量偏移"""
        first_iter = self.surr_offset is None or self.read_offset is None
        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            # 初始read_offset为输出起点，surr_offset提前一小段避免断裂
            self.surr_offset = max(self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
        all_ids = self.origin_input_ids_unpadded + self.output_ids
        return all_ids[self.surr_offset:], self.read_offset - self.surr_offset

    def get_next_inc_detokenization(self):
        """流式输出增量解码"""
        if self.tokenizer is None: return False, ""
        # 从周围偏移量起始的token id，该token序列对应的已读偏移量
        read_ids, surr_len = self.init_incremental_detokenize()
        surr_ids = read_ids[:surr_len]
        surr_text = self.tokenizer.decode(
            surr_ids,
            skip_special_tokens=self.sampling_params.skip_special_tokens,
            spaces_between_special_tokens=self.sampling_params.spaces_between_special_tokens,
        )
        new_text = self.tokenizer.decode(
            read_ids,
            skip_special_tokens=self.sampling_params.skip_special_tokens,
            spaces_between_special_tokens=self.sampling_params.spaces_between_special_tokens,
        )
        if len(new_text) > len(surr_text) and not new_text.endswith("�"):
            # 成功解码
            return True, new_text[len(surr_text) :]
        return False, ""

    def set_out_queue_time(self):
        self.out_queue_timestamp = time.time()

    def set_prefill_finish_time(self):
        self.prefill_finish_timestamp = time.time()

    def set_finish_time(self):
        self.finish_timestamp = time.time()

    def append_decode_time(self):
        decode_time = time.time()
        self.decode_timestamps.append(decode_time)

    def check_finished(self):
        """检查终止条件"""
        if self.finished(): return
        # 长度超限终止
        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(length=self.sampling_params.max_new_tokens)
            return
        # EOS输出终止
        last_token_id = self.output_ids[-1]
        matched_eos = False
        if self.sampling_params.stop_token_ids:
            matched_eos = last_token_id in self.sampling_params.stop_token_ids
        if self.tokenizer is not None:
            matched_eos |= last_token_id == self.tokenizer.eos_token_id
            if self.tokenizer.additional_stop_token_ids:
                matched_eos |= last_token_id in self.tokenizer.additional_stop_token_ids
        if matched_eos and not self.sampling_params.ignore_eos:
            self.finished_reason = FINISH_MATCHED_TOKEN(matched=last_token_id)
            return
        # 指定字符终止
        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(self.output_ids[-(self.sampling_params.stop_str_max_len + 1):])
            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str or stop_str in self.decoded_text:
                    self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                    return

    def jump_forward_and_retokenize(self, jump_forward_str, next_state):
        """跳跃补全token"""
        if self.origin_input_text is None:
            self.origin_input_text = self.tokenizer.decode(self.origin_input_ids_unpadded)
        all_text = self.origin_input_text + self.decoded_text + jump_forward_str
        all_ids = self.tokenizer.encode(all_text)
        if not all_ids:
            logger.warning("Encoded all_text resulted in empty all_ids")
            return False
        prompt_tokens = len(self.origin_input_ids_unpadded)
        if prompt_tokens > len(all_ids):
            logger.warning("prompt_tokens is larger than encoded all_ids")
            return False
        if all_ids[prompt_tokens-1] != self.origin_input_ids_unpadded[-1]: # TODO: 发生token fusion，输入末尾和输出开头混合了
            logger.warning("Token fusion between input and output, try to avoid this by removing the space at the end of the input.")
            return False
        old_output_ids = self.output_ids
        self.output_ids = all_ids[prompt_tokens:]
        self.decoded_text = self.decoded_text + jump_forward_str
        self.read_offset = len(all_ids)
        # NOTE: 确定surr_offset，从末尾逐渐增大decode窗口
        self.surr_offset = prompt_tokens
        for i in range(0, INIT_INCREMENTAL_DETOKENIZATION_OFFSET):
            surr_text_ = self.tokenizer.decode(all_ids[self.read_offset-i: self.read_offset])
            if not surr_text_.endswith("�"):
                self.surr_offset = self.read_offset - i
                break
        # 更新状态机
        self.regex_fsm_state = next_state
        # 更新logprobs输出
        if self.return_logprob:
            k = 0
            for i, old_id in enumerate(old_output_ids):
                if old_id == self.output_ids[i]: k += 1
                else: break
            self.output_token_logprobs = self.output_token_logprobs[:k]
            self.output_top_logprobs = self.output_top_logprobs[:k]
            self.logprob_start_len = prompt_tokens + k
            self.last_update_decode_tokens = len(self.output_ids) - k
        return True

    def __repr__(self):
        return f"rid(n={self.rid}, " f"input_ids={self.origin_input_ids}, "


bid = 0


@dataclasses.dataclass
class ScheduleBatch:
    """Batch相关信息"""
    
    reqs: List[Req] # 请求队列
    
    # 内存池
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: BaseTokenToKVPool = None
    tree_cache: BasePrefixCache = None
    
    # 推理功能
    model_config: ModelConfig = None # 模型参数
    forward_mode: ForwardMode = None # 推理阶段
    sampling_info: SamplingBatchInfo = None # 推理参数
    
    # 批参数
    input_ids: torch.Tensor = None # 所有输入token拼接
    req_pool_indices: torch.Tensor = None # 每个请求的Req->Token Cache地址
    seq_lens: torch.Tensor = None
    seq_lens_sum: int = None # 请求token总和
    
    # 输出地址
    out_cache_loc: torch.Tensor = None # 输出缓冲区：每一步输出token KV Cache（encoder、decoder结果）地址
    output_ids: torch.Tensor = None # 上次decode输出token，作为下次decode输入
    
    # logprobs日志
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None

    # 跟踪extend阶段
    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_num_tokens: int = None
    decoding_reqs: List[Req] = None

    # 非文本encoder
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None # 存储图像输入编码

    # 其他参数
    has_stream: bool = False # 流式输出
    has_regex: bool = False # 正则输出约束
    device: str = "cuda"

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
    ):
        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            tree_cache=tree_cache,
            model_config=model_config,
            return_logprob=any(req.return_logprob for req in reqs),
            has_stream=any(req.stream for req in reqs),
            has_regex=any(req.regex_fsm for req in reqs),
            device=req_to_token_pool.device,
        )

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def alloc_req_slots(self, num_reqs):
        """Req->Token内存池分配slots，每条指令一个slot"""
        req_pool_indices = self.req_to_token_pool.alloc(num_reqs)
        if req_pool_indices is None:
            raise RuntimeError("Out of memory. Please set a smaller number for `--max-running-requests`.")
        return req_pool_indices

    def alloc_token_slots(self, num_tokens: int):
        """Token->KV内存池分配slots，每个token一个slot"""
        out_cache_loc = self.token_to_kv_pool.alloc(num_tokens) # Tensor存储页索引
        # 分配失败，retry 20 times
        if out_cache_loc is None:
            for i in range(20):
                logger.warning(f"Failed to allocate {num_tokens} tokens, retrying...")
                time.sleep(0.001 * i)
                if self.tree_cache is not None:
                    self.tree_cache.evict(num_tokens, self.token_to_kv_pool.free) # pass
                out_cache_loc = self.token_to_kv_pool.alloc(num_tokens)
                if out_cache_loc is not None: break
        if out_cache_loc is None:
            if self.tree_cache is not None:
                self.tree_cache.evict(num_tokens, self.token_to_kv_pool.free) # pass
                out_cache_loc = self.token_to_kv_pool.alloc(num_tokens)
            if out_cache_loc is None: # 根据推理阶段报错
                phase_str = "Prefill" if self.forward_mode.is_extend() else "Decode"
                logger.error(
                    f"{phase_str} out of memory. Try to lower your batch size.\n"
                    f"Try to allocate {num_tokens} tokens.\n"
                    f"Avaliable tokens: {self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()}"
                )
                if self.tree_cache is not None: self.tree_cache.pretty_print() # pass
                exit(1)
        return out_cache_loc

    def prepare_encoder_info_extend(self, input_ids: List[List[int]], seq_lens: List[int]):
        """分离Encoder-Decoder信息
        Args:
            input_ids (List[List[int]]): 多模态输入tokens
            seq_lens (List[int]): 多模态输入长度
        根据请求多模态情况对输入输出分离信息
        图像数据加入cache，此后不参与token统计
        输入input_ids、seq_lens分离encoder部分，只保留纯文本内容
        更新对应prefix、extend统计数据
        分离并重构encoder、decoder cache，按req划分
        """
        # 添加图像信息，prefill阶段应先encode图像并加入prefix
        self.encoder_lens_cpu = []
        self.encoder_cached = []
        for req in self.reqs:
            img = req.image_inputs
            if img is None or img.num_image_tokens is None:
                self.encoder_lens_cpu.append(0)
                self.encoder_cached.append(True)
            else:
                self.encoder_lens_cpu.append(img.num_image_tokens)
                self.encoder_cached.append(self.forward_mode.is_decode() or len(req.prefix_indices) >= img.num_image_tokens)
        self.encoder_lens = torch.tensor(self.encoder_lens_cpu, dtype=torch.int32).to(self.device, non_blocking=True)
        
        # 分离encoder、decoder统计数据
        ptr = 0 # 指向当前请求输出起始位置，可能为图像encode结果或文本extend结果
        encoder_out_cache_loc = []
        decoder_out_cache_loc = []
        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            seq_lens[i] -= encoder_len # 保留纯文本长度
            if len(req.prefix_indices) < encoder_len: # encoder部分未缓存
                assert len(req.prefix_indices) == 0 # encoder必须为整体
                input_ids[i] = input_ids[i][encoder_len:] # 保留纯文本token
                # encoder cache添加encode部分
                encoder_out_cache_loc.append(self.out_cache_loc[ptr: ptr+encoder_len])
                # decoder cache添加extend部分
                decoder_out_cache_loc.append(self.out_cache_loc[ptr+encoder_len: ptr+req.extend_input_len])
                # 保留纯文本extend长度
                self.extend_lens[i] -= encoder_len
                self.extend_num_tokens -= encoder_len
            else: # encoder部分有缓存或无图像输入
                # decoder cache添加extend部分
                decoder_out_cache_loc.append(self.out_cache_loc[ptr: ptr+req.extend_input_len])
                self.prefix_lens[i] -= encoder_len
            ptr += req.extend_input_len
        
        # 重构encoder、decoder cache
        self.input_ids = torch.tensor(sum(input_ids, []), dtype=torch.int32).to(self.device, non_blocking=True) # 拼接输入token
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32).to(self.device, non_blocking=True)
        # 重构输出cache，保留decoder输出，按请求划分
        if not decoder_out_cache_loc:
            self.out_cache_loc = torch.empty(0, dtype=torch.int32).to(self.device, non_blocking=True)
        else:
            self.out_cache_loc = torch.cat(decoder_out_cache_loc)
        assert len(self.out_cache_loc) == self.extend_num_tokens
        # 重构图像cache，按请求划分
        if not encoder_out_cache_loc:
            self.encoder_out_cache_loc = torch.empty(0, dtype=torch.int32).to(self.device, non_blocking=True)
        else:
            self.encoder_out_cache_loc = torch.cat(encoder_out_cache_loc)

    def prepare_for_extend(self):
        """extend阶段准备
        分配显存，维护显存映射
        更新统计数据
        """
        self.forward_mode = ForwardMode.EXTEND
        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices):] for r in reqs] # 待extend token
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []
        # 显存分配
        req_pool_indices = self.alloc_req_slots(len(self.reqs))
        out_cache_loc = self.alloc_token_slots(extend_num_tokens)
        
        # 更新prefixed tokens
        ptr = 0
        for i, req in enumerate(reqs):
            prefix_len, seq_len = len(req.prefix_indices), len(req.fill_ids)
            already_computed = (
                req.extend_logprob_start_len + 1 + req.cached_tokens
                if req.extend_logprob_start_len > 0 else 0
            )
            req.cached_tokens += max(prefix_len - already_computed, 0)
            req.req_pool_idx = req_pool_indices[i]
            seq_lens.append(seq_len)
            assert seq_len - prefix_len == req.extend_input_len
            # encode结果写入缓存，写入prefixed token地址
            if prefix_len > 0:
                self.req_to_token_pool.write((req.req_pool_idx, slice(0, prefix_len)), req.prefix_indices)
            # encode结果写入缓存，写入当前extend部分token地址
            self.req_to_token_pool.write((req.req_pool_idx, slice(prefix_len, seq_len)), out_cache_loc[ptr: ptr+req.extend_input_len])
            # 更新logprob_start_len
            if req.logprob_start_len >= prefix_len:
                req.extend_logprob_start_len = min(req.logprob_start_len - prefix_len, req.extend_input_len - 1)
            else:
                req.extend_logprob_start_len = req.extend_input_len - 1
            ptr += req.extend_input_len
            
        # 向量化加载
        self.input_ids = torch.tensor(sum(input_ids, []), dtype=torch.int32).to(self.device, non_blocking=True)
        self.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int32).to(self.device, non_blocking=True)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32).to(self.device, non_blocking=True)
        # 输出地址向量映射
        self.out_cache_loc = out_cache_loc
        # 更新统计数据
        self.seq_lens_sum = sum(seq_lens)
        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [len(r.prefix_indices) for r in reqs]
        self.extend_lens = [r.extend_input_len for r in reqs]
        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        if self.return_logprob: self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
        if self.model_config.is_encoder_decoder:
            # 分离encoder-decoder输出，只对mllama模型调用
            self.prepare_encoder_info_extend(input_ids, seq_lens)
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
            global_server_args_dict["disable_penalizer"],
        )

    def mix_with_running(self, running_batch: "ScheduleBatch"):
        """将当前batch与decoding batch合并运行
        Args:
            running_batch (ScheduleBatch): 待合并running batch
        """
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()
        for req in running_batch.reqs:
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.extend_input_len = 1 # AR阶段每次新增一个extend token
        # 合并地址映射
        input_ids = torch.cat([self.input_ids, running_batch.input_ids])
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])
        # 合并批次信息
        logger.info(f"Merging new batch {self} with running batch {running_batch}")
        self.merge_batch(running_batch)
        self.input_ids = input_ids
        self.out_cache_loc = out_cache_loc
        self.extend_num_tokens += running_bs
        # AR阶段每次新增一个extend token，该token不在cache中
        self.prefix_lens.extend([len(r.origin_input_ids) + len(r.output_ids) - 1 for r in running_batch.reqs])
        self.extend_lens.extend([1] * running_bs)
        self.extend_logprob_start_lens.extend([0] * running_bs)

    def check_decode_mem(self):
        """检查剩余KV Cache页数是否足够分配"""
        bs = len(self.reqs)
        if self.token_to_kv_pool.available_size() >= bs: return True
        # chunked cache时pass
        self.tree_cache.evict(bs, self.token_to_kv_pool.free)
        if self.token_to_kv_pool.available_size() >= bs: return True
        return False

    def retract_decode(self) -> Tuple[List[Req], float]:
        """撤回decode请求以释放空间"""
        sorted_indices = [i for i in range(len(self.reqs))]
        # 当前请求按输出长度倒序，输入长度正序排列
        sorted_indices.sort(
            key=lambda i: (len(self.reqs[i].output_ids), -len(self.reqs[i].origin_input_ids)),
            reverse=True,
        )
        retracted_reqs = []
        seq_lens_cpu = self.seq_lens.cpu().numpy()
        first_iter = True
        retract_decode_steps = global_config.retract_decode_steps # 20
        while (self.token_to_kv_pool.available_size() < len(sorted_indices) * global_config.retract_decode_steps or first_iter):
            # comment out for now and it seems to be working fine
            if len(sorted_indices) == 1 and retract_decode_steps > 1:
                logger.warning("RetractDecode: Only one request in the batch, sleep a while and try again with half steps")
                time.sleep(PHYSICAL_MEM_CHECK_FREQ)  # sleep a while and try again
                retract_decode_steps = max(1, retract_decode_steps // 2)
                continue
            if len(sorted_indices) == 0: break
            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            logger.info(
                f"RetractDecode: Req {idx} "
                f"(input={len(req.origin_input_ids)}, output={len(req.output_ids)}) "
                f"is retracted to release memory"
            )
            retracted_reqs.append(req)
            if isinstance(self.tree_cache, ChunkCache):
                # 释放占用内存
                token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :seq_lens_cpu[idx]]
                self.req_to_token_pool.free(req.req_pool_idx)
                self.token_to_kv_pool.free(token_indices)
                if req.rid in self.tree_cache.entries:
                    del self.tree_cache.entries[req.rid]
            else:
                # TODO: apply more fine-grained retraction
                last_uncached_pos = len(req.prefix_indices)
                token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, last_uncached_pos:seq_lens_cpu[idx]]
                self.token_to_kv_pool.free(token_indices)
                self.req_to_token_pool.free(req.req_pool_idx)
                # release the last node
                self.tree_cache.dec_lock_ref(req.last_node)
                # NOTE(lsyin): we should use the newly evictable memory instantly.
                residual_size = (len(sorted_indices) * retract_decode_steps - self.token_to_kv_pool.available_size())
                residual_size = max(0, residual_size)
                self.tree_cache.evict(residual_size, self.token_to_kv_pool.free)
            req.prefix_indices = []
            req.last_node = None
            req.extend_input_len = 0
            # For incremental logprobs
            req.last_update_decode_tokens = 0
            req.logprob_start_len = 10**9
        # 更新batch信息
        self.filter_batch(keep_indices=sorted_indices) # 过滤撤回请求
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)
        if total_max_new_tokens == 0:
            new_estimate_ratio = 0.0
            logger.warning("Total max_new_tokens is 0, set new_estimate_ratio to 0.0")
        else:
            new_estimate_ratio = (total_decoded_tokens + retract_decode_steps * len(self.reqs)) / total_max_new_tokens
        new_estimate_ratio = min(1.0, new_estimate_ratio)
        return retracted_reqs, new_estimate_ratio

    def check_for_jump_forward(self, pad_input_ids_func):
        jump_forward_reqs = []
        keep_indices = set(i for i in range(len(self.reqs)))

        for i, req in enumerate(self.reqs):
            if req.jump_forward_map is not None:
                jump_forward_bytes = req.jump_forward_map.jump_forward_byte(
                    req.regex_fsm_state
                )
                if jump_forward_bytes is not None and len(jump_forward_bytes) > 1:
                    suffix_bytes = []
                    continuation_range = range(0x80, 0xC0)
                    cur_state = req.regex_fsm_state
                    while (
                        len(jump_forward_bytes)
                        and jump_forward_bytes[0][0] in continuation_range
                    ):
                        # continuation bytes
                        byte_edge = jump_forward_bytes.pop(0)
                        suffix_bytes.append(byte_edge[0])
                        cur_state = byte_edge[1]

                    suffix_tokens = [f"<0x{hex(b)[2:].upper()}>" for b in suffix_bytes]
                    suffix_ids = req.tokenizer.convert_tokens_to_ids(suffix_tokens)

                    # Current ids, for cache and revert
                    cur_all_ids = tuple(req.origin_input_ids + req.output_ids)[:-1]
                    cur_output_ids = req.output_ids

                    req.output_ids.extend(suffix_ids)
                    decode_res, new_text = req.get_next_inc_detokenization()
                    if not decode_res:
                        req.output_ids = cur_output_ids
                        continue

                    (
                        jump_forward_str,
                        next_state,
                    ) = req.jump_forward_map.jump_forward_symbol(cur_state)

                    # Make the incrementally decoded text part of jump_forward_str
                    # so that the UTF-8 will not corrupt
                    jump_forward_str = new_text + jump_forward_str
                    if not req.jump_forward_and_retokenize(
                        jump_forward_str, next_state
                    ):
                        req.output_ids = cur_output_ids
                        continue

                    # The decode status has diverged from detokenizer_manager
                    req.vid += 1

                    # insert the old request into tree_cache
                    self.tree_cache.cache_finished_req(req, cur_all_ids)

                    # re-applying image padding
                    if req.image_inputs is not None:
                        req.origin_input_ids = pad_input_ids_func(
                            req.origin_input_ids_unpadded, req.image_inputs
                        )

                    jump_forward_reqs.append(req)
                    keep_indices.remove(i)

        self.filter_batch(keep_indices=list(keep_indices))

        return jump_forward_reqs

    def prepare_encoder_info_decode(self):
        """重置encoder cache状态（清空图像编码缓存）"""
        self.encoder_cached = [True] * len(self.reqs)

    def prepare_for_decode(self, enable_overlap: bool = False):
        """decode阶段准备
        分配显存
        更新显存映射
        """
        self.forward_mode = ForwardMode.DECODE
        # AR式生成，上一步decode输出为下一步decode输入
        self.input_ids = self.output_ids
        self.output_ids = None
        if self.sampling_info.penalizer_orchestrator:
            self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(self.input_ids)
        # 显存分配
        bs = len(self.reqs)
        self.out_cache_loc = self.alloc_token_slots(bs)

        # decode结果写入缓存，写入每句最后一token位置
        if self.model_config.is_encoder_decoder:
            # 分离encoder-decoder输出，只对mllama模型调用
            locs = self.encoder_lens + self.seq_lens
            self.prepare_encoder_info_decode()
        else:
            locs = self.seq_lens
        self.req_to_token_pool.write((self.req_pool_indices, locs), self.out_cache_loc)
        # 更新尾部token指针
        if enable_overlap:
            # Do not use in-place operations in the overlap mode
            self.seq_lens = self.seq_lens + 1
        else:
            # A faster in-place version
            self.seq_lens.add_(1)
        self.seq_lens_sum += bs

    def filter_batch(
        self,
        current_inflight_req: Optional[List[Req]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        """过滤请求，仅保留非inflight请求或按索引保留"""
        if keep_indices is None: # 所有非inflight请求索引
            if current_inflight_req:
                keep_indices = [
                    i for i in range(len(self.reqs))
                    if not self.reqs[i].finished()
                    and self.reqs[i] not in current_inflight_req
                ]
            else:
                keep_indices = [i for i in range(len(self.reqs)) if not self.reqs[i].finished()]
        if len(keep_indices) == 0: # 过滤全部请求
            self.reqs = []
            return
        if len(keep_indices) == len(self.reqs): return # 无过滤请求

        # 对剩余请求更新数据
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]
        self.reqs = [self.reqs[i] for i in keep_indices]
        new_indices = torch.tensor(keep_indices, dtype=torch.int32).to(self.device, non_blocking=True)
        self.req_pool_indices = self.req_pool_indices[new_indices]
        self.seq_lens = self.seq_lens[new_indices]
        self.out_cache_loc = None
        self.seq_lens_sum = self.seq_lens.sum().item()
        self.output_ids = self.output_ids[new_indices]
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
        self.has_stream = any(req.stream for req in self.reqs)
        self.has_regex = any(req.regex_fsm for req in self.reqs)
        self.sampling_info.filter_batch(keep_indices, new_indices)

    def merge_batch(self, other: "ScheduleBatch"):
        """合并batch"""
        # TODO: Penalizer orchestrator must be merged before Batch.reqs is merged. This is because orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it needs to be called with pre-merged Batch.reqs.
        self.sampling_info.merge_batch(other.sampling_info)

        # 只对mllama模型调用，合并encoder信息
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = torch.cat([self.encoder_lens, other.encoder_lens])
            self.encoder_lens_cpu.extend(other.encoder_lens_cpu)
        
        # 合并请求列表、缓存索引及信息
        self.reqs.extend(other.reqs)
        self.req_pool_indices = torch.concat([self.req_pool_indices, other.req_pool_indices])
        self.seq_lens = torch.concat([self.seq_lens, other.seq_lens])
        self.out_cache_loc = None # 清空输出缓冲区
        self.seq_lens_sum += other.seq_lens_sum
        if self.output_ids is not None:
            self.output_ids = torch.concat([self.output_ids, other.output_ids])
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums.extend(other.top_logprobs_nums)
        elif self.return_logprob:
            self.top_logprobs_nums.extend([0] * len(other.reqs))
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
        self.return_logprob = self.return_logprob or other.return_logprob
        self.has_stream = self.has_stream or other.has_stream
        self.has_regex = self.has_regex or other.has_regex

    def get_model_worker_batch(self):
        """创建ModelWorkerBatch对象"""
        if self.forward_mode.is_decode(): # decode阶段清空extend信息
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            extend_seq_lens = self.extend_lens
            extend_prefix_lens = self.prefix_lens
            extend_logprob_start_lens = self.extend_logprob_start_lens

        if self.has_regex:
            self.sampling_info.regex_fsms = [req.regex_fsm for req in self.reqs]
            self.sampling_info.regex_fsm_states = [req.regex_fsm_state for req in self.reqs]
        else:
            self.sampling_info.regex_fsms = None
        mrope_positions_delta = [req.mrope_position_delta for req in self.reqs]
        # 更新全局batch id
        global bid
        bid += 1
        return ModelWorkerBatch(
            bid=bid,
            forward_mode=self.forward_mode,
            input_ids=self.input_ids,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            out_cache_loc=self.out_cache_loc,
            seq_lens_sum=self.seq_lens_sum,
            req_to_token_pool_records=self.req_to_token_pool.get_write_records(),
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            image_inputs=[r.image_inputs for r in self.reqs],
            encoder_cached=self.encoder_cached,
            encoder_lens=self.encoder_lens,
            encoder_lens_cpu=self.encoder_lens_cpu,
            encoder_out_cache_loc=self.encoder_out_cache_loc,
            lora_paths=[req.lora_path for req in self.reqs],
            sampling_info=self.sampling_info,
            mrope_positions_delta=mrope_positions_delta,
        )

    def copy(self):
        """创建副本，仅保留process_batch_result所需属性"""
        return ScheduleBatch(
            reqs=self.reqs,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
        )

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name}, "
            f"#req={(len(self.reqs))})"
        )


@dataclasses.dataclass
class ModelWorkerBatch:
    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # The memory pool operation records
    req_to_token_pool_records: Optional[List[Tuple[Tuple, torch.Tensor]]]

    # For logprob
    return_logprob: bool
    top_logprobs_nums: Optional[List[int]]

    # For extend
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]
    extend_logprob_start_lens: Optional[List[int]]

    # For multimodal
    image_inputs: Optional[List[ImageInputs]]

    # For encoder-decoder
    encoder_cached: Optional[List[bool]]
    encoder_lens: Optional[torch.Tensor]
    encoder_lens_cpu: Optional[List[int]]
    encoder_out_cache_loc: Optional[torch.Tensor]

    # For LoRA
    lora_paths: Optional[List[str]]

    # Sampling info
    sampling_info: SamplingBatchInfo

    # For Qwen2-VL
    mrope_positions_delta: List[List[int]]

    def copy(self):
        return dataclasses.replace(self, sampling_info=self.sampling_info.copy())

    def to(self, device: str):
        self.input_ids = self.input_ids.to(device, non_blocking=True)
        self.req_pool_indices = self.req_pool_indices.to(device, non_blocking=True)
        self.seq_lens = self.seq_lens.to(device, non_blocking=True)
        self.out_cache_loc = self.out_cache_loc.to(device, non_blocking=True)
        self.req_to_token_pool_records = [
            (x, y.to(device, non_blocking=True))
            for x, y in self.req_to_token_pool_records
        ]
        self.sampling_info.to(device)
