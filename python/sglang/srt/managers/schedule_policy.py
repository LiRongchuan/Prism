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

"""Request scheduler policy"""

import os
import random
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import Dict, List, Optional

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.radix_cache import TreeNode

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS = int(os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS", "512"))
CHUNK_ENABLE_THRESHOLD = 100

class SchedulePolicy:
    def __init__(self, policy: str, tree_cache: BasePrefixCache):
        if tree_cache.disable and policy in ["lpm", "dfs-weight"]:
            # LPM and DFS-weight is meaningless when the tree cache is disabled.
            policy = "fcfs"
        self.policy = policy
        self.tree_cache = tree_cache

    def calc_priority(self, waiting_queue: List[Req]):
        # Compute matched prefix length
        prefix_computed = False
        if self.policy == "lpm" or self.policy == "dfs-weight":
            for r in waiting_queue:
                # NOTE: the prefix_indices must always be aligned with last_node
                r.prefix_indices, r.last_node = self.tree_cache.match_prefix(rid=r.rid, key=r.adjust_max_prefix_ids())
            prefix_computed = True

        if self.policy == "lpm":
            # Longest Prefix Match
            waiting_queue.sort(key=lambda x: -len(x.prefix_indices))
        elif self.policy == "fcfs":
            # first come first serve
            pass
        elif self.policy == "lof":
            # longest output first
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)
        elif self.policy == "random":
            random.shuffle(waiting_queue)
        elif self.policy == "dfs-weight":
            last_node_to_reqs = defaultdict(list)
            for req in waiting_queue:
                last_node_to_reqs[req.last_node].append(req)

            node_to_weight = defaultdict(int)
            for node in last_node_to_reqs:
                node_to_weight[node] = len(last_node_to_reqs[node])
            self.calc_weight(self.tree_cache.root_node, node_to_weight)

            waiting_queue.clear()
            self.get_dfs_priority(
                self.tree_cache.root_node,
                node_to_weight,
                last_node_to_reqs,
                waiting_queue,
            )
        else:
            raise ValueError(f"Unknown schedule_policy: {self.policy}")

        return prefix_computed

    def calc_weight(self, cur_node: TreeNode, node_to_weight: Dict):
        for child in cur_node.children.values():
            self.calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    def get_dfs_priority(
        self,
        cur_node: TreeNode,
        node_to_priority: Dict,
        last_node_to_reqs: Dict,
        q: List,
    ):
        childs = [child for child in cur_node.children.values()]
        childs.sort(key=lambda x: -node_to_priority[x])
        for child in childs:
            self.get_dfs_priority(child, node_to_priority, last_node_to_reqs, q)
        q.extend(last_node_to_reqs[cur_node])


class AddReqResult(Enum):
    CONTINUE = auto()  # 可继续添加请求
    NO_TOKEN = auto()  # 无剩余token
    OTHER = auto()     # 不可添加，原因未知


class PrefillAdder:
    def __init__(
        self,
        tree_cache: BasePrefixCache,
        running_batch: ScheduleBatch, # 当前decode batch
        new_token_ratio: float, # 0.1
        rem_total_tokens: int, # KV Cache剩余空间 (token数)
        max_prefill_tokens: int, # 服务器参数max_prefill_tokens = 16384
        chunked_prefill_size: Optional[int], # chunk大小
        mixed_with_decode_tokens: int = 0, # decoding请求数
        enable_elastic_memory: bool = False,
    ):
        self.tree_cache = tree_cache
        self.running_batch = running_batch
        # 限制参数
        self.new_token_ratio = new_token_ratio
        # TODO: 可能保守
        self.cur_rem_tokens = rem_total_tokens - mixed_with_decode_tokens   # 剩余总token限制 (运行限制), 真实跟踪
        self.rem_total_tokens = rem_total_tokens - mixed_with_decode_tokens # 剩余总token限制 (运行限制), 用于估计
        self.rem_input_tokens = max_prefill_tokens - mixed_with_decode_tokens # 输入总token限制 (系统限制)
        self.chunked_prefill_size = chunked_prefill_size
        # if self.chunked_prefill_size is not None:
        #     self.chunked_prefill_size -= mixed_with_decode_tokens
        self.enable_elastic_memory = enable_elastic_memory
        self.token_to_kv_pool = tree_cache.token_to_kv_pool
        # 调度相关
        self.req_states = None # 全部请求 [(最大剩余输出, 当前token占用)]，按最大剩余输出升序排列
        self.can_run_list: List[Req] = []
        self.new_inflight_req: List[Req] = []
        self.log_hit_tokens = 0
        self.log_input_tokens = 0
        if running_batch is not None:
            # 为当前decode请求估计并预留空间
            self.rem_total_tokens -= sum([
                min(r.sampling_params.max_new_tokens - len(r.output_ids), CLIP_MAX_NEW_TOKENS)
                * self.new_token_ratio
                for r in running_batch.reqs
            ])

    def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN
        if self.rem_input_tokens <= 0 or (self.chunked_prefill_size is not None and self.chunked_prefill_size <= 0):
            # 输入超出限制
            return AddReqResult.OTHER
        return AddReqResult.CONTINUE

    def _prefill_one_req(self, prefix_len: int, extend_input_len: int, max_new_tokens: int):
        """对待prefill请求更新内存预算
        Args:
            prefix_len (int): 请求已缓存输入长度
            extend_input_len (int): 当前chunk待缓存输入长度
            max_new_tokens (int): 最大输出，因chunk导致prefill不完整则为0
        """
        # 扣除未缓存输入预算
        self.cur_rem_tokens -= extend_input_len # Prefill后剩余内存
        self.rem_total_tokens -= extend_input_len + max_new_tokens # 预计Prefill + Decode剩余内存
        self.rem_input_tokens -= extend_input_len
        # if self.chunked_prefill_size is not None:
        #     self.chunked_prefill_size -= extend_input_len
        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len

    def add_inflight_req(self, inflight_reqs: List[Req]) -> List[Req]:
        """对chunked prefill请求更新内存预算及请求数据
        Args:
            req (Req): 待prefill请求
        Returns:
            List[Req]: chunked导致不能完整prefill则返回原请求
        """
        truncated_reqs: List[Req] = []
        for req in inflight_reqs:
            truncated = req.extend_input_len > self.chunked_prefill_size # 截出一个chunk的输入token
            req.extend_input_len = min(req.extend_input_len, self.chunked_prefill_size)
            req.fill_ids = req.fill_ids[:len(req.prefix_indices) + req.extend_input_len]
            self.can_run_list.append(req)
            self._prefill_one_req(
                len(req.prefix_indices),
                req.extend_input_len,
                (min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS) if not truncated else 0)
            )
            if truncated: truncated_reqs.append(req) # 因chunked被截断则返回原请求
        return truncated_reqs

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        """节点锁，chunked cache不起作用"""
        try:
            delta = self.tree_cache.inc_lock_ref(last_node)
            self.rem_total_tokens += delta
            yield None
        finally:
            delta = self.tree_cache.dec_lock_ref(last_node)
            self.rem_total_tokens += delta

    def add_one_req_ignore_eos(self, req: Req):
        """添加不可终止请求，需要保守的性能估计"""
        def add_req_state(req: Req, insert_sort=False):
            """req_state队列插入请求"""
            new_token_ratio = 1.0 if req.sampling_params.ignore_eos else self.new_token_ratio
            tokens_left = req.sampling_params.max_new_tokens * new_token_ratio - len(req.output_ids) # 最大剩余输出
            tokens_occupied = len(req.origin_input_ids) + len(req.output_ids) # 当前输入+输出占用token
            # 插入队列
            if tokens_left > 0:
                if not insert_sort:
                    self.req_states.append((tokens_left, tokens_occupied))
                else: # 按剩余输出升序排列
                    for i in range(len(self.req_states)):
                        if tokens_left <= self.req_states[i][0]:
                            self.req_states.insert(i, (tokens_left, tokens_occupied))
                            return
                    self.req_states.append((tokens_left, tokens_occupied))
        # 维护请求预算队列
        if self.req_states is None: # 初始化，插入running batch及待prefill队列全部请求
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            # self.req_states.sort(key=lambda x: x[0])
        else: # 维护队列，插入新请求
            # add_req_state(req, insert_sort=True)
            add_req_state(req)
        # 动态预测剩余显存预算
        cur_rem_tokens = self.cur_rem_tokens - len(req.origin_input_ids)
        tokens_freed = 0 # 随请求结束而释放的token
        # 对队列中每条指令检查显存是否会溢出
        for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
            # 始于当前req的batch最大占用 < 剩余显存
            decode_steps = self.req_states[i+1][0] if i + 1 < len(self.req_states) else tokens_left
            bs = len(self.req_states) - i
            if cur_rem_tokens + tokens_freed - decode_steps * bs <= 0: return AddReqResult.NO_TOKEN
            tokens_freed += tokens_occupied
        # 更新预算及请求数据
        if (self.chunked_prefill_size is None or req.extend_input_len <= self.chunked_prefill_size):
            extend_len = req.extend_input_len
            max_output_len = min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
        else:
            extend_len = self.chunked_prefill_size
            req.extend_input_len = extend_len
            req.fill_ids = req.fill_ids[:extend_len]
            max_output_len = 0
            self.new_inflight_req.append(req)
        self.can_run_list.append(req)
        self._prefill_one_req(0, extend_len, max_output_len)
        return self.budget_state()

    def add_one_req(self, req: Req):
        if req.sampling_params.ignore_eos and self.tree_cache.disable:
            return self.add_one_req_ignore_eos(req)
        # 估计输入/全部预算
        input_tokens = req.extend_input_len
        total_tokens = req.extend_input_len + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
        # 预算不足时添加失败
        if input_tokens > self.rem_input_tokens and len(self.can_run_list) != 0: return AddReqResult.OTHER
        if total_tokens >= self.rem_total_tokens: return AddReqResult.NO_TOKEN
        with self._lock_node(req.last_node):
            if total_tokens > self.rem_total_tokens: return AddReqResult.NO_TOKEN
            prefix_len = len(req.prefix_indices)
            if (
                self.chunked_prefill_size is None
                or input_tokens <= self.chunked_prefill_size
                or (req.return_logprob and req.normalized_prompt_logprob is None)
            ):
                # 整体加入batch
                self.can_run_list.append(req)
                self.tree_cache.inc_lock_ref(req.last_node)
                self._prefill_one_req(
                    prefix_len,
                    input_tokens,
                    min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
                )
            else:
                # 首个chunk加入batch
                trunc_len = self.chunked_prefill_size
                if trunc_len == 0: return AddReqResult.OTHER
                req.extend_input_len = trunc_len
                req.fill_ids = req.fill_ids[:len(req.prefix_indices) + trunc_len]
                self.can_run_list.append(req)
                self.new_inflight_req.append(req)
                self.tree_cache.inc_lock_ref(req.last_node)
                self._prefill_one_req(prefix_len, trunc_len, 0)
        return self.budget_state()
