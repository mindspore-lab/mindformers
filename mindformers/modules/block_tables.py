# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""manage simple block tables for paged attention."""
from typing import List

import numpy as np

from mindformers.tools.logger import logger
from mindformers.modules.cache_engine import BlockMemPool, CacheEngine


class BlockTables:
    """
    The Block Table records on which physical block the key and value of each seq are distributed.
    By dividing the cache of each seq's key and value into fixed size physical blocks,
    each block contains the key and value of several tokens in each sentence.
    Paged Attention obtains the corresponding key and value through the block table and calculates the attention.

    Args:
        num_blocks (int): The count of block.
        block_size (int): The size of block.
        seq_length (int): The seq length.

        Examples:
            >>> num_blocks = 1024
            >>> block_size = 16
            >>> seq_length = 1024
            >>> batch_size = 1
            >>> block_mgr = BlockTables(num_blocks, block_size, seq_length)
            >>> block_mgr.init_cache_engine(batch_size)
    """

    def __init__(self, num_blocks, block_size, seq_length):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.seq_length = seq_length
        self.max_num_blocks_per_seq = self.seq_length // self.block_size
        self.block_mem_pool = BlockMemPool(self.num_blocks, self.block_size)
        self.cache_engines = []

    def init_cache_engine(self, batch_size):
        self.cache_engines.clear()
        if batch_size * self.seq_length // self.block_size > self.num_blocks:
            logger.warning(
                f"Argument `num blocks` is less than the maximum possible block numbers. "
                f"May cause `block pool is out of memory` error")
        for _ in range(batch_size):
            self.cache_engines.append(CacheEngine(self.block_size, self.block_mem_pool))
        logger.info("init cache engine success.")

    def assemble_pa_inputs(self, is_first_iteration, batch_valid_length: np.array, is_finished: List[bool]):
        if is_first_iteration:
            return self._assemble_pa_full_inputs(batch_valid_length, is_finished)
        return self._assemble_pa_inc_inputs(batch_valid_length, is_finished)

    def _assemble_pa_full_inputs(self, batch_valid_length: np.array, is_finished: List[bool]):
        """Prepare prefill inputs for Paged Attention."""
        bs = batch_valid_length.shape[0]

        block_tables = []
        slot_mapping = []
        for i in range(bs):
            if not is_finished[i]:
                logger.info("prepare cache for full: %s", batch_valid_length[i])
                self.cache_engines[i].prepare_cache(batch_valid_length[i] + self.block_size)

            null_block_id = self.cache_engines[i].block_table[0]
            block_table = self.cache_engines[i].block_table[1:]
            padded_table = block_table + [-1 for _ in range(
                self.max_num_blocks_per_seq - len(self.cache_engines[i].block_table) + 1)]
            block_tables.append(padded_table)

            slots = [block_table[k // self.block_size] * self.block_size + k % self.block_size
                     for k in range(batch_valid_length[i])]
            null_slot_idx = null_block_id * self.block_size + null_block_id % self.block_size
            slots = slots + [null_slot_idx for _ in range(self.seq_length - batch_valid_length[i])]
            slot_mapping = slot_mapping + slots
        block_tables = np.array(block_tables, dtype=np.int32)
        slot_mapping = np.array(slot_mapping, dtype=np.int32)
        return block_tables, slot_mapping

    def _assemble_pa_inc_inputs(self, batch_valid_length: np.array, is_finished: List[bool]):
        """Prepare incremental inputs for Paged Attention."""
        bs = batch_valid_length.shape[0]

        block_tables = []
        slot_mapping = []
        for i in range(bs):
            if not is_finished[i]:
                logger.info("prepare cache for inc: %s", batch_valid_length[i])
                self.cache_engines[i].prepare_cache(1)

            block_table = self.cache_engines[i].block_table[1:]
            padded_table = block_table + [-1 for _ in range(
                self.max_num_blocks_per_seq - len(self.cache_engines[i].block_table) + 1)]
            block_tables.append(padded_table)

            curent_idx = batch_valid_length[i] - 1
            slots = [block_table[curent_idx // self.block_size] * self.block_size + curent_idx % self.block_size]
            slot_mapping = slot_mapping + slots
        block_tables = np.array(block_tables, dtype=np.int32)
        slot_mapping = np.array(slot_mapping, dtype=np.int32)
        return block_tables, slot_mapping
