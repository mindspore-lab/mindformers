# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Run LanguageModelEmbedding accuracy test with configurable parameters via args"""
import argparse
import os
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore.communication import init
from mindformers.parallel_core.training_graph.transformer.language_model import LanguageModelEmbedding
from mindformers.parallel_core.transformer_config import TransformerConfig
from data_gen_utils import get_init_params

SCRIPT_DIR = Path(__file__).parent.resolve()
HIDDEN_SIZE = 100

class LanguageModelEmbeddingRunner:
    """Class to manage LanguageModelEmbedding model and weights"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.vocab_size = self.args.vocab_size
        self.max_sequence_length = self.args.max_sequence_length
        self.position_embedding_type = self.args.position_embedding_type
        self.num_tokentypes = self.args.num_tokentypes

        self.compute_dtype = ms.bfloat16
        self.param_init_dtype = ms.float32

        init_params = get_init_params(self.vocab_size, HIDDEN_SIZE)

        self.net_weight = init_params.get("weight")
        self.input_ids = ms.Tensor(init_params.get("input_ids"), dtype=ms.int32)
        self.position_ids = ms.Tensor(init_params.get("position_ids"), dtype=ms.int32)
        if self.num_tokentypes > 0:
            self.tokentype_ids = ms.Tensor(init_params.get("tokentype_ids"), dtype=ms.int32)
        else:
            self.tokentype_ids = None


        # RANK_ID and worker_num are set by msrun environment
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))


        # Set parallel context
        if self.rank_id is not None:
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True
            )
            init()

        # Transformer config
        self.config = TransformerConfig(
            data_parallel_size=self.worker_num // self.args.tensor_parallel,
            tensor_model_parallel_size=self.args.tensor_parallel,
            hidden_size=HIDDEN_SIZE,
            compute_dtype=self.compute_dtype,
            params_dtype=self.param_init_dtype,
            num_attention_heads=self.args.tensor_parallel,
            num_layers=1
        )

    def build_model(self):
        """Build and initialize LanguageModelEmbedding model"""
        net = LanguageModelEmbedding(
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
            config=self.config,
            position_embedding_type=self.position_embedding_type,
            num_tokentypes=self.num_tokentypes,
        )
        state_dict = {
            "word_embeddings.weight": ms.Parameter(self.net_weight)
        }
        ms.load_param_into_net(net, state_dict)
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

        output = net(self.input_ids, self.position_ids, self.tokentype_ids)
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {
                k: v.asnumpy().astype(np.float32)
                for k, v in output_ms.items()
                if v is not None
            }
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run LanguageModelEmbedding test")
    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument("--max_sequence_length", type=int, default=64)
    parser.add_argument("--num_tokentypes", type=int, default=0)
    parser.add_argument("--position_embedding_type", type=str, default="none")
    parser.add_argument("--output_path", type=str, default="output_ms.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)

    args = parser.parse_args()

    ms.set_deterministic(True)
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_seed(42)

    # Prepare input
    runner = LanguageModelEmbeddingRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
