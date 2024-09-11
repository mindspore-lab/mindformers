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
""" Language Model """
import mindspore as ms
from mindspore import mint
from mindspore.nn import Cell
import mindspore.ops as P
import mindspore.nn as nn

from mindformers.experimental.parallel_core.pynative.tensor_parallel import GatherFromSequenceParallelRegion, \
                                                    VocabParallelEmbedding, ScatterToSequenceParallelRegion
from mindformers.experimental.parallel_core.pynative.tensor_parallel.random import get_rng_tracer
from mindformers.experimental.parallel_core.pynative.parallel_state import get_pp_world_size

from .module import Module
from .transformer import ParallelTransformer
from .mlp import ParallelMLP
from .rotary_pos_embedding import RotaryEmbedding


class Pooler(Cell):
    """
    Add a linear transformation to the hidden state corresponding to a specific token.

    Args:
        hidden_size: hidden states size for dense layer
        init_method: dense layer weight init method
        config: model config
    """

    def __init__(self, hidden_size, init_method, config, **kwargs):
        super().__init__(**kwargs)
        param_init_dtype = config.param_init_dtype
        self.dense = nn.Dense(hidden_size,
                              hidden_size,
                              weight_init=init_method,
                              bias_init='zeros',
                              dtype=param_init_dtype,
                              activation='tanh')
        self.sequence_parallel = config.parallel_config.use_sequence_parallel
        self.gather_from_sequence_parallel_region = GatherFromSequenceParallelRegion(
            need_to_swapaxes=False,
            tensor_parallel_output_grad=False
        )

    def construct(self, hidden_states, sequence_index=0):
        """ pooler forward """
        if self.sequence_parallel:
            hidden_states = self.gather_from_sequence_parallel_region(hidden_states)

        pooled = hidden_states[:, sequence_index, :]
        pooled = self.dense(pooled)
        return pooled


class Embedding(Module):
    """
    A embedding layer contain word embedding, position embedding and tokentypes embedding.

    Args:
        - **hidden_size** : hidden states size for embedding layer
        - **vocab_size** : vocabulary size
        - **max_sequence_length** : if using position embedding, it is necessary to set the maximum sequence length
        - **embedding_dropout_prob** : dropout rate for embedding layer
        - **init_method** : embedding layer weight init method
        - **num_tokentypes** : if > 0, using tokentypes embedding

    Outputs:
        - **embeddings** - the embedding output
        - **word_embedding_table** - the word embedding table

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 config,
                 num_tokentypes=0,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_init_dtype = config.param_init_dtype
        self.embedding_init_dtype = config.embedding_init_dtype
        self.compute_dtype = config.compute_dtype
        self.init_method = config.init_method
        self.sequence_parallel = config.parallel_config.use_sequence_parallel
        self.num_tokentypes = num_tokentypes
        self.data_layout = config.dataset_config.data_layout

        # init word embedding
        self.word_embeddings = VocabParallelEmbedding(vocab_size,
                                                      hidden_size,
                                                      config=config,
                                                      init_method=self.init_method,
                                                      param_init_dtype=self.embedding_init_dtype)

        # init position embedding
        self.use_position_embedding = config.position_embedding_type == 'absolute'
        self.parallel_position_embedding = config.parallel_position_embedding
        if self.use_position_embedding:
            if self.parallel_position_embedding:
                self.position_embeddings = VocabParallelEmbedding(max_sequence_length,
                                                                  hidden_size,
                                                                  config=config,
                                                                  init_method=self.init_method,
                                                                  param_init_dtype=self.param_init_dtype)
            else:
                self.position_embeddings = nn.Embedding(max_sequence_length,
                                                        hidden_size,
                                                        embedding_table=self.init_method,
                                                        dtype=ms.int32)

        # init tokentypes embedding
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = nn.Embedding(num_tokentypes,
                                                     hidden_size,
                                                     embedding_table=self.init_method,
                                                     dtype=ms.int32)

        # init dropout
        self.embedding_dropout_prob = embedding_dropout_prob
        self.dropout = mint.nn.Dropout(self.embedding_dropout_prob)

        # init comm op
        self.scatter_to_sequence_parallel_region = ScatterToSequenceParallelRegion(
            need_to_swapaxes=self.data_layout == "BSH"
        )

    def set_zero_parameters(self):
        """ set zero value for all embedding parameters """
        P.assign(self.word_embeddings, P.zeros_like(self.word_embeddings))
        self.word_embeddings.weight.shared = True
        if self.use_position_embedding:
            P.assign(self.position_embeddings, P.zeros_like(self.position_embeddings))
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            P.assign(self.tokentype_embeddings, P.zeros_like(self.tokentype_embeddings))
            self.tokentype_embeddings.weight.shared = True

    def construct(self, input_ids, position_ids, tokentype_ids=None):
        """ embedding layer forward """
        # word embedding
        embeddings = self.word_embeddings(input_ids)

        # position embedding
        if self.use_position_embedding:
            position_embedding = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embedding

        # tokentype embedding
        if tokentype_ids is not None:
            if self.num_tokentypes < 1:
                raise RuntimeError("Embedding layer got 'tokentype_ids' input, "
                                   "but 'tokentype_embeddings' layer is not initialized")
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids)
            embeddings = embeddings + tokentype_embedding
        else:
            if self.num_tokentypes > 0:
                raise RuntimeError("The 'tokentype_ids' input for Embedding layer is None, "
                                   "but 'tokentype_embeddings' layer is initialized")

        if self.data_layout == "SBH":
            embeddings = embeddings.swapaxes(0, 1)

        # dropout
        if self.sequence_parallel:
            embeddings = self.scatter_to_sequence_parallel_region(embeddings)
            with get_rng_tracer().rng_fork():
                embeddings = self.dropout(embeddings)
        else:
            embeddings = self.dropout(embeddings)

        # convert dtype to compute dtype
        embeddings = embeddings.astype(self.compute_dtype)
        return embeddings


class TransformerLanguageModel(Module):
    """
    Transformer language model.

    Args:
        - **config** : model config
        - **encoder_attn_mask_type** : encoder attention mask type
        - **num_tokentypes** : if > 0, using tokentypes embedding
        - **use_encoder** : if True, use encoder
        - **use_decoder** : if True, use decoder
        - **decoder_attn_mask_type** : decoder attention mask type
        - **add_pooler** : if True, use pooler
        - **pre_process** : when using pipeline parallel, indicate whether it's the first stage
        - **post_process** : when using pipeline parallel, indicate whether it's the last stage
        - **visual_encoder** : visual encoder

    Outputs:
        - **encoder_output** - the hidden states
        - **embedding_table** - the word embedding table
        - **pooled_output** - the pooler layer output

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self,
                 config,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_encoder=True,
                 add_decoder=False,
                 decoder_attn_mask_type=None,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True,
                 visual_encoder=None,
                 **kwargs):
        super().__init__(config, **kwargs)
        if add_decoder:
            raise NotImplementedError("use_decoder is not supported for now.")
        if config.use_retriever:
            raise NotImplementedError("retriever is not supported for now.")
        if encoder_attn_mask_type is not None:
            raise NotImplementedError("encoder_attn_mask_type is not supported for now.")
        if decoder_attn_mask_type is not None:
            raise NotImplementedError("decoder_attn_mask_type is not supported for now.")
        if visual_encoder is not None:
            raise NotImplementedError("visual_encoder is not supported for now.")

        self.pre_process = pre_process
        self.post_process = post_process
        self.pipeline_parallel = get_pp_world_size() > 1
        self.use_encoder = add_encoder
        self.num_tokentypes = num_tokentypes
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.use_pooler = add_pooler
        self.encoder_hidden_state = None
        self.init_method = config.init_method
        self.use_decoder = add_decoder

        # get value from config
        self.seq_length = config.seq_length
        self.compute_dtype = config.compute_dtype
        self.untie_embeddings_and_output_weights = config.untie_embeddings_and_output_weights
        post_norm = config.use_final_norm
        param_init_dtype = config.param_init_dtype
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        hidden_dropout_rate = config.hidden_dropout_rate
        num_heads = config.num_heads
        reduce_scatter_embeddings = config.parallel_config.use_sequence_parallel

        if self.pre_process:
            # init embedding layer
            self.embedding = Embedding(hidden_size,
                                       vocab_size,
                                       self.seq_length,
                                       hidden_dropout_rate,
                                       config,
                                       self.num_tokentypes)

            # init visual encoder
            if visual_encoder is not None:
                self.visual_encoder = visual_encoder
                self.visual_mlp = ParallelMLP(config.visual_config)
            else:
                self.visual_encoder = None

        # init rotary embeddings
        self.use_rotary_embeddings = config.position_embedding_type == 'rope'
        if self.use_rotary_embeddings:
            rotary_dim = hidden_size // num_heads
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=rotary_dim,
                rotary_percent=config.rotary_percent,
                rotary_interleaved=config.rotary_interleaved,
                seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
                rotary_base=config.rotary_base)

        # init encoder
        if self.use_encoder:
            self.encoder = ParallelTransformer(config,
                                               model_type=None,
                                               self_attn_mask_type=self.encoder_attn_mask_type,
                                               pre_process=self.pre_process,
                                               post_process=self.post_process,
                                               post_norm=post_norm
                                               )
        else:
            self.encoder = None

        # init pooler
        if self.post_process:
            if self.use_pooler:
                self.pooler = Pooler(hidden_size, self.init_method, config)

            if self.untie_embeddings_and_output_weights or self.pipeline_parallel:
                init_method = self.init_method if self.untie_embeddings_and_output_weights else 'zeros'
                self.output_layer = VocabParallelEmbedding(vocab_size,
                                                           hidden_size,
                                                           config=config,
                                                           init_method=init_method,
                                                           reduce_scatter_embeddings=reduce_scatter_embeddings,
                                                           param_init_dtype=param_init_dtype)

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        if not isinstance(input_tensor, (list, tuple)):
            input_tensor = [input_tensor]

        if len(input_tensor) != 1:
            raise RuntimeError("When using `set_input_tensor` function, "
                               "length of `input_tensor` must be equal to 1")
        self.encoder.set_input_tensor(input_tensor[0])

    def visual_forward(self, input_image):
        """ visual encoder forward """
        n_image = 1
        if input_image.ndim == 5:
            bs, n_image, channel, height, width = input_image.shape
            input_image = input_image.reshape(-1, channel, height, width)
        image_embedding = self.visual_encoder(input_image)
        image_embedding, _ = self.visual_mlp(image_embedding)
        image_embedding = image_embedding.reshape(bs, n_image, image_embedding.shape[1], image_embedding.shape[2])
        image_embedding = image_embedding.astype(self.compute_dtype)
        return image_embedding

    def mixed_embedding(self, text_embedding, image_embedding, delimiter_position):
        """ mixing text embedding and image embedding """
        mix_embeddings = []
        for cur_batch in range(text_embedding.shape[0]):
            mix_embedding = []
            image_num = int(len(delimiter_position[cur_batch]) / 2)
            image_delimiter_position = [i + 1 for i in range(image_num)]
            split_text_embedding = P.tensor_split(text_embedding[cur_batch], delimiter_position[cur_batch], axis=0)
            split_image_embedding = P.tensor_split(image_embedding[cur_batch], image_delimiter_position, axis=0)
            split_image_embedding = [split_image_embedding[i][0] for i in range(image_num)]
            for i in range(len(split_text_embedding)):
                mix_embedding.append(split_text_embedding[i] if i % 2 == 0 \
                                        else split_image_embedding[int((i - 1) / 2)])
            mix_embedding = mint.cat(mix_embedding, dim=0)
            mix_embeddings.append(mix_embedding)
        mix_embeddings = mint.cat(mix_embeddings, dim=0)
        return mix_embeddings

    def construct(self, enc_input_ids, enc_position_ids, enc_attn_mask,
                  dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
                  retriever_input_ids=None, retriever_position_ids=None, retriever_attn_mask=None,
                  enc_dec_attn_mask=None, tokentype_ids=None, inference_params=None,
                  pooling_sequence_index=0, enc_hidden_states=None, output_enc_hidden=False,
                  input_image=None, delimiter_position=None, image_embedding=None):
        """ language model forward """
        self._check_inputs(dec_input_ids, dec_position_ids, dec_attn_mask,
                           retriever_input_ids, retriever_position_ids, retriever_attn_mask,
                           enc_dec_attn_mask, inference_params, output_enc_hidden,
                           input_image, delimiter_position, image_embedding)

        # visual encoder
        image_embedding_out = None
        if self.pre_process:
            if image_embedding is not None:
                image_embedding_out = image_embedding
            else:
                if self.visual_encoder is not None:
                    if input_image is None:
                        raise TypeError("When 'visual_encoder' is not None, 'input_image' can't be None")
                    image_embedding_out = self.visual_forward(input_image)

        # encoder
        text_embedding_out = None
        encoder_input = None
        if self.pre_process:
            text_embedding_out = self.embedding(enc_input_ids, enc_position_ids,
                                                tokentype_ids=tokentype_ids)

            # mix embedding out if image_embedding_out is not None
            # Now, only support below mix order:
            # one_text_embedding -> one_image_embedding (loop)
            if image_embedding_out is None:
                encoder_input = text_embedding_out
            else:
                if delimiter_position is None:
                    raise TypeError("When 'visual_encoder' is not None, 'delimiter_position' can't be None")
                encoder_input = self.mixed_embedding(text_embedding_out, image_embedding_out, delimiter_position)

        # rotary embedding
        rotary_pos_emb = None
        if self.use_rotary_embeddings:
            rotary_pos_emb = self.rotary_pos_emb(self.seq_length)

        # encoder
        encoder_output = None
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(encoder_input,
                                              enc_attn_mask,
                                              retriever_input=None,
                                              retriever_attn_mask=retriever_attn_mask,
                                              inference_params=inference_params,
                                              rotary_pos_emb=rotary_pos_emb)
        else:
            encoder_output = enc_hidden_states.astype(self.compute_dtype)

        # pooler
        if self.post_process and self.use_pooler:
            pooled_output = self.pooler(encoder_output,
                                        pooling_sequence_index)

        if self.use_pooler and self.post_process:
            return encoder_output, pooled_output
        return encoder_output

    def _check_inputs(self, dec_input_ids, dec_position_ids, dec_attn_mask,
                      retriever_input_ids, retriever_position_ids, retriever_attn_mask,
                      enc_dec_attn_mask, inference_params, output_enc_hidden,
                      input_image, delimiter_position, image_embedding):
        """check inputs function"""
        if dec_input_ids is not None:
            raise NotImplementedError("dec_input_ids is not supported for now.")
        if dec_position_ids is not None:
            raise NotImplementedError("dec_position_ids is not supported for now.")
        if dec_attn_mask is not None:
            raise NotImplementedError("dec_attn_mask is not supported for now.")
        if retriever_input_ids is not None:
            raise NotImplementedError("dec_input_ids is not supported for now.")
        if retriever_position_ids is not None:
            raise NotImplementedError("dec_position_ids is not supported for now.")
        if retriever_attn_mask is not None:
            raise NotImplementedError("dec_attn_mask is not supported for now.")
        if enc_dec_attn_mask is not None:
            raise NotImplementedError("enc_dec_attn_mask is not supported for now.")
        if inference_params is not None:
            raise NotImplementedError("inference_params is not supported for now.")
        if output_enc_hidden:
            raise NotImplementedError("output_enc_hidden is not supported for now.")
        if input_image is not None:
            raise NotImplementedError("input_image is not supported for now.")
        if delimiter_position is not None:
            raise NotImplementedError("delimiter_position is not supported for now.")
        if image_embedding is not None:
            raise NotImplementedError("image_embedding is not supported for now.")


def get_language_model(config, num_tokentypes, add_pooler,
                       encoder_attn_mask_type,
                       add_encoder=True,
                       add_decoder=False,
                       decoder_attn_mask_type=None,
                       pre_process=True, post_process=True):
    """
    get language model

    Args:
        - **config** : model config
        - **num_tokentypes** : if > 0, using tokentypes embedding
        - **add_pooler** : if True, use pooler
        - **encoder_attn_mask_type** : encoder attention mask type
        - **add_encoder** : if True, use encoder
        - **add_decoder** : if True, use decoder
        - **decoder_attn_mask_type** : decoder attention mask type
        - **pre_process** : when using pipeline parallel, indicate whether it's the first stage
        - **post_process** : when using pipeline parallel, indicate whether it's the last stage
    """
    language_model = TransformerLanguageModel(
        config=config,
        encoder_attn_mask_type=encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process
    )
    language_model_key = 'language_model'
    return language_model, language_model_key
