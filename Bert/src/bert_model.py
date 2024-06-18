# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Bert model."""

import math
import copy
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.functional as F
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


class BertConfig:
    """
    Configuration for `BertModel`.

    Args:
        seq_length (int): Length of input sequence. Default: 128.
        vocab_size (int): The shape of each embedding vector. Default: 32000.
        hidden_size (int): Size of the bert encoder layers. Default: 768.
        num_hidden_layers (int): Number of hidden layers in the BertTransformer encoder
                           cell. Default: 12.
        num_attention_heads (int): Number of attention heads in the BertTransformer
                             encoder cell. Default: 12.
        intermediate_size (int): Size of intermediate layer in the BertTransformer
                           encoder cell. Default: 3072.
        hidden_act (str): Activation function used in the BertTransformer encoder
                    cell. Default: "gelu".
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.1.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        type_vocab_size (int): Size of token type vocab. Default: 16.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        use_position_embedding (bool): Specifies whether to use positions embedding. Default: False.
        dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
        use_recompute (bool): Specifies whether to use recompute. Default: False.
        return_all_encoders (bool): Specifies whether to return all encoders. Default: False.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: False.
        max_relative_position (int): Max value of relative position. Default: 16.
        use_token_type (bool): Specifies whether to use token type embeddings. Default: True.

    """
    def __init__(self,
                 seq_length=128,
                 vocab_size=32000,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 use_relative_positions=False,
                 use_position_embedding=False,
                 dtype=mstype.float32,
                 compute_type=mstype.float32,
                 use_recompute=False,
                 return_all_encoders=False,
                 has_attention_mask=True,
                 max_relative_position=16,
                 use_token_type=True):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.use_relative_positions = use_relative_positions
        self.dtype = dtype
        self.compute_type = compute_type
        self.use_recompute = use_recompute
        self.use_position_embedding = use_position_embedding
        self.return_all_encoders = return_all_encoders
        self.has_attention_mask = has_attention_mask
        self.max_relative_position = max_relative_position
        self.use_token_type = use_token_type

class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional and token type embeddings to word embeddings.

    Args:
        config (Class): Configuration for BertModel.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
    """
    def __init__(self,config,embedding_shape,):
        super(EmbeddingPostprocessor, self).__init__()
        self.embedding_dim = config.hidden_size
        self.use_token_type = config.use_token_type
        self.token_type_vocab_size = config.type_vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.token_type_embedding = nn.extend.Embedding(
            num_embeddings=self.token_type_vocab_size,
            embedding_dim=self.embedding_dim)
        self.reshape = P.Reshape()
        self.shape = tuple(embedding_shape)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.use_position_embedding = config.use_position_embedding
        _, seq, _ = self.shape
        self.full_position_embedding = nn.extend.Embedding(
            num_embeddings=self.max_position_embeddings,
            embedding_dim=self.embedding_dim)
        self.layernorm = nn.extend.LayerNorm((self.embedding_dim,))
        self.position_ids = Tensor(np.arange(seq).reshape(-1, seq).astype(np.int32))
        self.add = ops.extend.add

    def construct(self, token_type_ids, word_embeddings):
        """Postprocessors apply positional and token type embeddings to word embeddings."""
        output = word_embeddings
        if self.use_token_type:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            output = self.add(output, token_type_embeddings)
        if self.use_position_embedding:
            shape = F.shape(output)
            position_ids = self.position_ids[:, :shape[1]]
            position_embeddings = self.full_position_embedding(position_ids)
            output = self.add(output, position_embeddings)
        output = self.layernorm(output)
        output = self.dropout(output)
        return output


class BertOutput(nn.Cell):
    """
    Apply a linear computation to hidden status and a residual computation to input.

    Args:
        config (Class): Configuration for BertModel.
        in_channels (int): Input channels.
        out_channels (int): Output channels.
    """
    def __init__(self,config,in_channels,out_channels):
        super(BertOutput, self).__init__()
        self.dropout_prob = config.hidden_dropout_prob
        self.compute_type = config.compute_type
        self.out_channels = out_channels
        self.dense = nn.extend.Linear(in_channels, self.out_channels,
                              weight_init=TruncatedNormal(config.initializer_range)).to_float(self.compute_type)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.add = ops.extend.add
        self.layernorm = nn.extend.LayerNorm((self.out_channels,)).to_float(self.compute_type)
        self.cast = ops.cast

    def construct(self, hidden_status, input_tensor):
        output = self.dense(hidden_status)
        output = self.dropout(output)
        output = self.add(input_tensor, output)
        output = self.layernorm(output)
        return output


class RelaPosMatrixGenerator(nn.Cell):
    """
    Generates matrix of relative positions between inputs.

    Args:
        length (int): Length of one dim for the matrix to be generated.
        max_relative_position (int): Max value of relative position.
    """
    def __init__(self, max_relative_position):
        super(RelaPosMatrixGenerator, self).__init__()
        self._max_relative_position = max_relative_position
        self._min_relative_position = -max_relative_position

        self.tile = ops.tile
        self.range_mat = P.Reshape()
        self.sub = ops.extend.sub
        self.expanddims = P.ExpandDims()
        self.cast = ops.cast

    def construct(self, length):
        """Generates matrix of relative positions between inputs."""
        range_vec_row_out = self.cast(F.tuple_to_array(F.make_range(length)), mstype.int32)
        range_vec_col_out = self.range_mat(range_vec_row_out, (length, -1))
        tile_row_out = self.tile(range_vec_row_out, (length,))
        tile_col_out = self.tile(range_vec_col_out, (1, length))
        range_mat_out = self.range_mat(tile_row_out, (length, length))
        transpose_out = self.range_mat(tile_col_out, (length, length))
        distance_mat = self.sub(range_mat_out, transpose_out)

        distance_mat_clipped = C.clip_by_value(distance_mat,
                                               self._min_relative_position,
                                               self._max_relative_position)

        # Shift values to be >=0. Each integer still uniquely identifies a
        # relative position difference.
        final_mat = distance_mat_clipped + self._max_relative_position
        return final_mat


class RelaPosEmbeddingsGenerator(nn.Cell):
    """
    Generates tensor of size [length, length, depth].

    Args:
        length (int): Length of one dim for the matrix to be generated.
        depth (int): Size of each attention head.
        max_relative_position (int): Maxmum value of relative position.
        initializer_range (float): Initialization value of TruncatedNormal.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,
                 depth,
                 max_relative_position,
                 initializer_range,
                 use_one_hot_embeddings=False):
        super(RelaPosEmbeddingsGenerator, self).__init__()
        self.depth = depth
        self.vocab_size = max_relative_position * 2 + 1
        self.use_one_hot_embeddings = use_one_hot_embeddings

        self.embeddings_table = Parameter(
            initializer(TruncatedNormal(initializer_range),
                        [self.vocab_size, self.depth]))

        self.relative_positions_matrix = RelaPosMatrixGenerator(max_relative_position=max_relative_position)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.gather = P.Gather()  # index_select
        self.matmul = ops.extend.bmm

    def construct(self, length):
        """Generate embedding for each relative position of dimension depth."""
        relative_positions_matrix_out = self.relative_positions_matrix(length)

        if self.use_one_hot_embeddings:
            flat_relative_positions_matrix = self.reshape(relative_positions_matrix_out, (-1,))
            one_hot_relative_positions_matrix = ops.one_hot(flat_relative_positions_matrix,
                                                            self.vocab_size, 1.0, 0.0)
            embeddings = self.matmul(one_hot_relative_positions_matrix, self.embeddings_table)
            my_shape = self.shape(relative_positions_matrix_out) + (self.depth,)
            embeddings = self.reshape(embeddings, my_shape)
        else:
            embeddings = self.gather(self.embeddings_table,
                                     relative_positions_matrix_out,0)
        return embeddings


class SaturateCast(nn.Cell):
    """
    Performs a safe saturating cast. This operation applies proper clamping before casting to prevent
    the danger that the value will overflow or underflow.

    Args:
        dst_type (:class:`mindspore.dtype`): The type of the elements of the output tensor. Default: mstype.float32.
    """
    def __init__(self, dst_type=mstype.float32):
        super(SaturateCast, self).__init__()
        np_type = mstype.dtype_to_nptype(dst_type)

        self.tensor_min_type = float(np.finfo(np_type).min)
        self.tensor_max_type = float(np.finfo(np_type).max)

        self.min_op = ops.minimum
        self.max_op = ops.maximum
        self.cast = ops.cast
        self.dst_type = dst_type

    def construct(self, x):
        out = self.max_op(x, self.tensor_min_type)
        out = self.min_op(out, self.tensor_max_type)
        return self.cast(out, self.dst_type)


class BertAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".

    Args:
        config (Class): Configuration for BertModel.
        from_tensor_width (int): Size of last dim of from_tensor.
        to_tensor_width (int): Size of last dim of to_tensor.
        size_per_head (int): Size of each attention head. Default: 512.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,config,
                 from_tensor_width,
                 to_tensor_width,
                 size_per_head=512,
                 use_one_hot_embeddings=False):

        super(BertAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.size_per_head = size_per_head
        self.has_attention_mask = config.has_attention_mask
        self.use_relative_positions = config.use_relative_positions
        self.compute_type = config.compute_type
        self.scores_mul = 1.0 / math.sqrt(float(self.size_per_head))
        self.reshape = P.Reshape()
        self.shape_from_2d = (-1, from_tensor_width)
        self.shape_to_2d = (-1, to_tensor_width)
        weight = TruncatedNormal(config.initializer_range)
        units = self.num_attention_heads * size_per_head
        self.query_layer = nn.extend.Linear(from_tensor_width,
                                    units,
                                    weight_init=weight).to_float(self.compute_type)
        self.key_layer = nn.extend.Linear(to_tensor_width,
                                  units,
                                  weight_init=weight).to_float(self.compute_type)
        self.value_layer = nn.extend.Linear(to_tensor_width,
                                    units,
                                    weight_init=weight).to_float(self.compute_type)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = ops.mul
        self.transpose = ops.permute
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = -10000.0
        self.matmul = P.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = ops.extend.sub
            self.add = ops.extend.add
            self.cast = ops.cast
            self.get_dtype = P.DType()

        self.shape_return = (-1, self.num_attention_heads * size_per_head)

        self.cast_compute_type = SaturateCast(dst_type=self.compute_type)
        if self.use_relative_positions:
            self._generate_relative_positions_embeddings = \
                RelaPosEmbeddingsGenerator(depth=size_per_head,
                                           max_relative_position=config.max_relative_position,
                                           initializer_range=config.initializer_range,
                                           use_one_hot_embeddings=use_one_hot_embeddings)

    def construct(self, from_tensor, to_tensor, attention_mask):
        """reshape 2d/3d input tensors to 2d"""
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        shape_from = F.shape(attention_mask)[2]
        from_tensor = F.depend(from_tensor, shape_from)
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)

        query_layer = self.reshape(query_out, (-1, shape_from, self.num_attention_heads, self.size_per_head))
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, (-1, shape_from, self.num_attention_heads, self.size_per_head))
        key_layer = self.transpose(key_layer, self.trans_shape)
        # `attention_scores` = [B, N, F, T]
        attention_scores = self.matmul_trans_b(query_layer, key_layer)

        # use_relative_position, supplementary logic
        # Self-Attention with Relative Position Representations
        if self.use_relative_positions:
            # relations_keys is [F|T, F|T, H]
            relations_keys = self._generate_relative_positions_embeddings(shape_from)
            relations_keys = self.cast_compute_type(relations_keys)
            # query_layer_t is [F, B, N, H]
            query_layer_t = self.transpose(query_layer, self.trans_shape_relative)
            # query_layer_r is [F, B * N, H]
            query_layer_r = self.reshape(query_layer_t,
                                         (shape_from,
                                          -1,
                                          self.size_per_head))
            # key_position_scores is [F, B * N, F|T]
            key_position_scores = self.matmul_trans_b(query_layer_r,
                                                      relations_keys)
            # key_position_scores_r is [F, B, N, F|T]
            key_position_scores_r = self.reshape(key_position_scores,
                                                 (shape_from,
                                                  -1,
                                                  self.num_attention_heads,
                                                  shape_from))
            # key_position_scores_r_t is [B, N, F, F|T]
            key_position_scores_r_t = self.transpose(key_position_scores_r,
                                                     self.trans_shape_position)
            attention_scores = attention_scores + key_position_scores_r_t

        attention_scores = self.multiply(self.scores_mul, attention_scores)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))

            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, (-1, shape_from, self.num_attention_heads, self.size_per_head))
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)

        # use_relative_position, supplementary logic
        if self.use_relative_positions:
            # relations_values is [F|T, F|T, H]
            relations_values = self._generate_relative_positions_embeddings(shape_from)
            relations_values = self.cast_compute_type(relations_values)
            # attention_probs_t is [F, B, N, T]
            attention_probs_t = self.transpose(attention_probs, self.trans_shape_relative)
            # attention_probs_r is [F, B * N, T]
            attention_probs_r = self.reshape(
                attention_probs_t,
                (shape_from,
                 -1,
                 shape_from))
            # value_position_scores is [F, B * N, H]
            value_position_scores = self.matmul(attention_probs_r,
                                                relations_values)
            # value_position_scores_r is [F, B, N, H]
            value_position_scores_r = self.reshape(value_position_scores,
                                                   (shape_from,
                                                    -1,
                                                    self.num_attention_heads,
                                                    self.size_per_head))
            # value_position_scores_r_t is [B, N, F, H]
            value_position_scores_r_t = self.transpose(value_position_scores_r,
                                                       self.trans_shape_position)
            context_layer = context_layer + value_position_scores_r_t

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, self.shape_return)

        return context_layer


class BertSelfAttention(nn.Cell):
    """
    Apply self-attention.

    Args:
        config (Class): Configuration for BertModel.
        use_one_hot_embeddings (bool): Specifies whether to use one_hot encoding form. Default: False.
    """
    def __init__(self,config,use_one_hot_embeddings=False):
        super(BertSelfAttention, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (self.hidden_size, self.num_attention_heads))

        self.size_per_head = int(self.hidden_size / self.num_attention_heads)

        self.attention = BertAttention(
            config=config,
            from_tensor_width=self.hidden_size,
            to_tensor_width=self.hidden_size,
            size_per_head=self.size_per_head,
            use_one_hot_embeddings=use_one_hot_embeddings,
            )

        self.output = BertOutput(
                                config=config,
                                in_channels=self.hidden_size,
                                out_channels=self.hidden_size
                                )
        self.reshape = P.Reshape()
        self.shape = (-1, self.hidden_size)

    def construct(self, input_tensor, attention_mask):
        attention_output = self.attention(input_tensor, input_tensor, attention_mask)
        output = self.output(attention_output, input_tensor)
        return output


class BertEncoderCell(nn.Cell):
    """
    Encoder cells used in BertTransformer.

    Args:
        config (Class): Configuration for BertModel.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,config,use_one_hot_embeddings=False):
        super(BertEncoderCell, self).__init__()
        self.hidden_size = config.hidden_size
        self.compute_type = config.compute_type
        self.intermediate_size = config.intermediate_size
        self.attention = BertSelfAttention(
            config=config,
            use_one_hot_embeddings=use_one_hot_embeddings)
        self.intermediate = nn.extend.Linear(config=config,
                                    in_features=self.hidden_size,
                                    out_features=self.intermediate_size,
                                    weight_init=TruncatedNormal(config.initializer_range)).to_float(self.compute_type)
        self.output = BertOutput(config=config,
                                 in_channels=self.intermediate_size,
                                 out_channels=self.hidden_size)

    def construct(self, hidden_states, attention_mask):
        # self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        # feed construct
        intermediate_output = self.intermediate(attention_output)
        # add and normalize
        output = self.output(intermediate_output, attention_output)
        return output


class BertTransformer(nn.Cell):
    """
    Multi-layer bert transformer.

    Args:
        config (Class): Configuration for BertModel.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,config,use_one_hot_embeddings=False):
        super(BertTransformer, self).__init__()
        self.return_all_encoders = config.return_all_encoders
        self.hidden_size = config.hidden_size

        layers = []
        for _ in range(config.num_hidden_layers):
            layer = BertEncoderCell(config=config,
                                    use_one_hot_embeddings=use_one_hot_embeddings)
            layers.append(layer)

        self.layers = nn.CellList(layers)
        if config.use_recompute:
            for layer in self.layers:
                self.recompute(layer)
        self.reshape = P.Reshape()
        self.shape = (-1, self.hidden_size)
    def recompute(self, b):
        b.recompute()
    def construct(self, input_tensor, attention_mask):
        """Multi-layer bert transformer."""
        prev_output = self.reshape(input_tensor, self.shape)

        all_encoder_layers = ()
        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask)
            prev_output = layer_output

            if self.return_all_encoders:
                shape = F.shape(input_tensor)
                layer_output = self.reshape(layer_output, shape)
                all_encoder_layers = all_encoder_layers + (layer_output,)

        if not self.return_all_encoders:
            shape = F.shape(input_tensor)
            prev_output = self.reshape(prev_output, shape)
            all_encoder_layers = all_encoder_layers + (prev_output,)
        return all_encoder_layers


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (Class): Configuration for BertModel.
    """
    def __init__(self, config):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.input_mask = None

        self.cast = ops.cast
        self.reshape = P.Reshape()

    def construct(self, input_mask):
        seq_length = F.shape(input_mask)[1]
        attention_mask = self.cast(self.reshape(input_mask, (-1, 1, seq_length)), mstype.float32)
        return attention_mask


class BertModel(nn.Cell):
    """
    Bidirectional Encoder Representations from Transformers.

    Args:
        config (Class): Configuration for BertModel.
        is_training (bool): True for training mode. False for eval mode.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,config,is_training,use_one_hot_embeddings=False):
        super(BertModel, self).__init__()
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size
        self.token_type_ids = None
        self.compute_type = config.compute_type

        self.last_idx = self.num_hidden_layers - 1
        output_embedding_shape = [-1, config.seq_length, self.embedding_size]

        self.bert_embedding_lookup = nn.Embedding(
            vocab_size=config.vocab_size,
            embedding_size=self.embedding_size,
            use_one_hot=use_one_hot_embeddings,
            embedding_table=TruncatedNormal(config.initializer_range))

        self.bert_embedding_postprocessor = EmbeddingPostprocessor(
            config=config,
            embedding_shape=output_embedding_shape)

        self.bert_encoder = BertTransformer(
            config=config,
            use_one_hot_embeddings=use_one_hot_embeddings)

        self.cast = ops.cast
        self.dtype = config.dtype
        self.cast_compute_type = SaturateCast(dst_type=self.compute_type)
        self.slice = P.StridedSlice()

        self.squeeze_1 = P.Squeeze(axis=1)
        self.dense = nn.Dense(self.hidden_size, self.hidden_size,
                              activation="tanh",
                              weight_init=TruncatedNormal(config.initializer_range)).to_float(self.compute_type)
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)

    def construct(self, input_ids, token_type_ids, input_mask):
        """Bidirectional Encoder Representations from Transformers."""
        # embedding
        # Perform embedding lookup on the word ids.
        embedding_tables = self.bert_embedding_lookup.embedding_table
        word_embeddings = self.bert_embedding_lookup(input_ids)
        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        embedding_output = self.bert_embedding_postprocessor(token_type_ids,
                                                             word_embeddings)

        # attention mask [batch_size, 1, seq_length]
        attention_mask = self._create_attention_mask_from_input_mask(input_mask)

        # bert encoder
        encoder_output = self.bert_encoder(self.cast_compute_type(embedding_output),
                                           attention_mask)

        sequence_output = self.cast(encoder_output[self.last_idx], self.dtype)

        # pooler
        batch_size = P.Shape()(input_ids)[0]
        sequence_slice = self.slice(sequence_output,
                                    (0, 0, 0),
                                    (batch_size, 1, self.hidden_size),
                                    (1, 1, 1))
        first_token = self.squeeze_1(sequence_slice)
        pooled_output = self.dense(first_token)
        pooled_output = self.cast(pooled_output, self.dtype)

        return sequence_output, pooled_output, embedding_tables
