#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SequenceBias(nn.Module):
    r"""
    Adds one bias element to the end of the sequence.
    so if the input has a shape ``(L, N, E)``, (``batch_first = False``),
    where ``L`` is the sequence length, ``N`` is the batch size, and ``E`` is
    the embedding dimension, the output will have a shape
    ``(L+1, N, E)``. When ``batch_first = True``, input has a shape ``(N, L, E)``
    and the output will have a shape ``(N, L+1, E)``

    Attributes:
        bias (:class:`torch.nn.parameter.Parameter`): the learnable bias of
            the module of shape ``(E)``, where ``E`` is the embedding dimension.

    Example:
        >>> m = SequenceBias(16, batch_first=False)
        >>> input = torch.randn(20, 4, 16)
        >>> output = m(input)
        >>> output.size()
        torch.Size([21, 4, 16])
    """

    def __init__(self, embed_dim: int, batch_first: bool = False):
        r"""
        Args:
            embed_dim: Embedding dimension
        """
        super(SequenceBias, self).__init__()
        self.batch_first = batch_first
        self.bias = Parameter(torch.empty(embed_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        r"""
        assigns Normally distributed random values to bias.
        """
        nn.init.normal_(self.bias)

    def forward(self, x):
        if self.batch_first:
            bsz, _, _ = x.shape
            return torch.cat([x, self.bias.repeat(bsz, 1, 1)], 1)
        else:
            _, bsz, _ = x.shape
            return torch.cat([x, self.bias.repeat(1, bsz, 1)])


class InputProjection(nn.Module):
    def __init__(self, embed_dim: int, kdim: int, vdim: int, bias: bool = False):
        super().__init__()
        self.q_end_index = embed_dim
        self.k_end_index = embed_dim + kdim

        self.qlinear_weight = Parameter(torch.empty((embed_dim, embed_dim)))
        self.klinear_weight = Parameter(torch.empty((embed_dim, kdim)))
        self.vlinear_weight = Parameter(torch.empty((embed_dim, vdim)))
        if bias:
            self.bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.bias = None

    def _reset_parameters(self):
        r"""
        assigns Normally distributed random values to bias.
        """
        # TODO: nn.MultiheadAttention uses xavier_uniform_, while .nn.Linear uses kaiming_uniform_.
        # This module was based on the latter, but should it be migrated to use the former?
        nn.init.kaiming_uniform_(self.qlinear_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.klinear_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.vlinear_weight, a=math.sqrt(5))

        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = x[:, :, : self.q_end_index]
        key = x[:, :, self.q_end_index : self.k_end_index]
        value = x[:, :, self.k_end_index :]

        if self.bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = self.bias.chunk(3)

        projected_values = F._in_projection(
            query,
            key,
            value,
            self.qlinear_weight,
            self.klinear_weight,
            self.vlinear_weight,
            b_q,
            b_k,
            b_v,
        )

        return torch.stack(projected_values, dim=-1)


class PackedInputProjection(nn.Module):
    def __init__(self, embed_dim: int, bias: bool = False):
        super().__init__()
        self.q_end_index = embed_dim
        self.k_end_index = 2 * embed_dim

        self.weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.bias = None

    def _reset_parameters(self):
        r"""
        assigns Normally distributed random values to bias.
        """
        # TODO: nn.MultiheadAttention uses xavier_uniform_, while .nn.Linear uses kaiming_uniform_.
        # This module was based on the latter, but should it be migrted to the former?
        # Biases also are initialized using different distributions....
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = x[:, :, : self.q_end_index]
        key = x[:, :, self.q_end_index : self.k_end_index]
        value = x[:, :, self.k_end_index :]

        projected_values = F._in_projection_packed(
            query, key, value, self.weight, self.bias
        )

        return torch.stack(projected_values, dim=-1)


class DPMultiheadAttention(nn.Module):
    r"""
    This is DP-friendly implementation of nn.MultiheadAttention.
    For full reference see original module refer to
    :class:`torch.nn.MultiheadAttention`.

    Current implementation leverages pytorch modules as building blocks
    to allow DP engine to calculate per-sample gradients.
    This is in contrast with original implementation based on nn.functional.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(DPMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj = PackedInputProjection(embed_dim, bias)
        else:
            self.in_proj = InputProjection(embed_dim, self.kdim, self.vdim, bias)

        # torch.nn.MultiHeadAttention out_proj is _LinearWithBias
        # explicilty setting bias=True for consistent mimicry
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.add_bias_kv = add_bias_kv
        if self.add_bias_kv:
            self.seq_bias_k = SequenceBias(embed_dim)
            self.seq_bias_v = SequenceBias(embed_dim)

        self.add_zero_attn = add_zero_attn

        self.dropout = nn.Dropout(dropout)

    def load_state_dict(self, state_dict):
        r"""
        Loads module from previously saved state.

        Supports loading from both :class:`torch.nn.MultiheadAttention` and
        :class:`opacus.layers.dp_multihead_attention.DPMultiheadAttention`.

        Args:
            state_dict: Please refer to
                https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html.
        """
        if self._qkv_same_embed_dim:
            if "in_proj_weight" in state_dict:
                state_dict["in_proj.weight"] = state_dict["in_proj_weight"]
                del state_dict["in_proj_weight"]
        else:
            if "q_proj_weight" in state_dict:
                state_dict["in_proj.qlinear_weight"] = state_dict["q_proj_weight"]
                del state_dict["q_proj_weight"]

            if "k_proj_weight" in state_dict:
                state_dict["in_proj.klinear_weight"] = state_dict["k_proj_weight"]
                del state_dict["k_proj_weight"]

            if "v_proj_weight" in state_dict:
                state_dict["in_proj.vlinear_weight"] = state_dict["v_proj_weight"]
                del state_dict["v_proj_weight"]

        if "in_proj_bias" in state_dict:
            state_dict["in_proj.bias"] = state_dict["in_proj_bias"]
            del state_dict["in_proj_bias"]

        if "bias_k" in state_dict:
            state_dict["seq_bias_k.bias"] = state_dict["bias_k"].squeeze()
            del state_dict["bias_k"]

        if "bias_v" in state_dict:
            state_dict["seq_bias_v.bias"] = state_dict["bias_v"].squeeze()
            del state_dict["bias_v"]

        super(DPMultiheadAttention, self).load_state_dict(state_dict)

    # flake8: noqa C901
    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
    ):
        tgt_len, bsz, embed_dim = query.size()
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"query has as size of {embed_dim} while the embedding"
                " size is {self.embed_dim}"
            )

        head_dim = embed_dim // self.num_heads
        if head_dim * self.num_heads != embed_dim:
            raise ValueError(
                f"embedding dimension {embed_dim} not divisible "
                "by number of heads {num_heads}"
            )
        scaling = float(head_dim) ** -0.5

        key_seq_len = key.size()[0]
        value_seq_len = value.size()[0]
        if key_seq_len != value_seq_len:
            raise ValueError(
                f"key sequence lenght {key_seq_len} doesn't match "
                f"the value sequence length {value_seq_len}"
            )
        # We need to combine all inputs into a single tensor, because grad_sampler
        # does not support modules whose forward function takes more than 1 argument.
        # At the same time, the way the both the packed and unpacked transformation
        # is implemented requires passig all 3 inputs at the same time.

        # "query" may have a different size of the first dimension than "key" and "value", which
        # prevents a simple concatentation. In order to enable it, we'll pad key and value
        # and then truncate them after the transformation.
        kv_pad_len = query.shape[0] - key.shape[0]

        if kv_pad_len > 0:
            key = F.pad(key, (0, 0, 0, 0, 0, kv_pad_len))
            value = F.pad(value, (0, 0, 0, 0, 0, kv_pad_len))
        elif kv_pad_len < 0:
            query = F.pad(query, (0, 0, 0, 0, 0, -kv_pad_len))

        # Concatenate along the last dimension which might be different for each of the input values.
        x = torch.cat((query, key, value), dim=-1)
        qkv_proj = self.in_proj(x)
        q, k, v = qkv_proj.unbind(-1)

        # this is where we store the former shapes
        if kv_pad_len > 0:
            k = k[:-kv_pad_len]
            v = v[:-kv_pad_len]
        elif kv_pad_len < 0:
            q = q[:kv_pad_len]

        q = q * scaling

        if attn_mask is not None:
            if attn_mask.dtype not in (
                torch.float32,
                torch.float64,
                torch.uint8,
                torch.bool,
            ):
                raise ValueError(
                    f"Only float, byte, and bool types are supported for attn_mask, "
                    f"not {attn_mask.dtype}."
                )

            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated."
                    "Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise ValueError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * self.num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise ValueError("The size of the 3D attn_mask is not correct.")
            else:
                raise ValueError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention"
                "is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        if self.add_bias_kv:
            k = self.seq_bias_k(k)
            v = self.seq_bias_v(v)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat(
                [
                    k,
                    torch.zeros(
                        (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                    ),
                ],
                dim=1,
            )
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None
