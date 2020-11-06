#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import init


class LSTMLinear(nn.Linear):
    r"""
    This function is the same as a nn.Linear layer, except that in the backward pass
    the grad_samples get accumulated (instead of being concatenated as in the standard
    nn.Linear)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)


class DPLSTM(nn.Module):
    r"""
    DP-friendly abstraction in place of the ``torch.nn.LSTM`` module with a similar interface.

    The dimensionality of each timestep input tensor for a sequence of length ``T`` is
    ``[B, D]`` where ``B`` is the batch size. The ``DPLSTM`` output at timestep ``t``,
    ``h_t`` is of shape ``[B, H]`` with the cell state ``c_t`` also of shape ``[B, H]``.

    Attributes:
        input_size (int): The number of expected features in the input ``x``.
        hidden_size (int):  The number of features in the hidden state ``h``.
        batch_first (bool): If ``True``, then the input and output tensors are provided as
            (batch, seq, feature). The default is ``False``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.validate_parameters()

        self.ih = LSTMLinear(input_size, 4 * hidden_size)
        self.hh = LSTMLinear(hidden_size, 4 * hidden_size)

        self.reset_parameters()

    def validate_parameters(self):
        r"""
        Validates the DPLSTM configuration and raises a ``NotImplementedError`` if the number of
        layers is more than 1, the DPLSTM is bidirectional, uses dropout at the output
        or, it does not have a bias term.

        Raises:
            NotImplementedError
                If the number of layers is more than 1, the DPLSTM is bidirectional,
                uses dropout at the output, or it does not have a bias term.
        """
        if self.num_layers > 1 or not self.bias or self.dropout or self.bidirectional:
            raise NotImplementedError(
                "DPLSTM Layer initialized with unsupported non-default flag. "
                "Only supported flags are bias=True, bidirectional=False "
                "num_layers=1, dropout=False, and initial state set to zero tensors"
            )

    def initialize_weights(self, weight_params):
        weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0 = weight_params
        self.ih.weight.data.copy_(weight_ih_l0)
        self.ih.bias.data.copy_(bias_ih_l0)
        self.hh.weight.data.copy_(weight_hh_l0)
        self.hh.bias.data.copy_(bias_hh_l0)

    def reset_parameters(self):
        r"""
        Resets parameters of the DPLSTM by initializing them from an uniform distribution.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def _rearrange_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:  # batch is by default in second dimension
            x = x.transpose(0, 1)
        return x

    def _forward_cell(
        self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ):
        gates = self.ih(x) + self.hh(h_prev)  # [B, 4*D]
        i_t_input, f_t_input, g_t_input, o_t_input = torch.split(
            gates, self.hidden_size, 1
        )
        i_t = torch.sigmoid(i_t_input)  # [B, D]
        f_t = torch.sigmoid(f_t_input)  # [B, D]
        g_t = torch.tanh(g_t_input)  # [B, D]
        o_t = torch.sigmoid(o_t_input)  # [B, D]
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

    def forward(
        self, x: torch.Tensor, state_init: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Implements the forward pass of the DPLSTM when a sequence is input.

        Args:
            x: Input sequence to the DPLSTM of shape ``[T, B, D]``
            state_init: Initial state of the LSTM as a tuple ``(h_init, c_init)``
                where ``h_init`` is the initial hidden state and ``c_init`` is the
                initial cell state of the DPLSTM (The default is ``None``, in which case both
                ``h_init`` and ``c_init`` default to zero tensors).

        Returns:
            ``output, (h_n, c_n)`` where ``output`` is of shape ``[T, B, H]`` and is a
            tensor containing the output features (``h_t``) from the last layer of the
            DPLSTM for each timestep ``t``. ``h_n`` is of shape ``[B,H]`` and is a
            tensor containing the hidden state for ``t = T``. ``c_n`` is of shape
            ``[B, H]`` tensor containing the cell state for ``t = T``.
        """

        x = self._rearrange_batch_dim(x)
        seq_length, batch_sz, _ = x.shape

        hc = [(None, None)] * seq_length
        x = torch.unbind(x, dim=0)

        if state_init:
            h_init, c_init = state_init
        else:
            h_init = torch.zeros(
                self.num_layers,
                batch_sz,
                self.hidden_size,
                dtype=x[0].dtype,
                device=x[0].device,
            )
            c_init = torch.zeros(
                self.num_layers,
                batch_sz,
                self.hidden_size,
                dtype=x[0].dtype,
                device=x[0].device,
            )

        layer = 0
        hc[0] = self._forward_cell(x[0], h_init[layer], c_init[layer])
        for t in range(1, seq_length):
            h_prev, c_prev = hc[t - 1]
            hc[t] = self._forward_cell(x[t], h_prev, c_prev)

        return (
            self._rearrange_batch_dim(torch.stack([hc_t[0] for hc_t in hc])),
            (hc[-1][0], hc[-1][1]),
        )
