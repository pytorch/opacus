#!/usr/bin/env python3
# Copyright (c) Facebook, Dnc. and its affiliates. All Rights Reserved

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class DPLSTMCell(nn.Module):
    r"""
    Encapsulates an LSTM cell in the DP-friendly implementation of LSTM.
    This provides access to the forward and backward passes and enables
    computation of the per-example gradients required in DP-SGD.

    The dimensionality of each timestep input tensor for a sequence of length
    ``T`` is ``[B, D]`` where ``B`` is the batch size. The ``DPLSTMCell``
    output at timestep ``t``, ``h_t`` is of shape ``[B, H]`` with the cell
    state ``c_t`` also of shape ``[B, H]``.

    Attributes:
        sigmoid_fn (Callable): Sigmoid activation function used in internal cell gates.
        tanh_fn (Callable): Tanh activation function used in internal cell gates.
    """

    def __init__(self, input_dim: int, lstm_out_dim: int):
        r"""
        Args:
            input_dim: Dimensionality ``D`` at each timestep of the input
                sequence to LSTM.
            lstm_out_dim: Output dimensionality ``H`` at each timestep of the LSTM output.
        """
        super(DPLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.lstm_out_dim = lstm_out_dim
        self.sigmoid_fn = nn.Sigmoid()
        self.tanh_fn = nn.Tanh()

    def initialize_weights(self, weight_params: List[nn.Parameter]):
        r"""
        Loads internal gate parameter weights and biases from an input list of parameters
        ``[weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0]``.

        Args:
            weight_params: List of parameters ``[weight_ih_l0, weight_hh_l0, ...
                bias_ih_l0, bias_hh_l0]`` from which to load internal cell weights
                (for input, output and forget gates).
        """
        [
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
        ] = weight_params

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Implements the forward pass for the LSTM cell. This includes computation
        of the cell states as well as the internal gate activations for the input,
        output, cell and forget gates.

        Args:
            x_t: Input tensor for the timestep ``t`` of shape ``[B, D]``.
            h_prev: Hidden state ``h`` for the previous timestep ``t-1``
                of shape ``[B, H]``.
            c_prev: Cell state ``c`` for the previous timestep ``t-1``
                of shape ``[B, H]``.

        Returns:
            Hidden state for the particular timestep on which the cell is run.
        """
        self.split_input = (
            F.linear(x_t, self.weight_ih_l0, self.bias_ih_l0)
            + F.linear(h_prev, self.weight_hh_l0, self.bias_hh_l0)
        )[0, :]
        self.i_t_input, self.f_t_input, self.g_t_input, self.o_t_input = torch.split(
            self.split_input, self.lstm_out_dim, 1
        )
        self.i_t = self.sigmoid_fn(self.i_t_input)
        self.f_t = self.sigmoid_fn(self.f_t_input)
        self.g_t = self.tanh_fn(self.g_t_input)
        self.o_t = self.sigmoid_fn(self.o_t_input)
        self.c_t = self.f_t * c_prev + self.i_t * self.g_t
        self.h_t = self.o_t * self.tanh_fn(self.c_t)
        return self.h_t[0, :]

    def backward(
        self,
        x_t: torch.Tensor,
        delta_h_t: torch.Tensor,
        delta_t: torch.Tensor,
        f_next: torch.Tensor,
        dc_next: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Implements the backward pass for the LSTM cell.

        The steps implemented closely correspond to the description provided in the Medium post
        https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
        """
        self.dh_t = delta_t + delta_h_t
        self.dc_t = (
            self.dh_t * self.o_t * (1 - torch.pow(self.tanh_fn(self.c_t), 2))
            + dc_next * f_next
        )
        self.dg_t = self.dc_t * self.i_t * (1 - torch.pow(self.g_t, 2))
        self.di_t = self.dc_t * self.g_t * self.i_t * (1 - self.i_t)
        self.df_t = self.dc_t * c_prev * self.f_t * (1 - self.f_t)
        self.do_t = self.dh_t * self.tanh_fn(self.c_t) * self.o_t * (1 - self.o_t)
        self.dgates_t = torch.cat([self.di_t, self.df_t, self.dg_t, self.do_t], 2)[0, :]
        self.delta_h_prev = torch.matmul(self.dgates_t, self.weight_hh_l0)
        return self.delta_h_prev


class DPLSTM(nn.Module):
    r"""
    Encapsulates a DPLSTM module which provides a DP-friendly abstraction in place of the
    ``torch.nn.LSTM`` module while having a similar interface. Includes functionality
    for resetting parameters, loading them from an external source as well as unrolling
    the network for multiple timesteps and implementing the forward pass.

    The dimensionality of each timestep input tensor for a sequence of length ``T`` is
    ``[B, D]`` where ``B`` is the batch size. The ``DPLSTM`` output at timestep ``t``,
    ``h_t`` is of shape ``[B, H]`` with the cell state ``c_t`` also of shape ``[B, H]``.

    Attributes:
        input_size: The number of expected features in the input ``x``.
        hidden_size:  The number of features in the hidden state ``h``.
        batch_first: If ``True``, then the input and output tensors are provided as
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
        super(DPLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.cells_initialized = False
        self.cells = nn.ModuleList([])

        self.validate_parameters()

        self.weight_ih_l0 = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.Tensor(4 * hidden_size))

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

    def initialize_weights(self, weight_params: List[nn.Parameter]):
        r"""
        Loads the LSTM weights and biases for the internal cell gates from
        an external input.

        Args:
            weight_params: List of parameters ``[weight_ih_l0, weight_hh_l0, ...
                bias_ih_l0, bias_hh_l0]`` from which to load internal cell weights
                (for input, output and forget gates).
        """
        [
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
        ] = weight_params

    def _unroll_and_initialize_cells(self, seq_length: int, device: torch.device):
        r"""
        Unrolls and initializes the DPLSTM cells when presented with a new sequence
        input during the forward pass.

        Args:
            seq_length: Length ``T`` of the input sequence in the forward pass.
            device: Device (cpu/cuda) on which the unrolled and initialized cells are
                stored.
        """
        self.cells = nn.ModuleList([])
        for t in range(0, seq_length):
            self.cells.append(DPLSTMCell(self.input_size, self.hidden_size).to(device))
            self.cells[t].initialize_weights(
                [self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0]
            )

    def forward(
        self, x: torch.Tensor, state_init: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Implements the forward pass of the DPLSTM when a sequence is input.
        The constituent cells are initialized and unrolled, and the hidden state
        for each ``DPLSTMCell`` in the sequence is obtained as a function of the
        input at the timestep and the previous timestep hidden and cell states.

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
        device = next(self.parameters()).device
        self._unroll_and_initialize_cells(seq_length, device)

        x = torch.unbind(x, dim=0)
        h = [None] * seq_length

        if state_init:
            h_init, c_init = state_init
        else:
            h_init = torch.zeros(self.num_layers, batch_sz, self.hidden_size)
            c_init = torch.zeros(self.num_layers, batch_sz, self.hidden_size)

        h_init = h_init.to(device)
        c_init = c_init.to(device)

        h[0] = self.cells[0](x[0].unsqueeze(0), h_init, c_init)
        for t in range(1, seq_length):
            h[t] = self.cells[t](
                x[t].unsqueeze(0), self.cells[t - 1].h_t, self.cells[t - 1].c_t
            )
        return (
            self._rearrange_batch_dim(torch.stack(h)),
            (self.cells[-1].h_t, self.cells[-1].c_t),
        )
