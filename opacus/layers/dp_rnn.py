#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import numbers
import warnings
from typing import List, Optional, Tuple, Union, Type, Literal

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from .param_rename import ParamRenamedMixin


def apply_permutation(tensor: Tensor, dim: int, permutation: Optional[Tensor]):
    """
    Permute elements of a tensor along a dimension `dim`. If permutation is None do nothing.
    """
    if permutation is None:
        return tensor
    return tensor.index_select(dim, permutation)


def compute_seq_lengths(batch_sizes: Tensor) -> List[int]:
    """
    Computes the sequence lengths of a PackedSequence represented with batch_sizes.

    Args:
        batch_sizes: Contains the batch sizes as stored in a PackedSequence

    Returns:
        running_seq_lengths: the length parameter used in the torch.nn.utils.rnn.packed_padded_sequence function
        to create a PackedSequence. It's a list of the same length as batch_sizes.
    """

    max_batch_size = batch_sizes[0]
    if len(batch_sizes) == 1:
        return [1] * max_batch_size

    running_seq = 0
    running_seq_lengths = []
    for i in range(1, len(batch_sizes)):
        delta = batch_sizes[i - 1].item() - batch_sizes[i].item()
        running_seq += 1
        running_seq_lengths += delta * [running_seq]

    running_seq += 1
    running_seq_lengths += batch_sizes[-1].item() * [running_seq]
    running_seq_lengths.reverse()
    return running_seq_lengths


class RNNLinear(nn.Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module is the same as a ``torch.nn.Linear``` layer, except that in the backward pass
    the grad_samples get accumulated (instead of being concatenated as in the standard
    nn.Linear).

    Attributes:
        weight, bias: refer to ``nn.Linear`` documentation
        max_batch_length:

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.max_batch_length = 0


class DPRNNCellBase(nn.Module):
    has_cell_state: bool = False

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.ih = RNNLinear(input_size, num_chunks * hidden_size, bias)
        self.hh = RNNLinear(hidden_size, num_chunks * hidden_size, bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def set_max_batch_length(self, max_batch_length: int) -> None:
        self.ih.max_batch_len = max_batch_length
        self.hh.max_batch_len = max_batch_length


class DPRNNCell(DPRNNCellBase):
    """An Elman RNN cell with tanh or ReLU non-linearity.

    TODO
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool, nonlinearity: str = 'tanh') -> None:
        super().__init__(input_size, hidden_size, bias, num_chunks=1)
        self.nonlinearity = nonlinearity

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tensor] = None,
        batch_size_t: Optional[int] = None,
    ) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.shape[0], self.hidden_size, dtype=input.dtype, device=input.device)

        h_prev = hx
        gates = self.ih(input) + self.hh(h_prev if batch_size_t is None else h_prev[:batch_size_t, :])
        if self.nonlinearity == 'tanh':
            h_t = torch.tanh(gates)
        elif self.nonlinearity == 'relu':
            h_t = torch.relu(gates)
        else:
            raise RuntimeError(f"Unknown nonlinearity: {self.nonlinearity}")
        return h_t


class DPGRUCell(DPRNNCellBase):
    """A gated recurrent unit (GRU) cell

    TODO
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool) -> None:
        super().__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tensor] = None,
        batch_size_t: Optional[int] = None,
    ) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.shape[0], self.hidden_size, dtype=input.dtype, device=input.device)

        h_prev = hx if batch_size_t is None else hx[:batch_size_t, :]
        gates_x = self.ih(input)
        gates_h = self.hh(h_prev)
        r_t_input_x, z_t_input_x, n_t_input_x = torch.split(gates_x, self.hidden_size, 1)
        r_t_input_h, z_t_input_h, n_t_input_h = torch.split(gates_h, self.hidden_size, 1)
        r_t = torch.sigmoid(r_t_input_x + r_t_input_h)
        z_t = torch.sigmoid(z_t_input_x + z_t_input_h)
        n_t = torch.tanh(n_t_input_x + r_t * n_t_input_h)
        h_t = (1 - z_t) * n_t + z_t * h_prev
        return h_t


class DPLSTMCell(DPRNNCellBase):
    """A long short-term memory (LSTM) cell.

    TODO
    """
    has_cell_state = True

    def __init__(self, input_size: int, hidden_size: int, bias: bool) -> None:
        super().__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
        batch_size_t: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        if hx is None:
            zeros = torch.zeros(input.shape[0], self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)

        h_prev, c_prev = hx

        if batch_size_t is None:
            gates = self.ih(input) + self.hh(h_prev)  # [B, 4*D]
        else:
            gates = self.ih(input) + self.hh(
                h_prev[:batch_size_t, :]
            )  # [batch_size_t, 4*D]

        i_t_input, f_t_input, g_t_input, o_t_input = torch.split(gates, self.hidden_size, 1)

        # [B, D] or [batch_size_t, D] if batch_size_t is not None
        i_t = torch.sigmoid(i_t_input)
        f_t = torch.sigmoid(f_t_input)
        g_t = torch.tanh(g_t_input)
        o_t = torch.sigmoid(o_t_input)

        if batch_size_t is None:
            c_t = f_t * c_prev + i_t * g_t
        else:
            c_t = f_t * c_prev[:batch_size_t, :] + i_t * g_t

        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


RNN_CELL_TYPES = {
    "RNN_TANH": (DPRNNCell, dict(nonlinearity="tanh")),
    "RNN_RELU": (DPRNNCell, dict(nonlinearity="relu")),
    "GRU": (DPGRUCell, {}),
    "LSTM": (DPLSTMCell, {}),
}


class DPRNNBase(ParamRenamedMixin, nn.Module):
    def __init__(
        self,
        mode: Union[str, Type[DPRNNCellBase]],
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.,
        bidirectional: bool = False,
        proj_size: int = 0,
        cell_params: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.cell_params = {}
        if isinstance(mode, str):
            if mode not in RNN_CELL_TYPES:
                raise ValueError(f"Invalid RNN mode '{mode}', available options: {list(RNN_CELL_TYPES.keys())}")
            self.cell_type, default_params = RNN_CELL_TYPES[mode]
            self.cell_params.update(default_params)
        else:
            self.cell_type = mode
        if cell_params is not None:
            self.cell_params.update(cell_params)
        self.has_cell_state = self.cell_type.has_cell_state

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if proj_size > 0:
            raise NotImplementedError("proj_size > 0 is not supported")
        if proj_size < 0:
            raise ValueError("proj_size should be a positive integer or zero to disable projections")
        if proj_size >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.cells = self.initialize_cells()

    def forward(
        self,
        input: Union[Tensor, PackedSequence],
        state_init: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None
    ) -> Tuple[Union[Tensor, PackedSequence], Union[Tensor, Tuple[Tensor, Tensor]]]:
        """
        TODO

        Args:
            input:
            state_init:

        Returns:

        """
        num_directions = 2 if self.bidirectional else 1

        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input_data, batch_sizes, sorted_indices, unsorted_indices = input
            x = input_data.split(tuple(batch_sizes))

            seq_length = len(batch_sizes)
            max_batch_size = int(batch_sizes[0])

            for cell in self.cells:
                cell.set_max_batch_length(max_batch_size)
        else:
            batch_sizes = None
            sorted_indices = None
            unsorted_indices = None

            x = input

            # Rearrange batch dim. Batch is by default in second dimension.
            if self.batch_first:
                x = x.transpose(0, 1)

            seq_length = x.shape[0]
            max_batch_size = x.shape[1]

        if self.has_cell_state:
            h_0s, c_0s = state_init or (None, None)
        else:
            h_0s = state_init

        if h_0s is None:
            h_0s = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype if not is_packed else input_data.dtype,
                device=input.device if not is_packed else input_data.device,
            )
        else:
            h_0s = apply_permutation(h_0s, 1, sorted_indices)

        if self.has_cell_state:
            if c_0s is None:
                c_0s = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype if not is_packed else input_data.dtype,
                    device=input.device if not is_packed else input_data.device,
                    )
            else:
                c_0s = apply_permutation(c_0s, 1, sorted_indices)
        else:
            c_0s = [None] * len(h_0s)


        # TODO: fix checks
        #self.check_forward_args(input, hx, batch_sizes)

        #####################################################################################
        # RNN:
        # https://github.com/pytorch/pytorch/issues/4930

        # T = seq_length
        # B = max_batch_size
        # L = num_layers
        # P = num_directions

        # D = input_size
        # H = hidden_size

        # x = input
        # unpack: [T, B, D]
        # packed: tuple T x [B, D]

        # hx
        # unpack: [L*P, B, H]
        # packed: [L*P, B, H]

        # output
        # out: [T, B, P*H] / tuple T x [B, P*H]
        # h_0: [L*P, B, H]

        hs = []
        cs = [] # list of None if no cell state

        import itertools
        for layer, directions in self.iterate_layers(self.cells, h_0s, c_0s):

            layer_outs = []

            for direction, (cell, h0, c0) in directions:
                # apply single direction layer (with dropout)
                out_layer, h, c = self.forward_layer(
                    x if layer == 0 else output,  # [T, B, D/H/2H] / tuple T x [B, D/H/2H]
                    h0, # [B, H]
                    c0,
                    batch_sizes,
                    cell=cell,
                    max_batch_size=max_batch_size,
                    seq_length=seq_length,
                    is_packed=is_packed,
                    reverse_layer=(direction == 1),
                )

                # out_layer: [T, B, H] / tuple T x [B, H]
                # h_layer: [B, H]

                cs.append(c)
                hs.append(h)
                layer_outs.append(out_layer)

            if is_packed:
                output = [ # tuple T x [B, H*P]
                    torch.cat([
                        layer_out[i]
                        for layer_out in layer_outs
                    ], dim=1)
                    for i in range(seq_length)
                ]
            else:
                output = torch.cat(layer_outs, dim=2) # [T, B, P*H]

        if is_packed:
            packed_data = torch.cat(output, dim=0) # [TB, P*H]
            output = PackedSequence(packed_data, batch_sizes, sorted_indices, unsorted_indices)
        else:
            # Rearrange batch dim back
            if self.batch_first:
                output = output.transpose(0, 1)

        hs = torch.stack(hs, dim=0)  # [L * P, B, H]
        hs = apply_permutation(hs, 1, unsorted_indices)
        if self.has_cell_state:
            cs = torch.stack(cs, dim=0)  # [L * P, B, H]
            cs = apply_permutation(cs, 1, unsorted_indices)

        hidden = (hs, cs) if self.has_cell_state else hs

        return output, hidden

    def forward_layer(
        self,
        x: Union[Tensor, PackedSequence],
        h_0: Tensor,
        c_0: Optional[Tensor],
        batch_sizes: Tensor,
        cell: DPRNNCellBase,
        max_batch_size: int,
        seq_length: int,
        is_packed: bool,
        reverse_layer: bool,
    ) -> Tuple[Union[Tensor, List[Tensor]], Tensor, Tensor]:
        r"""
        TODO: Rewrite this
        Implements the forward pass of the DPLSTMLayer when a sequence is given in input.

        Args:
            x: Input sequence to the DPLSTMCell of shape ``[T, B, D]``.
            state_init: Initial state of the LSTMCell as a tuple ``(h_0, c_0)``
                where ``h_0`` is the initial hidden state and ``c_0`` is the
                initial cell state of the DPLSTMCell
            batch_sizes: Contains the batch sizes as stored in PackedSequence


        Returns:
            ``output, (h_n, c_n)`` where, ``output`` is of shape ``[T, B, H]`` and is a
            tensor containing the output features (``h_t``) from the last layer of the
            DPLSTMCell for each timestep ``t``. ``h_n`` is of shape ``[B, H]`` and is a
            tensor containing the hidden state for ``t = T``. ``c_n`` is of shape ``[B, H]``
            tensor containing the cell state for ``t = T``.
        """
        if is_packed:
            if reverse_layer:
                x = tuple(reversed(x))
                batch_sizes = batch_sizes.flip(0)
        else:
            if reverse_layer:
                x = x.flip(0)
            x = torch.unbind(x, dim=0)

        h_n = [h_0]
        c_n = [c_0]
        c_next = c_0
        batch_size_prev = h_0.shape[0]

        for t in range(seq_length):
            if is_packed:
                batch_size_t = batch_sizes[t].item()
                delta = batch_size_t - batch_size_prev
                if delta > 0:
                    h_cat = torch.cat((h_n[t], h_0[batch_size_prev:batch_size_t, :]), 0)
                    if self.has_cell_state:
                        c_cat = torch.cat((c_n[t], c_0[batch_size_prev:batch_size_t, :]), 0)
                        h_next, c_next = cell(x[t], (h_cat, c_cat), batch_size_t)
                    else:
                        h_next = cell(x[t], h_cat, batch_size_t)
                else:
                    if self.has_cell_state:
                        h_next, c_next = cell(x[t], (h_n[t], c_n[t]), batch_size_t)
                    else:
                        h_next = cell(x[t], h_n[t], batch_size_t)
            else:
                if self.has_cell_state:
                    h_next, c_next = cell(x[t], (h_n[t], c_n[t]))
                else:
                    h_next = cell(x[t], h_n[t])

            if self.dropout:
                h_next = self.dropout_layer(h_next)

            h_n.append(h_next)
            c_n.append(c_next)
            batch_size_prev = h_next.shape[0]

        if is_packed:
            h_temp = h_n[1:] # list T x [B, H]
            c_temp = c_n[1:]

            # Collect last states for all sequences
            seq_lengths = compute_seq_lengths(batch_sizes)
            h_last = torch.zeros(max_batch_size, self.hidden_size) # [B, H]
            c_last = torch.zeros(max_batch_size, self.hidden_size) if self.has_cell_state else None
            for i, seq_len in enumerate(seq_lengths):
                h_last[i, :] = h_temp[seq_len - 1][i, :]
                if self.has_cell_state:
                    c_last[i, :] = c_temp[seq_len - 1][i, :]
            if reverse_layer:
                h_temp = tuple(reversed(h_temp))

        else:
            h_n = torch.stack(h_n[1:], dim=0)  # [T, B, H], init step not part of output
            h_temp = h_n if not reverse_layer else h_n.flip(0)  # Flip the output...
            h_last = h_n[-1]  # ... But not the states
            c_last = c_n[-1]

        return h_temp, h_last, c_last

    def iterate_layers(self, *args):
        """
        TODO

        Args:
            *args:

        Returns:

        """
        for layer in range(self.num_layers):
            yield layer, (
                (direction, tuple(arg[self.num_directions*layer + direction] for arg in args))
                for direction in range(self.num_directions)
            )

    def initialize_cells(self):
        cells = []
        rename_map = {}
        for layer, directions in self.iterate_layers():
            for direction, _ in directions:
                layer_input_size = self.input_size if layer == 0 else self.hidden_size * self.num_directions

                cell = self.cell_type(layer_input_size, self.hidden_size, bias=self.bias, **self.cell_params)
                cells.append(cell)

                suffix = "_reverse" if direction == 1 else ""
                cell_name = f"l{layer}{suffix}"
                setattr(self, cell_name, cell)

                components = ["weight"] + ["bias" if self.bias else []]
                matrices = ["ih", "hh"]
                for c in components:
                    for m in matrices:
                        rename_map[f"{cell_name}.{m}.{c}"] = f"{c}_{m}_{cell_name}"

        self.set_rename_map(rename_map)
        return cells

    # TODO: rewrite
    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.shape[-1]:
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.shape[-1]))

    # TODO: rewrite
    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.shape[0] if self.batch_first else input.shape[1]
        num_directions = 2 if self.bidirectional else 1
        if self.proj_size > 0:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.proj_size)
        else:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.hidden_size)
        return expected_hidden_size

    # TODO: rewrite
    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    # TODO: rewrite
    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)


class DPRNN(DPRNNBase):
    r"""Applies a multi-layer Elman RNN with :math:`\tanh` or :math:`\text{ReLU}` non-linearity to an
    input sequence.

    TODO
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
        nonlinearity: Literal['tanh', 'relu'] = 'tanh',
    ) -> None:
        super().__init__(
            DPRNNCell,
            input_size,
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            cell_params=dict(
                nonlinearity=nonlinearity
            ),
        )


class DPGRU(DPRNNBase):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

    TODO
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            DPGRUCell,
            input_size,
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )


class DPLSTM(DPRNNBase):
    r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.

    TODO

    DP-friendly drop-in replacement of the ``torch.nn.LSTM`` module.

    Its state_dict matches that of nn.LSTM exactly, so that after training it can be exported
    and loaded by an nn.LSTM for inference.

    Refer to nn.LSTM's documentation for all parameters and inputs.

    Not supported: proj_size.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            DPLSTMCell,
            input_size,
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

