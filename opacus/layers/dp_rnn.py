#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import numbers
import warnings
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from ..utils.packed_sequences import compute_seq_lengths
from .param_rename import RenameParamsMixin


def apply_permutation(tensor: Tensor, dim: int, permutation: Optional[Tensor]):
    """
    Permute elements of a tensor along a dimension `dim`. If permutation is None do nothing.
    """
    if permutation is None:
        return tensor
    return tensor.index_select(dim, permutation)


class RNNLinear(nn.Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module is the same as a ``torch.nn.Linear``` layer, except that in the backward pass
    the grad_samples get accumulated (instead of being concatenated as in the standard
    nn.Linear).

    When used with `PackedSequence`s, additional attribute `max_batch_len` is defined to determine
    the size of per-sample grad tensor.
    """

    max_batch_len: int

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)


class DPRNNCellBase(nn.Module):
    has_cell_state: bool = False

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, num_chunks: int
    ) -> None:
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

    DP-friendly drop-in replacement of the ``torch.nn.RNNCell`` module to use in ``DPRNN``.
    Refer to ``torch.nn.RNNCell`` documentation for the model description, parameters and inputs/outputs.
    """

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool, nonlinearity: str = "tanh"
    ) -> None:
        super().__init__(input_size, hidden_size, bias, num_chunks=1)
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        self.nonlinearity = nonlinearity

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tensor] = None,
        batch_size_t: Optional[int] = None,
    ) -> Tensor:
        if hx is None:
            hx = torch.zeros(
                input.shape[0], self.hidden_size, dtype=input.dtype, device=input.device
            )

        h_prev = hx
        gates = self.ih(input) + self.hh(
            h_prev if batch_size_t is None else h_prev[:batch_size_t, :]
        )
        if self.nonlinearity == "tanh":
            h_t = torch.tanh(gates)
        elif self.nonlinearity == "relu":
            h_t = torch.relu(gates)
        else:
            raise RuntimeError(f"Unknown nonlinearity: {self.nonlinearity}")
        return h_t


class DPGRUCell(DPRNNCellBase):
    """A gated recurrent unit (GRU) cell

    DP-friendly drop-in replacement of the ``torch.nn.GRUCell`` module to use in ``DPGRU``.
    Refer to ``torch.nn.GRUCell`` documentation for the model description, parameters and inputs/outputs.
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
            hx = torch.zeros(
                input.shape[0], self.hidden_size, dtype=input.dtype, device=input.device
            )

        h_prev = hx if batch_size_t is None else hx[:batch_size_t, :]
        gates_x = self.ih(input)
        gates_h = self.hh(h_prev)
        r_t_input_x, z_t_input_x, n_t_input_x = torch.split(
            gates_x, self.hidden_size, 1
        )
        r_t_input_h, z_t_input_h, n_t_input_h = torch.split(
            gates_h, self.hidden_size, 1
        )
        r_t = torch.sigmoid(r_t_input_x + r_t_input_h)
        z_t = torch.sigmoid(z_t_input_x + z_t_input_h)
        n_t = torch.tanh(n_t_input_x + r_t * n_t_input_h)
        h_t = (1 - z_t) * n_t + z_t * h_prev
        return h_t


class DPLSTMCell(DPRNNCellBase):
    """A long short-term memory (LSTM) cell.

    DP-friendly drop-in replacement of the ``torch.nn.LSTMCell`` module to use in ``DPLSTM``.
    Refer to ``torch.nn.LSTMCell`` documentation for the model description, parameters and inputs/outputs.
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
            zeros = torch.zeros(
                input.shape[0], self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (zeros, zeros)

        h_prev, c_prev = hx

        if batch_size_t is None:
            gates = self.ih(input) + self.hh(h_prev)  # [B, 4*D]
        else:
            gates = self.ih(input) + self.hh(
                h_prev[:batch_size_t, :]
            )  # [batch_size_t, 4*D]

        i_t_input, f_t_input, g_t_input, o_t_input = torch.split(
            gates, self.hidden_size, 1
        )

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
    "RNN_TANH": (DPRNNCell, {"nonlinearity": "tanh"}),
    "RNN_RELU": (DPRNNCell, {"nonlinearity": "relu"}),
    "GRU": (DPGRUCell, {}),
    "LSTM": (DPLSTMCell, {}),
}


class DPRNNBase(RenameParamsMixin, nn.Module):
    """Base class for all RNN-like sequence models.

    DP-friendly drop-in replacement of the ``torch.nn.RNNBase`` module.
    After training this module can be exported and loaded by the original ``torch.nn`` implementation for inference.

    This module implements multi-layer (Type-2, see [this issue](https://github.com/pytorch/pytorch/issues/4930#issuecomment-361851298))
    bi-directional sequential model based on abstract cell. Cell should be a subclass of ``DPRNNCellBase``.

    Limitations:
    - proj_size > 0 is not implemented
    - this implementation doesn't use cuDNN
    """

    def __init__(
        self,
        mode: Union[str, Type[DPRNNCellBase]],
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        cell_params: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.cell_params = {}
        if isinstance(mode, str):
            if mode not in RNN_CELL_TYPES:
                raise ValueError(
                    f"Invalid RNN mode '{mode}', available options: {list(RNN_CELL_TYPES.keys())}"
                )
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

        if (
            not isinstance(dropout, numbers.Number)
            or not 0 <= dropout <= 1
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        if dropout > 0 and num_layers == 1:
            warnings.warn(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                "num_layers greater than 1, but got dropout={} and "
                "num_layers={}".format(dropout, num_layers)
            )

        if proj_size > 0:
            raise NotImplementedError("proj_size > 0 is not supported")
        if proj_size < 0:
            raise ValueError(
                "proj_size should be a positive integer or zero to disable projections"
            )
        if proj_size >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.cells = self.initialize_cells()

    # flake8: noqa C901
    def forward(
        self,
        input: Union[Tensor, PackedSequence],
        state_init: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Union[Tensor, PackedSequence], Union[Tensor, Tuple[Tensor, Tensor]]]:
        """
        Forward pass of a full RNN, containing one or many single- or bi-directional layers. Implemented for
        an abstract cell type.

        Note: ``proj_size > 0`` is not supported here. Cell state size is always equal to hidden state size.

        Inputs: input, h_0/(h_0, c_0)
            input: Input sequence. Tensor of shape ``[T, B, D]`` (``[B, T, D]`` if ``batch_first=True``)
                   or PackedSequence.
            h_0: Initial hidden state for each element in the batch. Tensor of shape ``[L*P, B, H]``. Default to zeros.
            c_0: Initial cell state for each element in the batch. Only for cell types with an additional state.
                 Tensor of shape ``[L*P, B, H]``. Default to zeros.

        Outputs: output, h_n/(h_n, c_n)
            output: Output features (``h_t``) from the last layer of the model for each ``t``. Tensor of
                    shape ``[T, B, P*H]`` (``[B, T, P*H]`` if ``batch_first=True``), or PackedSequence.
            h_n: Final hidden state for each element in the batch. Tensor of shape ``[L*P, B, H]``.
            c_n: Final cell state for each element in the batch. Tensor of shape ``[L*P, B, H]``.

        where
            T = sequence length
            B = batch size
            D = input_size
            H = hidden_size
            L = num_layers
            P = num_directions (2 if `bidirectional=True` else 1)
        """
        num_directions = 2 if self.bidirectional else 1

        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input_data, batch_sizes, sorted_indices, unsorted_indices = input
            dtype, device = input_data.dtype, input_data.device

            x = input_data.split(tuple(batch_sizes))  # tuple T x [B, D]

            seq_length = len(batch_sizes)
            max_batch_size = int(batch_sizes[0])

            for cell in self.cells:
                cell.set_max_batch_length(max_batch_size)
        else:
            dtype, device = input.dtype, input.device
            batch_sizes = None
            sorted_indices = None
            unsorted_indices = None

            # Rearrange batch dim. Batch is by default in second dimension.
            if self.batch_first:
                input = input.transpose(0, 1)

            x = input  # [T, B, D]

            seq_length = x.shape[0]
            max_batch_size = x.shape[1]

        if self.has_cell_state:
            h_0s, c_0s = state_init or (None, None)
        else:
            h_0s, c_0s = state_init, None

        if h_0s is None:
            h_0s = torch.zeros(  # [L*P, B, H]
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=dtype,
                device=device,
            )
        else:
            h_0s = apply_permutation(h_0s, 1, sorted_indices)

        if self.has_cell_state:
            if c_0s is None:
                c_0s = torch.zeros(  # [L*P, B, H]
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=dtype,
                    device=device,
                )
            else:
                c_0s = apply_permutation(c_0s, 1, sorted_indices)
        else:
            c_0s = [None] * len(h_0s)

        hs = []
        cs = []  # list of None if no cell state
        output = None

        for layer, directions in self.iterate_layers(self.cells, h_0s, c_0s):
            layer_outs = []

            for direction, (cell, h0, c0) in directions:
                # apply single direction layer (with dropout)
                out_layer, h, c = self.forward_layer(
                    x
                    if layer == 0
                    else output,  # [T, B, D/H/2H] / tuple T x [B, D/H/2H]
                    h0,  # [B, H]
                    c0,
                    batch_sizes,
                    cell=cell,
                    max_batch_size=max_batch_size,
                    seq_length=seq_length,
                    is_packed=is_packed,
                    reverse_layer=(direction == 1),
                )

                hs.append(h)  # h: [B, H]
                cs.append(c)
                layer_outs.append(out_layer)  # out_layer: [T, B, H] / tuple T x [B, H]

            if is_packed:
                output = [  # tuple T x [B, P*H]
                    torch.cat([layer_out[i] for layer_out in layer_outs], dim=1)
                    for i in range(seq_length)
                ]
            else:
                output = torch.cat(layer_outs, dim=2)  # [T, B, P*H]

        if is_packed:
            packed_data = torch.cat(output, dim=0)  # [TB, P*H]
            output = PackedSequence(
                packed_data, batch_sizes, sorted_indices, unsorted_indices
            )
        else:
            # Rearrange batch dim back
            if self.batch_first:
                output = output.transpose(0, 1)

        hs = torch.stack(hs, dim=0)  # [L*P, B, H]
        hs = apply_permutation(hs, 1, unsorted_indices)
        if self.has_cell_state:
            cs = torch.stack(cs, dim=0)  # [L*P, B, H]
            cs = apply_permutation(cs, 1, unsorted_indices)

        hidden = (hs, cs) if self.has_cell_state else hs

        return output, hidden

    # flake8: noqa C901
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
        """
        Forward pass of a single RNN layer (one direction). Implemented for an abstract cell type.

        Inputs: x, h_0, c_0
            x: Input sequence. Tensor of shape ``[T, B, D]`` or PackedSequence if `is_packed = True`.
            h_0: Initial hidden state. Tensor of shape ``[B, H]``.
            c_0: Initial cell state. Tensor of shape ``[B, H]``. Only for cells with additional
                 state `c_t`, e.g. DPLSTMCell.

        Outputs: h_t, h_last, c_last
            h_t: Final hidden state, output features (``h_t``) for each timestep ``t``. Tensor of
                shape ``[T, B, H]`` or list of length ``T`` with tensors ``[B, H]`` if PackedSequence is used.
            h_last: The last hidden state. Tensor of shape ``[B, H]``.
            c_last: The last cell state. Tensor of shape ``[B, H]``. None if cell has no additional state.

        where
            T = sequence length
            B = batch size
            D = input_size (for this specific layer)
            H = hidden_size (output size, for this specific layer)

        Args:
            batch_sizes: Contains the batch sizes as stored in PackedSequence
            cell: Module implementing a single cell of the network, must be an instance of DPRNNCell
            max_batch_size: batch size
            seq_length: sequence length
            is_packed: whether PackedSequence is used as input
            reverse_layer: if True, it will run forward pass for a reversed layer
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
                        c_cat = torch.cat(
                            (c_n[t], c_0[batch_size_prev:batch_size_t, :]), 0
                        )
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
            h_temp = h_n[1:]  # list T x [B, H]
            c_temp = c_n[1:]

            # Collect last states for all sequences
            seq_lengths = compute_seq_lengths(batch_sizes)
            h_last = torch.zeros(max_batch_size, self.hidden_size)  # [B, H]
            c_last = (
                torch.zeros(max_batch_size, self.hidden_size)
                if self.has_cell_state
                else None
            )
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
        Iterate through all the layers and through all directions within each layer.

        Arguments should be list-like of length ``num_layers * num_directions`` where
        each element corresponds to (layer, direction) pair. The corresponding elements
        of each of these lists will be iterated over.

        Example:
            num_layers = 3
            bidirectional = True

            for layer, directions in self.iterate_layers(self.cell, h):
                for dir, (cell, hi) in directions:
                    print(layer, dir, hi)

            # 0 0 h[0]
            # 0 1 h[1]
            # 1 0 h[2]
            # 1 1 h[3]
            # 2 0 h[4]
            # 2 1 h[5]

        """
        for layer in range(self.num_layers):
            yield layer, (
                (
                    direction,
                    tuple(arg[self.num_directions * layer + direction] for arg in args),
                )
                for direction in range(self.num_directions)
            )

    def initialize_cells(self):
        cells = []
        rename_map = {}
        for layer, directions in self.iterate_layers():
            for direction, _ in directions:
                layer_input_size = (
                    self.input_size
                    if layer == 0
                    else self.hidden_size * self.num_directions
                )

                cell = self.cell_type(
                    layer_input_size,
                    self.hidden_size,
                    bias=self.bias,
                    **self.cell_params,
                )
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


class DPRNN(DPRNNBase):
    """Applies a multi-layer Elman RNN with :math:`\tanh` or :math:`\text{ReLU}` non-linearity to an
    input sequence.

    DP-friendly drop-in replacement of the ``torch.nn.RNN`` module.
    Refer to ``torch.nn.RNN`` documentation for the model description, parameters and inputs/outputs.

    After training this module can be exported and loaded by the original ``torch.nn`` implementation for inference.
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
        proj_size: int = 0,
        nonlinearity: str = "tanh",
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
            proj_size=proj_size,
            cell_params={"nonlinearity": nonlinearity},
        )


class DPGRU(DPRNNBase):
    """Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

    DP-friendly drop-in replacement of the ``torch.nn.GRU`` module.
    Refer to ``torch.nn.GRU`` documentation for the model description, parameters and inputs/outputs.

    After training this module can be exported and loaded by the original ``torch.nn`` implementation for inference.
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
        proj_size: int = 0,
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
            proj_size=proj_size,
        )


class DPLSTM(DPRNNBase):
    """Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.

    DP-friendly drop-in replacement of the ``torch.nn.LSTM`` module.
    Refer to ``torch.nn.LSTM`` documentation for the model description, parameters and inputs/outputs.

    After training this module can be exported and loaded by the original ``torch.nn`` implementation for inference.
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
        proj_size: int = 0,
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
            proj_size=proj_size,
        )
