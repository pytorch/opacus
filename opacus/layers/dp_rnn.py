#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import numbers
import warnings
from typing import List, Optional, Tuple, Union, Type, Literal

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_sequence

from .param_rename import ParamRenamedMixin


# TODO: explain max_batch_len in grad_sample.dp_rnn

def apply_permutation(tensor: torch.Tensor, dim: int, permutation: Optional[torch.Tensor]):
    if permutation is None:
        return tensor
    return tensor.index_select(dim, permutation)


def _compute_seq_lengths(batch_sizes: torch.Tensor) -> List[int]:
    r"""
    Computes the sequence lengths (the length parameter used in the packed_padded_sequence function to create a PackedSequence).

    Args:
        batch_sizes: Contains the batch sizes as stored in a PackedSequence

    Returns:
        running_seq_lengths: the length parameter used in the torch.nn.utils.rnn.packed_padded_sequence function to create a PackedSequence.
        It's a list of the same length as batch_sizes.
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


def _compute_last_states(
        h_n: List[torch.Tensor], seq_lengths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_batch_size = len(seq_lengths)
    hidden_size = h_n[0].shape[-1]
    h_last = torch.zeros(max_batch_size, hidden_size)

    for i, seq_len in enumerate(seq_lengths):
        h_last[i, :] = h_n[seq_len - 1][i, :]

    return h_last


def _compute_last_states_lstm(
    h_n: List[torch.Tensor], c_n: List[torch.Tensor], seq_lengths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Given h and c values of all time steps, this function computes the h and c values for each sequence at their last timestep (this can vary across sequences with different sequence lengths).

    Args:
        h_n: A list of hidden state values across all timesteps.
        c_n: A list of cell state values across all timesteps.
        seq_lengths: the length parameter used in the torch.nn.utils.rnn.packed_padded_sequence function to create a PackedSequence. This can be computed using the _compute_seq_lengths function.

    Returns:
        h_last: Contains the last hidden state values for each of the sequences.
                If the i'th sequence has a length of l_i, then h_last[i,:] contains the hidden state corresponding to the i'th sequence at timestep l_i.
        c_last: The structure is the same as h_last, except that it contains the last cell state values for each of the sequences.
    """

    max_batch_size = len(seq_lengths)
    hidden_size = h_n[0].shape[-1]
    h_last = torch.zeros(max_batch_size, hidden_size)
    c_last = torch.zeros(max_batch_size, hidden_size)

    for i, seq_len in enumerate(seq_lengths):
        h_last[i, :] = h_n[seq_len - 1][i, :]
        c_last[i, :] = c_n[seq_len - 1][i, :]

    return h_last, c_last


def _concat_sequence_directions(
    forward: Union[List[torch.Tensor], Tuple[torch.Tensor]],
    reverse: Union[List[torch.Tensor], Tuple[torch.Tensor]],
    dim: int,
) -> Tuple[torch.Tensor]:
    r"""
    Given two list/tuple of same length containing tensors, this function returns a concatenation along dimension d. So, output[i] : concatenation of forward[i] and reverse[i] along dimension dim.
    forward[i] and reverse[i] should have the same shape. This function is used for concatenating the outputs of the forward and reverse layer of a bidirectional LSTM.

    Args:
        forward: list/tuple containing n tensors, representing the output of the forward layer.
        reverse: list/tuple containing n tensors, representing the output of the backward layer.
        dim: the dimension along which the sequence of tensors within forward and reverse will be concatenated.
    Returns:
        output: list/tuple containing n concatenated tensors.
    """

    if len(forward) != len(reverse):
        raise ValueError(
            "The forward and reverse layer output sequences should have the same length"
        )

    seq_length = len(forward)
    output = [0] * seq_length

    for i in range(seq_length):
        output[i] = torch.cat((forward[i], reverse[i]), dim=dim)

    return output


class RNNLinear(nn.Linear):
    r"""
    This function is the same as a nn.Linear layer, except that in the backward pass
    the grad_samples get accumulated (instead of being concatenated as in the standard
    nn.Linear)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)


class DPRNNCellBase(nn.Module):
    has_cell_state: bool = False

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int) -> None:
        super(DPRNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.ih = RNNLinear(input_size, num_chunks * hidden_size, bias)
        self.hh = RNNLinear(hidden_size, num_chunks * hidden_size, bias)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets parameters by initializing them from an uniform distribution.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def set_max_batch_length(self, max_batch_length: int) -> None:
        """
        Sets max batch length
        """
        self.ih.max_batch_len = max_batch_length
        self.hh.max_batch_len = max_batch_length


class DPRNNCell(DPRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool, nonlinearity: str = 'tanh'):
        super(DPRNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1)
        self.nonlinearity = nonlinearity

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        batch_size_t: Optional[int] = None,
    ) -> torch.Tensor:
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)

        h_prev = hx
        gates = self.ih(input) + self.hh(h_prev if batch_size_t is None else h_prev[:batch_size_t, :])
        if self.nonlinearity == 'tanh':
            h_t = torch.tanh(gates)
        elif self.nonlinearity == 'relu':
            h_t = torch.relu(gates)
        else:
            h_t = gates
            raise RuntimeError("Unknown nonlinearity: {}".format(self.nonlinearity))
        return h_t


class DPGRUCell(DPRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool):
        super(DPGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        batch_size_t: Optional[int] = None,
    ) -> torch.Tensor:
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)

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
    has_cell_state = True

    def __init__(self, input_size: int, hidden_size: int, bias: bool):
        super(DPLSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        batch_size_t: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
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
        i_t = torch.sigmoid(
            i_t_input
        )  # [B, D] or [batch_size_t, D] if batch_size_t is not None
        f_t = torch.sigmoid(
            f_t_input
        )  # [B, D] or [batch_size_t, D] if batch_size_t is not None
        g_t = torch.tanh(
            g_t_input
        )  # [B, D] or [batch_size_t, D] if batch_size_t is not None
        o_t = torch.sigmoid(
            o_t_input
        )  # [B, D] or [batch_size_t, D] if batch_size_t is not None
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
    ):
        super(DPRNNBase, self).__init__()

        self.cell_params = {}
        if isinstance(mode, str):
            if mode not in RNN_CELL_TYPES:
                raise ValueError(f"Invalid RNN mode='{mode}', available options: {list(RNN_CELL_TYPES.keys())}")
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
        num_directions = 2 if bidirectional else 1

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
        self.cells = []
        self.cell_layer = []
        self.cell_direction = []

        rename_map = {}
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                cell = self.cell_type(layer_input_size, hidden_size, bias=bias, **self.cell_params)

                self.cells.append(cell)
                self.cell_layer.append(layer)
                self.cell_direction.append(direction)

                suffix = "_reverse" if direction == 1 else ""
                cell_name = f'l{layer}{suffix}'
                setattr(self, cell_name, cell)

                components = ["weight"] + ["bias" if self.bias else []]
                matrices = ["ih", "hh"]
                for c in components:
                    for m in matrices:
                        rename_map[f"{cell_name}.{m}.{c}"] = f"{c}_{m}_{cell_name}"
        self.set_rename_map(rename_map)

    def forward(
            self,
            input: Union[torch.Tensor, PackedSequence],
            state_init: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[Union[torch.Tensor, PackedSequence], Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
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

            assert isinstance(input, torch.Tensor)
            x = self._rearrange_batch_dim(input)

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

        layer_outs = []
        hs = []
        cs = []

        for cell, layer, direction, h0, c0 in zip(self.cells, self.cell_layer, self.cell_direction, h_0s, c_0s):

            # apply single direction layer (with dropout)
            out_layer, state_layer = self.forward_layer(
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

            if self.has_cell_state:
                h, c = state_layer
                cs.append(c)
            else:
                h = state_layer
            hs.append(h)

            layer_outs.append(out_layer)

            if direction == num_directions - 1:
                # aggregate all outputs to y

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

                layer_outs = []

        if is_packed:
            packed_data = torch.cat(output, dim=0) # [TB, P*H]
            output = PackedSequence(packed_data, batch_sizes, sorted_indices, unsorted_indices)
        else:
            output = self._rearrange_batch_dim(output)

        hs = torch.stack(hs, dim=0)  # [L * P, B, H]
        hs = apply_permutation(hs, 1, unsorted_indices)
        if self.has_cell_state:
            cs = torch.stack(cs, dim=0)  # [L * P, B, H]
            cs = apply_permutation(cs, 1, unsorted_indices)

        hidden = (hs, cs) if self.has_cell_state else hs

        return output, hidden

    def forward_layer(
            self,
            x: Union[torch.Tensor, PackedSequence],
            h_0: torch.Tensor,
            c_0: Optional[torch.Tensor],
            batch_sizes: torch.Tensor,
            cell: DPRNNCellBase,
            max_batch_size: int,
            seq_length: int,
            is_packed: bool,
            reverse_layer: bool,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
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
        if self.has_cell_state:
            c_n = [c_0]
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
            if self.has_cell_state:
                c_n.append(c_next)
            batch_size_prev = h_next.shape[0]

        if is_packed:
            seq_lengths = _compute_seq_lengths(batch_sizes)
            h_temp = h_n[1:] # list T x [B, H]
            if self.has_cell_state:
                c_temp = c_n[1:]

            # h_last = _compute_last_states(h_temp, seq_lengths)
            h_last = torch.zeros(max_batch_size, self.hidden_size) # [B, H]
            if self.has_cell_state:
                c_last = torch.zeros(max_batch_size, self.hidden_size) # [B, H]
            for i, seq_len in enumerate(seq_lengths):
                h_last[i, :] = h_temp[seq_len - 1][i, :]
                if self.has_cell_state:
                    c_last[i, :] = c_temp[seq_len - 1][i, :]
            if reverse_layer:
                h_temp = tuple(reversed(h_temp))

        else:
            h_n = torch.stack(h_n[1:], dim=0)  # [T, B, H], init step not part of output
            h_temp = h_n.flip(0) if reverse_layer else h_n  # Flip the output...
            h_last = h_n[-1]  # ... But not the states
            if self.has_cell_state:
                c_last = c_n[-1]

        if self.has_cell_state:
            return h_temp, (h_last, c_last)
        else:
            return h_temp, h_last

    def _rearrange_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:  # batch is by default in second dimension
            x = x.transpose(0, 1)
        return x

    def check_input(self, input: torch.Tensor, batch_sizes: Optional[torch.Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.shape[-1]:
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.shape[-1]))

    def get_expected_hidden_size(self, input: torch.Tensor, batch_sizes: Optional[torch.Tensor]) -> Tuple[int, int, int]:
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

    def check_hidden_size(self, hx: torch.Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    def check_forward_args(self, input: torch.Tensor, hidden: torch.Tensor, batch_sizes: Optional[torch.Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)


class DPRNN(DPRNNBase):

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
    ):
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

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0,
            bidirectional: bool = False,
    ):
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
    r"""
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
    ):
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

