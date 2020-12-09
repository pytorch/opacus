import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .param_rename import ParamRenamedModule


class LSTMLinear(nn.Linear):
    r"""
    This function is the same as a nn.Linear layer, except that in the backward pass
    the grad_samples get accumulated (instead of being concatenated as in the standard
    nn.Linear)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)


class DPLSTMCell(nn.Module):
    r"""
    Internal-only class. Implements *one* step of LSTM so that a LSTM layer can be seen as repeated
    applications of this class.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.ih = LSTMLinear(input_size, 4 * hidden_size, bias=self.bias)
        self.hh = LSTMLinear(hidden_size, 4 * hidden_size, bias=self.bias)

        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Resets parameters by initializing them from an uniform distribution.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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


class DPLSTMLayer(nn.Module):
    r"""
    Implements *one* layer of LSTM in a way amenable to differential privacy.
    We don't expect you to use this directly: use DPLSTM instead :)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        dropout: float,
        reverse: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.reverse = reverse

        self.cell = DPLSTMCell(
            input_size=input_size, hidden_size=hidden_size, bias=bias
        )

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        state_init: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Implements the forward pass of the DPLSTMLayer when a sequence is given in input.

        Args:
            x: Input sequence to the DPLSTMCell of shape ``[T, B, D]``.
            state_init: Initial state of the LSTMCell as a tuple ``(h_0, c_0)``
                where ``h_0`` is the initial hidden state and ``c_0`` is the
                initial cell state of the DPLSTMCell

        Returns:
            ``output, (h_n, c_n)`` where:
            - ``output`` is of shape ``[T, B, H]`` and is a tensor containing the output
                features (``h_t``) from the last layer of the DPLSTMCell for each timestep ``t``.
            - ``h_n`` is of shape ``[B, H]`` and is a tensor containing the hidden state for ``t = T``.
            - ``c_n`` is of shape ``[B, H]`` tensor containing the cell state for ``t = T``.
        """

        seq_length, batch_sz, _ = x.shape
        if self.reverse:
            x = x.flip(0)
        x = torch.unbind(x, dim=0)

        h_0, c_0 = state_init

        h_n = [h_0]
        c_n = [c_0]

        for t in range(seq_length):
            h_next, c_next = self.cell(x[t], h_n[t], c_n[t])
            if self.dropout:
                h_next = self.dropout_layer(h_next)
            h_n.append(h_next)
            c_n.append(c_next)

        h_n = torch.stack(h_n[1:], dim=0)  # [T, B, H], init step not part of output

        return (
            h_n.flip(0) if self.reverse else h_n,  # Flip the output...
            (h_n[-1], c_n[-1]),  # ... But not the states
        )


class BidirectionalDPLSTMLayer(nn.Module):
    r"""
    Implements *one* layer of Bidirectional LSTM in a way amenable to differential privacy.
    We don't expect you to use this directly: use DPLSTM instead :)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        dropout: float,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout

        # nn.LSTM (as of November 2020) only implements a "type 2" multilayer bidirectional LSTM.
        # See https://github.com/pytorch/pytorch/issues/4930 for the definition of type 1 and type 2
        # and for discussion. When the PR to extend nn.LSTM to Type 1 lands, we will extend this
        # accordingly.

        self.forward_layer = DPLSTMLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            dropout=dropout,
            reverse=False,
        )
        self.reverse_layer = DPLSTMLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            dropout=dropout,
            reverse=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state_init: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Implements the forward pass of the DPLSTM when a sequence is input.

        Dimensions as follows:
            - B: Batch size
            - T: Sequence length
            - D: LSTM input hidden size (eg from a word embedding)
            - H: LSTM output hidden size
            - P: number of directions (2 if bidirectional, else 1)

        Args:
            x: Input sequence to the DPLSTM of shape ``[T, B, D]``
            state_init: Initial state of the LSTM as a tuple ``(h_0, c_0)``, where:
                    - h_0 of shape ``[P, B, H]  contains the initial hidden state
                    - c_0 of shape ``[P, B, H]  contains the initial cell state
                    This argument can be (and defaults to) None, in which case zero tensors will be used.

         Returns:
            ``output, (h_n, c_n)`` where:
            - ``output`` is of shape ``[T, B, H * P]`` and is a tensor containing the output
                features (``h_t``) from the last layer of the DPLSTM for each timestep ``t``.
            - ``h_n`` is of shape ``[P, B, H]`` and contains the hidden state for ``t = T``.
            - ``c_n`` is of shape ``[P, B, H]`` and contains the cell state for ``t = T``.
        """

        h0, c0 = state_init

        h0_f, h0_r = h0.unbind(0)  # each of shape [B, H] for their layer
        c0_f, c0_r = c0.unbind(0)  # each of shape [B, H] for their layer

        out_f, (h_f, c_f) = self.forward_layer(x, (h0_f, c0_f))
        out_r, (h_r, c_r) = self.reverse_layer(x, (h0_r, c0_r))

        out = torch.cat([out_f, out_r], dim=-1)  # [T, B, H * P]

        h = torch.stack([h_f, h_r], dim=0)  # [P, B, H]
        c = torch.stack([c_f, c_r], dim=0)  # [P, B, H]
        return out, (h, c)


class DPLSTM(ParamRenamedModule):
    r"""
    DP-friendly drop-in replacement of the ``torch.nn.LSTM`` module.

    Its state_dict matches that of nn.LSTM exactly, so that after training it can be exported
    and loaded by an nn.LSTM for inference.

    Refer to nn.LSTM's documentation for all parameters and inputs.
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
        rename_dict = self._make_rename_dict(num_layers, bias, bidirectional)
        super().__init__(rename_dict)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        LayerClass = BidirectionalDPLSTMLayer if bidirectional else DPLSTMLayer

        self.layers = nn.ModuleList(
            [
                LayerClass(
                    input_size=self.input_size
                    if i == 0
                    else self.hidden_size * self.num_directions,
                    hidden_size=self.hidden_size,
                    bias=self.bias,
                    dropout=self.dropout if i < self.num_layers - 1 else 0,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        state_init: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Implements the forward pass of the DPLSTM when a sequence is input.

        Dimensions as follows:
            - B: Batch size
            - T: Sequence length
            - D: LSTM input hidden size (eg from a word embedding)
            - H: LSTM output hidden size
            - L: number of layers in the LSTM
            - P: number of directions (2 if bidirectional, else 1)

        Args:
            x: Input sequence to the DPLSTM of shape ``[T, B, D]``
            state_init: Initial state of the LSTM as a tuple ``(h_0, c_0)``, where:
                    - h_0 of shape ``[L*P, B, H]  contains the initial hidden state
                    - c_0 of shape ``[L*P, B, H]  contains the initial cell state
                    This argument can be (and defaults to) None, in which case zero tensors will be used.

         Returns:
            ``output, (h_n, c_n)`` where:
            - ``output`` is of shape ``[T, B, H * P]`` and is a tensor containing the output
                features (``h_t``) from the last layer of the DPLSTM for each timestep ``t``.
            - ``h_n`` is of shape ``[L * P, B, H]`` and contains the hidden state for ``t = T``.
            - ``c_n`` is of shape ``[L * P, B, H]`` and contains the cell state for ``t = T``.
        """

        x = self._rearrange_batch_dim(x)
        T, B, D = x.shape
        L = self.num_layers
        P = 2 if self.bidirectional else 1
        H = self.hidden_size

        h_0s, c_0s = state_init or (None, None)

        if h_0s is None:
            h_0s = torch.zeros(
                L,
                P,
                B,
                self.hidden_size,
                dtype=x[0].dtype,
                device=x[0].device,
            )
        else:
            h_0s = h_0s.reshape([L, P, B, H])
        if c_0s is None:
            c_0s = torch.zeros(
                L,
                P,
                B,
                self.hidden_size,
                dtype=x[0].dtype,
                device=x[0].device,
            )
        else:
            c_0s = c_0s.reshape([L, P, B, H])

        hs: List[torch.Tensor] = []
        cs: List[torch.Tensor] = []

        for layer, h0, c0 in zip(self.layers, h_0s, c_0s):
            if not self.bidirectional:
                h0 = h0.squeeze()
                c0 = c0.squeeze()
            x, (h, c) = layer(x, (h0, c0))
            if not self.bidirectional:
                h = h.unsqueeze(0)  # [1, B, H]
                c = c.unsqueeze(0)  # [1, B, H]

            hs.append(h)
            cs.append(c)

        hs = torch.cat(hs, dim=0)  # [L * P, B, H]
        cs = torch.cat(cs, dim=0)  # [L * P, B, H]
        out = self._rearrange_batch_dim(x)
        return out, (hs, cs)

    def _rearrange_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:  # batch is by default in second dimension
            x = x.transpose(0, 1)
        return x

    def __repr__(self):
        s = f"DPLSTM({self.input_size}, {self.hidden_size}, bias={self.bias}"

        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"

        if self.dropout:
            s += f", dropout={self.dropout}"

        if self.bidirectional:
            s += f", bidirectional={self.bidirectional}"

        return s

    def _make_rename_dict(self, num_layers, bias, bidirectional):
        """
        Programmatically constructs a dictionary old_name -> new_name to align with the param
        names used in ``torch.nn.LSTM``.
        """
        d = {}
        components = ["weight"] + ["bias" if bias else []]
        matrices = ["ih", "hh"]
        for i in range(num_layers):
            for c in components:
                for m in matrices:
                    nn_name = f"{c}_{m}_l{i}"
                    if bidirectional:
                        d[f"layers.{i}.forward_layer.cell.{m}.{c}"] = nn_name
                        d[f"layers.{i}.reverse_layer.cell.{m}.{c}"] = (
                            nn_name + "_reverse"
                        )
                    else:
                        d[f"layers.{i}.cell.{m}.{c}"] = nn_name

        return d
