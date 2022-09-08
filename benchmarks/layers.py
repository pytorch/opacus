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

from enum import Enum
from typing import Callable, Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModule
from opacus.grad_sample.utils import wrap_model
from opacus.layers import DPGRU, DPLSTM, DPRNN, DPMultiheadAttention
from opacus.layers.dp_rnn import DPRNNBase


class LayerType:
    LINEAR: str = "linear"
    CONV: str = "conv"
    LAYERNORM: str = "layernorm"
    INSTANCENORM: str = "instancenorm"
    GROUPNORM: str = "groupnorm"
    EMBEDDING: str = "embedding"
    MHA: str = "mha"
    DPMHA: str = "dpmha"
    RNN: str = "rnn"
    DPRNN: str = "dprnn"
    GRU: str = "gru"
    DPGRU: str = "dpgru"
    LSTM: str = "lstm"
    DPLSTM: str = "dplstm"


class Layer:
    _input_tensor: torch.Tensor
    _module: nn.Module
    _labels: torch.Tensor

    def __init__(
        self,
        *,
        random_seed: Optional[int] = None,
        criterion: Callable = F.cross_entropy,
    ) -> None:
        """Sets random seed and criterion."""
        if random_seed is not None:
            torch.manual_seed(random_seed)
        self._criterion = criterion

    @staticmethod
    def _get_memory_difference(device: torch.device, stats: Dict[str, int]) -> int:
        """If applicable, computes the device's CUDA memory difference between the
        sum of the values in the stats dict and the current allocated CUDA memory.

        Args:
            device: torch.device
            stats: dictionary from item names (e.g. input_tensor, module, labels)
            to their respective sizes

        Returns:
            device's CUDA memory difference between the sum of the values in the
            stats dict and the current allocated CUDA memory if given a CUDA
            device. 0 if given a CPU device.
        """
        if device.type == "cuda":
            return torch.cuda.memory_allocated(device) - sum(stats.values())
        return 0

    def _inputs_to(self, device: torch.device, stats: Dict[str, int]) -> Dict[str, int]:
        """Some modules (e.g. RNNs) take additional layer inputs such as initial
        hidden state. These modules should override this function accordingly.

        Args:
            device: torch.device
            stats: dictionary from item names (e.g. input_tensor, module, labels)
            to their respective sizes

        Returns: updated stats dictionary with input tensor sizes
        """
        self._input_tensor = self._input_tensor.to(device)
        stats["input_tensor"] = self._get_memory_difference(device=device, stats=stats)
        return stats

    def to(self, device: torch.device) -> Dict[str, int]:
        """Moves input_tensor, additional inputs, module, and labels to device.

        Args:
            device: torch.device

        Returns:
            Dictionary from item names (e.g. input_tensor, module, labels) to
            their respective sizes if CUDA device, else respective sizes are all
            set to 0.
        """
        stats: Dict[str, int] = {}
        stats["offset"] = self._get_memory_difference(device=device, stats=stats)

        # some modules take additional inputs such as hidden state
        stats = self._inputs_to(device=device, stats=stats)

        self._module = self._module.to(device)
        stats["layer"] = self._get_memory_difference(device=device, stats=stats)

        self._labels = self._labels.to(device)
        stats["labels"] = self._get_memory_difference(device=device, stats=stats)

        # check that all memory is accounted for
        if device.type == "cuda":
            assert torch.cuda.memory_allocated(device) == sum(stats.values())

        return stats

    def forward_only(self) -> torch.Tensor:
        return self._module(self._input_tensor)

    def forward_backward(self) -> None:
        preds = self.forward_only()
        loss = self._criterion(preds, self._labels)
        loss.backward()
        self._module.zero_grad()

    def make_private(self, gsm_mode: str = "hooks") -> None:
        self._module = wrap_model(self._module, grad_sample_mode=gsm_mode)

    @property
    def module(self):
        return self._module

    def __del__(self):
        self.to(torch.device("cpu"))


class LinearBase(Layer):
    def __init__(
        self,
        *,
        batch_size: int,
        input_shape: Tuple[int, ...],
        in_features: int,
        out_features: int,
        bias: bool = True,
        random_seed: Optional[int] = None,
        criterion: Callable = F.cross_entropy,
    ) -> None:
        super().__init__(random_seed=random_seed, criterion=criterion)
        self._input_tensor = torch.randn(batch_size, *input_shape, in_features)
        self._module = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        self._labels = torch.randn(batch_size, *input_shape, out_features)


class ConvBase(Layer):
    def __init__(
        self,
        *,
        batch_size: int,
        in_channels: int,
        input_shape: Tuple[int, ...],
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[str, int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        random_seed: Optional[int] = None,
        criterion: Callable = F.cross_entropy,
    ) -> None:
        super().__init__(random_seed=random_seed, criterion=criterion)

        D = len(input_shape)
        if D == 1:
            self._module_name = nn.Conv1d
        elif D == 2:
            self._module_name = nn.Conv2d
        elif D == 3:
            self._module_name = nn.Conv3d
        else:
            raise Exception("Input shape must be between 1 and 3 long")

        self._input_tensor = torch.randn(batch_size, in_channels, *input_shape)
        self._module = self._module_name(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        outputs = self._module(self._input_tensor)
        self._labels = torch.randn(outputs.shape)
        del outputs


class LayerNormBase(Layer):
    def __init__(
        self,
        *,
        batch_size: int,
        input_shape: Tuple[int, ...],
        D: int,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        random_seed: Optional[int] = None,
        criterion: Callable = F.cross_entropy,
    ) -> None:
        super().__init__(random_seed=random_seed, criterion=criterion)

        self._input_tensor = torch.randn(batch_size, *input_shape)
        self._module = nn.LayerNorm(
            normalized_shape=self._input_tensor.shape[-D:],
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
        self._labels = torch.randn(self._input_tensor.shape)


class InstanceNormBase(Layer):
    def __init__(
        self,
        *,
        batch_size: int,
        num_features: int,
        input_shape: Tuple[int, ...],
        eps: float = 1e-05,
        affine: bool = False,
        track_running_stats: bool = False,
        random_seed: Optional[int] = None,
        criterion: Callable = F.cross_entropy,
    ) -> None:
        super().__init__(random_seed=random_seed, criterion=criterion)

        D = len(input_shape)
        if D == 1:
            self._module_name = nn.InstanceNorm1d
        elif D == 2:
            self._module_name = nn.InstanceNorm2d
        elif D == 3:
            self._module_name = nn.InstanceNorm3d
        else:
            raise Exception("Input shape must be between 1 and 3 long")

        self._input_tensor = torch.randn(batch_size, num_features, *input_shape)
        self._module = self._module_name(
            num_features=num_features,
            eps=eps,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self._labels = torch.randn(self._input_tensor.shape)


class GroupNormBase(Layer):
    def __init__(
        self,
        *,
        batch_size: int,
        input_shape: Tuple[int, ...],
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
        random_seed: Optional[int] = None,
        criterion: Callable = F.cross_entropy,
    ) -> None:
        super().__init__(random_seed=random_seed, criterion=criterion)

        self._input_tensor = torch.randn(batch_size, num_channels, *input_shape)
        self._module = nn.GroupNorm(
            num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine
        )
        self._labels = torch.randn(self._input_tensor.shape)


class EmbeddingBase(Layer):
    def __init__(
        self,
        *,
        batch_size: int,
        input_shape: Tuple[int, ...],
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        random_seed: Optional[int] = None,
        criterion: Callable = F.cross_entropy,
    ) -> None:
        super().__init__(random_seed=random_seed, criterion=criterion)

        self._input_tensor = torch.randint(
            high=num_embeddings,
            size=(batch_size, *input_shape),
            dtype=torch.long,
        )
        self._module = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        self._labels = torch.randn(batch_size, *input_shape, embedding_dim)


class MHABase(Layer):
    def __init__(
        self,
        *,
        layer: Union[Type[nn.MultiheadAttention], Type[DPMultiheadAttention]],
        batch_size: int,
        source_seq_len: int,
        targ_seq_len: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        random_seed: Optional[int] = None,
        criterion: Callable = F.cross_entropy,
    ) -> None:
        super().__init__(random_seed=random_seed, criterion=criterion)

        kdim = kdim if kdim else embed_dim
        vdim = vdim if vdim else embed_dim

        self._input_tensor = (
            torch.randn(targ_seq_len, batch_size, embed_dim)
            if not batch_first
            else torch.randn(batch_size, targ_seq_len, embed_dim)
        )
        self._key = (
            torch.randn(source_seq_len, batch_size, kdim)
            if not batch_first
            else torch.randn(batch_size, source_seq_len, kdim)
        )
        self._value = (
            torch.randn(source_seq_len, batch_size, vdim)
            if not batch_first
            else torch.randn(batch_size, source_seq_len, vdim)
        )
        self._module = layer(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )
        self._labels = (
            torch.randn(targ_seq_len, batch_size, embed_dim)
            if not batch_first
            else torch.randn(batch_size, targ_seq_len, embed_dim)
        )

    def _inputs_to(self, device: torch.device, stats: Dict[str, int]) -> Dict[str, int]:
        """MultiheadAttention takes additional layer inputs key and value.

        Args:
            device: torch.device
            stats: dictionary from item names (e.g. input_tensor, module, labels)
            to their respective sizes

        Returns: updated stats dictionary with input tensor, key, and value size
        """
        stats = super()._inputs_to(device=device, stats=stats)
        self._key = self._key.to(device)
        stats["key"] = self._get_memory_difference(device=device, stats=stats)
        self._value = self._value.to(device)
        stats["value"] = self._get_memory_difference(device=device, stats=stats)
        return stats

    def forward_only(self) -> torch.Tensor:
        return self._module(self._input_tensor, self._key, self._value)[0]


class RNNBase(Layer):
    def __init__(
        self,
        *,
        layer: Union[Type[DPRNNBase], Type[nn.RNNBase]],
        batch_size: int,
        seq_len: int,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = False,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
        random_seed: Optional[int] = None,
        criterion: Callable = F.cross_entropy,
        **kwargs,
    ) -> None:
        super().__init__(random_seed=random_seed, criterion=criterion)

        self._input_tensor = (
            torch.randn(
                seq_len,
                batch_size,
                input_size,
            )
            if not batch_first
            else torch.randn(batch_size, seq_len, input_size)
        )

        D = 2 if bidirectional else 1
        self._h_0 = torch.randn(D * num_layers, batch_size, hidden_size)

        self._module = layer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            **kwargs,
        )

        self._labels = (
            torch.randn(seq_len, batch_size, D * hidden_size)
            if not batch_first
            else torch.randn(batch_size, seq_len, D * hidden_size)
        )

    def _inputs_to(self, device: torch.device, stats: Dict[str, int]) -> Dict[str, int]:
        """RNNs take additional layer inputs h_0.

        Args:
            device: torch.device
            stats: dictionary from item names (e.g. input_tensor, module, labels)
            to their respective sizes

        Returns: updated stats dictionary with input tensor and h_0 size
        """
        stats = super()._inputs_to(device=device, stats=stats)
        self._h_0 = self._h_0.to(device)
        stats["h_0"] = self._get_memory_difference(device=device, stats=stats)
        return stats

    def forward_only(self) -> torch.Tensor:
        return self._module(self._input_tensor, self._h_0)[0]


class LSTMBase(RNNBase):
    def __init__(
        self,
        *,
        layer: Union[Type[nn.LSTM], Type[DPLSTM]],
        batch_size: int,
        seq_len: int,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = False,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
        proj_size: int = 0,
        random_seed: Optional[int] = None,
        criterion: Callable = F.cross_entropy,
    ) -> None:
        super().__init__(
            layer=layer,
            batch_size=batch_size,
            seq_len=seq_len,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
            random_seed=random_seed,
            criterion=criterion,
        )
        h_out = proj_size if proj_size > 0 else hidden_size
        D = 2 if bidirectional else 1
        self._h_0 = torch.randn(D * num_layers, batch_size, h_out)
        self._c_0 = torch.randn(D * num_layers, batch_size, hidden_size)

        self._labels = (
            torch.randn(seq_len, batch_size, D * h_out)
            if not batch_first
            else torch.randn(batch_size, seq_len, D * h_out)
        )

    def _inputs_to(self, device: torch.device, stats: Dict[str, int]) -> Dict[str, int]:
        """LSTMs take additional layer inputs h_0, c_0.

        Args:
            device: torch.device
            stats: dictionary from item names (e.g. input_tensor, module, labels)
            to their respective sizes

        Returns: updated stats dictionary with input tensor, h_0, and c_0 size
        """
        stats = super()._inputs_to(device=device, stats=stats)
        self._h_0 = self._h_0.to(device)
        stats["h_0"] = self._get_memory_difference(device=device, stats=stats)

        self._c_0 = self._c_0.to(device)
        stats["c_0"] = self._get_memory_difference(device=device, stats=stats)

        return stats

    def forward_only(self) -> torch.Tensor:
        return self._module(self._input_tensor, (self._h_0, self._c_0))[0]


class LayerFactory:
    @staticmethod
    # flake8: noqa C901
    def create(
        layer_name: str, gsm_mode: str = "baseline", **kwargs
    ) -> Optional[Layer]:
        if gsm_mode not in ("baseline", "hooks", "ew", "functorch"):
            raise ValueError(f"Unexpected grad_sample_mode={gsm_mode}")

        if layer_name == LayerType.LINEAR:
            module = LinearBase(**kwargs)
        elif layer_name == LayerType.CONV:
            module = ConvBase(**kwargs)
        elif layer_name == LayerType.LAYERNORM:
            module = LayerNormBase(**kwargs)
        elif layer_name == LayerType.INSTANCENORM:
            module = InstanceNormBase(**kwargs)
        elif layer_name == LayerType.GROUPNORM:
            module = GroupNormBase(**kwargs)
        elif layer_name == LayerType.EMBEDDING:
            module = EmbeddingBase(**kwargs)
        elif layer_name == LayerType.RNN:
            module = RNNBase(layer=nn.RNN, **kwargs)
        elif layer_name == LayerType.DPRNN:
            module = RNNBase(layer=DPRNN, **kwargs)
        elif layer_name == LayerType.GRU:
            module = RNNBase(layer=nn.GRU, **kwargs)
        elif layer_name == LayerType.DPGRU:
            module = RNNBase(layer=DPGRU, **kwargs)
        elif layer_name == LayerType.LSTM:
            module = LSTMBase(layer=nn.LSTM, **kwargs)
        elif layer_name == LayerType.DPLSTM:
            module = LSTMBase(layer=DPLSTM, **kwargs)
        elif layer_name == LayerType.MHA:
            module = MHABase(layer=nn.MultiheadAttention, **kwargs)
        elif layer_name == LayerType.DPMHA:
            module = MHABase(layer=DPMultiheadAttention, **kwargs)
        else:
            raise Exception(f"Invalid layer type: {layer_name}.")

        if gsm_mode != "baseline":
            module.make_private(gsm_mode=gsm_mode)

        return module
