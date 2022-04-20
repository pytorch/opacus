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

import copy
import random
from typing import Any, Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from helpers import skipifnocuda
from layers import LayerFactory
from opacus.grad_sample import GradSampleModule
from opacus.layers import DPGRU, DPLSTM, DPRNN, DPMultiheadAttention
from utils import reset_peak_memory_stats


PARAMETERS = [
    (
        [("linear", nn.Linear), ("gsm_linear", nn.Linear)],
        {"input_shape": [], "in_features": 512, "out_features": 512},
    ),
    (
        [("conv", nn.Conv2d), ("gsm_conv", nn.Conv2d)],
        {
            "in_channels": 64,
            "input_shape": [50, 100],
            "out_channels": 64,
            "kernel_size": 8,
        },
    ),
    (
        [("layernorm", nn.LayerNorm), ("gsm_layernorm", nn.LayerNorm)],
        {"input_shape": [64], "D": 1},
    ),
    (
        [
            ("instancenorm", nn.InstanceNorm1d),
            ("gsm_instancenorm", nn.InstanceNorm1d),
        ],
        {"num_features": 256, "input_shape": [64], "affine": True},
    ),
    (
        [("groupnorm", nn.GroupNorm), ("gsm_groupnorm", nn.GroupNorm)],
        {"input_shape": [], "num_groups": 16, "num_channels": 256},
    ),
    (
        [("embedding", nn.Embedding), ("gsm_embedding", nn.Embedding)],
        {"input_shape": [], "num_embeddings": 20000, "embedding_dim": 100},
    ),
    (
        [
            ("mha", nn.MultiheadAttention),
            ("dpmha", DPMultiheadAttention),
            ("gsm_dpmha", DPMultiheadAttention),
        ],
        {
            "source_seq_len": 128,
            "targ_seq_len": 64,
            "embed_dim": 100,
            "num_heads": 4,
        },
    ),
    (
        [
            ("rnn", nn.RNN),
            ("dprnn", DPRNN),
            ("gsm_dprnn", DPRNN),
            ("gru", nn.GRU),
            ("dpgru", DPGRU),
            ("gsm_dpgru", DPGRU),
            ("lstm", nn.LSTM),
            ("dplstm", DPLSTM),
            ("gsm_dplstm", DPLSTM),
        ],
        {"seq_len": 128, "input_size": 100, "hidden_size": 100},
    ),
]


@pytest.mark.parametrize("layer_list, layer_config", PARAMETERS)
def test_layer_modules(
    layer_list: List[Tuple[str, nn.Module]], layer_config: Dict[str, Any]
) -> None:
    """For each supported layer, tests that it is instantiated with the correct module
    and DP support.

    Args:
        layer_list: list of tuples of form (layer_name, module)
        layer_config: config for instantiating the layers in layer_list
    """
    for layer_name, module in layer_list:
        layer = LayerFactory.create(
            layer_name=layer_name,
            batch_size=64,
            **layer_config,
        )

        if "gsm" in layer_name:
            assert isinstance(layer.module, GradSampleModule)
            assert isinstance(layer.module.to_standard_module(), module)
        else:
            assert isinstance(layer.module, module)


@skipifnocuda
@pytest.mark.parametrize("layer_list, layer_config", PARAMETERS)
def test_to_device(
    layer_list: List[Tuple[str, nn.Module]], layer_config: Dict[str, Any]
) -> None:
    """Tests that inputs, labels, and module are initialized on CPU, and that moving
    them to GPU and CPU works correctly.

    Args:
        layer_list: list of tuples of form (layer_name, module)
        layer_config: config for instantiating the layers in layer_list
    """
    cuda = torch.device("cuda:0")
    cpu = torch.device("cpu")
    assert reset_peak_memory_stats(cuda).cur_mem == 0

    for layer_name, module in layer_list:
        layer = LayerFactory.create(
            layer_name=layer_name,
            batch_size=64,
            **layer_config,
        )
        # layer should be initialized on CPU
        assert torch.cuda.memory_allocated(cuda) == 0

        mem_stats = layer.to(cuda)
        allocated = torch.cuda.memory_allocated(cuda)
        assert allocated > 0
        # all allocated memory should be accounted for in the memory statistics
        assert allocated == sum(v for _, v in mem_stats.items())

        mem_stats = layer.to(cpu)
        allocated = torch.cuda.memory_allocated(cuda)
        assert allocated == 0
        assert allocated == sum(v for _, v in mem_stats.items())

    assert reset_peak_memory_stats(cuda).cur_mem == 0


@pytest.mark.parametrize("layer_list, layer_config", PARAMETERS)
def test_layer_outputs(
    layer_list: List[Tuple[str, nn.Module]], layer_config: Dict[str, Any]
) -> None:
    """Layers in layer_list that share the same underlying module (either a
    torch.nn.Module or opacus.layers.DPModule) should produce the same output
    given the same random seed and different outputs given different random seeds.

    Args:
        layer_list: list of tuples of form (layer_name, module)
        layer_config: config for instantiating the layers in layer_list
    """
    random_seed_a = random.randint(0, 100000)
    random_seed_b = random.randint(100000, 200000)
    outputs: Dict[int, Dict[str, torch.Tensor]] = {
        random_seed_a: {},
        random_seed_b: {},
    }

    for random_seed in (random_seed_a, random_seed_b):
        for layer_name, module in layer_list:
            layer = LayerFactory.create(
                layer_name=layer_name,
                batch_size=64,
                random_seed=random_seed,
                **layer_config,
            )
            if str(module) not in outputs[random_seed]:
                outputs[random_seed][str(module)] = layer.forward_only()

            # same module with same seed should result in same output
            assert torch.equal(outputs[random_seed][str(module)], layer.forward_only())

    # same module with different seed should result in different output
    for module_name in outputs[random_seed_a]:
        assert not torch.equal(
            outputs[random_seed_a][module_name], outputs[random_seed_b][module_name]
        )


@pytest.mark.parametrize("layer_list, layer_config", PARAMETERS)
def test_forward_backward(
    layer_list: List[Tuple[str, nn.Module]], layer_config: Dict[str, Any]
) -> None:
    """Tests that Layer.forward_backward() runs for each layer in layer_list and that
    the Layer is not modified.

    Args:
        layer_list: list of tuples of form (layer_name, module)
        layer_config: config for instantiating the layers in layer_list
    """
    for layer_name, module in layer_list:
        layer = LayerFactory.create(
            layer_name=layer_name,
            batch_size=64,
            **layer_config,
        )
        layer_copy = copy.deepcopy(layer)
        layer.forward_backward()
        for p1, p2 in zip(layer.module.parameters(), layer_copy.module.parameters()):
            assert torch.equal(p1.data, p2.data)
