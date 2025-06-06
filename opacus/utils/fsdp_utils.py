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

from typing import Iterable

import torch.nn as nn
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp import (
    GradSampleModuleFastGradientClippingFSDP,
)
from opacus.utils.module_utils import has_trainable_params
from torch.distributed._composable.fsdp import fully_shard


def has_params(module: nn.Module) -> bool:
    return len(list(module.parameters(recurse=False))) > 0


def iterate_submodules(module: nn.Module) -> Iterable[nn.Module]:
    if has_params(module):
        yield module

    for m in module.children():
        yield from iterate_submodules(m)


def FSDP2Wrapper(model: nn.Module) -> nn.Module:
    sampler_classes = set(
        list(GradSampleModuleFastGradientClippingFSDP.GRAD_SAMPLERS.keys())
        + list(GradSampleModuleFastGradientClippingFSDP.NORM_SAMPLERS.keys())
    )
    for module in iterate_submodules(model):
        if (type(module) in sampler_classes) or (not has_trainable_params(module)):
            fully_shard(module)
    model = fully_shard(model)
    return model
