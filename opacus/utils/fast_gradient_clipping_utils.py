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

import torch
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.optimizers import DPOptimizerFastGradientClipping


def double_backward(
    module: GradSampleModuleFastGradientClipping,
    optimizer: DPOptimizerFastGradientClipping,
    loss_per_sample: torch.Tensor,
) -> None:
    """
    Packages the training loop for Fast Gradient and Ghost Clipping. It does the two backward passes, as well as the loss rescaling and hook operations in between.
    This function also works with DistributedDPOptimizer.

    Args:
        module: The DP gradient sample module to train
        optimizer: The DP optimizer used to train the module
        loss_per_sample: loss on each sample in the mini-batch of size [batch_size, 1]

    Returns:
        None
    """

    torch.mean(loss_per_sample).backward(retain_graph=True)
    optimizer.zero_grad()
    rescaled_loss_per_sample = module.get_coeff() * loss_per_sample
    rescaled_loss = torch.sum(rescaled_loss_per_sample)
    module.disable_hooks()
    rescaled_loss.backward()
    module.enable_hooks()
