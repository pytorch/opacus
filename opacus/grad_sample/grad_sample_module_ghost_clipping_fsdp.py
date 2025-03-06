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

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.utils.module_utils import requires_grad, trainable_parameters


logger = logging.getLogger(__name__)
logger.disabled = True


class GradSampleModuleGhostClippingFSDP(GradSampleModuleFastGradientClipping):
    """
    Hooks-based implementation of GradSampleModule with Fast Gradient and Ghost Clipping

    Computes norms of gradients without gradient instantiation
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
        max_grad_norm=1,
    ):
        """

        Args:
            m: nn.Module to be wrapped
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            max_grad_norm: The value at which gradients are to be clipped.
            strict: If set to True, the input module will be validated to make sure that
                it does not have buffers in all its submodules.

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) includes a buffer.
        """

        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=False,
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
        )

    def get_clipping_coef(self) -> torch.Tensor:
        """Get per-example gradient scaling factor for clipping."""
        norm_sample = self.get_norm_sample()
        return (self.max_grad_norm / (norm_sample + 1e-6)).clamp(max=1.0)

    def get_norm_sample(self) -> torch.Tensor:
        """Get per-example gradient norms."""
        norm_sample = torch.stack(
            [
                per_param_norm
                for module in self.iterate_submodules(self._module)
                for per_param_norm in module.norm_sample
            ],
            dim=0,
        ).norm(2, dim=0)

        self.per_sample_gradient_norms = norm_sample
        return norm_sample

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        if (
            not requires_grad(module)
            or not module.training
            or not torch.is_grad_enabled()
            or not self.hooks_enabled
        ):
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append([t.detach() for t in forward_input])  # pyre-ignore

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        """
        Computes norms of per sample gradient given the current backprops and activations
        stored by the associated forward hook. Computed per sample gradient norms are
        stored in ``norm_sample`` field in each parameter.

        Args:
            module: nn.Module,
            _forward_input: torch.Tensor,
            forward_output: torch.Tensor,
            loss_reduction: str,
            batch_first: bool,
        """
        if not self.hooks_enabled:
            return

        backprops = forward_output[0].detach()

        activations, backprops = self.rearrange_grad_samples(
            module=module,
            backprops=backprops,
            loss_reduction=loss_reduction,
            batch_first=batch_first,
        )

        if not hasattr(module, "norm_sample"):
            # currently, we don't support freezing and unfreezing params in between training. Making this a dictionary and mapping with param names might fix this.
            module.norm_sample = []
            for _, param in trainable_parameters(module):
                module.norm_sample.append(
                    torch.zeros(
                        torch.Size([module.max_batch_len, 1]),
                        device=param.device,
                        dtype=param.dtype,
                    )
                )

        if self.use_ghost_clipping and isinstance(module, nn.Linear):
            norm_sampler_fn = self.NORM_SAMPLERS[nn.Linear]
            norm_samples = norm_sampler_fn(module, activations, backprops)

            for idx, (_, ns) in enumerate(
                (item for item in norm_samples.items() if item[0].requires_grad)
            ):
                module.norm_sample[idx] = ns

        else:
            raise NotImplementedError(
                "FSDP is only supported with Ghost Clipping and Linear layers"
            )

        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len

    @property
    def per_sample_gradient_norms(self) -> torch.Tensor:
        """Returns per sample gradient norms. Note that these are not privatized and should only be used for debugging purposes or in non-private settings"""
        if self._per_sample_gradient_norms is not None:
            return self._per_sample_gradient_norms
        else:
            raise AttributeError(
                "per_sample_gradient_norms is not set. Please call forward and backward on the model before accessing this property."
            )

    @per_sample_gradient_norms.setter
    def per_sample_gradient_norms(self, value):
        self._per_sample_gradient_norms = value
