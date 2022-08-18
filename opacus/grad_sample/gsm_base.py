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

import logging
from abc import ABC, abstractmethod

import torch.nn as nn
from opacus.utils.module_utils import trainable_parameters


logger = logging.getLogger(__name__)

OPACUS_PARAM_MONKEYPATCH_ATTRS = ["_forward_counter", "_current_grad_sample"]


class AbstractGradSampleModule(nn.Module, ABC):
    r"""
    Extends nn.Module so that its parameter tensors have an extra field called .grad_sample.
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
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

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) doesn't have a registered grad sampler function.
        """
        super().__init__()

        self._module = m
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction

        for _, p in trainable_parameters(self):
            p.grad_sample = None
            p._forward_counter = 0

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError as e:
            submodules = dict(self._module.named_modules())
            if item and item in submodules:
                return submodules[item]
            raise e

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.

        Clears ``p.grad`` and ``p.grad_sample`` for all of it's parameters

        Notes:
            ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` is
            never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list

        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """
        if set_to_none is False:
            logger.debug(
                "Despite set_to_none is set to False, "
                "opacus will set p.grad_sample to None due to "
                "non-trivial gradient accumulation behaviour"
            )
        self.set_grad_sample_to_none()
        super().zero_grad(set_to_none)

    def set_grad_sample_to_none(self):
        """
        Sets ``.grad_sample`` to None
        """
        for p in self.parameters():
            p.grad_sample = None

    def del_grad_sample(self):
        """
        Deleted ``.grad_sample`` attribute from all model parameters
        """
        for p in self.parameters():
            del p.grad_sample

    def to_standard_module(self) -> nn.Module:
        """
        Returns the standard nn.Module wrapped by this, eliminating all traces
        of grad samples and hooks

        Returns:
            The wrapped module
        """
        self._close()
        return self._module

    def _close(self):
        self.del_grad_sample()
        self._clean_up_attributes()

    def __repr__(self):
        return f"{type(self).__name__}({self._module.__repr__()})"

    def _clean_up_attributes(self):
        for attr in OPACUS_PARAM_MONKEYPATCH_ATTRS:
            for p in self.parameters():
                if hasattr(p, attr):
                    delattr(p, attr)
