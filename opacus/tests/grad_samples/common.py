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

import unittest
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.per_sample_gradients_utils import (
    compute_grad_samples_microbatch_and_opacus,
    compute_opacus_grad_sample,
    is_batch_empty,
)
from torch.nn.utils.rnn import PackedSequence
from torch.testing import assert_close


def expander(x, factor: int = 2):
    return x * factor


def shrinker(x, factor: int = 2):
    return max(1, x // factor)  # if avoid returning 0 for x == 1


class GradSampleHooks_test(unittest.TestCase):
    """
    Set of common testing utils. It is meant to be subclassed by your test.
    See other tests as an example of how this is done.
    """

    def run_test(
        self,
        x: Union[torch.Tensor, PackedSequence, Tuple],
        module: nn.Module,
        batch_first=True,
        atol=10e-6,
        rtol=10e-5,
        ew_compatible=True,
        chunk_method=iter,
    ):
        grad_sample_modes = ["hooks", "functorch"]

        if type(module) is nn.EmbeddingBag or (
            type(x) is not PackedSequence and is_batch_empty(x)
        ):
            grad_sample_modes = ["hooks"]

        if ew_compatible and batch_first and torch.__version__ >= (1, 13):
            grad_sample_modes += ["ew"]

        for loss_reduction in ["sum", "mean"]:
            for grad_sample_mode in grad_sample_modes:
                with self.subTest(
                    grad_sample_mode=grad_sample_mode, loss_reduction=loss_reduction
                ):
                    self.run_test_with_reduction(
                        x,
                        module,
                        batch_first=batch_first,
                        loss_reduction=loss_reduction,
                        atol=atol,
                        rtol=rtol,
                        grad_sample_mode=grad_sample_mode,
                        chunk_method=chunk_method,
                    )

    def run_test_with_reduction(
        self,
        x: Union[torch.Tensor, PackedSequence],
        module: nn.Module,
        batch_first=True,
        loss_reduction="mean",
        atol=10e-6,
        rtol=10e-5,
        grad_sample_mode="hooks",
        chunk_method=iter,
    ):
        if not type(x) is PackedSequence and is_batch_empty(x):
            _ = compute_opacus_grad_sample(
                x,
                module,
                batch_first=batch_first,
                loss_reduction=loss_reduction,
                grad_sample_mode=grad_sample_mode,
            )
            # We've checked opacus can handle 0-sized batch. Microbatch doesn't make sense
            return
        (
            microbatch_grad_samples,
            opacus_grad_samples,
        ) = compute_grad_samples_microbatch_and_opacus(
            x,
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
            chunk_method=chunk_method,
        )

        self.check_shapes(microbatch_grad_samples, opacus_grad_samples, loss_reduction)
        self.check_values(
            microbatch_grad_samples, opacus_grad_samples, loss_reduction, atol, rtol
        )

    def check_shapes(
        self,
        microbatch_grad_samples,
        opacus_grad_samples,
        loss_reduction,
    ) -> None:
        failed = []
        for name, opacus_grad_sample in opacus_grad_samples.items():
            microbatch_grad_sample = microbatch_grad_samples[name]
            msg = (
                f"Param '{name}': "
                f"from Opacus: {opacus_grad_sample.shape}, "
                f"from Microbatch: {microbatch_grad_sample.shape}. "
            )
            try:
                self.assertEqual(
                    opacus_grad_sample.shape,
                    microbatch_grad_sample.shape,
                    msg=msg,
                )

            except AssertionError:
                failed.append(msg)

        if failed:
            failed_str = "\n\t".join(f"{i}. {s}" for i, s in enumerate(failed, 1))
            raise AssertionError(
                f"A total of {len(failed)} shapes do not match "
                f"for loss_reduction={loss_reduction}: \n\t{failed_str}"
            )

    def check_values(
        self,
        microbatch_grad_samples,
        opacus_grad_samples,
        loss_reduction,
        atol,
        rtol,
    ) -> None:
        failed = []
        for name, opacus_grad_sample in opacus_grad_samples.items():
            microbatch_grad_sample = microbatch_grad_samples[name]
            msg = (
                f"Param {name}: Opacus L2 norm = : {opacus_grad_sample.norm(2)}, ",
                f"Microbatch L2 norm = : {microbatch_grad_sample.norm(2)}, ",
                f"MSE = {F.mse_loss(opacus_grad_sample, microbatch_grad_sample)}, ",
                f"L1 Loss = {F.l1_loss(opacus_grad_sample, microbatch_grad_sample)}",
            )
            try:
                assert_close(
                    actual=microbatch_grad_sample,
                    expected=opacus_grad_sample,
                    atol=atol,
                    rtol=rtol,
                )
            except AssertionError:
                failed.append(msg)
        if failed:
            failed_str = "\n\t".join(f"{i}. {s}" for i, s in enumerate(failed, 1))
            raise AssertionError(
                f"A total of {len(failed)} values do not match "
                f"for loss_reduction={loss_reduction}: \n\t{failed_str}"
            )
