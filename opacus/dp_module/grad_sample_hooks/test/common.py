#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import io
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.dp_module import GradSampleModule
from torch.testing import assert_allclose


class ModelWithLoss(nn.Module):
    """
    To test the gradients of a module, we need to have a loss.
    This module makes it easy to get a loss from any nn.Module, and automatically generates
    a target y vector for it in the forward (of all zeros of the correct size).
    This reduces boilerplate while testing.
    """

    def __init__(self, module: nn.Module, n_classes: int, loss_reduction: str = "mean"):
        """
        Instantiates this module.

        Args:
            module: The nn.Module you want to test.
            n_classes: The size of your outputs after flattening all but the batch dimension.
                Eg if after ``module`` runs you get a tensor of size [16, 3, 9], this would
                be 3*9 = 27.
            loss_reduction: What reduction to apply to the loss. Defaults to "mean".

        Raises:
            ValueError: If ``loss_reduction`` is not among those supported.
        """
        super().__init__()
        self.wrapped_module = module
        self.sequential = nn.Sequential(module, nn.Flatten())
        supported_reductions = ["mean", "sum"]
        if loss_reduction not in supported_reductions:
            raise ValueError(
                f"Passed loss_reduction={loss_reduction}. Only {supported_reductions} supported."
            )
        self.criterion = nn.CrossEntropyLoss(reduction=loss_reduction)
        self.n_classes = n_classes

    def forward(self, x):
        x = self.sequential(x)
        y = torch.zeros([x.size(0)]).long()
        loss = self.criterion(x, y)
        return loss


def clone_module(module):
    with io.BytesIO() as bytesio:
        torch.save(module, bytesio)
        bytesio.seek(0)
        module_copy = torch.load(bytesio)
    return module_copy


class GradSampleHooks_test(unittest.TestCase):
    """
    Set of common testing utils. It is meant to be subclassed by your test.
    See other tests as an example of how this is done.
    """

    def fill_microbatch_grad_sample(
        self, x: torch.Tensor, module: ModelWithLoss, batch_first=True
    ) -> None:
        """
        Computes per-sample gradients with the microbatch method, ie by computing normal gradients
        with batch_size set to 1, and manually accumulating them. This is our reference for testing
        as this method is obviously correct, but slow.

        The per-sample gradients will be put in each parameter under a new attribute called
        ``microbatch_grad_sample``. You can get it by eg doing:

        >>> [p.microbatch_grad_sample for p in module.parameters()]

        Args:
            x: The tensor in input to the ``module``
            module: The ``ModelWithLoss`` that wraps the nn.Module you want to test.
            batch_first: Whether batch size is the first dimension (as opposed to the second).
                Defaults to True.
        """
        print(f"Pre-transpose: {x.shape}")
        if not batch_first:
            x = x.transpose(0, 1)
        print(f"Post-transpose: {x.shape}")
        for p in module.parameters():
            p.microbatch_grad_sample = []

        printed = False
        for x_i in x:
            x_i = x_i.unsqueeze(0 if batch_first else 1)
            if not printed:
                print(f"x_i unsqueezed: {x_i.shape}")
            module.zero_grad()
            loss_i = module(x_i)
            loss_i.backward()
            for p in module.parameters():
                p.microbatch_grad_sample.append(p.grad.detach().clone())
            printed = True

        for p in module.parameters():
            p.microbatch_grad_sample = torch.stack(p.microbatch_grad_sample, dim=0)
        return

    def fill_gradsamplemodule_grad_sample(
        self, x: torch.Tensor, module: GradSampleModule, batch_first=True
    ) -> None:
        """
        Runs your ``GradSampleModule`` and in doing that fills the ``grad_sample`` attribute in each
        parameter of ``module``. You can get it by eg doing:

        >>> [p.grad_sample for p in module.parameters()]

        Args:
            x: The tensor in input to the ``module``
            module: The ``ModelWithLoss`` that wraps the nn.Module you want to test.
            batch_first: Whether batch size is the first dimension (as opposed to the second).
                Defaults to True.
        """
        if not batch_first:
            x = x.transpose(0, 1)
        module.zero_grad()
        loss = module(x)
        loss.backward()
        return

    def run_test(
        self,
        x: torch.Tensor,
        module: nn.Module,
        batch_first=True,
        atol=10e-6,
        rtol=10e-5,
    ):
        self.run_test_with_reduction(
            x,
            module,
            batch_first=batch_first,
            loss_reduction="mean",
            atol=atol,
            rtol=rtol,
        )
        self.run_test_with_reduction(
            x,
            module,
            batch_first=batch_first,
            loss_reduction="sum",
            atol=atol,
            rtol=rtol,
        )

    def run_test_with_reduction(
        self,
        x: torch.Tensor,
        module: nn.Module,
        batch_first=True,
        loss_reduction="mean",
        atol=10e-6,
        rtol=10e-5,
    ):
        self.fill_microbatch_grad_sample(x, module, batch_first)
        module_clone = clone_module(module)
        grad_sample_module = GradSampleModule(module_clone)
        self.fill_gradsamplemodule_grad_sample(x, grad_sample_module, batch_first)

        original_module_microbatch = module.wrapped_module

        original_module_grad_sample = (
            grad_sample_module._module
        )  # Unwrap GradSampleModule

        original_module_grad_sample = (
            original_module_grad_sample.wrapped_module
        )  # Unwrap ModelWithLoss

        microbatch_grad_samples = {
            name: p.microbatch_grad_sample
            for name, p in original_module_microbatch.named_parameters()
        }
        our_grad_samples = {
            name: p.grad_sample
            for name, p in original_module_grad_sample.named_parameters()
        }

        for name, our_grad_sample in our_grad_samples.items():
            microbatch_grad_sample = microbatch_grad_samples[name]
            self.assertEqual(
                our_grad_sample.shape,
                microbatch_grad_sample.shape,
                msg=(
                    f"Gradient shapes mismatch for param '{name}'! "
                    f"From GradSampleModule: {our_grad_sample.shape}, "
                    f"From Microbatch: {microbatch_grad_sample.shape}. "
                    f"Loss reduction = '{loss_reduction}'"
                ),
            )

            assert_allclose(
                actual=microbatch_grad_sample,
                expected=our_grad_sample,
                atol=atol,
                rtol=rtol,
                msg=(
                    f"Gradient value mismatch for param {name}! "
                    f"L1 Loss = {F.l1_loss(our_grad_sample, microbatch_grad_sample)}",
                    f"MSE = {F.mse_loss(our_grad_sample, microbatch_grad_sample)}",
                    f"Loss reduction = '{loss_reduction}'",
                ),
            )
