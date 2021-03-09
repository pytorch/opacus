#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import io
import unittest
from typing import Dict, List, Union

import numpy as np
import opacus
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.testing import assert_allclose


def expander(x, factor: int = 2):
    return x * factor


def shrinker(x, factor: int = 2):
    return max(1, x // factor)  # if avoid returning 0 for x == 1


class ModelWithLoss(nn.Module):
    """
    To test the gradients of a module, we need to have a loss.
    This module makes it easy to get a loss from any nn.Module, and automatically generates
    a target y vector for it in the forward (of all zeros of the correct size).
    This reduces boilerplate while testing.
    """

    supported_reductions = ["mean", "sum"]

    def __init__(self, module: nn.Module, loss_reduction: str = "mean"):
        """
        Instantiates this module.

        Args:
            module: The nn.Module you want to test.
            loss_reduction: What reduction to apply to the loss. Defaults to "mean".

        Raises:
            ValueError: If ``loss_reduction`` is not among those supported.
        """
        super().__init__()
        self.wrapped_module = module

        if loss_reduction not in self.supported_reductions:
            raise ValueError(
                f"Passed loss_reduction={loss_reduction}. Only {self.supported_reductions} supported."
            )
        self.criterion = nn.L1Loss(reduction=loss_reduction)

    def forward(self, x):
        x = self.wrapped_module(x)
        if type(x) is PackedSequence:
            loss = _compute_loss_packedsequences(self.criterion, x)
        else:
            y = torch.zeros_like(x)
            loss = self.criterion(x, y)
        return loss


def clone_module(module: nn.Module) -> nn.Module:
    """
    Handy utility to clone an nn.Module. PyTorch doesn't always support copy.deepcopy(), so it is
    just easier to serialize the model to a BytesIO and read it from there.

    Args:
        module: The module to clone

    Returns:
        The clone of ``module``
    """
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

    def compute_microbatch_grad_sample(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        module: nn.Module,
        batch_first=True,
        loss_reduction="mean",
    ) -> Dict[str, torch.tensor]:
        """
        Computes per-sample gradients with the microbatch method, ie by computing normal gradients
        with batch_size set to 1, and manually accumulating them. This is our reference for testing
        as this method is obviously correct, but slow.

        Args:
            x: The tensor in input to the ``module``
            module: The ``ModelWithLoss`` that wraps the nn.Module you want to test.
            batch_first: Whether batch size is the first dimension (as opposed to the second).
                Defaults to True.

        Returns:
            Dictionary mapping parameter_name -> per-sample-gradient for that parameter
        """
        torch.set_deterministic(True)
        torch.manual_seed(0)
        np.random.seed(0)

        module = ModelWithLoss(clone_module(module), loss_reduction)

        for p in module.parameters():
            p.microbatch_grad_sample = []

        if not batch_first and type(x) is not list:
            # This allows us to iterate with x_i
            x = x.transpose(0, 1)

        # Invariant: x is [B, T, ...]

        for x_i in x:
            # x_i is [T, ...]
            x_i = x_i.unsqueeze(
                0 if batch_first else 1
            )  # x_i of size [1, T, ...] if batch_first, else [T, 1, ...]
            module.zero_grad()
            loss_i = module(x_i)
            loss_i.backward()
            for p in module.parameters():
                p.microbatch_grad_sample.append(p.grad.detach().clone())

        for p in module.parameters():
            if batch_first:
                p.microbatch_grad_sample = torch.stack(
                    p.microbatch_grad_sample, dim=0  # [B, T, ...]
                )
            else:
                p.microbatch_grad_sample = torch.stack(
                    p.microbatch_grad_sample, dim=1  # [T, B, ...]
                ).transpose(
                    0, 1
                )  # Opacus's semantics is that grad_samples are ALWAYS batch_first: [B, T, ...]

        microbatch_grad_samples = {
            name: p.microbatch_grad_sample
            for name, p in module.wrapped_module.named_parameters()
        }
        return microbatch_grad_samples

    def compute_opacus_grad_sample(
        self,
        x: Union[torch.Tensor, PackedSequence],
        module: nn.Module,
        batch_first=True,
        loss_reduction="mean",
    ) -> Dict[str, torch.tensor]:
        """
        Runs Opacus to compute per-sample gradients and return them for testing purposes.

        Args:
            x: The tensor in input to the ``module``
            module: The ``ModelWithLoss`` that wraps the nn.Module you want to test.
            batch_first: Whether batch size is the first dimension (as opposed to the second).
                Defaults to True.
            loss_reduction: What reduction to apply to the loss. Defaults to "mean".

        Returns:
            Dictionary mapping parameter_name -> per-sample-gradient for that parameter
        """
        torch.set_deterministic(True)
        torch.manual_seed(0)
        np.random.seed(0)

        gs_module = clone_module(module)
        opacus.autograd_grad_sample.add_hooks(gs_module, loss_reduction, batch_first)
        grad_sample_module = ModelWithLoss(gs_module, loss_reduction)

        grad_sample_module.zero_grad()
        loss = grad_sample_module(x)
        loss.backward()

        opacus_grad_samples = {
            name: p.grad_sample
            for name, p in grad_sample_module.wrapped_module.named_parameters()
        }

        return opacus_grad_samples

    def run_test(
        self,
        x: Union[torch.Tensor, PackedSequence],
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
        x: Union[torch.Tensor, PackedSequence],
        module: nn.Module,
        batch_first=True,
        loss_reduction="mean",
        atol=10e-6,
        rtol=10e-5,
    ):
        if type(x) is PackedSequence:
            x_unpacked = _unpack_packedsequences(x)
            microbatch_grad_samples = self.compute_microbatch_grad_sample(
                x_unpacked,
                module,
                batch_first=batch_first,
                loss_reduction=loss_reduction,
            )
        else:
            microbatch_grad_samples = self.compute_microbatch_grad_sample(
                x, module, batch_first=batch_first, loss_reduction=loss_reduction
            )

        opacus_grad_samples = self.compute_opacus_grad_sample(
            x, module, batch_first=batch_first, loss_reduction=loss_reduction
        )

        assert microbatch_grad_samples.keys() == opacus_grad_samples.keys()

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
                assert_allclose(
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


def _unpack_packedsequences(X: PackedSequence) -> List[torch.Tensor]:
    r"""
    Produces a list of tensors from X (PackedSequence) such that this list was used to create X with batch_first=True

    Args:
        X: A PackedSequence from which the output list of tensors will be produced.

    Returns:
        unpacked_data: The list of tensors produced from X.
    """

    X_padded = pad_packed_sequence(X)
    X_padded = X_padded[0].permute((1, 0, 2))

    if X.sorted_indices is not None:
        X_padded = X_padded[X.sorted_indices]

    seq_lens = _compute_seq_lengths(X.batch_sizes)
    unpacked_data = [0] * len(seq_lens)
    for idx, length in enumerate(seq_lens):
        unpacked_data[idx] = X_padded[idx][:length, :]

    return unpacked_data


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


def _compute_loss_packedsequences(
    criterion: nn.L1Loss, x: PackedSequence
) -> torch.Tensor:
    r"""
    This function computes the loss in a different way for 'mean' reduced L1 loss while for 'sum' reduced L1 loss,
    it computes the same way as with non-packed data. For 'mean' reduced L1 loss, it transforms x (PackedSequence)
    into a list of tensors such that this list of tensors was used to create this PackedSequence in the first
    place using batch_first=True and then takes the mean of the loss values produced from applying criterion on
    each sequence sample.

    Args:
        criterion: An L1 loss function with reduction either set to 'sum' or 'mean'.
        x: Data in the form of a PackedSequence.

    Returns:
        A loss variable, reduced either using summation or averaging from L1 errors.
    """

    if criterion.reduction == "sum":
        y = torch.zeros_like(x[0])
        return criterion(x[0], y)
    elif criterion.reduction == "mean":
        x = _unpack_packedsequences(x)
        loss_sum = 0
        for x_i in x:
            y_i = torch.zeros_like(x_i)
            loss_sum += criterion(x_i, y_i)
        loss_mean = loss_sum / len(x)
        return loss_mean
