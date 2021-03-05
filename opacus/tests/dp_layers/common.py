#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import io
import unittest
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.testing import assert_allclose


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


def flatten(seq: Sequence) -> Sequence:
    """
    Utility function to flatten any sequence ie [1, [2, 3], [4, [5, 6]]] -> [1, 2, 3, 4, 5, 6]

    Args:
        seq: The sequence to flatten

    Returns:
        The flattened out sequence
    """

    def _flatten(seq, a):
        for i in seq:
            if isinstance(i, Sequence) and not isinstance(i, PackedSequence):
                _flatten(i, a)
            else:
                a.append(i)
        return a

    return _flatten(seq, [])


def default_train_fn(model: nn.Module, x: torch.Tensor, *args, **kwargs) -> None:
    """
    Example of a default train_fn to be passed to ``compare_gradients``.

    Args:
        Recommend to always have *args and **kwargs so you can pass whatever you want to it,
        plus anything else that you need (in this case, we directly refer to x so we add it to
        the list)

    Returns:
        Nothing. But it must call ``loss.backward()`` to fill in the gradients.
    """
    model.train()
    criterion = nn.MSELoss()
    logits = model(x)
    y = torch.zeros_like(logits)
    loss = criterion(logits, y)
    loss.backward()


class DPModules_test(unittest.TestCase):
    """
    Set of common testing utils. It is meant to be subclassed by your test.
    See other tests as an example of how this is done.

    The objective of these tests is to make sure that our DP-friendly reimplementations of
    standard nn.Modules such as LSTM are indeed drop-in replacements: we are checking that all
    outputs and states are the same between the two implementations. Here, we do NOT test for
    grad_samples, which is something we do in the grad_sample tests.
    """

    batch_first_nn, batch_first_dp = None, None

    def compare_forward_outputs(
        self,
        nn_module: nn.Module,
        dp_module: nn.Module,
        *module_args,
        output_names: Optional[Tuple[str]] = None,
        atol: float = 10e-6,
        rtol: float = 10e-5,
        **module_kwargs,
    ) -> None:
        """
        Runs forward through both the standard nn_module and the dp_module and checks that all
        outputs are indeed the same.

        Args:
            nn_module: The original ``nn.Module`` that will be our reference
            dp_module: Our ``dp_module`` reimplementation that we want to test against ``nn_module``
            *module_args: args to pass to the model's forward (ie we'll call
                ``module(*module_args, **module_kwargs))``.
            output_names: If provided, will make printing more informative (rather than say
                output number 3 does not match" we can say "output `h` does not match").
            atol: Absolute tolerance.
            rtol: Relative tolerance.
            **module_kwargs: kwargs to pass to the model's forward (ie we'll call
                ``module(*module_args, **module_kwargs))``.
        Returns:
            None

        Raises:
            AssertionError if either:
                - The number of outputs of the forward doesn't match
                - The shape of any output doesn't match
                - The values for any output ``nn_out`` in ``nn_outs`` differ by more
                    than `atol + rtol * abs(nn_out)`
        """
        torch.set_deterministic(True)
        torch.manual_seed(0)
        np.random.seed(0)

        batch_first_nn, batch_first_dp = (
            getattr(nn_module, "batch_first", None),
            getattr(dp_module, "batch_first", None),
        )

        nn_outs = nn_module(*module_args, **module_kwargs)
        nn_len = 0
        try:
            nn_len = len(nn_outs)
        except AttributeError:
            nn_outs = [nn_outs]
        nn_outs = flatten(nn_outs)

        dp_outs = dp_module(*module_args, **module_kwargs)
        dp_len = 0
        try:
            dp_len = len(dp_outs)
        except AttributeError:
            dp_outs = [dp_outs]
        dp_outs = flatten(dp_outs)

        self.assertEqual(
            dp_len,
            nn_len,
            f"The number of outputs does not match! Reference nn has {nn_len} outputs, and "
            f"DP reimplementation has {dp_len} outputs",
        )

        self._check_shapes(nn_outs, dp_outs, output_names=output_names)
        self._check_values(
            nn_outs,
            dp_outs,
            atol,
            rtol,
            output_names=output_names,
            batch_first_nn=batch_first_nn,
            batch_first_dp=batch_first_dp,
        )

    def compare_gradients(
        self,
        nn_module: nn.Module,
        dp_module: nn.Module,
        train_fn: Callable,
        *train_fn_args,
        atol: float = 10e-6,
        rtol: float = 10e-5,
        **train_fn_kwargs,
    ) -> None:
        """
        Runs forward and backward through both the standard nn_module and the dp_module and
        checks that all gradients are indeed the same.

        Args:
            nn_module: The original nn.Module that will be our reference
            dp_module: Our dp_module reimplementation that we want to test against ``nn_module``
            train_fn: A function that we can run to train the model on a single input batch.
                It should run forward and backward and stop there.
                Refer to ``default_train_fn`` in this file for an example.
            *train_fn_args: args to pass to the train_fn (ie we'll call
                ``train_fn(*train_fn_args, **train_fn_kwargs))``.
            atol: Absolute tolerance.
            rtol: Relative tolerance.
            **train_fn_kwargs: kwargs to pass to the train_fn (ie we'll call
                ``module(*module_args, **module_kwargs))``.
        Returns:
            None

        Raises:
            AssertionError if either:
                - nn_module has a gradient for a parameter that dp_module doesn't have.
                - dp_module has a gradient for a parameter that nn_module doesn't have.
                - The shape of any parameter gradient doesn't match.
                - The values for any parameter gradient ``nn_grad`` differ by more than
                    `atol + rtol * abs(nn_grad)`.
        """

        train_fn(nn_module, *train_fn_args, **train_fn_kwargs)
        train_fn(dp_module, *train_fn_args, **train_fn_kwargs)

        nn_params = dict(nn_module.named_parameters())
        dp_params = dict(dp_module.named_parameters())

        nn_only_grads = [
            param_name
            for param_name, param in nn_params.items()
            if param.grad is not None and param_name not in dp_params
        ]
        if nn_only_grads:
            failed_str = "\n\t".join(
                f"{i}. {s}" for i, s in enumerate(nn_only_grads, 1)
            )
            raise AssertionError(
                f"A total of {len(nn_only_grads)} gradients are in nn_module "
                f"but not in dp_module: \n\t{failed_str}"
            )

        dp_only_grads = [
            param_name
            for param_name, param in dp_params.items()
            if param.grad is not None and param_name not in nn_params
        ]
        if dp_only_grads:
            failed_str = "\n\t".join(
                f"{i}. {s}" for i, s in enumerate(nn_only_grads, 1)
            )
            raise AssertionError(
                f"A total of {len(nn_only_grads)} gradients are in dp_module "
                f"but not in nn_module: \n\t{failed_str}"
            )

        for param_name, nn_param in nn_module.named_parameters():
            dp_param = dp_params[param_name]
            self._check_shapes((nn_param), (dp_param), (param_name))
            self._check_values((nn_param), (dp_param), atol, rtol, (param_name))

    def _check_shapes(
        self,
        nn_outs: Tuple[Union[torch.Tensor, PackedSequence]],
        dp_outs: Tuple[Union[torch.Tensor, PackedSequence]],
        output_names: Optional[Tuple[str]] = None,
    ) -> None:
        output_names = output_names or [None] * len(nn_outs)
        failed = []
        for i, (out_name, nn_out, dp_out) in enumerate(
            zip(output_names, nn_outs, dp_outs)
        ):
            name = f"'{out_name}'" or f"#{i}"
            if not torch.is_tensor(nn_out):
                continue  # Won't have a shape, and value check between nontensors is done in self._check_values()

            msg = (
                f"Output {name}: "
                f"from our DP module: {dp_out.shape}, "
                f"from reference nn.Module: {nn_out.shape}. "
            )

            try:
                self.assertEqual(
                    dp_out.shape,
                    nn_out.shape,
                    msg=msg,
                )

            except AssertionError:
                failed.append(msg)

        if failed:
            failed_str = "\n\t".join(f"{i}. {s}" for i, s in enumerate(failed, 1))
            raise AssertionError(
                f"A total of {len(failed)} shapes do not match \n\t{failed_str}"
            )

    def _check_values(
        self,
        nn_outs: Tuple[Union[torch.Tensor, PackedSequence]],
        dp_outs: Tuple[Union[torch.Tensor, PackedSequence]],
        atol: float,
        rtol: float,
        output_names: Optional[Tuple[str]] = None,
        batch_first_nn: Optional[bool] = None,
        batch_first_dp: Optional[bool] = None,
    ) -> None:
        output_names = output_names or [None] * len(nn_outs)
        failed = []
        for i, (out_name, nn_out, dp_out) in enumerate(
            zip(output_names, nn_outs, dp_outs)
        ):
            name = f"'{out_name}'" or f"#{i}"

            if isinstance(nn_out, PackedSequence):
                self._check_packed_sequence(
                    name,
                    nn_out,
                    dp_out,
                    batch_first_nn,
                    batch_first_dp,
                    atol,
                    rtol,
                    failed,
                )
                continue

            msg = (
                f"Output {name}: DP module L2 norm = : {dp_out.norm(2)}, ",
                f"Reference nn.Module L2 norm = : {nn_out.norm(2)}, ",
                f"MSE = {F.mse_loss(dp_out, nn_out)}, ",
                f"L1 Loss = {F.l1_loss(dp_out, nn_out)}",
            )
            try:
                assert_allclose(
                    actual=dp_out,
                    expected=nn_out,
                    atol=atol,
                    rtol=rtol,
                )
            except AssertionError:
                failed.append(msg)
        if failed:
            failed_str = "\n\t".join(f"{i}. {s}" for i, s in enumerate(failed, 1))
            raise AssertionError(
                f"A total of {len(failed)} values do not match:\n\t{failed_str}"
            )

    def _check_packed_sequence(
        self,
        name: str,
        nn_out: PackedSequence,
        dp_out: PackedSequence,
        batch_first_nn: bool,
        batch_first_dp: bool,
        atol: float,
        rtol: float,
        failure_msgs: Optional[Sequence] = None,
    ) -> bool:

        try:
            padded_seq_nn, seq_lens_nn = pad_packed_sequence(nn_out, batch_first_nn)
        except ValueError:
            raise ValueError("Incorrect format of the nn.module output PackedSequence")

        try:
            padded_seq_dp, seq_lens_dp = pad_packed_sequence(dp_out, batch_first_dp)
        except ValueError:
            raise ValueError("Incorrect format of the DP module output PackedSequence")

        self._check_shapes(
            (padded_seq_nn, seq_lens_nn),
            (padded_seq_dp, seq_lens_dp),
            ("padded_sequence", "batch_sequence_lengths"),
        )

        msg = (
            f"Output PackedSequence {name}: DP module padded sequence L2 norm = {padded_seq_dp.norm(2)}, ",
            f"Reference nn.Module padded sequence L2 norm = {padded_seq_nn.norm(2)}, ",
            f"MSE = {F.mse_loss(padded_seq_dp, padded_seq_nn)}, ",
            f"L1 Loss = {F.l1_loss(padded_seq_dp, padded_seq_nn)}",
            f"Manhattan distance (L1) between batch sequence lengths = {(seq_lens_nn - seq_lens_dp).abs().sum()}",  # F.l1_loss is for floats, so we are computing this manually.
        )

        try:
            assert_allclose(
                actual=padded_seq_dp, expected=padded_seq_nn, atol=atol, rtol=rtol
            )
            assert_allclose(
                actual=seq_lens_dp, expected=seq_lens_nn, atol=atol, rtol=rtol
            )
        except AssertionError:
            if failure_msgs is not None:
                failure_msgs.append(msg)
            return False

        return True
