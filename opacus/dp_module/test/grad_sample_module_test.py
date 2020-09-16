#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.dp_module import GradSampleModule
from opacus.dp_module.module_checkers.multihead_attention import SequenceBias
from torch.testing import assert_allclose


class GradSampleModule_test(unittest.TestCase):
    """
    Checks the correctness of per sample gradient computation by splitting the data
    into batches of size 1 and computing vanilla gradients
    """

    def setUp(self):
        pass

    def get_gradsamplemodule_grad_sample(
        self,
        x: torch.Tensor,
        module: GradSampleModule,
        criterion: nn.Module,
        batch_first=True,
    ) -> torch.Tensor:
        if not batch_first:
            x = x.transpose(0, 1)
        y = torch.zeros_like(x)
        module.zero_grad()
        logits = module(x)
        loss = criterion(logits, y)
        loss.backward()
        per_sample_grads = torch.cat(
            [
                p.grad_sample.detach().reshape(p.grad_sample.size(0), -1)
                for p in module.parameters()
            ],
            dim=1,
        )
        return per_sample_grads

    def get_microbatch_grad_sample(
        self,
        x: torch.Tensor,
        module: GradSampleModule,
        criterion: nn.Module,
        batch_first=True,
    ) -> torch.Tensor:
        if not batch_first:
            x = x.transpose(0, 1)
        per_sample_grads = []
        print(x.shape)
        for x_i in x:
            module.zero_grad()
            x_i = x_i.unsqueeze(0)
            y_i = torch.zeros_like(x_i)
            logits_i = module(x_i)
            loss_i = criterion(logits_i, y_i.unsqueeze(0))
            loss_i.backward()
            grad_sample_i = torch.cat(
                [p.grad.detach().flatten() for p in module.parameters()]
            )
            per_sample_grads.append(grad_sample_i)
        return torch.stack(per_sample_grads, dim=0)

    def _check_one_module(self, module, data, batch_first=True):
        module = GradSampleModule(module)
        self._check_one_module_with_criterion(
            module, nn.L1Loss(reduction="mean"), data, batch_first
        )
        self._check_one_module_with_criterion(
            module, nn.L1Loss(reduction="sum"), data, batch_first
        )

    def _check_one_module_with_criterion(
        self, module, criterion, data, batch_first=True
    ):
        microbatch_grad_sample = self.get_microbatch_grad_sample(
            data, module, criterion, batch_first
        )

        our_grad_sample = self.get_gradsamplemodule_grad_sample(
            data, module, criterion, batch_first
        )

        self.assertEqual(
            our_grad_sample.shape,
            microbatch_grad_sample.shape,
            msg=(
                "Gradient shapes mismatch! "
                f"Computed: {our_grad_sample.shape}, "
                f"Actual: {microbatch_grad_sample.shape}"
            ),
        )

        assert_allclose(
            actual=microbatch_grad_sample,
            expected=our_grad_sample,
            atol=10e-5,
            rtol=10e-3,
            msg=(
                f"Gradient value mismatch!. "
                f"Parameter: {module}.{param_name}, loss: {criterion.reduction}. ",
                f"L1 Delta = {F.l1_loss(our_grad_sample, microbatch_grad_sample)}",
                f"MSE = {F.mse_loss(our_grad_sample, microbatch_grad_sample)}",
            ),
        )

    def test_conv1d(self):
        x = torch.randn(24, 16, 24)
        module = nn.Conv1d(16, 32, 3, 1)

        self._check_one_module(module, x)

    def test_conv2d(self):
        x = torch.randn(24, 16, 24, 24)
        module = nn.Conv2d(16, 32, 3, 1)

        self._check_one_module(module, x)

    def test_linear(self):
        self._check_one_module(nn.Linear(8, 4), torch.randn(16, 8))
        self._check_one_module(nn.Linear(8, 4), torch.randn(24, 8, 8))

    def test_LayerNorm(self):
        x = torch.randn(64, 16, 24, 24)

        self._check_one_module(nn.LayerNorm(24), x)
        self._check_one_module(nn.LayerNorm((24, 24)), x)
        self._check_one_module(nn.LayerNorm((16, 24, 24)), x)

    def test_groupnorm(self):
        self._check_one_module(nn.GroupNorm(4, 16), torch.randn(16, 16, 10))
        self._check_one_module(nn.GroupNorm(4, 16), torch.randn(16, 16, 10, 9))
        self._check_one_module(nn.GroupNorm(4, 16), torch.randn(16, 16, 10, 9, 8))

    def test_instancenorm(self):
        self._check_one_module(
            nn.InstanceNorm1d(16, affine=True), torch.randn(16, 16, 10)
        )
        self._check_one_module(
            nn.InstanceNorm2d(16, affine=True), torch.randn(16, 16, 10, 9)
        )
        self._check_one_module(
            nn.InstanceNorm3d(16, affine=True), torch.randn(16, 16, 10, 9, 8)
        )

    def test_sequence_bias(self):
        x = torch.randn(4, 3, 2)
        module = SequenceBias(2)

        self._check_one_module(module, x, batch_first=False)

    def test_embedding(self):
        module = nn.Embedding(256, 100)
        x1 = torch.randint(0, 255, (24, 42)).long()
        x2 = torch.randint(0, 255, (12, 64, 4)).long()
        self._check_one_module(module, x1)
        self._check_one_module(module, x2)
