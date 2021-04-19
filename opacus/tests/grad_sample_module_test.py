#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModule
from torch.testing import assert_allclose
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData
from torchvision.models import mobilenet_v3_small


class GradSampleModule_test(unittest.TestCase):
    def setUp(self):
        self.original_model = mobilenet_v3_small()
        copy_of_original_model = mobilenet_v3_small()
        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(), strict=True
        )

        self.grad_sample_module = GradSampleModule(
            copy_of_original_model, batch_first=True, loss_reduction="mean"
        )
        self.DATA_SIZE = 8
        self.setUp_data()
        self.criterion = nn.L1Loss()

    def setUp_data(self):
        self.ds = FakeData(
            size=self.DATA_SIZE,
            image_size=(3, 28, 28),
            num_classes=10,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        self.dl = DataLoader(self.ds, batch_size=self.DATA_SIZE)

    def test_outputs_unaltered(self):
        """
        Test that boxing with GradSampleModule won't alter any outputs.
        Gradients are tested in the various `grad_samples` tests.
        """
        x, _ = next(iter(self.dl))
        print(f"SHAPE: {x.shape}")
        self.original_model = self.original_model.eval()
        self.grad_sample_module = self.grad_sample_module.eval()
        with torch.no_grad():
            normal_out = self.original_model(x)
            gs_out = self.grad_sample_module(x)
        msg = (
            f"GradSample L2 norm = : {gs_out.norm(2)}, ",
            f"Original L2 norm = : {normal_out.norm(2)}, ",
            f"MSE = {F.mse_loss(gs_out, normal_out)}, ",
            f"L1 Loss = {F.l1_loss(gs_out, normal_out)}",
        )
        assert_allclose(gs_out, normal_out, atol=1e-7, rtol=1e-5, msg=msg)

    def test_zero_grad(self):
        x, _ = next(iter(self.dl))
        print(f"SHAPE: {x.shape}")
        self.original_model = self.original_model.train()
        self.grad_sample_module = self.grad_sample_module.train()
        gs_out = self.grad_sample_module(x)
        loss = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss.backward()

        self.grad_sample_module.zero_grad()
        params_with_gs = [
            n
            for n, p in self.grad_sample_module.named_parameters()
            if hasattr(p, "grad_sample")
        ]
        msg = (
            "After calling .zero_grad() on the GradSampleModule, the following parameters still "
            f"have a grad_sample: {params_with_gs}"
        )
        assert len(params_with_gs) == 0, msg

    def test_to_standard_module(self):
        copy_of_original_model = mobilenet_v3_small()
        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(),
            strict=True,
        )
        new_grad_sample_module = GradSampleModule(
            copy_of_original_model, batch_first=True, loss_reduction="mean"
        )

        new_grad_sample_module = new_grad_sample_module.to_standard_module()

        assert isinstance(new_grad_sample_module, type(self.original_model))

        original_state_dict = self.original_model.state_dict()
        gs_state_dict = new_grad_sample_module.state_dict()

        missing_keys = gs_state_dict.keys() - original_state_dict.keys()
        assert not missing_keys, f"The following keys are missing: {missing_keys}"

        extra_keys = original_state_dict.keys() - gs_state_dict.keys()
        assert not extra_keys, f"The following keys are extra: {extra_keys}"

        for key in original_state_dict:
            original_tensor = original_state_dict[key].float()
            gs_tensor = gs_state_dict[key].float()
            msg = (
                f"Param {key}: GradSample L2 norm = : {gs_tensor.norm(2)}, ",
                f"Original L2 norm = : {original_tensor.norm(2)}, ",
                f"MSE = {F.mse_loss(gs_tensor, original_tensor)}, ",
                f"L1 Loss = {F.l1_loss(gs_tensor, original_tensor)}",
            )

            assert_allclose(gs_tensor, original_tensor, atol=1e-6, rtol=1e-4, msg=msg)

    def test_remove_hooks(self):
        """
        Test that after calling .remove_hooks() no hooks are left
        """
        copy_of_original_model = mobilenet_v3_small()
        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(),
            strict=True,
        )
        new_grad_sample_module = GradSampleModule(
            copy_of_original_model, batch_first=True, loss_reduction="mean"
        )
        new_grad_sample_module.remove_hooks()

        remaining_forward_hooks = {
            module: module._forward_hooks
            for module in new_grad_sample_module.modules()
            if module._forward_hooks
        }
        assert (
            not remaining_forward_hooks
        ), f"Some forward hooks remain after .remove_hooks(): {remaining_forward_hooks}"

        remaining_backward_hooks = {
            module: module._backward_hooks
            for module in new_grad_sample_module.modules()
            if module._backward_hooks
        }
        assert (
            not remaining_backward_hooks
        ), f"Some backward hooks remain after .remove_hooks(): {remaining_backward_hooks}"

    def test_enable_hooks(self):
        self.grad_sample_module.enable_hooks()
        assert self.grad_sample_module.hooks_enabled

    def test_disable_hooks(self):
        self.grad_sample_module.disable_hooks()
        assert not self.grad_sample_module.hooks_enabled
