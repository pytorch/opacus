#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PerSampleGradientClipper
from opacus.utils.clipping import ConstantFlatClipper, ConstantPerLayerClipper
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 3)
        self.conv2 = nn.Conv1d(16, 32, 3, 1)
        self.convf = nn.Conv1d(32, 32, 1, 1)
        for p in self.convf.parameters():
            p.requires_grad = False
        self.fc1 = nn.Linear(23, 17)
        self.fc2 = nn.Linear(32 * 17, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 10, 10]
        x = F.max_pool2d(x, 2, 2)  # -> [B, 16, 5, 5]
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # -> [B, 16, 25]
        x = F.relu(self.conv2(x))  # -> [B, 32, 23]
        x = self.convf(x)  # -> [B, 32, 23]
        x = self.fc1(x)  # -> [B, 32, 17]
        x = x.view(-1, x.shape[-2] * x.shape[-1])  # -> [B, 32 * 17]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


class PerSampleGradientClipper_test(unittest.TestCase):
    def setUp(self):
        self.DATA_SIZE = 64
        self.criterion = nn.CrossEntropyLoss()

        self.setUp_data()
        self.setUp_original_model()
        self.setUp_clipped_model(clip_value=0.003, run_clipper_step=True)

    def setUp_data(self):
        self.ds = FakeData(
            size=self.DATA_SIZE,
            image_size=(1, 35, 35),
            num_classes=10,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        self.dl = DataLoader(self.ds, batch_size=self.DATA_SIZE)

    def setUp_original_model(self):
        self.original_model = SampleConvNet()
        for x, y in self.dl:
            logits = self.original_model(x)
            loss = self.criterion(logits, y)
            loss.backward()  # puts grad in self.original_model.parameters()
        self.original_grads_norms = torch.stack(
            [
                p.grad.norm()
                for p in self.original_model.parameters()
                if p.requires_grad
            ],
            dim=-1,
        )

    def setUp_clipped_model(self, clip_value=0.003, run_clipper_step=True):
        # Deep copy
        self.clipped_model = SampleConvNet()  # create the structure
        self.clipped_model.load_state_dict(self.original_model.state_dict())  # fill it

        # Intentionally clipping to a very small value
        norm_clipper = (
            ConstantFlatClipper(clip_value)
            if not isinstance(clip_value, list)
            else ConstantPerLayerClipper(clip_value)
        )
        self.clipper = PerSampleGradientClipper(self.clipped_model, norm_clipper)

        for x, y in self.dl:
            logits = self.clipped_model(x)
            loss = self.criterion(logits, y)
            loss.backward()  # puts grad in self.clipped_model.parameters()
            if run_clipper_step:
                self.clipper.clip_and_accumulate()
                self.clipper.pre_step()
        self.clipped_grads_norms = torch.stack(
            [p.grad.norm() for p in self.clipped_model.parameters() if p.requires_grad],
            dim=-1,
        )

    def test_clipped_grad_norm_is_smaller(self):
        """
        Test that grad are clipped and their value changes
        """
        for original_layer_norm, clipped_layer_norm in zip(
            self.original_grads_norms, self.clipped_grads_norms
        ):
            self.assertLess(float(clipped_layer_norm), float(original_layer_norm))

    def test_clipped_grad_norm_is_smaller_perlayer(self):
        """
        Test that grad are clipped and their value changes
        """
        # there are 8 parameter sets [bias, weight] * [conv1, conv2, fc1, fc2]
        self.setUp_clipped_model(clip_value=[0.001] * 8)
        for original_layer_norm, clipped_layer_norm in zip(
            self.original_grads_norms, self.clipped_grads_norms
        ):
            self.assertLess(float(clipped_layer_norm), float(original_layer_norm))

    def test_clipped_grad_norms_not_zero(self):
        """
        Test that grads aren't killed by clipping
        """
        allzeros = torch.zeros_like(self.clipped_grads_norms)
        self.assertFalse(torch.allclose(self.clipped_grads_norms, allzeros))

    def test_clipped_grad_norms_not_zero_per_layer(self):
        """
        Test that grads aren't killed by clipping
        """
        # there are 8 parameter sets [bias, weight] * [conv1, conv2, fc1, fc2]
        self.setUp_clipped_model(clip_value=[0.001] * 8)
        allzeros = torch.zeros_like(self.clipped_grads_norms)
        self.assertFalse(torch.allclose(self.clipped_grads_norms, allzeros))

    def test_clipping_to_high_value_does_nothing(self):
        self.setUp_clipped_model(
            clip_value=9999, run_clipper_step=True
        )  # should be a no-op
        self.assertTrue(
            torch.allclose(self.original_grads_norms, self.clipped_grads_norms)
        )

    def test_clipping_to_high_value_does_nothing_per_layer(self):
        self.setUp_clipped_model(
            clip_value=[9999] * 8, run_clipper_step=True
        )  # should be a no-op
        self.assertTrue(
            torch.allclose(self.original_grads_norms, self.clipped_grads_norms)
        )

    def test_grad_norms_untouched_without_clip_step(self):
        """
        Test that grad are not clipped until clipper.step() is called
        """
        self.setUp_clipped_model(clip_value=0.003, run_clipper_step=False)
        self.assertTrue(
            torch.allclose(self.original_grads_norms, self.clipped_grads_norms)
        )
