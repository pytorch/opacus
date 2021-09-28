#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import torch
import torch.nn as nn
from opacus import dp_model_inspector as dp_inspector
from opacus.utils.module_modification import convert_batchnorm_modules
from torchvision import models


class dp_model_inspector_test(unittest.TestCase):
    def test_raises_exception(self):
        inspector = dp_inspector.DPModelInspector()
        model = models.resnet50()
        with self.assertRaises(dp_inspector.IncompatibleModuleException):
            inspector.validate(model)

    def test_returns_False(self):
        inspector = dp_inspector.DPModelInspector(should_throw=False)
        model = models.resnet50()
        self.assertFalse(inspector.validate(model))

    def test_raises_for_eval_mode(self):
        inspector = dp_inspector.DPModelInspector()
        model = models.resnet50()
        model = model.eval()
        with self.assertRaises(dp_inspector.IncompatibleModuleException):
            inspector.validate(model)

    def test_convert_batchnorm(self):
        inspector = dp_inspector.DPModelInspector()
        model = convert_batchnorm_modules(models.resnet50())
        self.assertTrue(inspector.validate(model))

    def test_running_stats(self):
        inspector = dp_inspector.DPModelInspector(should_throw=False)

        self.assertTrue(inspector.validate(nn.InstanceNorm1d(16)))
        self.assertTrue(inspector.validate(nn.InstanceNorm1d(16, affine=True)))
        self.assertTrue(
            inspector.validate(nn.InstanceNorm1d(16, track_running_stats=True))
        )
        self.assertFalse(
            inspector.validate(
                nn.InstanceNorm1d(16, affine=True, track_running_stats=True)
            )
        )

    def test_extra_param(self):
        inspector = dp_inspector.DPModelInspector(should_throw=False)

        class SampleNetWithExtraParam(nn.Module):
            def __init__(self):
                super().__init__()

                self.fc = nn.Linear(8, 16)
                self.extra_param = nn.Parameter(torch.Tensor(16, 2))

            def forward(self, x):
                x = self.fc(x)
                x = x.matmul(self.extra_param)
                return x

        model = SampleNetWithExtraParam()
        self.assertFalse(inspector.validate(model))

        model.extra_param.requires_grad = False
        self.assertTrue(inspector.validate(model))

    def test_unsupported_layer(self):
        class SampleNetWithTransformer(nn.Module):
            def __init__(self):
                super().__init__()

                self.fc = nn.Linear(8, 16)
                self.encoder = nn.Transformer()

            def forward(self, x):
                x = self.fc(x)
                x = self.encoder(x)
                return x

        model = SampleNetWithTransformer()
        inspector = dp_inspector.DPModelInspector(should_throw=False)
        self.assertFalse(inspector.validate(model))

    def test_conv2d(self):
        inspector = dp_inspector.DPModelInspector(should_throw=False)

        self.assertTrue(
            inspector.validate(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, groups=1)
            )
        )
        self.assertTrue(
            inspector.validate(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, groups=3)
            )
        )
        self.assertFalse(
            inspector.validate(
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, groups=2)
            )
        )
