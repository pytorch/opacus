#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import torch.nn as nn
from torchdp import dp_model_inspector as dp_inspector, utils
from torchvision import models


class dp_model_inspector_test(unittest.TestCase):
    def test_raises_exception(self):
        inspector = dp_inspector.DPModelInspector()
        model = models.resnet50()
        with self.assertRaises(dp_inspector.IncompatibleModuleException):
            inspector.validate(model)

    def test_returns_False(self):
        inspector = dp_inspector.DPModelInspector()
        model = models.resnet50()
        inspector.should_throw = False
        self.assertFalse(inspector.validate(model))

    def test_returns_true(self):
        inspector = dp_inspector.DPModelInspector()
        model = utils.convert_batchnorm_modules(models.resnet50())
        self.assertTrue(inspector.validate(model))

    def test_running_stats(self):
        inspector = dp_inspector.DPModelInspector()
        inspector.should_throw = False

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
