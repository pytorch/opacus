#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
import torch.nn as nn
from opacus.validators.errors import UnsupportedModuleError
from opacus.validators.module_validator import ModuleValidator
from torchvision.models import mobilenet_v3_small


class ModuleValidator_test(unittest.TestCase):
    def setUp(self):
        self.original_model = mobilenet_v3_small()
        self.fixed_model = ModuleValidator.fix(self.original_model)

    def test_is_valid(self):
        self.assertFalse(ModuleValidator.is_valid(self.original_model))
        self.assertTrue(ModuleValidator.is_valid(self.fixed_model))

    def test_validate_invalid_model(self):
        with self.assertRaises(UnsupportedModuleError):
            ModuleValidator.validate(self.original_model, strict=True)
        errors = ModuleValidator.validate(self.original_model)
        self.assertGreater(len(errors), 0)

    def test_validate_valid_model(self):
        errors = ModuleValidator.validate(self.fixed_model)
        self.assertEqual(len(errors), 0)

    def test_validate_training_mode(self):
        self.fixed_model.eval()
        self.assertFalse(ModuleValidator.is_valid(self.fixed_model))
        self.fixed_model.train()
        self.assertTrue(ModuleValidator.is_valid(self.fixed_model))

    def test_is_valid_unsupported_grdsample_module(self):
        unsupported_module = nn.Bilinear(2, 2, 2)  # currently not implemented
        self.assertFalse(ModuleValidator.is_valid(unsupported_module))

    def test_is_valid_extra_param(self):
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
        self.assertFalse(ModuleValidator.is_valid(model))

        model.extra_param.requires_grad = False
        self.assertTrue(ModuleValidator.is_valid(model))

    def test_fix(self):
        with self.assertLogs(level="INFO") as log_cm:
            ModuleValidator.fix(self.original_model)
            self.assertGreater(len(log_cm.records), 0)
            for log_record in log_cm.records:
                log_msg = log_record.getMessage()
                self.assertRegex(
                    log_msg,
                    "Replaced sub_module .+ with .*"
                    "|"
                    "The default batch_norm fixer replaces BatchNorm with GroupNorm",
                )

    def test_is_valid_non_learnable_bn(self):
        class SampleNetWithNonLearnableBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(8, 3, 4, 4)
                self.b1 = nn.BatchNorm2d(3)
                self.c2 = nn.Conv2d(8, 3, 4, 4)
                self.b2 = nn.BatchNorm2d(3, affine=False)

            def forward(self, x):
                x = self.c1(x)
                x = self.b1(x)
                x = self.c2(x)
                x = self.b2(x)
                return x

        model = SampleNetWithNonLearnableBN()
        self.assertFalse(ModuleValidator.is_valid(model))

        model.b1.weight.requires_grad = False
        model.b1.bias.requires_grad = False
        self.assertTrue(ModuleValidator.is_valid(model))
