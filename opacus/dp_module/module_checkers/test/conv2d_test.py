#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import torch
import torch.nn as nn
from opacus.dp_module.module_checkers.conv2d import Conv2dChecker
from opacus.dp_module.module_checkers.errors import NotYetSupportedModuleError


class Conv2dChecker_test(unittest.TestCase):
    def setUp(self):
        self.checker = Conv2dChecker()
        self.N, self.C, self.H, self.W = 2, 8, 10, 12

    def test_raises_for_wrong_groups(self):
        depthwise_conv = nn.Conv2d(self.C, self.C * 2, kernel_size=(3, 3), groups=2)
        with self.assertRaises(NotYetSupportedModuleError):
            self.checker.validate(depthwise_conv)

    def test_not_raises_for_one_group(self):
        self.checker.validate(nn.Conv2d(self.C, self.C * 2, kernel_size=3, groups=1))

    def test_not_raises_for_C_groups(self):
        self.checker.validate(
            nn.Conv2d(self.C, self.C * 2, kernel_size=(3, 3), groups=self.C)
        )

    def test_not_raises_for_no_conv2d(self):
        self.checker.validate(nn.Linear(2, 4))
        self.checker.validate(nn.Conv1d(4, 8, kernel_size=2))
        self.checker.validate(nn.Conv3d(12, 4, kernel_size=3))

    def test_replaces_groups(self):
        replacement = self.checker.recommended_replacement(
            nn.Conv2d(self.C, self.C * 2, kernel_size=3)
        )
        self.assertIsInstance(replacement, nn.Conv2d)
        self.assertEqual(replacement.groups, self.C)

    def test_replacement_still_works(self):
        m = nn.Conv2d(self.C, self.C * 2, kernel_size=3)
        x = torch.randn([self.N, self.C, self.H, self.W])
        y_old = m(x)
        replacement = self.checker.recommended_replacement(m)
        y_new = replacement(x)
        self.assertEqual(y_old.shape, y_new.shape)
