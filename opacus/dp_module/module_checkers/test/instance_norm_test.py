#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import torch
import torch.nn as nn
from opacus.dp_module.module_checkers.errors import UnsupportableModuleError
from opacus.dp_module.module_checkers.instance_norm import InstanceNormChecker


class InstanceNormChecker_test(unittest.TestCase):
    def setUp(self):
        self.checker = InstanceNormChecker()
        N, C, Z, H, W = 2, 3, 4, 8, 10

        self.C = C

        self.x_1d = torch.randn([N, C, W])
        self.x_2d = torch.randn([N, C, H, W])
        self.x_3d = torch.randn([N, C, Z, H, W])

        self.m_1d = nn.InstanceNorm1d(C, track_running_stats=False)
        self.m_2d = nn.InstanceNorm2d(C, track_running_stats=False)
        self.m_3d = nn.InstanceNorm3d(C, track_running_stats=False)

        self.linear = nn.Linear(W, W + 2)
        self.conv2d = nn.Conv2d(H, W, kernel_size=3)

    def test_raises_for_runningstats1d(self):
        with self.assertRaises(UnsupportableModuleError):
            self.checker.validate(nn.InstanceNorm1d(self.C, track_running_stats=True))

    def test_raises_for_runningstats2d(self):
        with self.assertRaises(UnsupportableModuleError):
            self.checker.validate(nn.InstanceNorm2d(self.C, track_running_stats=True))

    def test_raises_for_runningstats3d(self):
        with self.assertRaises(UnsupportableModuleError):
            self.checker.validate(nn.InstanceNorm3d(self.C, track_running_stats=True))

    def test_not_raises_for_notrunningstats1d(self):
        self.checker.validate(self.m_1d)

    def test_not_raises_for_notrunningstats2d(self):
        self.checker.validate(self.m_2d)

    def test_not_raises_for_notrunningstats3d(self):
        self.checker.validate(self.m_3d)

    def test_not_raises_for_no_instancenorm(self):
        self.checker.validate(self.linear)
        self.checker.validate(self.conv2d)

    def test_replaces_instancenorm1d(self):
        replacement = self.checker.recommended_replacement(self.m_1d)
        self.assertIsInstance(replacement, nn.InstanceNorm1d)
        self.assertFalse(replacement.track_running_stats)

    def test_replaces_instancenorm2d(self):
        replacement = self.checker.recommended_replacement(self.m_2d)
        self.assertIsInstance(replacement, nn.InstanceNorm2d)
        self.assertFalse(replacement.track_running_stats)

    def test_replaces_instancenorm3d(self):
        replacement = self.checker.recommended_replacement(self.m_3d)
        self.assertIsInstance(replacement, nn.InstanceNorm3d)
        self.assertFalse(replacement.track_running_stats)

    def test_replacement_still_works_1d(self):
        y_old = self.m_1d(self.x_1d)
        replacement = self.checker.recommended_replacement(self.m_1d)
        y_new = replacement(self.x_1d)
        self.assertEqual(y_old.shape, y_new.shape)

    def test_replacement_still_works_2d(self):
        y_old = self.m_2d(self.x_2d)
        replacement = self.checker.recommended_replacement(self.m_2d)
        y_new = replacement(self.x_2d)
        self.assertEqual(y_old.shape, y_new.shape)

    def test_replacement_still_works_3d(self):
        y_old = self.m_3d(self.x_3d)
        replacement = self.checker.recommended_replacement(self.m_3d)
        y_new = replacement(self.x_3d)
        self.assertEqual(y_old.shape, y_new.shape)
