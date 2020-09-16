#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import torch
import torch.nn as nn
from opacus.dp_module.module_checkers.batch_norm import BatchNormChecker
from opacus.dp_module.module_checkers.errors import UnsupportableModuleError


class BatchNormChecker_test(unittest.TestCase):
    def setUp(self):
        self.checker = BatchNormChecker()
        N, C, Z, H, W = 2, 3, 4, 8, 10

        self.x_1d = torch.randn([N, C, W])
        self.x_2d = torch.randn([N, C, H, W])
        self.x_3d = torch.randn([N, C, Z, H, W])

        self.m_1d = nn.BatchNorm1d(C)
        self.m_2d = nn.BatchNorm2d(C)
        self.m_3d = nn.BatchNorm3d(C)
        # We can't test SyncBatchNorm directly because it is a GPU wrapper for multi-GPU

        self.linear = nn.Linear(W, W + 2)
        self.conv2d = nn.Conv2d(H, W, kernel_size=3)

    def test_raises_for_batchnorm1d(self):
        with self.assertRaises(UnsupportableModuleError):
            self.checker.validate(self.m_1d)

    def test_raises_for_batchnorm2d(self):
        with self.assertRaises(UnsupportableModuleError):
            self.checker.validate(self.m_2d)

    def test_raises_for_batchnorm3d(self):
        with self.assertRaises(UnsupportableModuleError):
            self.checker.validate(self.m_3d)

    def test_not_raises_for_no_batchnorm(self):
        self.checker.validate(self.linear)
        self.checker.validate(self.conv2d)

    def test_replaces_batchnorm1d(self):
        replacement = self.checker.recommended_replacement(self.m_1d)
        self.assertIsInstance(replacement, nn.GroupNorm)

    def test_replaces_batchnorm2d(self):
        replacement = self.checker.recommended_replacement(self.m_2d)
        self.assertIsInstance(replacement, nn.GroupNorm)

    def test_replaces_batchnorm3d(self):
        replacement = self.checker.recommended_replacement(self.m_3d)
        self.assertIsInstance(replacement, nn.GroupNorm)

    def test_replacement_still_works_1d(self):
        y_bn = self.m_1d(self.x_1d)
        replacement = self.checker.recommended_replacement(self.m_1d)
        y_gn = replacement(self.x_1d)
        self.assertEqual(y_bn.shape, y_gn.shape)

    def test_replacement_still_works_2d(self):
        y_bn = self.m_2d(self.x_2d)
        replacement = self.checker.recommended_replacement(self.m_2d)
        y_gn = replacement(self.x_2d)
        self.assertEqual(y_bn.shape, y_gn.shape)

    def test_replacement_still_works_3d(self):
        y_bn = self.m_3d(self.x_3d)
        replacement = self.checker.recommended_replacement(self.m_3d)
        y_gn = replacement(self.x_3d)
        self.assertEqual(y_bn.shape, y_gn.shape)
