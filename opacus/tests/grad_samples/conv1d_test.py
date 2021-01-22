#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test


class Conv1d_test(GradSampleHooks_test):
    def test_3d_input_expanding_1_group_kernel2(self):
        N, C, W = 32, 3, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, 2 * C, kernel_size=2)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_shrinking_1_group_kernel2(self):
        N, C, W = 32, 12, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, C // 2, kernel_size=2)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_expanding_1_group_kernel2_stride2(self):
        N, C, W = 32, 3, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, 2 * C, kernel_size=2, stride=2)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_shrinking_1_group_kernel2_stride2(self):
        N, C, W = 32, 12, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, C // 2, kernel_size=2, stride=2)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_expanding_1_group_kernel2_padding2(self):
        N, C, W = 32, 3, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, 2 * C, kernel_size=2, padding=2)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_shrinking_1_group_kernel2_padding2(self):
        N, C, W = 32, 12, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, C // 2, kernel_size=2, padding=2)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_expanding_1_group_kernel3(self):
        N, C, W = 32, 4, 11
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, 2 * C, kernel_size=3)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_shrinking_1_group_kernel3(self):
        N, C, W = 32, 12, 11
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, C // 2, kernel_size=3)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_expanding_2_groups_kernel2(self):
        N, C, W = 32, 12, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, 2 * C, kernel_size=2, groups=2)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_shrinking_2_groups_kernel2(self):
        N, C, W = 32, 12, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, C // 2, kernel_size=2, groups=2)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_expanding_3_groups_kernel2(self):
        N, C, W = 32, 12, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, 2 * C, kernel_size=2, groups=3)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_shrinking_3_groups_kernel2(self):
        N, C, W = 32, 12, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, C // 2, kernel_size=2, groups=3)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_expanding_12_groups_kernel2(self):
        N, C, W = 32, 12, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, 2 * C, kernel_size=2, groups=12)
        self.run_test(x, conv, batch_first=True)

    def test_3d_input_shrinking_12_groups_kernel2(self):
        N, C, W = 32, 24, 10
        x = torch.randn([N, C, W])
        conv = nn.Conv1d(C, C // 2, kernel_size=2, groups=12)
        self.run_test(x, conv, batch_first=True)
