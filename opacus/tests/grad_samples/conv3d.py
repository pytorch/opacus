#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test


class Conv3d_test(GradSampleHooks_test):
    def test_square_image_kernel2(self):
        N, C, D, H, W = 32, 1, 10, 10, 10
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, C, kernel_size=2)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_landscape_image_kernel2(self):
        N, C, D, H, W = 32, 3, 4, 9, 16
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_portrait_image_kernel2(self):
        N, C, D, H, W = 32, 3, 4, 16, 9
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_square_image_kernel3(self):
        N, C, D, H, W = 32, 3, 10, 10, 10
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=3)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_landscape_image_kernel3(self):
        N, C, D, H, W = 32, 3, 5, 9, 16
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=3)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_portrait_image_kernel3(self):
        N, C, D, H, W = 32, 3, 5, 16, 9
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=3)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_square_image_stride_2(self):
        N, C, D, H, W = 32, 3, 10, 10, 10
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, stride=2)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_landscape_image_stride_2(self):
        N, C, D, H, W = 32, 3, 5, 9, 16
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, stride=2)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_portrait_image_stride_2(self):
        N, C, D, H, W = 32, 3, 5, 16, 9
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, stride=2)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_square_image_pad_2(self):
        N, C, D, H, W = 32, 3, 10, 10, 10
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, padding=2)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_landscape_image_pad_2(self):
        N, C, D, H, W = 32, 3, 5, 9, 16
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, padding=2)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_portrait_image_pad_2(self):
        N, C, D, H, W = 32, 3, 5, 16, 9
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, padding=2)
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_square_image_variable_pad(self):
        N, C, D, H, W = 32, 3, 10, 10, 10
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, padding=(1, 2, 3))
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_landscape_image_variable_pad(self):
        N, C, D, H, W = 32, 3, 5, 9, 16
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, padding=(1, 2, 3))
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_portrait_image_variable_pad(self):
        N, C, D, H, W = 32, 3, 5, 16, 9
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, padding=(1, 2, 3))
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_square_image_variable_kernel(self):
        N, C, D, H, W = 32, 3, 10, 10, 10
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=(1, 2, 3))
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_landscape_image_variable_kernel(self):
        N, C, D, H, W = 32, 3, 4, 9, 16
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=(1, 2, 3))
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_portrait_image_variable_kernel(self):
        N, C, D, H, W = 32, 3, 4, 16, 9
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=(1, 2, 3))
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_square_image_variable_stride(self):
        N, C, D, H, W = 32, 3, 10, 10, 10
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, stride=(1, 2, 3))
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_landscape_image_variable_stride(self):
        N, C, D, H, W = 32, 3, 5, 9, 16
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, stride=(1, 2, 3))
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    def test_portrait_image_variable_stride(self):
        N, C, D, H, W = 32, 3, 5, 16, 9
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, 2 * C, kernel_size=2, stride=(1, 2, 3))
        self.run_test(x, conv, batch_first=True, atol=10e-8, rtol=10e-4)

    """
    These generally show up in upper layers. At this point, C is just generic "channels"
    and it generally shrinks rather than expanding
    """

    def test_4d_input_2_groups_expanding(self):
        N, C, D, H, W = 4, 16, 12, 12, 12
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, C * 2, kernel_size=3, stride=1, groups=2)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)

    def test_4d_input_2_groups_shrinking(self):
        N, C, D, H, W = 4, 32, 12, 12, 12
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, C // 2, kernel_size=3, stride=1, groups=2)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)

    def test_4d_input_4_groups_expanding(self):
        N, C, D, H, W = 4, 16, 12, 12, 12
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, C * 2, kernel_size=3, stride=1, groups=4)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)

    def test_4d_input_4_groups_shrinking(self):
        N, C, D, H, W = 4, 32, 12, 12, 12
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, C // 2, kernel_size=3, stride=1, groups=4)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)

    def test_4d_input_16_groups_expanding(self):
        N, C, D, H, W = 4, 16, 12, 12, 12
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, C * 2, kernel_size=3, stride=1, groups=16)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)

    def test_4d_input_16_groups_shrinking(self):
        N, C, D, H, W = 4, 32, 12, 12, 12
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(C, C // 2, kernel_size=3, stride=1, groups=16)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)
