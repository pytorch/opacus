#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test


class Conv2d_test(GradSampleHooks_test):
    def test_square_image_kernel2(self):
        N, C, H, W = 32, 3, 10, 10
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=2)
        self.run_test(x, conv, batch_first=True)

    def test_landscape_image_kernel2(self):
        N, C, H, W = 32, 3, 9, 16
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=2)
        self.run_test(x, conv, batch_first=True)

    def test_portrait_image_kernel2(self):
        N, C, H, W = 32, 3, 16, 9
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=2)
        self.run_test(x, conv, batch_first=True)

    def test_square_image_kernel3(self):
        N, C, H, W = 32, 3, 10, 10
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=3)
        self.run_test(x, conv, batch_first=True)

    def test_landscape_image_kernel3(self):
        N, C, H, W = 32, 3, 9, 16
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=3)
        self.run_test(x, conv, batch_first=True)

    def test_portrait_image_kernel3(self):
        N, C, H, W = 32, 3, 16, 9
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=3)
        self.run_test(x, conv, batch_first=True)

    def test_square_image_stride_2(self):
        N, C, H, W = 32, 3, 10, 10
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=2, stride=2)
        self.run_test(x, conv, batch_first=True)

    def test_landscape_image_stride_2(self):
        N, C, H, W = 32, 3, 9, 16
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=2, stride=2)
        self.run_test(x, conv, batch_first=True)

    def test_portrait_image_stride_2(self):
        N, C, H, W = 32, 3, 16, 9
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=2, stride=2)
        self.run_test(x, conv, batch_first=True)

    def test_square_image_pad_2(self):
        N, C, H, W = 32, 3, 10, 10
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=2, padding=2)
        self.run_test(x, conv, batch_first=True)

    def test_landscape_image_pad_2(self):
        N, C, H, W = 32, 3, 9, 16
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=2, padding=2)
        self.run_test(x, conv, batch_first=True)

    def test_portrait_image_pad_2(self):
        N, C, H, W = 32, 3, 16, 9
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, 2 * C, kernel_size=2, padding=2)
        self.run_test(x, conv, batch_first=True)

    """
    These generally show up in upper layers. At this point, C is just generic "channels"
    and it generally shrinks rather than expanding
    """

    def test_4d_input_2_groups_expanding(self):
        N, C, H, W = 48, 16, 24, 24
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, C * 2, kernel_size=3, stride=1, groups=2)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)

    def test_4d_input_2_groups_shrinking(self):
        N, C, H, W = 48, 32, 24, 24
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, C // 2, kernel_size=3, stride=1, groups=2)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)

    def test_4d_input_4_groups_expanding(self):
        N, C, H, W = 48, 16, 24, 24
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, C * 2, kernel_size=3, stride=1, groups=4)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)

    def test_4d_input_4_groups_shrinking(self):
        N, C, H, W = 48, 32, 24, 24
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, C // 2, kernel_size=3, stride=1, groups=4)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)

    def test_4d_input_16_groups_expanding(self):
        N, C, H, W = 48, 16, 24, 24
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, C * 2, kernel_size=3, stride=1, groups=16)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)

    def test_4d_input_16_groups_shrinking(self):
        N, C, H, W = 48, 32, 24, 24
        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(C, C // 2, kernel_size=3, stride=1, groups=16)
        self.run_test(x, conv, batch_first=True, atol=10e-5, rtol=10e-4)
