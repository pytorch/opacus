import unittest

import torch
import torch.nn as nn
from opacus.dp_module import DPModule, UnsupportableModuleError
from torchvision.models import resnet18


class SampleModel(nn.Module):
    def __init__(self, C, W, normalize=True):
        super().__init__()
        self.fc1 = nn.Linear(W, W + 2)
        self.norm = nn.BatchNorm2d(C) if normalize else nn.Identity()
        self.fc2 = nn.Linear(W + 2, W + 4)

    def forward(self, x):
        N, C, H, W = x.shape
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class DPModule_test(unittest.TestCase):
    def setUp(self):
        N, C, H, W = 8, 3, 10, 12
        self.model_with_bn = SampleModel(C, W, normalize=True)
        self.model_without_bn = SampleModel(C, W, normalize=False)
        self.resnet = resnet18(pretrained=False)
        self.x = torch.randn([N, C, H, W])

    def test_raises_for_sample_model_containing_batchnorm(self):
        with self.assertRaises(UnsupportableModuleError):
            DPModule(self.model_with_bn, strict=True)

    def test_raises_for_resnet(self):
        with self.assertRaises(UnsupportableModuleError):
            DPModule(self.resnet, strict=True)

    def test_not_raises_for_valid(self):
        DPModule(self.model_without_bn, strict=True)

    def test_replacement_works_for_sample_model_containing_batchnorm(self):
        y = self.model_with_bn(self.x)
        dp_module = DPModule(self.model_with_bn, strict=False)
        dp_y = dp_module(self.x)
        self.assertEquals(y.shape, dp_y.shape)

    def test_replacement_works_for_resnet(self):
        y = self.resnet(self.x)
        dp_module = DPModule(self.resnet, strict=False)
        dp_y = dp_module(self.x)
        self.assertEquals(y.shape, dp_y.shape)
