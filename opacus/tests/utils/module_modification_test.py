#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest
from typing import Tuple

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.dp_model_inspector import IncompatibleModuleException
from opacus.utils import module_modification as mm
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import FakeData


class replace_all_modules_test(unittest.TestCase):
    def checkModulePresent(self, root: nn.Module, targetclass):
        result = any(isinstance(module, targetclass) for module in root.modules())
        self.assertTrue(result)

    def checkModuleNotPresent(self, root: nn.Module, targetclass):
        for module in root.modules():
            self.assertFalse(
                isinstance(module, targetclass),
                msg=f"{module} has the given targetclass type",
            )

    def test_replace_basic_case(self):
        model = nn.BatchNorm1d(10)
        model = mm.replace_all_modules(
            model, nn.BatchNorm1d, lambda _: nn.BatchNorm2d(10)
        )
        self.checkModulePresent(model, nn.BatchNorm2d)
        self.checkModuleNotPresent(model, nn.BatchNorm1d)

    def test_replace_sequential_case(self):
        model = nn.Sequential(nn.Conv1d(1, 2, 3), nn.Sequential(nn.Conv2d(4, 5, 6)))

        def conv(m: nn.Conv2d):
            return nn.Linear(4, 5)

        model = mm.replace_all_modules(model, nn.Conv2d, conv)
        self.checkModulePresent(model, nn.Linear)
        self.checkModuleNotPresent(model, nn.Conv2d)

    def test_nullify_resnet18(self):
        model = models.resnet18()
        # check module BatchNorms is there
        self.checkModulePresent(model, nn.BatchNorm2d)
        # nullify the module (replace with Idetity)
        model = mm.nullify_batchnorm_modules(model)
        # check module is not present
        self.checkModuleNotPresent(model, nn.BatchNorm2d)

    def test_convert_batchnorm_modules_resnet50(self):
        model = models.resnet50()
        # check module BatchNorms is there
        self.checkModulePresent(model, nn.BatchNorm2d)
        # replace the module with instancenorm
        model = mm.convert_batchnorm_modules(model)
        # check module is not present
        self.checkModuleNotPresent(model, nn.BatchNorm2d)
        self.checkModulePresent(model, nn.GroupNorm)


class BasicModel(nn.Module):
    def __init__(self, imgSize):
        super().__init__()
        self.size = imgSize[0] * imgSize[1] * imgSize[2]
        self.bn = nn.BatchNorm2d(imgSize[0])
        self.fc = nn.Linear(self.size, 2)

    def forward(self, input):
        x = self.bn(input)
        x = x.view(-1, self.size)
        x = self.fc(x)
        return x


class convert_batchnorm_modules_test(unittest.TestCase):
    def setUp(self):
        self.criterion = nn.CrossEntropyLoss()

    def setUpOptimizer(
        self, model: nn.Module, data_loader: DataLoader, privacy_engine: bool = False
    ):
        # sample parameter values
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        optimizer.zero_grad()
        if privacy_engine:
            pe = PrivacyEngine(
                model,
                sample_rate=data_loader.batch_size / len(data_loader.dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=1.3,
                max_grad_norm=1,
            )
            pe.attach(optimizer)
        return optimizer

    def genFakeData(
        self, imgSize: Tuple[int, int, int], batch_size: int = 1, num_batches: int = 1
    ) -> DataLoader:
        self.ds = FakeData(
            size=num_batches,
            image_size=imgSize,
            num_classes=2,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        return DataLoader(self.ds, batch_size=batch_size)

    def runOneBatch(
        self,
        model: nn.Module,
        imgsize: Tuple[int, int, int],
        privacy_engine: bool = True,
    ):
        dl = self.genFakeData(imgsize, 1, 1)
        optimizer = self.setUpOptimizer(model, dl, privacy_engine)
        for x, y in dl:
            # forward
            try:
                logits = model(x)
            except Exception as err:
                self.fail(f"Failed forward step with exception: {err}")
            loss = self.criterion(logits, y)
            # backward
            try:
                loss.backward()
            except Exception as err:
                self.fail(f"Failed backward step with exception: {err}")
            # optimizer
            try:
                optimizer.step()
            except Exception as err:
                self.fail(f"Failed optimizer step with exception: {err}")
        optimizer.zero_grad()

    def test_run_basic_case(self):
        imgSize = (3, 4, 5)
        # should throw because privacy engine does not work with batch norm
        # remove the next two lines when we support batch norm
        with self.assertRaises(IncompatibleModuleException):
            self.runOneBatch(BasicModel(imgSize), imgSize)
        self.runOneBatch(mm.convert_batchnorm_modules(BasicModel(imgSize)), imgSize)

    def test_run_resnet18(self):
        imgSize = (3, 224, 224)
        # should throw because privacy engine does not work with batch norm
        # remove the next two lines when we support batch norm
        with self.assertRaises(IncompatibleModuleException):
            self.runOneBatch(models.resnet18(), imgSize)
        self.runOneBatch(mm.convert_batchnorm_modules(models.resnet18()), imgSize)

    def test_run_resnet34(self):
        imgSize = (3, 224, 224)
        # should throw because privacy engine does not work with batch norm
        # remove the next two lines when we support batch norm
        with self.assertRaises(IncompatibleModuleException):
            self.runOneBatch(models.resnet34(), imgSize)
        self.runOneBatch(mm.convert_batchnorm_modules(models.resnet34()), imgSize)

    def test_run_resnet50(self):
        imgSize = (3, 224, 224)
        # should throw because privacy engine does not work with batch norm
        # remove the next two lines when we support batch norm
        with self.assertRaises(IncompatibleModuleException):
            self.runOneBatch(models.resnet50(), imgSize)
        self.runOneBatch(mm.convert_batchnorm_modules(models.resnet50()), imgSize)

    def test_run_resnet101(self):
        imgSize = (3, 224, 224)
        # should throw because privacy engine does not work with batch norm
        # remove the next two lines when we support batch norm
        with self.assertRaises(IncompatibleModuleException):
            self.runOneBatch(models.resnet101(), imgSize)
        self.runOneBatch(mm.convert_batchnorm_modules(models.resnet101()), imgSize)
