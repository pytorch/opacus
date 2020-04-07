#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchdp import PrivacyEngine, utils
from torchvision import models, transforms
from torchvision.datasets import FakeData


class utils_replace_all_modules_test(unittest.TestCase):
    def checkModulePresent(self, root: nn.Module, targetclass):
        result = False
        for module in root.modules():
            result |= isinstance(module, targetclass)
        self.assertTrue(result)

    def checkModuleNotPresent(self, root: nn.Module, targetclass):
        for module in root.modules():
            self.assertFalse(
                isinstance(module, targetclass),
                msg=f"{module} has the given targetclass type",
            )

    def test_replace_basic_case(self):
        model = nn.BatchNorm1d(10)
        model = utils.replace_all_modules(
            model, nn.BatchNorm1d, lambda _: nn.BatchNorm2d(10)
        )
        self.checkModulePresent(model, nn.BatchNorm2d)
        self.checkModuleNotPresent(model, nn.BatchNorm1d)

    def test_replace_sequential_case(self):
        model = nn.Sequential(nn.Conv1d(1, 2, 3), nn.Sequential(nn.Conv2d(4, 5, 6)))

        def conv(m: nn.Conv2d):
            return nn.Linear(4, 5)

        model = utils.replace_all_modules(model, nn.Conv2d, conv)
        self.checkModulePresent(model, nn.Linear)
        self.checkModuleNotPresent(model, nn.Conv2d)

    def test_nullify_resnet18(self):
        model = models.resnet18()
        # check module BatchNorms is there
        self.checkModulePresent(model, nn.BatchNorm2d)
        # nullify the module (replace with Idetity)
        model = utils.nullify_batchnorm_modules(model, nn.BatchNorm2d)
        # check module is not present
        self.checkModuleNotPresent(model, nn.BatchNorm2d)

    def test_convert_batchnorm_modules_resnet50(self):
        model = models.resnet50()
        # check module BatchNorms is there
        self.checkModulePresent(model, nn.BatchNorm2d)
        # replace the module with instancenorm
        model = utils.convert_batchnorm_modules(model)
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


class utils_convert_batchnorm_modules_test(unittest.TestCase):
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
                batch_size=data_loader.batch_size,
                sample_size=len(data_loader.dataset),
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
        with self.assertRaises(Exception):
            self.runOneBatch(BasicModel(imgSize), imgSize)
        self.runOneBatch(
            utils.convert_batchnorm_modules(BasicModel(imgSize)), imgSize)

    def test_run_resnet18(self):
        imgSize = (3, 224, 224)
        # should throw because privacy engine does not work with batch norm
        # remove the next two lines when we support batch norm
        with self.assertRaises(Exception):
            self.runOneBatch(models.resnet18(), imgSize)
        self.runOneBatch(
            utils.convert_batchnorm_modules(models.resnet18()), imgSize)

    def test_run_resnet34(self):
        imgSize = (3, 224, 224)
        # should throw because privacy engine does not work with batch norm
        # remove the next two lines when we support batch norm
        with self.assertRaises(Exception):
            self.runOneBatch(models.resnet34(), imgSize)
        self.runOneBatch(
            utils.convert_batchnorm_modules(models.resnet34()), imgSize)

    def test_run_resnet50(self):
        imgSize = (3, 224, 224)
        # should throw because privacy engine does not work with batch norm
        # remove the next two lines when we support batch norm
        with self.assertRaises(Exception):
            self.runOneBatch(models.resnet50(), imgSize)
        self.runOneBatch(utils.convert_batchnorm_modules(models.resnet50()), imgSize)

    def test_run_resnet101(self):
        imgSize = (3, 224, 224)
        # should throw because privacy engine does not work with batch norm
        # remove the next two lines when we support batch norm
        with self.assertRaises(Exception):
            self.runOneBatch(models.resnet101(), imgSize)
        self.runOneBatch(
            utils.convert_batchnorm_modules(models.resnet101()), imgSize)


class utils_ModelInspector_test(unittest.TestCase):
    def setUp(self):
        def pred_supported(module):
            return isinstance(module, (nn.Conv2d, nn.Linear))

        def pred_not_unsupported(module):
            return not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d))

        def pred_requires_grad(module):
            requires_grad = True
            for p in module.parameters(recurse=False):
                requires_grad &= p.requires_grad
            return requires_grad

        self.pred_supported = pred_supported
        self.pred_not_unsupported = pred_not_unsupported
        self.pred_mix = lambda m: (not pred_requires_grad(m)) | pred_not_unsupported(m)

    def test_validate_basic(self):
        inspector = utils.ModelInspector(
            'pred', lambda model: isinstance(model, nn.Linear)
        )
        model = nn.Conv1d(1, 1, 1)
        valid = inspector.validate(model)
        self.assertFalse(valid, inspector.violators)

    def test_validate_positive_predicate_valid(self):
        # test when a positive predicate (e.g. supported) returns true
        inspector = utils.ModelInspector('pred', self.pred_supported)
        model = nn.Conv2d(1, 1, 1)
        valid = inspector.validate(model)
        self.assertTrue(valid)
        list_len = len(inspector.violators)
        self.assertEqual(list_len, 0, f'violators = {inspector.violators}')

    def test_validate_positive_predicate_invalid(self):
        # test when a positive predicate (e.g. supported) returns false
        inspector = utils.ModelInspector('pred', self.pred_supported)
        model = nn.Conv1d(1, 1, 1)
        valid = inspector.validate(model)
        self.assertFalse(valid)
        list_len = len(inspector.violators)
        self.assertEqual(list_len, 1, f'violators = {inspector.violators}')

    def test_validate_negative_predicate_ture(self):
        # test when a negative predicate (e.g. not unsupported) returns true
        inspector = utils.ModelInspector('pred1', self.pred_not_unsupported)
        model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Linear(1, 1))
        valid = inspector.validate(model)
        self.assertTrue(valid)
        list_len = len(inspector.violators)
        self.assertEqual(list_len, 0)

    def test_validate_negative_predicate_False(self):
        # test when a negative predicate (e.g. not unsupported) returns false
        inspector = utils.ModelInspector('pred', self.pred_not_unsupported)
        model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
        valid = inspector.validate(model)
        self.assertFalse(valid)
        list_len = len(inspector.violators)
        self.assertEqual(list_len, 1, f'violators = {inspector.violators}')

    def test_validate_mix_predicate(self):
        # check with a mix predicate not requires grad or is not unsupported
        inspector = utils.ModelInspector('pred1', self.pred_mix)
        model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
        for p in model[1].parameters():
            p.requires_grad = False
        valid = inspector.validate(model)
        self.assertTrue(valid)

    def test_check_everything_flag(self):
        # check to see if a model does not containt nn.sequential
        inspector = utils.ModelInspector(
            'pred', lambda model: not isinstance(model, nn.Sequential),
            check_leaf_nodes_only=False
        )
        model = nn.Sequential(nn.Conv1d(1, 1, 1))
        valid = inspector.validate(model)
        self.assertFalse(
            valid, f'violators = {inspector.violators}')

    def test_complicated_case(self):
        def good(x):
            return isinstance(x, (nn.Conv2d, nn.Linear))

        def bad(x):
            return isinstance(x, nn.modules.batchnorm._BatchNorm)

        inspector1 = utils.ModelInspector(
            'good_or_bad', lambda x: good(x) | bad(x))
        inspector2 = utils.ModelInspector(
            'not_bad', lambda x: not bad(x))
        model = models.resnet50()
        valid = inspector1.validate(model)
        self.assertTrue(valid, f'violators = {inspector1.violators}')
        self.assertEqual(
            len(inspector1.violators),
            0,
            f'violators = {inspector1.violators}')
        valid = inspector2.validate(model)
        self.assertFalse(valid, f'violators = {inspector2.violators}')
        self.assertEqual(
            len(inspector2.violators),
            53,
            f'violators = {inspector2.violators}')
