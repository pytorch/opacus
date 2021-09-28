#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import torch.nn as nn
from opacus.utils import module_inspection as mi
from torchvision import models


class utils_ModelInspector_test(unittest.TestCase):
    def setUp(self):
        def pred_supported(module):
            return isinstance(module, (nn.Conv2d, nn.Linear))

        def pred_not_unsupported(module):
            return not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d))

        def pred_requires_grad(module):
            return all(p.requires_grad for p in module.parameters(recurse=False))

        self.pred_supported = pred_supported
        self.pred_not_unsupported = pred_not_unsupported
        self.pred_mix = lambda m: (not pred_requires_grad(m)) or pred_not_unsupported(m)

    def test_validate_basic(self):
        inspector = mi.ModelInspector(
            "pred", lambda model: isinstance(model, nn.Linear)
        )
        model = nn.Conv1d(1, 1, 1)
        valid = inspector.validate(model)
        self.assertFalse(valid, inspector.violators)

    def test_validate_positive_predicate_valid(self):
        # test when a positive predicate (e.g. supported) returns true
        inspector = mi.ModelInspector("pred", self.pred_supported)
        model = nn.Conv2d(1, 1, 1)
        valid = inspector.validate(model)
        self.assertTrue(valid)
        list_len = len(inspector.violators)
        self.assertEqual(list_len, 0, f"violators = {inspector.violators}")

    def test_validate_positive_predicate_invalid(self):
        # test when a positive predicate (e.g. supported) returns false
        inspector = mi.ModelInspector("pred", self.pred_supported)
        model = nn.Conv1d(1, 1, 1)
        valid = inspector.validate(model)
        self.assertFalse(valid)
        list_len = len(inspector.violators)
        self.assertEqual(list_len, 1, f"violators = {inspector.violators}")

    def test_validate_negative_predicate_ture(self):
        # test when a negative predicate (e.g. not unsupported) returns true
        inspector = mi.ModelInspector("pred1", self.pred_not_unsupported)
        model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Linear(1, 1))
        valid = inspector.validate(model)
        self.assertTrue(valid)
        list_len = len(inspector.violators)
        self.assertEqual(list_len, 0)

    def test_validate_negative_predicate_False(self):
        # test when a negative predicate (e.g. not unsupported) returns false
        inspector = mi.ModelInspector("pred", self.pred_not_unsupported)
        model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
        valid = inspector.validate(model)
        self.assertFalse(valid)
        list_len = len(inspector.violators)
        self.assertEqual(list_len, 1, f"violators = {inspector.violators}")

    def test_validate_mix_predicate(self):
        # check with a mix predicate not requires grad or is not unsupported
        inspector = mi.ModelInspector("pred1", self.pred_mix)
        model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1))
        for p in model[1].parameters():
            p.requires_grad = False
        valid = inspector.validate(model)
        self.assertTrue(valid)

    def test_check_everything_flag(self):
        # check to see if a model does not containt nn.sequential
        inspector = mi.ModelInspector(
            "pred",
            lambda model: not isinstance(model, nn.Sequential),
            check_leaf_nodes_only=False,
        )
        model = nn.Sequential(nn.Conv1d(1, 1, 1))
        valid = inspector.validate(model)
        self.assertFalse(valid, f"violators = {inspector.violators}")

    def test_complicated_case(self):
        def good(x):
            return isinstance(x, (nn.Conv2d, nn.Linear))

        def bad(x):
            return isinstance(x, nn.modules.batchnorm._BatchNorm)

        inspector1 = mi.ModelInspector("good_or_bad", lambda x: good(x) or bad(x))
        inspector2 = mi.ModelInspector("not_bad", lambda x: not bad(x))
        model = models.resnet50()
        valid = inspector1.validate(model)
        self.assertTrue(valid, f"violators = {inspector1.violators}")
        self.assertEqual(
            len(inspector1.violators), 0, f"violators = {inspector1.violators}"
        )
        valid = inspector2.validate(model)
        self.assertFalse(valid, f"violators = {inspector2.violators}")
        self.assertEqual(
            len(inspector2.violators), 53, f"violators = {inspector2.violators}"
        )
