#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch.nn as nn
from opacus.validators.errors import IllegalModuleConfigurationError
from opacus.validators.module_validator import ModuleValidator


class InstanceNormValidator_test(unittest.TestCase):
    def setUp(self):
        self.in1 = nn.InstanceNorm1d(4, affine=True, track_running_stats=True)
        self.in2 = nn.InstanceNorm2d(4, affine=False, track_running_stats=True)
        self.in3 = nn.InstanceNorm3d(4, affine=False, track_running_stats=True)
        self.in_no_stats = nn.InstanceNorm3d(4, affine=True)

        self.mv = ModuleValidator.VALIDATORS
        self.mf = ModuleValidator.FIXERS

    def test_validate(self):
        val1 = self.mv[type(self.in1)](self.in1)
        val2 = self.mv[type(self.in2)](self.in2)
        val3 = self.mv[type(self.in3)](self.in3)
        vals = self.mv[type(self.in_no_stats)](self.in_no_stats)

        self.assertEqual(len(val1), 1)
        self.assertEqual(len(val2), 1)
        self.assertEqual(len(val3), 1)
        self.assertEqual(len(vals), 0)

        self.assertTrue(isinstance(val1[0], IllegalModuleConfigurationError))
        self.assertTrue(isinstance(val2[0], IllegalModuleConfigurationError))
        self.assertTrue(isinstance(val3[0], IllegalModuleConfigurationError))

    def test_fix(self):
        fix1 = self.mf[type(self.in1)](self.in1)
        fix2 = self.mf[type(self.in2)](self.in2)
        fix3 = self.mf[type(self.in3)](self.in3)
        fixs = self.mf[type(self.in_no_stats)](self.in_no_stats)

        self.assertTrue(isinstance(fix1, nn.InstanceNorm1d))
        self.assertFalse(fix1.track_running_stats)

        self.assertTrue(isinstance(fix2, nn.InstanceNorm2d))
        self.assertFalse(fix2.track_running_stats)

        self.assertTrue(isinstance(fix3, nn.InstanceNorm3d))
        self.assertFalse(fix3.track_running_stats)

        self.assertTrue(isinstance(fixs, nn.InstanceNorm3d))
        self.assertTrue(fixs is self.in_no_stats)
