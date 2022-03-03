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
from opacus.layers import DPLSTM
from opacus.utils.module_utils import are_state_dict_equal
from opacus.validators.errors import ShouldReplaceModuleError
from opacus.validators.module_validator import ModuleValidator


class LSTMValidator_test(unittest.TestCase):
    def setUp(self):
        self.lstm = nn.LSTM(8, 4)
        self.mv = ModuleValidator.VALIDATORS
        self.mf = ModuleValidator.FIXERS

    def test_validate(self):
        val_lstm = self.mv[type(self.lstm)](self.lstm)
        self.assertEqual(len(val_lstm), 1)
        self.assertTrue(isinstance(val_lstm[0], ShouldReplaceModuleError))

    def test_fix(self):
        fix_lstm = self.mf[type(self.lstm)](self.lstm)
        self.assertTrue(isinstance(fix_lstm, DPLSTM))
        self.assertTrue(
            are_state_dict_equal(self.lstm.state_dict(), fix_lstm.state_dict())
        )
