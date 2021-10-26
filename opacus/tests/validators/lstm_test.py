#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
