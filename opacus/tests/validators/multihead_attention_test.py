#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch.nn as nn
from opacus.layers import DPMultiheadAttention
from opacus.utils.module_utils import are_state_dict_equal
from opacus.validators.errors import ShouldReplaceModuleError
from opacus.validators.module_validator import ModuleValidator


class MultiheadAttentionValidator_test(unittest.TestCase):
    def setUp(self):
        self.mha = nn.MultiheadAttention(8, 4)
        self.mv = ModuleValidator.VALIDATORS
        self.mf = ModuleValidator.FIXERS

    def test_validate(self):
        val_mha = self.mv[type(self.mha)](self.mha)
        self.assertEqual(len(val_mha), 1)
        self.assertTrue(isinstance(val_mha[0], ShouldReplaceModuleError))

    def test_fix(self):
        fix_mha = self.mf[type(self.mha)](self.mha)
        self.assertTrue(isinstance(fix_mha, DPMultiheadAttention))
        self.assertFalse(  # state_dicts are not same
            are_state_dict_equal(self.mha.state_dict(), fix_mha.state_dict())
        )
