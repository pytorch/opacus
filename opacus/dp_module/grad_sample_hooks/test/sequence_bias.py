#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from opacus.dp_module.modules import SequenceBias

from .common import GradSampleHooks_test, ModelWithLoss


class SequenceBias_test(GradSampleHooks_test):
    def test_3d_input_batch_first(self):
        Z, N, W = 4, 32, 10
        seq_bias = SequenceBias(W)
        module = ModelWithLoss(seq_bias, n_classes=W)
        x = torch.randn([N, Z, W])
        self.run_test(x, module, batch_first=True)

    def test_3d_input_batch_second(self):
        Z, N, W = 4, 32, 10
        seq_bias = SequenceBias(W)
        module = ModelWithLoss(seq_bias, n_classes=W)
        x = torch.randn([Z, N, W])
        self.run_test(x, module, batch_first=False)
