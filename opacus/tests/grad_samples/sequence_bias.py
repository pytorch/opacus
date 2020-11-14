#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from opacus.layers import SequenceBias

from .common import GradSampleHooks_test


class SequenceBias_test(GradSampleHooks_test):
    def test_batch_second(self):
        N, T, D = 32, 20, 8
        seqbias = SequenceBias(D)
        x = torch.randn([T, N, D])
        self.run_test(x, seqbias, batch_first=False)
