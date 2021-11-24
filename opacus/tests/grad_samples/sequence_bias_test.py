#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hypothesis.strategies as st
import torch
from hypothesis import given, settings
from opacus.layers import SequenceBias

from .common import GradSampleHooks_test


class SequenceBias_test(GradSampleHooks_test):
    @given(
        N=st.integers(1, 4),
        T=st.integers(10, 20),
        D=st.integers(4, 8),
        batch_first=st.booleans(),
    )
    @settings(deadline=10000)
    def test_batch_second(
        self,
        N: int,
        T: int,
        D: int,
        batch_first: bool,
    ):

        seqbias = SequenceBias(D, batch_first)
        if batch_first:
            x = torch.randn([N, T, D])
        else:
            x = torch.randn([T, N, D])
        self.run_test(x, seqbias, batch_first)
