#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

from typing import Optional, Tuple, Callable, Union

import hypothesis.strategies as st
from hypothesis import given, settings

from opacus.layers import SequenceBias
from .common import GradSampleHooks_test


class SequenceBias_test(GradSampleHooks_test):
    @given(
        N=st.sampled_from([32]),
        T=st.sampled_from([20]),
        D=st.sampled_from([8]),
    )
    @settings(deadline=10000)
    def test_batch_second(
        self,
        N: int,
        T: int,
        D: int,
    ):

        seqbias = SequenceBias(D)
        x = torch.randn([T, N, D])
        self.run_test(x, seqbias, batch_first=False)
