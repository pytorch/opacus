#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from typing import Optional, Tuple, Callable, Union

import hypothesis.strategies as st
from hypothesis import given, settings

from .common import GradSampleHooks_test


class Embedding_test(GradSampleHooks_test):
    @given(
        N=st.sampled_from([32]),
        T=st.sampled_from([12]),
        Q=st.sampled_from([4]),
        R=st.sampled_from([2]),
        V=st.sampled_from([128]),
        D=st.sampled_from([17]),
        dim=st.sampled_from([1, 2, 3, 4]),
    )
    @settings(deadline=10000)
    def test_input_across_dims(
        self,
        N: int,
        T: int,
        Q: int,
        R: int,
        V: int,
        D: int,
        dim: int,
    ):

        if dim == 1:
            size = [T]
        elif dim == 2:
            size = [N, T]
        elif dim == 3:
            size = [N, T, Q]
        elif dim == 4:
            size = [N, T, Q, R]

        emb = nn.Embedding(V, D)
        x = torch.randint(low=0, high=V - 1, size=size)
        self.run_test(x, emb, batch_first=True)
