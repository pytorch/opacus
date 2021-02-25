#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings

from .common import GradSampleHooks_test


class Embedding_test(GradSampleHooks_test):
    @given(
        N=st.integers(1, 4),
        T=st.integers(1, 5),
        Q=st.integers(1, 4),
        R=st.integers(1, 2),
        V=st.integers(2, 32),
        D=st.integers(10, 17),
        dim=st.integers(1, 4),
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
