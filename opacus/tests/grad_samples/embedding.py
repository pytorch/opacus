#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test


class Embedding_test(GradSampleHooks_test):
    def test_1d_input(self):
        V, D = 128, 17
        T = 12

        emb = nn.Embedding(V, D)
        x = torch.randint(low=0, high=V - 1, size=[T])
        self.run_test(x, emb, batch_first=True)

    def test_2d_input(self):
        V, D = 128, 17
        N, T = 32, 12

        emb = nn.Embedding(V, D)
        x = torch.randint(low=0, high=V - 1, size=[N, T])
        self.run_test(x, emb, batch_first=True)

    def test_3d_input(self):
        V, D = 128, 17
        N, T, Q = 32, 12, 4

        emb = nn.Embedding(V, D)
        x = torch.randint(low=0, high=V - 1, size=[N, T, Q])
        self.run_test(x, emb, batch_first=True)

    def test_4d_input(self):
        V, D = 128, 17
        N, T, Q, R = 32, 12, 4, 2

        emb = nn.Embedding(V, D)
        x = torch.randint(low=0, high=V - 1, size=[N, T, Q, R])
        self.run_test(x, emb, batch_first=True)
