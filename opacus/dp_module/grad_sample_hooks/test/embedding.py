#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test, ModelWithLoss


class Embedding_test(GradSampleHooks_test):
    def test_2d_input(self):
        V, D = 128, 17
        N, S = 32, 12

        emb = nn.Embedding(V, D)
        x = torch.randint(low=0, high=V - 1, size=[N, S])

        module = ModelWithLoss(emb, n_classes=S * D)

        self.run_test(x, module, batch_first=True)

    def test_3d_input(self):
        V, D = 128, 17
        N, S, T = 32, 12, 4

        emb = nn.Embedding(V, D)
        x = torch.randint(low=0, high=V - 1, size=[N, S, T])

        module = ModelWithLoss(emb, n_classes=S * T * D)

        self.run_test(x, module, batch_first=True)
