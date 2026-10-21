#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings

from .common import GradSampleHooks_test


def chunk_method(x):
    input, offsets = x
    for i_offset, offset in enumerate(offsets):
        if i_offset < len(offsets) - 1:
            next_offset = offsets[i_offset + 1]
        else:
            next_offset = len(input)
        yield (input[offset:next_offset], torch.LongTensor([0]))


class Embedding_bag_test(GradSampleHooks_test):
    @given(
        N=st.integers(4, 8),
        sz=st.integers(3, 7),
        V=st.integers(10, 32),
        D=st.integers(10, 17),
        mode=st.sampled_from(["sum", "mean"]),
    )
    @settings(deadline=60000)
    def test_input_across_dims(
        self,
        N: int,
        sz: int,
        V: int,
        D: int,
        mode: str,
    ):
        emb = nn.EmbeddingBag(num_embeddings=V, embedding_dim=D, mode=mode)

        sizes = torch.randint(low=1, high=sz + 1, size=(N,))
        offsets = torch.LongTensor([0] + torch.cumsum(sizes, dim=0).tolist()[:-1])
        input = []
        for size in sizes:
            input += [torch.randperm(V)[:size]]

        input = torch.cat(input, dim=0)

        self.run_test(
            (input, offsets), emb, chunk_method=chunk_method, ew_compatible=False
        )

    def test_repeated_indices_in_bag(self):
        # Bags that contain the same index more than once, e.g. a repeated
        # word in a bag-of-words model.
        input = torch.LongTensor([3, 3, 5, 1, 2, 7, 7, 7])
        offsets = torch.LongTensor([0, 3, 5])
        for mode in ["sum", "mean"]:
            with self.subTest(mode=mode):
                emb = nn.EmbeddingBag(num_embeddings=10, embedding_dim=4, mode=mode)
                self.run_test(
                    (input, offsets),
                    emb,
                    chunk_method=chunk_method,
                    ew_compatible=False,
                )
