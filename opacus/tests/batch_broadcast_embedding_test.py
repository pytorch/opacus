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

import unittest

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.optimizers.perlayeroptimizer import DPPerLayerOptimizer
from opacus.optimizers.utils import check_grad_sample_batch_size_consistency


class BroadcastEmbeddingModel(nn.Module):
    def __init__(self, vocab_size=50, max_positions=32, embedding_dim=16):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embedding_dim)
        self.wpe = nn.Embedding(max_positions, embedding_dim)

    def forward(self, input_ids, position_ids=None):
        B, T = input_ids.shape
        if position_ids is None:
            # Broadcasted position embeddings with batch size 1 (like HuggingFace GPT-2)
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.wte(input_ids) + self.wpe(position_ids)


class BatchBroadcastEmbeddingTest(unittest.TestCase):
    def test_check_grad_sample_batch_size_consistency_unit(self):
        p1 = nn.Parameter(torch.zeros(10, 16))
        p1._param_name = "wte.weight"
        p2 = nn.Parameter(torch.zeros(10, 16))
        p2._param_name = "wpe.weight"

        # Matching batch sizes (batch_size=4)
        g1 = torch.zeros(4, 10, 16)
        g2 = torch.zeros(4, 10, 16)
        check_grad_sample_batch_size_consistency([p1, p2], [g1, g2])

        # Mismatched batch sizes (4 vs 1)
        g2_mismatch = torch.zeros(1, 10, 16)
        with self.assertRaises(ValueError) as ctx:
            check_grad_sample_batch_size_consistency([p1, p2], [g1, g2_mismatch])

        err_msg = str(ctx.exception)
        self.assertIn("Inconsistent per-sample gradient batch dimensions", err_msg)
        self.assertIn("wpe.weight", err_msg)
        self.assertIn("batch dimension 1, expected 4", err_msg)
        self.assertIn("position_ids.expand", err_msg)

    def test_unexpanded_position_ids_raises_informative_error(self):
        model = BroadcastEmbeddingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        privacy_engine = PrivacyEngine()
        model, optimizer, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=torch.utils.data.DataLoader(
                torch.randint(0, 50, (8, 10)), batch_size=4
            ),
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        input_ids = torch.randint(0, 50, (4, 10))
        # Forward with unexpanded position IDs (broadcasted internally)
        out = model(input_ids)
        loss = out.sum()
        loss.backward()

        with self.assertRaises(ValueError) as ctx:
            optimizer.step()

        err_msg = str(ctx.exception)
        self.assertIn("Inconsistent per-sample gradient batch dimensions", err_msg)
        self.assertIn("wpe.weight", err_msg)
        self.assertIn("batch dimension 1, expected 4", err_msg)

    def test_expanded_position_ids_succeeds(self):
        model = BroadcastEmbeddingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        privacy_engine = PrivacyEngine()
        model, optimizer, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=torch.utils.data.DataLoader(
                torch.randint(0, 50, (8, 10)), batch_size=4
            ),
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        B, T = 4, 10
        input_ids = torch.randint(0, 50, (B, T))
        position_ids = torch.arange(T).unsqueeze(0).expand(B, T)
        out = model(input_ids, position_ids=position_ids)
        loss = out.sum()
        loss.backward()

        # Should step without errors
        optimizer.step()
        self.assertEqual(model._module.wte.weight.grad_sample.shape[0], B)
        self.assertEqual(model._module.wpe.weight.grad_sample.shape[0], B)

    def test_frozen_position_embedding_succeeds(self):
        model = BroadcastEmbeddingModel()
        model.wpe.weight.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=0.1)
        privacy_engine = PrivacyEngine()
        model, optimizer, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=torch.utils.data.DataLoader(
                torch.randint(0, 50, (8, 10)), batch_size=4
            ),
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        input_ids = torch.randint(0, 50, (4, 10))
        out = model(input_ids)  # uses unexpanded position IDs, but wpe is frozen
        loss = out.sum()
        loss.backward()

        optimizer.step()
        self.assertEqual(model._module.wte.weight.grad_sample.shape[0], 4)
        self.assertFalse(hasattr(model._module.wpe.weight, "grad_sample"))

    def test_adaclip_mismatch_raises_informative_error(self):
        model = BroadcastEmbeddingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_opt = AdaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            expected_batch_size=4,
            target_unclipped_quantile=0.8,
            clipbound_learning_rate=0.1,
            max_clipbound=10.0,
            min_clipbound=0.1,
            unclipped_num_std=1.0,
        )
        p1, p2 = model.wte.weight, model.wpe.weight
        p1._param_name = "wte.weight"
        p2._param_name = "wpe.weight"
        p1.grad_sample = torch.zeros(4, 50, 16)
        p2.grad_sample = torch.zeros(1, 32, 16)

        with self.assertRaises(ValueError) as ctx:
            dp_opt.clip_and_accumulate()

        self.assertIn(
            "Inconsistent per-sample gradient batch dimensions", str(ctx.exception)
        )

    def test_perlayer_mismatch_raises_informative_error(self):
        model = BroadcastEmbeddingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_opt = DPPerLayerOptimizer(
            optimizer=optimizer,
            noise_multiplier=1.0,
            max_grad_norm=[1.0, 1.0],
            expected_batch_size=4,
        )
        p1, p2 = model.wte.weight, model.wpe.weight
        p1._param_name = "wte.weight"
        p2._param_name = "wpe.weight"
        p1.grad_sample = torch.zeros(4, 50, 16)
        p2.grad_sample = torch.zeros(1, 32, 16)

        with self.assertRaises(ValueError) as ctx:
            dp_opt.clip_and_accumulate()

        self.assertIn(
            "Inconsistent per-sample gradient batch dimensions", str(ctx.exception)
        )

    def test_gpt2_full_finetuning_informative_error_and_resolution(self):
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError:
            self.skipTest("transformers not installed")

        config = GPT2Config(
            vocab_size=100, n_positions=64, n_embd=32, n_layer=2, n_head=2
        )
        model = GPT2LMHeadModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        privacy_engine = PrivacyEngine()
        model, optimizer, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=torch.utils.data.DataLoader(
                torch.randint(0, 100, (8, 16)), batch_size=4
            ),
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        B, T = 4, 16
        input_ids = torch.randint(0, 100, (B, T))

        # 1. Without explicit position_ids (GPT-2 broadcasts [1, T] position embeddings):
        # Must raise informative ValueError mentioning transformer.wpe.weight
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        with self.assertRaises(ValueError) as ctx:
            optimizer.step()

        err_msg = str(ctx.exception)
        self.assertIn("Inconsistent per-sample gradient batch dimensions", err_msg)
        self.assertIn("transformer.wpe.weight", err_msg)
        self.assertIn("batch dimension 1, expected 4", err_msg)

        # Clear grads for second test
        optimizer.zero_grad()

        # 2. With expanded position_ids: Must succeed!
        position_ids = torch.arange(T).unsqueeze(0).expand(B, T)
        outputs = model(
            input_ids=input_ids, position_ids=position_ids, labels=input_ids
        )
        outputs.loss.backward()
        optimizer.step()
        self.assertEqual(model._module.transformer.wpe.weight.grad_sample.shape[0], B)


if __name__ == "__main__":
    unittest.main()
