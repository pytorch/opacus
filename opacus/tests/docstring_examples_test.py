#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine, privacy_analysis
from opacus.dp_model_inspector import DPModelInspector, IncompatibleModuleException
from opacus.layers.dp_multihead_attention import SequenceBias
from opacus.utils import stats
from opacus.utils.module_inspection import ModelInspector
from opacus.utils.module_modification import (
    convert_batchnorm_modules,
    replace_all_modules,
)
from opacus.utils.tensor_utils import (
    calc_sample_norms,
    sum_over_all_but_batch_and_last_n,
)


class DocstringExamplesTest(unittest.TestCase):
    """
    This test checks the correctness of the code snippets we use across the docstrings in the project.

    We want to make sure code examples are always up-to-date and the quality of the documentation doesn't degrade over time.
    This TestCase is a collection of all the examples we use at the moment.
    It is intended to catch breaking changes and signal to update the docstring alongside with the code.
    """

    def setUp(self):
        self.validator = DPModelInspector()

    def test_dp_model_inspector_example(self):
        # IMPORTANT: When changing this code you also need to update
        # the docstring for opacus.dp_model_inspector.DPModelInspector.validate()

        inspector = DPModelInspector()
        valid_model = nn.Linear(16, 32)
        is_valid = inspector.validate(valid_model)
        self.assertTrue(is_valid)

        invalid_model = nn.BatchNorm1d(2)
        with self.assertRaises(IncompatibleModuleException):
            is_valid = inspector.validate(invalid_model)

    def test_privacy_analysis_example(self):
        # IMPORTANT: When changing this code you also need to update
        # the docstring for opacus.privacy_analysis module
        parameters = [(1e-5, 1.0, 10), (1e-4, 3.0, 4)]
        delta = 1e-5

        max_order = 32
        orders = range(2, max_order + 1)
        rdp = np.zeros_like(orders, dtype=float)
        for q, sigma, steps in parameters:
            rdp += privacy_analysis.compute_rdp(q, sigma, steps, orders)

        epsilon, opt_order = privacy_analysis.get_privacy_spent(orders, rdp, delta)

    def test_privacy_engine_class_example(self):
        # IMPORTANT: When changing this code you also need to update
        # the docstring for opacus.privacy_engine.PrivacyEngine
        batch_size = 8
        sample_size = 64
        sample_rate = batch_size / sample_size

        model = torch.nn.Linear(16, 32)  # An example model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=sample_rate,
            noise_multiplier=1.3,
            max_grad_norm=1.0,
        )
        privacy_engine.attach(optimizer)  # That's it! Now it's business as usual.

    def test_privacy_engine_to_example(self):
        # IMPORTANT: When changing this code you also need to update
        # the docstring for opacus.privacy_engine.PrivacyEngine.to()
        batch_size = 8
        sample_size = 64
        sample_rate = batch_size / sample_size

        model = torch.nn.Linear(16, 32)  # An example model. Default device is CPU
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=sample_rate,
            noise_multiplier=0.8,
            max_grad_norm=0.5,
        )
        device = "cpu"
        model.to(
            device
        )  # If we move the model to GPU, we should call the to() method of the privacy engine (next line)
        privacy_engine.to(device)

    def test_privacy_engine_virtual_step_example(self):
        # IMPORTANT: When changing this code you also need to update
        # the docstring for opacus.privacy_engine.PrivacyEngine.virtual_step()
        model = nn.Linear(16, 2)
        dataloader = []
        batch_size = 64
        sample_size = 256
        sample_rate = batch_size / sample_size

        for _ in range(64):
            data = torch.randn(4, 16)
            labels = torch.randint(0, 2, (4,))
            dataloader.append((data, labels))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

        privacy_engine = PrivacyEngine(
            model,
            sample_rate=sample_rate,
            noise_multiplier=0.8,
            max_grad_norm=0.5,
        )
        privacy_engine.attach(optimizer)

        for i, (X, y) in enumerate(dataloader):
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            if i % 16 == 15:
                optimizer.step()  # this will call privacy engine's step()
                optimizer.zero_grad()
            else:
                optimizer.virtual_step()  # this will call privacy engine's virtual_step()

    def test_sequence_bias_example(self):
        # IMPORTANT: When changing this code you also need to update
        # the docstring for opacus.layers.dp_multihead_attention.SequenceBias
        m = SequenceBias(16)
        input = torch.randn(20, 4, 16)
        output = m(input)
        self.assertEqual(output.size(), (21, 4, 16))

    def test_module_inspection_example(self):
        # IMPORTANT: When changing this code you also need to update
        # the docstring for opacus.utils.module_inspection.ModelInspector
        inspector = ModelInspector("simple", lambda x: isinstance(x, nn.Conv2d))
        self.assertTrue(inspector.validate(nn.Conv2d(1, 1, 1)))

    def test_module_modification_replace_example(self):
        # IMPORTANT: When changing this code you also need to update
        # the docstring for opacus.utils.module_modification.replace_all_modules()
        from torchvision.models import resnet18

        model = resnet18()
        self.assertTrue(isinstance(model.layer1[0].bn1, nn.BatchNorm2d))

        model = replace_all_modules(model, nn.BatchNorm2d, lambda _: nn.Identity())
        self.assertTrue(isinstance(model.layer1[0].bn1, nn.Identity))

    def test_module_modification_convert_example(self):
        # IMPORTANT: When changing this code you also need to update
        # the docstring for opacus.utils.module_modification.convert_batchnorm_modules()
        from torchvision.models import resnet50

        model = resnet50()
        self.assertTrue(isinstance(model.layer1[0].bn1, nn.BatchNorm2d))

        model = convert_batchnorm_modules(model)
        self.assertTrue(isinstance(model.layer1[0].bn1, nn.GroupNorm))

    def test_tensor_utils_examples(self):
        # IMPORTANT: When changing this code you also need to update
        # the docstrings for opacus.utils.tensor_utils

        t1 = torch.rand((2, 5))
        t2 = torch.rand((2, 5))

        self.assertTrue(
            calc_sample_norms([("1", t1), ("2", t2)])[0].shape, torch.Size([1, 2])
        )

        tensor = torch.ones(1, 2, 3, 4, 5)
        self.assertTrue(
            sum_over_all_but_batch_and_last_n(tensor, n_dims=2).shape,
            torch.Size([1, 4, 5]),
        )

    def test_stats_example(self):
        # IMPORTANT: When changing this code you also need to update
        # the docstrings for opacus.utils.stats.Stat
        class MockSummaryWriter:
            def __init__(self):
                self.logs = defaultdict(dict)

            def add_scalar(self, name, value, iter):
                self.logs[name][iter] = value

        mock_summary_writer = MockSummaryWriter()
        stats.set_global_summary_writer(mock_summary_writer)

        stat = stats.Stat(stats.StatType.GRAD, "sample_stats", frequency=0.1)
        for i in range(21):
            stat.log({"val": i})

        self.assertEqual(len(mock_summary_writer.logs["GRAD:sample_stats/val"]), 2)

        stats.add(stats.Stat(stats.StatType.TEST, "accuracy", frequency=1.0))
        stats.update(stats.StatType.TEST, acc1=1.0)
