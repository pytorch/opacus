from collections import defaultdict

import pytest
import torch
from opacus.utils import stats
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine

class MyCustomModel(nn.Module):
    """Demo module to use in doctests"""

    def __init__(self):
        super().__init__()
        self.f = nn.Linear(5, 2)

    def forward(self, x):
        x = self.f(x)


def create_demo_dataloader():
    dataset = TensorDataset(
        torch.randn(64, 5),
        torch.randint(0, 2, (64,))
    )
    dataloader = DataLoader(dataset, batch_size=4)
    return dataloader


class MockSummaryWriter:
    def __init__(self):
        self.logs = defaultdict(dict)

    def add_scalar(self, name, value, iter):
        self.logs[name][iter] = value


def _init_private_training():
    model = MyCustomModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    data_loader = create_demo_dataloader()
    privacy_engine = PrivacyEngine()

    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    return model, optimizer, data_loader


mock_summary_writer = MockSummaryWriter()
stats.set_global_summary_writer(mock_summary_writer)


@pytest.fixture(autouse=True)
def create_namespace(doctest_namespace):
    """
    Initialize namespace for doctest.
    Everything added to `doctest_namespace` will be available in the doctest.
    """
    from typing import Any, Dict, List, Set, Tuple, Union  # noqa

    import numpy as np  # noqa
    import opacus  # noqa
    import torch  # noqa
    from torch import nn  # noqa

    # Adding all imports in the doctest namespace
    doctest_namespace.update(**locals())

    doctest_namespace["MyCustomModel"] = MyCustomModel
    doctest_namespace["demo_dataloader"] = create_demo_dataloader()
    doctest_namespace["mock_summary_writer"] = mock_summary_writer
