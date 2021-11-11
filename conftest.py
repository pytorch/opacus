from collections import defaultdict

import pytest
import torch
from opacus.utils import stats
from torch import nn


class MyCustomModel(nn.Module):
    """Demo module to use in doctests"""

    def __init__(self):
        super().__init__()
        self.f = nn.Linear(5, 3)

    def forward(self, x):
        x = self.f(x)


def create_demo_dataloader():
    dataloader = []
    for _ in range(64):
        data = torch.randn(4, 16)
        labels = torch.randint(0, 2, (4,))
        dataloader.append((data, labels))
    return dataloader


class MockSummaryWriter:
    def __init__(self):
        self.logs = defaultdict(dict)

    def add_scalar(self, name, value, iter):
        self.logs[name][iter] = value


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
