import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MyCustomModel(nn.Module):
    """Demo module to use in doctests"""

    def __init__(self):
        super().__init__()
        self.f = nn.Linear(5, 2)

    def forward(self, x):
        x = self.f(x)


def create_demo_dataloader():
    dataset = TensorDataset(torch.randn(64, 5), torch.randint(0, 2, (64,)))
    dataloader = DataLoader(dataset, batch_size=4)
    return dataloader


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
