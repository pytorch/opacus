import pytest

import torch
from torch import nn


class MyCustomModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear(5, 3)

    def forward(self, x):
        x = self.f(x)


@pytest.fixture(autouse=True)
def create_namespace(doctest_namespace):
    """
    Initialize namespace for doctest.
    Everything added to `doctest_namespace` will be available in the doctest.
    """
    import opacus  # noqa
    import torch  # noqa
    from torch import nn  # noqa
    from typing import Any, List, Tuple, Dict, Set, Union  # noqa

    doctest_namespace.update(**locals())
    doctest_namespace["MyCustomModule"] = MyCustomModule