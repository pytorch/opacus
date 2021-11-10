import pytest
import torch


@pytest.fixture(autouse=True)
def create_namespace(doctest_namespace):
    """
    Initialize namespace for doctest.
    Everything added to `doctest_namespace` will be available in the doctest.
    """
    import opacus  # noqa
    import torch  # noqa
    from torch import nn  # noqa
    from typing import Any, List, Tuple, Dict, Set  # noqa

    doctest_namespace.update(**locals())