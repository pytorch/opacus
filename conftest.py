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

import pytest
import torch
from opacus import PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MyCustomModel(nn.Module):
    """Demo module to use in doctests"""

    def __init__(self):
        super().__init__()
        self.f = nn.Linear(5, 2)

    def forward(self, x):
        x = self.f(x)
        return x


def create_demo_dataloader():
    dataset = TensorDataset(torch.randn(64, 5), torch.randint(0, 2, (64,)))
    dataloader = DataLoader(dataset, batch_size=8)
    return dataloader


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
    doctest_namespace["TensorDataset"] = TensorDataset
    doctest_namespace["demo_dataloader"] = create_demo_dataloader()
    doctest_namespace["_init_private_training"] = _init_private_training
