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

from typing import Dict, Type

from .accountant import IAccountant
from .gdp import GaussianAccountant
from .prv import PRVAccountant
from .rdp import RDPAccountant


_ACCOUNTANTS: Dict[str, Type[IAccountant]] = {
    "rdp": RDPAccountant,
    "gdp": GaussianAccountant,
    "prv": PRVAccountant,
}


def register_accountant(
    mechanism: str, accountant: Type[IAccountant], force: bool = False
):
    r"""
    Register a new accountant class to be used with a specified mechanism name.

    Args:
        mechanism: Name of the mechanism to register the accountant for
        accountant: Accountant class (subclass of IAccountant) to register
        force: If True, overwrites existing accountant for the specified mechanism.

    Raises:
        ValueError: If the mechanism is already registered.
    """
    if mechanism in _ACCOUNTANTS and not force:
        raise ValueError(f"Accountant for mechanism {mechanism} is already registered")

    _ACCOUNTANTS[mechanism] = accountant


def create_accountant(mechanism: str) -> IAccountant:
    r"""
    Creates and returns an accountant instance for the specified privacy mechanism.

    Args:
        mechanism: Name of the privacy accounting mechanism to use.

    Returns:
        An instance of the appropriate accountant class (subclass of IAccountant)
        for the specified mechanism.

    Raises:
        ValueError: If the specified mechanism is not registered.
    """
    if mechanism in _ACCOUNTANTS:
        return _ACCOUNTANTS[mechanism]()

    raise ValueError(f"Unexpected accounting mechanism: {mechanism}")
