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

from typing import Sequence, Union

from .module_validator import ModuleValidator


DEFAULT_MODULE_VALIDATOR = ModuleValidator


def register_module_validator(
    target_class_or_classes: Union[type, Sequence[type]],
    validator_class: type = DEFAULT_MODULE_VALIDATOR,
):
    """
    Registers the decorated function as the ``validator`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to validate that a module is compatible
    for training with Opacus.
    You may supply your own validator_class that holds the registry of VALIDATORS.
    The signature of every validator is always the same:

    >>> @register_module_validator(MyCustomModel)
    ... def validate(module: nn.Module, **kwargs) -> List[opacus.validators.errors.UnsupportedError]:
    ...    pass

    It may help you to take a look at the existing validator inside Opacus, under ``opacus.validators.``
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            validator_class.VALIDATORS[target_class] = f
        return f

    return decorator


def register_module_fixer(
    target_class_or_classes: Union[type, Sequence[type]],
    validator_class: type = DEFAULT_MODULE_VALIDATOR,
):
    """
    Registers the decorated function as the ``fixer`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to fix an incompatoble module to make
    it work for training with Opacus.
    You may supply your own validator_class that holds the registry of FIXERS.
    The signature of every fixer is always the same:

    >>> @register_module_fixer(MyCustomModel)
    ... def fix(module: nn.Module, **kwargs) -> nn.Module:
    ...    pass

    It may help you to take a look at the existing fixers inside Opacus, under ``opacus.validators.``
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            validator_class.FIXERS[target_class] = f
        return f

    return decorator
