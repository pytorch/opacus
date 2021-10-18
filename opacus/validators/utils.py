#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Sequence, Union

import torch
import torch.nn as nn

from .module_validator import ModuleValidator


def register_module_validator(target_class_or_classes: Union[type, Sequence[type]]):
    """
    Registers the decorated function as the ``validator`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to validate that a module is compatible
    for training with Opacus.
    The signature of every validator is always the same:

    >>> @register_module_validator(nn.MyCustomClass)
    >>> def validate(module: nn.Module, **kwargs) -> List[opacus.validators.errors.UnsupportedError]:
    >>>    pass

    It may help you to take a look at the existing validator inside Opacus, under ``opacus.validators.``
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            ModuleValidator.VALIDATORS[target_class] = f
        return f

    return decorator


def register_module_fixer(target_class_or_classes: Union[type, Sequence[type]]):
    """
    Registers the decorated function as the ``fixer`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to fix an incompatoble module to make
    it work for training with Opacus.
    The signature of every fixer is always the same:

    >>> @register_module_fixer(nn.MyCustomClass)
    >>> def fix(module: nn.Module, **kwargs) -> nn.Module:
    >>>    pass

    It may help you to take a look at the existing fixers inside Opacus, under ``opacus.validators.``
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            ModuleValidator.FIXERS[target_class] = f
        return f

    return decorator
