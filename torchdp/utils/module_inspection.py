#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
    utils for inspecting model layers, replacing layers, ...
"""
from typing import Callable

from torch import nn


class ModelInspector:
    """
    Class to inspect models for a specific predicate. If a module has
    children the predicate is checked on all children recursively.

    Args:
        name: A string to represent the predicate.
        predicate: A callable boolean function which tests a hypothesis
        on a module.
        check_leaf_nodes_only: Flag to check only leaf nodes of a module.
        Here leaf nodes are the ones that have parameters of their own.
        message: optional value to hold a message about violating this
        predicate.

    Attributes:
        name: A string to represent the predicate.
        predicate: A callable boolean function which tests a hypothesis
        on the module.
        message: optional value to record a message about violating this
        predicate.
        violators: list of module names that have violated the predicate. The list
        does not get automatically emptied if the predicate is applied on multiple
        modules.

    Note:
        The predicates will not be applied on non-leaf modules unless
        `check_leaf_nodes_only` is set to False. E.g. A predicate like:
        `lambda model: isinstance(model, nn.Sequential)` will always return
        `True` unless `check_leaf_nodes_only` is set.

    Examples:

        inspector = ModelInspector('simple', lambda x: issubclass(x, Conv2d))
        print(inspector(nn.Conv2d(1, 1, 1)))  # prints True
    """

    def __init__(
        self,
        name: str,
        predicate: Callable[[nn.Module], bool],
        check_leaf_nodes_only: bool = True,
        message: str = None,
    ):
        self.name = name
        if check_leaf_nodes_only:
            self.predicate = lambda x: has_no_param(x) or predicate(x)
        else:
            self.predicate = predicate
        self.message = message
        self.violators = []

    def validate(self, model: nn.Module) -> bool:
        valid = True
        for name, module in model.named_modules(prefix="Main"):
            if not self.predicate(module):
                valid = False
                self.violators.append(name)
        return valid


def has_no_param(module: nn.Module):
    """
    Returns True if module is a leaf module.
    A leaf module is a one that has no children or has its own
    parameters.
    """
    has_params = any(p is not None for p in module.parameters(recurse=False))
    return not has_params


def requires_grad(module: nn.Module, recurse: bool = False):
    requires_grad = all(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


def get_layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__
