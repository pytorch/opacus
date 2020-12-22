#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
r"""
This module includes utils for inspecting model layers using specified
predicates to check for conditions, getting layer type etc.
"""
from typing import Callable, Optional

from torch import nn


class ModelInspector:
    """
    An inspector of models given a specific predicate. If a module
    has children the predicate is checked on all children recursively.

    Example:
        >>>  inspector = ModelInspector('simple', lambda x: isinstance(x, Conv2d))
        >>>  print(inspector.validate(nn.Conv2d(1, 1, 1)))
        True
    """

    def __init__(
        self,
        name: str,
        predicate: Callable[[nn.Module], bool],
        check_leaf_nodes_only: bool = True,
        message: Optional[str] = None,
    ):
        """
        Args:
            name: String to represent the predicate.
            predicate: Callable boolean function which tests a hypothesis on a module.
            check_leaf_nodes_only: Flag to check only leaf nodes of a module. Here
                leaf nodes are the ones that have parameters of their own.
            message: Optional value to hold a message about violating this predicate.

        Notes:
            The predicates will not be applied on non-leaf modules unless
            ``check_leaf_nodes_only`` is set to False. E.g. A predicate like:

            ``lambda model: isinstance(model, nn.Sequential)``

            will always return True unless ``check_leaf_nodes_only`` is set.
        """
        self.name = name
        if check_leaf_nodes_only:
            self.predicate = (
                lambda x: has_no_param(x) or not requires_grad(x) or predicate(x)
            )
        else:
            self.predicate = predicate
        self.message = message
        self.violators = []
        # List that contains the module names that have violated the
        # predicate. The list does not get automatically emptied if
        # the predicate is applied on multiple modules.

    def validate(self, model: nn.Module) -> bool:
        """
        Checks if the provided module satisfies the predicate specified
        upon creation of the :class:`~opacus.utils.ModelInspector`.

        Args:
            model: PyTorch module on which the predicate must be evaluated
                and satisfied.

        Returns:
            Flag indicate if predicate is satisfied.
        """
        valid = True
        for name, module in model.named_modules(prefix="Main"):
            if not self.predicate(module):
                valid = False
                self.violators.append(name)
        return valid


def has_no_param(module: nn.Module) -> bool:
    """
    Checks if a module does not have any parameters.

    Args:
        module: The module on which this function is being evaluated.

    Returns:
        Flag indicating if the provided module does not have any
        parameters.
    """
    has_params = any(p is not None for p in module.parameters(recurse=False))
    return not has_params


def requires_grad(module: nn.Module, recurse: bool = False) -> bool:
    """
    Checks if any parameters in a specified module require gradients.

    Args:
        module: PyTorch module whose parameters are examined
        recurse: Flag specifying if the gradient requirement check should
            be applied recursively to sub-modules of the specified module

    Returns:
        Flag indicate if any parameters require gradients
    """
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


def get_layer_type(layer: nn.Module) -> str:
    """
    Returns the name of the type of the given layer.

    Args:
        layer: The module corresponding to the layer whose type
            is being queried.

    Returns:
        Name of the class of the layer
    """
    return layer.__class__.__name__
