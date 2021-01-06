#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
r"""
This module includes utils for modifying model layers, replacing layers etc.
"""
from typing import Callable, Type

from torch import nn


def _replace_child(
    root: nn.Module, child_name: str, converter: Callable[[nn.Module], nn.Module]
) -> None:
    """
    Converts a sub-module to a new module given a helper
    function, the root module and a string representing
    the name of the submodule to be replaced.

    Args:
        root: Root module whose sub module must be replaced.
        child_name: Name of submodule that must be replaced.
        converter: Function or a lambda that takes a module
            (the submodule to be replaced) and returns its
            replacement.
    """
    # find the immediate parent
    parent = root
    nameList = child_name.split(".")
    for name in nameList[:-1]:
        parent = parent._modules[name]
    # set to identity
    parent._modules[nameList[-1]] = converter(parent._modules[nameList[-1]])


def replace_all_modules(
    root: nn.Module,
    target_class: Type[nn.Module],
    converter: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Converts all the submodules (of root) that have the same
    type as target_class, given a converter, a module root,
    and a target class type.

    This method is useful for replacing modules that are not
    supported by the Privacy Engine.

    Args:
        root: Model instance, potentially with sub-modules
        target_class: Target class that needs to be replaced.
        converter: Function or a lambda that converts an instance
            of a given target_class to another nn.Module.

    Returns:
        Module with all the target_class types replaced using the
        converter. root is modified and is equal to the return value.

    Example:
        >>>  from torchvision.models import resnet18
        >>>  from torch import nn
        >>>  model = resnet18()
        >>>  print(model.layer1[0].bn1)
        BatchNorm2d(64, eps=1e-05, ...
        >>>  model = replace_all_modules(model, nn.BatchNorm2d, lambda _: nn.Identity())
        >>>  print(model.layer1[0].bn1)
        Identity()
    """
    # base case
    if isinstance(root, target_class):
        return converter(root)

    for name, obj in root.named_modules():
        if isinstance(obj, target_class):
            _replace_child(root, name, converter)
    return root


def _batchnorm_to_instancenorm(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
    """
    Converts a BatchNorm module to the corresponding InstanceNorm module

    Args:
        module: BatchNorm module to be replaced

    Returns:
        InstanceNorm module that can replace the BatchNorm module provided
    """

    def matchDim():
        if isinstance(module, nn.BatchNorm1d):
            return nn.InstanceNorm1d
        elif isinstance(module, nn.BatchNorm2d):
            return nn.InstanceNorm2d
        elif isinstance(module, nn.BatchNorm3d):
            return nn.InstanceNorm3d

    return matchDim()(module.num_features)


def _batchnorm_to_groupnorm(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
    """
    Converts a BatchNorm ``module`` to GroupNorm module.
    This is a helper function.

    Args:
        module: BatchNorm module to be replaced

    Returns:
        GroupNorm module that can replace the BatchNorm module provided

    Notes:
        A default value of 32 is chosen for the number of groups based on the
        paper *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*
        https://arxiv.org/pdf/1706.02677.pdf
    """
    return nn.GroupNorm(min(32, module.num_features), module.num_features, affine=True)


def nullify_batchnorm_modules(root: nn.Module) -> nn.Module:
    """
    Replaces all the BatchNorm submodules (e.g. :class:`torch.nn.BatchNorm1d`,
    :class:`torch.nn.BatchNorm2d` etc.) in ``root`` with :class:`torch.nn.Identity`.

    Args:
        root: Module for which to replace BatchNorm submodules.

    Returns:
        Module with all the BatchNorm sub modules replaced with
        Identity. ``root`` is modified and is equal to the return value.

    Notes:
        Most of the times replacing a BatchNorm module with Identity
        will heavily affect convergence of the model.
    """
    return replace_all_modules(
        root, nn.modules.batchnorm._BatchNorm, lambda _: nn.Identity()
    )


def convert_batchnorm_modules(
    model: nn.Module,
    converter: Callable[
        [nn.modules.batchnorm._BatchNorm], nn.Module
    ] = _batchnorm_to_groupnorm,
) -> nn.Module:
    """
    Converts all BatchNorm modules to another module
    (defaults to GroupNorm) that is privacy compliant.

    Args:
        model: Module instance, potentially with sub-modules
        converter: Function or a lambda that converts an instance of a
            Batchnorm to another nn.Module.

    Returns:
        Model with all the BatchNorm types replaced by another operation
        by using the provided converter, defaulting to GroupNorm if one
        isn't provided.

    Example:
        >>>  from torchvision.models import resnet50
        >>>  from torch import nn
        >>>  model = resnet50()
        >>>  print(model.layer1[0].bn1)
        BatchNorm2d module details
        >>>  model = convert_batchnorm_modules(model)
        >>>  print(model.layer1[0].bn1)
        GroupNorm module details
    """
    return replace_all_modules(model, nn.modules.batchnorm._BatchNorm, converter)
