#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import Callable, Type

from torch import nn


def requires_grad(module: nn.Module, recurse: bool = False):
    return all((p.requires_grad for p in module.parameters(recurse)))


def _replace_child(
    root: nn.Module, child_name: str, converter: Callable[[nn.Module], nn.Module]
) -> None:
    """ A helper function, given the root module (e.g. x) and
    string representing the name of the submodule
    (e.g. y.z) converts the sub-module to a new module
    with the given converter.
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
    Given a module `root`, and a `target_class` of type nn.Module,
    all the submodules (of root) that have the same
    type as target_class will be converted given the converter. This will
    be useful for replacing modules that are not supported by the Privacy
    Engine.

    Args:
        root: The model or a module instance with potentially sub-modules
        target_class: The target class of type nn.module that needs
                      to be replaced.
        converter: Is a function or a lambda that converts an instance
                   of a given target_class to another nn.Module.

    Returns:
        The module with all the `target_class` types replaced using
        the `converter`. `root` is modified and is equal to the return value.

    Examples:
            from torchvision.models import resnet18
            from torch import nn

            model = resnet18()
            print(model.layer1[0].bn1)
            # prints BatchNorm2d(64, eps=1e-05, ...
            model = replace_all_modules(model, nn.BatchNorm2d, lambda _: nn.Identity())
            print(model.layer1[0].bn1)
            # prints Identity()
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
    Converts a BatchNorm `module` to the corresponding InstanceNorm module
    """

    def matchDim():
        if isinstance(module, nn.BatchNorm1d):
            return nn.InstanceNorm1d
        elif isinstance(module, nn.BatchNorm2d):
            return nn.InstanceNorm2d
        elif isinstance(module, nn.BatchNorm3d):
            return nn.InstanceNorm3d

    return matchDim()(module.num_features)


class DPGroupNorm(nn.GroupNorm):
    """
    An extension of the `nn.GroupNorm` which is DP compatible.
    We do not support internal affine parameters of a
    `nn.GroupNorm`. Therefore we need to MonkeyPatch groupnorm
    as a normal non-affine groupnorm followed by a depth-wise
    Conv1x1. This class does this. It is exactly equivalent to
    a normal `nn.GroupNorm` with `affine=True`.
    """
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        super().__init__(num_groups, num_channels, eps, affine=False)
        self.affine = None
        if affine:
            self.affine = nn.Conv2d(num_channels, num_channels,
                                    1, groups=num_channels)
            self.affine.weight.data.fill_(1.0)
            self.affine.bias.data.fill_(0.0)

    def forward(self, x):
        x = super().forward(x)
        return self.affine(x) if self.affine else x

    def __repr__(self):
        return f'DPGroupNorm({self.num_groups}, {self.num_channels}, ' +\
               f'eps={self.eps}, affine={self.affine is not None})'


def _batchnorm_to_groupnorm(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
    """
    Converts a BatchNorm `module` to GroupNorm module.
    This is a helper function.

    Note:
        A default value of 32 is chosen for the number of groups based on the
        paper *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*
        https://arxiv.org/pdf/1706.02677.pdf
    """
    return DPGroupNorm(
        min(32, module.num_features), module.num_features, affine=True)


def nullify_batchnorm_modules(root: nn.Module, target_class):
    """
    Replaces all the submodules of type nn.BatchNormXd in `root` with
    nn.Identity.

    Note:
        Most of the times replacing a `BatchNorm` module with `Identity`
        will heavily affect convergance of the model.
    """
    return replace_all_modules(
        root, nn.modules.batchnorm._BatchNorm, lambda _: nn.Identity()
    )


def convert_batchnorm_modules(
    model: nn.Module,
    converter: Callable[
        [nn.modules.batchnorm._BatchNorm], nn.Module
    ] = _batchnorm_to_groupnorm,
):
    """
    Converts all BatchNorm modules to another module
    (defaults to GroupNorm) that is privacy compliant.

    Args:
        model: The model or a module instance with potentially sub-modules
        converter: Is a function or a lambda that converts an instance
                   of a Batchnorm to another nn.Module. Defaults to
                   `_batchnorm_to_groupnorm`.

    Returns:
        The model with all the BatchNorm types replaced by a GroupNorm operation.

    Examples:
        from torchvision.models import resnet18
        from torch import nn

        model = resnet18()
        print(model.layer1[0].bn1)
        # prints BatchNorm2d module details
        model = convert_batchnorm_moduleta(model)
        print(model.layer1[0].bn1)
        # prints GroupNorm module details
    """
    return replace_all_modules(model, nn.modules.batchnorm._BatchNorm, converter)


def has_no_param(module: nn.Module):
    """
    Returns True if module is a leaf module.
    A leaf module is a one that has no children or has its own
    parameters.
    """
    has_param = False
    for _ in module.parameters(recurse=False):
        has_param = True
        break
    return (not has_param)


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

    def __init__(self, name: str,
                 predicate: Callable[[nn.Module], bool],
                 check_leaf_nodes_only: bool = True,
                 message: str = None):
        self.name = name
        if check_leaf_nodes_only:
            self.predicate = lambda x: has_no_param(x) or predicate(x)
        else:
            self.predicate = predicate
        self.message = message
        self.violators = []

    def validate(self, model: nn.Module) -> bool:
        valid = True
        for name, module in model.named_modules(prefix='Main'):
            if not self.predicate(module):
                valid = False
                self.violators.append(name)
        return valid
