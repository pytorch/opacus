#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from abc import ABC
from enum import IntEnum
from itertools import cycle
from typing import Callable, Dict, Iterator, List, Tuple, Type

import torch
from torch import nn


try:
    from skimage.filters import threshold_otsu as otsu
except ImportError:

    def otsu(*args, **kwargs):
        raise NotImplementedError("Install skimage!")


#####################################################################
## utils for inspecting model layers, replacing layers, ...
#####################################################################


def requires_grad(module: nn.Module, recurse: bool = False):
    is_req_grad = [p.requires_grad for p in module.parameters(recurse)]
    return len(is_req_grad) > 0 and all(is_req_grad)


def get_layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


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


def _batchnorm_to_groupnorm(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
    """
    Converts a BatchNorm `module` to GroupNorm module.
    This is a helper function.

    Note:
        A default value of 32 is chosen for the number of groups based on the
        paper *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*
        https://arxiv.org/pdf/1706.02677.pdf
    """
    return nn.GroupNorm(min(32, module.num_features), module.num_features, affine=True)


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
    return not has_param


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


#####################################################################
## utils for generating stats from torch tensors.
#####################################################################
def calc_sample_norms(
    named_params: Iterator[Tuple[str, torch.Tensor]], flat: bool = True
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Calculates the (overall) norm of the given tensors over each sample,
    assuming dim=0 is represnting the sample in the batch.

    Returns:
        A tuple with first element being a list of torch tensors all of size
        B (look at `named_params`). Each element in the list corresponds to
        the norms of the parameter appearing in the same order of the
        `named_params`.

    Arguments:
        named_params: An iterator of tuples each representing a named tensor with
            name being a string and param a tensor of shape [B, XYZ...] where B
            is the size of the batch and is the 0th dimension
        flat: a flag, when set to `True` returns a flat norm over all
            layers, i.e. norm of all the norms across layers for each sample.
        stats_required: a flag, when set to True, the function will provide some
            statistics over the batch, including mean, median, and max values
    """
    norms = [param.view(len(param), -1).norm(2, dim=-1) for name, param in named_params]
    # calc norm over all layer norms if flat = True
    if flat:
        norms = [torch.stack(norms, dim=0).norm(2, dim=0)]
    return norms


def sum_over_all_but_batch_and_last_n(
    tensor: torch.Tensor, n_dims: int
) -> torch.Tensor:
    """
    Returns the sum of the input tensor over all dimensions except
    the first (batch) and last n_dims.

    Args:
        tensor: input tensor of shape (B, * , X[0], X[1], ..., X[n_dims-1])
        n_dims: Number of input tensor dimensions to keep

    Returns:
        New tensor of shape (B, X[0], X[1], ..., X[n_dims-1]).
        Will return the unchanged input tensor if `tensor.dim() == n_dims + 1`

    Examples:
        import torch

        A = torch.ones(2,3,4)
        print(sum_over_all_but_batch_and_last_n(A, 1))
        # prints torch.Size([2, 4])
        print(sum_over_all_but_batch_and_last_n(A, 2))
        # prints torch.Size([2, 3, 4])
    """
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)


def _mean_var(data: torch.Tensor, ratio: float = 0, **kwargs):
    """
    Finds mean(x) + ratio * std(x)
    """
    return max(data.min().item(), data.mean().item() + ratio * data.std().item() + 1e-8)


def _pvalue(data: torch.Tensor, ratio: float = 0.25, **kwargs):
    """
    Finds the P-(ratio* 100)'s value in the tensor, equivalent
    to the kth largest element where k = ratio * len(data)
    """
    cut = max(1, int(data.numel() * (1 - ratio)))
    return torch.kthvalue(data, cut)[0].item()


def _static(data: torch.Tensor, current_thresh, **kwargs):
    """
    Simple path through
    """
    return current_thresh


def _otsu(data: torch.Tensor, **kwargs):
    """
    Use Otsu's method, which assumes a GMM with 2 components
    but uses some heuristic to maximize the variance differences.
    """
    h = 2 ** int(1 + math.log2(data.shape[0]) / 2)
    fake_img = data.view(h, -1).cpu().numpy()
    return otsu(fake_img, h)


class ClippingMethod(IntEnum):
    STATIC = 0
    PVALUE = 1
    MEAN = 2
    GMM = 3
    OTSU = 4


_thresh_ = {
    ClippingMethod.STATIC: _static,
    ClippingMethod.PVALUE: _pvalue,
    ClippingMethod.MEAN: _mean_var,
    ClippingMethod.OTSU: _otsu,
}


def _calculate_thresh_value(
    data: torch.Tensor,
    current_thresh: float,
    clipping_mehod: ClippingMethod = ClippingMethod.STATIC,
    ratio: float = -1,
) -> float:
    """
    Calculates the clipping threshold by looking at the layer norms
    of each example. Three methods are supported: static threshold,
    threshold calculated based on mean and variance of the norms, and
    threshold calculated based on percentile values of the norms.

    Arguments:
        data: 1-D tensor
        current_thresh: value of the current threshold
        clipping_method: enum value defining the clipping strategy
                         current options are STATIC, PVALUE, MEAN, and OTSU
        ratio: has different meaning for differnet strategies, it is the p-value
        for PVALUE, and a multiplier for standard deviation for MEAN.

    """
    return _thresh_[clipping_mehod](data, ratio=ratio, current_thresh=current_thresh)


class NormClipper(ABC):
    """
    Abstract class to calculate the clipping factor
    """

    def calc_clipping_factors(self, named_norms):
        """
        Calculates the clipping factor based on the given
        parameters
        """
        pass

    @property
    def thresholds(self):
        pass

    @property
    def is_per_layer(self):
        pass


class ConstantFlatClipper(NormClipper):
    def __init__(self, flat_value: float):
        self.flat_value = float(flat_value)

    def calc_clipping_factors(self, norms):
        # Expects a list of size one.
        if len(norms) != 1:
            raise ValueError(
                "Waring: flat norm selected but "
                f"received norm for {len(norms)} layers"
            )
        per_sample_clip_factor = self.flat_value / (norms[0] + 1e-6)
        # We are *clipping* the gradient, so if the factor is ever >1 we set it to 1
        per_sample_clip_factor = per_sample_clip_factor.clamp(max=1.0)
        # return this clipping factor for all layers
        return cycle([per_sample_clip_factor])

    @property
    def thresholds(self):
        return torch.tensor([self.flat_value])

    @property
    def is_per_layer(self):
        return False


class ConstantPerLayerClipper(NormClipper):
    def __init__(self, flat_values: List[float]):
        self.flat_values = [float(fv) for fv in flat_values]

    def calc_clipping_factors(self, norms):
        if len(norms) != len(self.flat_values) and len(self.flat_values) != 1:
            raise ValueError(
                f"{len(norms)} layers have provided norms but the "
                f"number of clipping thresholds is {len(self.flat_values)}"
            )

        self.flat_values = self.flat_values * (
            len(norms) if len(self.flat_values) == 1 else 1
        )

        clipping_factor = []
        for norm, threshold in zip(norms, self.flat_values):
            per_sample_clip_factor = threshold / (norm + 1e-6)
            clipping_factor.append(per_sample_clip_factor.clamp(max=1.0))
        return clipping_factor

    @property
    def thresholds(self):
        return torch.tensor(self.flat_values)

    @property
    def is_per_layer(self):
        return True


class _Experimental_Clipper_(NormClipper):
    def __init__(
        self,
        flat_values: List[float],
        clip_per_layer: bool = False,
        clipping_method: ClippingMethod = ClippingMethod.STATIC,
        ratio: float = 0.0,
    ):
        self.flat_values = [float(float_value) for float_value in flat_values]
        self.clip_per_layer = clip_per_layer
        if clipping_method != ClippingMethod.STATIC:
            print(
                "Warning! Current implementations of dynamic clipping "
                "are not privacy safe; Caclulated privacy loss is not "
                "indicative of a proper bound."
            )
        self.clipping_method = clipping_method
        self.ratio = ratio
        self.thresh = [0.0]

    def calc_clipping_factors(self, norms):
        if len(self.flat_values) == 1:
            current_threshs = self.flat_values * (
                len(norms) if self.clip_per_layer else 1
            )
        clipping_factor = []
        self.thresh = []

        if len(norms) != len(current_threshs):
            raise ValueError(
                f"Provided grad norm max's size {len(self.current_max)}"
                f" does not match the number of layers {len(norms)}"
            )

        for norm, current_thresh in zip(norms, current_threshs):
            thresh = _calculate_thresh_value(
                norm, current_thresh, self.clipping_method, self.ratio
            )
            self.thresh.append(thresh)
            per_sample_clip_factor = thresh / (norm + 1e-6)
            clipping_factor.append(per_sample_clip_factor.clamp(max=1.0))
        return clipping_factor

    @property
    def thresholds(self):
        return torch.tensor(self.thresh)

    @property
    def is_per_layer(self):
        return self.clip_per_layer
