#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from abc import ABC
from enum import IntEnum
from itertools import cycle
from typing import List

import torch


try:
    from skimage.filters import threshold_otsu as otsu
except ImportError:

    def otsu(*args, **kwargs):
        raise NotImplementedError("Install skimage!")


def _mean_plus_r_var(data: torch.Tensor, ratio: float = 0, **kwargs):
    """
    Function caclulates mean + ratio * stdv.
    and returns the largest of this value and the smallest element
    in the list (can happen when ratio is negative).
    """
    return max(
        data.min().item(),
        data.mean().item() + ratio * data.std().item() + 1e-8
    )


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
    ClippingMethod.MEAN: _mean_plus_r_var,
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


class _Dynamic_Clipper_(NormClipper):
    """
    This is a generic clipper, that is in an experimental phase.
    The clipper uses different stats to find a clipping threshold
    based on the given per sample norms.
        Note:
            This clipper is not private [use only for experimentation]
    """
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
        return clipping_factor if self.is_per_layer else cycle(clipping_factor)

    @property
    def thresholds(self):
        return torch.tensor(self.thresh)

    @property
    def is_per_layer(self):
        return self.clip_per_layer
