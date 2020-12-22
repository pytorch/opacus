#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
from abc import ABC
from enum import IntEnum
from itertools import cycle
from typing import Iterator, List, Union

import torch


try:
    from skimage.filters import threshold_otsu as otsu
except ImportError:

    def otsu(*args, **kwargs) -> float:
        raise NotImplementedError("Install skimage!")


def _mean_plus_r_var(data: torch.Tensor, ratio: float = 0, **kwargs) -> float:
    """
    Caclulates mean + ratio x standard_deviation of the provided tensor
    and returns the larger of this value and the smallest element in
    the tensor (can happen when ratio is negative).

    Args:
        data: Pytorch tensor containing the data on which the mean and stdv.
            is evaluated.
        ratio: Value of the scaling factor in the value calculated by the
            function.

    Returns:
        The result of the function.

    """
    return max(data.min().item(), data.mean().item() + ratio * data.std().item() + 1e-8)


def _pvalue(data: torch.Tensor, ratio: float = 0.25, **kwargs) -> torch.Tensor:
    """
    Finds the pth largest value in the tensor, where p = ratio x len(data).

    Args:
        data: Pytorch tensor against which the function is evaluated.
        ratio: Value of the scaling factor in the value calculated by
            the function.

    Returns:
        Tensor of dimension ``(1,)`` with the result of the function.
    """
    cut = max(1, int(data.numel() * (1 - ratio)))
    return torch.kthvalue(data, cut)[0].item()


def _static(data: torch.Tensor, current_thresh: float, **kwargs) -> float:
    """
    Passes through the specified input ``current_threshold``.

    Args:
        data: Pytorch tensor containing the data.
        current_thresh: The threshold value.

    Returns:
        The threshold value.
    """
    return current_thresh


def _otsu(data: torch.Tensor, **kwargs) -> float:
    """
    Returns an intensity threshold for an image that separates it
    into backgorund and foreground pixels.

    The implementation uses Otsu's method, which assumes a GMM with
    2 components but uses some heuristic to maximize the variance
    differences. The input data is shaped into a 2D image for the
    purpose of evaluating the threshold value.

    Args:
        data: Pytorch tensor containing the data.

    Returns:
        Threshold value determined via Otsu's method.
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
    clipping_ratio: float = -1,
) -> float:
    """
    Calculates a clipping threshold by looking at the layer norms
    of each example.

    Four methods are supported: static threshold, threshold calculated
    based on mean and variance of the norms, and threshold calculated
    based on percentile values of the norms.

    Args:
        data: Pytorch tensor containing the data
        current_thresh: Value of the current threshold.
        clipping_method: Enum value defining the clipping strategy. Current
            options are STATIC, PVALUE, MEAN, and OTSU.
        clipping_ratio: Value that has different meaning for differnet strategies, it
            is the percentile parameter for PVALUE, and a multiplier for
            standard deviation for MEAN. It has no significance for OTSU and
            STATIC.

    Returns:
        Clipping threshold value
    """
    return _thresh_[clipping_mehod](
        data, ratio=clipping_ratio, current_thresh=current_thresh
    )


class NormClipper(ABC):
    """
    An abstract class to calculate the clipping factor
    """

    def calc_clipping_factors(
        self, norms: List[torch.Tensor]
    ) -> Union[List[torch.Tensor], Iterator[torch.Tensor]]:
        """
        Calculates the clipping factor(s) based on the given
        parameters. A concrete subclass must implement this.

        Returns:
            The clipping factors
        """
        pass

    @property
    def thresholds(self) -> torch.Tensor:
        """
        Depending on the type of clipper, returns threshold values.

        Returns:
            The threshold values
        """
        pass

    @property
    def is_per_layer(self) -> bool:
        """
        Depending on type of clipper, returns indicator as to whether
        different clipping is applied to each layer in the model.

        Returns:
            Flag indicator as to whether different clipping is applied
            to each layer in the model.
        """
        pass


class ConstantFlatClipper(NormClipper):
    """
    A clipper that clips all gradients in such a way that their norm is
    at most equal to a specified value. This value is shared for all
    layers in a model. Note that the process of clipping really involves
    multiplying all gradients by a scaling factor. If this scaling factor
    is > 1.0, it is instead capped at 1.0. The net effect is that the final
    norm of the scaled gradients will be less than the specified value in
    such a case. Thus it is better to think of the specified value as an
    upper bound on the norm of final clipped gradients.
    """

    def __init__(self, flat_value: float):
        """
        Args:
            flat_value: Constant value that is used to normalize gradients
                such that their norm equals this value before clipping.
                This threshold value is used for all layers.
        """
        self.flat_value = float(flat_value)

    def calc_clipping_factors(
        self, norms: List[torch.Tensor]
    ) -> Iterator[torch.Tensor]:
        """
        Calculates the clipping factor based on the given
        norm of gradients for all layers, so that the new
        norm of clipped gradients is at most equal to
        ``self.flat_value``.

        Args:
            norms: List containing a single tensor of dimension (1,)
                with the norm of all gradients.

        Returns:
            Tensor containing the single threshold value to be used
            for all layers.
        """
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
    def thresholds(self) -> torch.Tensor:
        """
        Returns singleton tensor of dimension (1,) containing
        the common threshold value used for clipping all
        layers in the model.

        Returns:
            Threshold values
        """
        return torch.tensor([self.flat_value])

    @property
    def is_per_layer(self) -> bool:
        """
        Returns indicator as to whether different clipping is applied
        to each layer in the model. For this clipper, it is False.

        Returns:
            Flag with value False
        """
        return False


class ConstantPerLayerClipper(NormClipper):
    """
    A clipper that clips all gradients in such a way that their norm is
    at most equal to a specified value. This value is specified for each
    layer in a model. Note that the process of clipping really involves
    multiplying all gradients by a scaling factor. If this scaling factor
    is > 1.0, it is instead capped at 1.0. The net effect is that the final
    norm of the scaled gradients will be less than the specified value in
    such a case. Thus it is better to think of the specified value as an
    upper bound on the norm of final clipped gradients.
    """

    def __init__(self, flat_values: List[float]):
        """
        Args:
            flat_values: List of values that is used to normalize gradients
                for each layer such that the norm equals the corresponding
                value before clipping.
        """
        self.flat_values = [float(fv) for fv in flat_values]

    def calc_clipping_factors(self, norms: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Calculates separate clipping factors for each layer based on
        its corresponding norm of gradients, such that its new norm is
        at most equal to the flat value specified for that layer when
        instantiating the object of
        :class:`~opacus.utils.clipping.ConstantPerLayerClipper`.

        Args:
            norms: List containing the desired norm of gradients for each layer.

        Returns:
            List of tensors, each containing a single value specifying the
            clipping factor per layer.
        """
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
    def thresholds(self) -> torch.Tensor:
        """
        Returns a tensor of values that are used to normalize gradients for
        each layer such that the norm at most equals the corresponding
        value before clipping.

        Returns:
            Tensor of thresholds
        """
        return torch.tensor(self.flat_values)

    @property
    def is_per_layer(self) -> bool:
        """
        Returns indicator as to whether different clipping is applied
        to each layer in the model. For this clipper, it is True.

        Returns:
            Flag with value True
        """
        return True


class _Dynamic_Clipper_(NormClipper):
    """
    This is a generic clipper, that is in an experimental phase.
    The clipper uses different stats to find a clipping threshold
    based on the given per sample norms.

    Notes:
        This clipper breaks DP guarantees [use only for experimentation]
    """

    def __init__(
        self,
        flat_values: List[float],
        clip_per_layer: bool = False,
        clipping_method: ClippingMethod = ClippingMethod.STATIC,
        clipping_ratio: float = 0.0,
        clipping_momentum: float = 0.9,
    ):
        """
        Args:
            flat_value: List of float values that is used to normalize gradients
                for each layer such that the norm equals the corresponding
                value before clipping.
            clip_per_layer: Flag indicating if a separate desired norm value is
                specified per layer or if a single value is shared for all.
            clipping_method: Value in the enum ClippingMethod that specifies one
                of the currently supported clipping types.
            clipping_ratio: Value that can be used to evaluate the clipping threshold
                for certain clipping types.
            clipping_momentum: value defines the decaing factor of an ubiased estimator
                 of exponential averaging of clipping thresholds, i.e. weight used to
                 combine the threshold from the current batch and the previous one.
        """
        self.flat_values = [float(float_value) for float_value in flat_values]
        self.clip_per_layer = clip_per_layer
        if clipping_method != ClippingMethod.STATIC:
            print(
                "Warning! Current implementations of dynamic clipping "
                "are not privacy safe; Caclulated privacy loss is not "
                "indicative of a proper bound."
            )
        self.clipping_method = clipping_method
        self.clipping_ratio = clipping_ratio
        self.clipping_momentum = clipping_momentum
        self.thresh = []

    def calc_clipping_factors(
        self, norms: List[torch.Tensor]
    ) -> Union[List[torch.Tensor], Iterator[torch.Tensor]]:
        """
        Calculates separate clipping factors for each layer based on
        stats such as a threshold determined by Otsu's method, combinations
        of mean and std. deviation, kth median value etc.

        This is experimental and does not guarantee privacy and is not recommended
        for production use.

        Args:
            norms: List containing the desired norm of gradients for each layer.

        Returns:
            Singleton list specifying a common clippng factor for all layers,
            or an iterator of tensors specifying a clipping factor per layer
        """

        if len(self.thresh) == 0:
            current_threshs = self.flat_values
            if len(self.flat_values) == 1 and self.clip_per_layer:
                # a constant clipping factor applied to all non-rozen layers
                # need to replicate it by the number of number of those layers
                # (= number of norms).
                current_threshs *= len(norms)
        else:
            current_threshs = self.thresh

        clipping_factor = []
        self.thresh = []

        if len(norms) != len(current_threshs):
            raise ValueError(
                f"Provided grad norm max's size {len(current_threshs)}"
                f" does not match the number of layers {len(norms)}"
            )

        for norm, current_thresh in zip(norms, current_threshs):
            thresh = _calculate_thresh_value(
                norm, current_thresh, self.clipping_method, self.clipping_ratio
            )
            thresh = float(
                (1 - self.clipping_momentum) * thresh
                + self.clipping_momentum * current_thresh
            )
            self.thresh.append(thresh)
            per_sample_clip_factor = thresh / (norm + 1e-6)
            clipping_factor.append(per_sample_clip_factor.clamp(max=1.0))
        return clipping_factor if self.is_per_layer else cycle(clipping_factor)

    @property
    def thresholds(self) -> torch.Tensor:
        """
        Returns a tensor of values that are used to normalize gradients
        for each layer such that the norm at most equals the corresponding
        value before clipping.

        Returns:
            Tensor of thresholds
        """
        return torch.tensor(self.thresh)

    @property
    def is_per_layer(self) -> bool:
        """
        Returns indicator as to whether different clipping is applied
        to each layer in the model.

        Returns:
            Value of the flag
        """
        return self.clip_per_layer
