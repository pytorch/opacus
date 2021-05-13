#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
r"""
The process of adding differential privacy to a model involves bounds its sensitivity prior to
applying the Gaussian mechanism. This is achieved by clipping the per-sample gradients.
Normally for a parameterized layer if you have a tensor of parameters of size ``[m, n]``,
the size of the gradients will match it. This means that they get aggregated over the batch.
Here, we will keep them per-sample i.e., we will have a tensor of size ``[b_sz, m, n]``, where
the slice ``[i, :, :]`` corresponds to the per-example gradients for the i-th example in the batch.

Per-sample gradient clipping has to be achieved under the following constraints:

1. The norm of the grad_sample of the loss with respect to all model parameters has
to be clipped so that if they were to be put in a single vector together. If ``C`` is the clipping
threshold, this ensures the total norm will be at most ``C``.

Example:
    >>> T = torch.cat([p.grad_sample.flatten() for p in model.parameters()])

    ``T`` will have shape ``[B, N_TOTAL_PARAMS]``. The total L2 norm of each row of ``T``
    cannot be greater than ``C``.

2. This clipping should not backpropagate. This means that clipping in the layer ``i+1``
should not affect computing the gradient of layer ``i``. To make sure this is followed
we will first compute the grad_sample of all layers **without clipping**. In a second pass, we will
go back to the per-sample gradients, clip them, and accumulate them in ``.grad``
(thus replacing the "real" gradients).

Notes:
    There is only a single .backward() call as the second pass just works on top of
    the stored grad_sample.
"""

from typing import Callable, Iterator, Optional, Tuple

import torch
from opacus.grad_sample import GradSampleModule
from torch import nn

from .utils.clipping import NormClipper
from .utils.tensor_utils import calc_sample_norms


class PerSampleGradientClipper:
    r"""
    Class to define a per-sample gradient clipper for a module. Per-sample gradient clipping
    bounds the sensitivity of the computation before applying the Gaussian mechanism.
    """

    def __init__(
        self,
        module: GradSampleModule,
        norm_clipper: NormClipper,
        batch_first: bool = True,
        loss_reduction: str = "mean",
    ):
        r"""
        Attaches to a module, and clips all grad_sample in the backward
        pass. It then puts them in each parameter's ``.grad``.

        Args:
            module: Module to which backward hooks are added and for which per-sample
                gradients are clipped

            norm_clipper: A norm clipper object of class
                :class:`~opacus.utils.clipping.NormClipper` which encapsulated different
                clipping strategies (such as flat clipping for the entire model, or
                per-layer clipping)

            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension represent the batch, for example of shape
                [batch_size, ..., ...]. Set to True if batch appears in first
                dimension else set to False (batch_first=False implies that the batch
                is always in the second dimension).

            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values ``sum`` or ``mean``
        """
        self.module = module
        self.norm_clipper = norm_clipper
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction

        self._reset_aggregated_state()

        self.on_batch_clip_func = None

    def set_on_batch_clip_func(self, on_batch_clip_func: Callable[..., None]) -> None:
        r"""
        Sets the function to be called after clipping to the input callable parameter
        (for example clipping stats collection)

        Args:
            on_batch_clip_func: Function to be called after clipping
        """
        self.on_batch_clip_func = on_batch_clip_func

    def __repr__(self):
        return f"PerSampleGradientClipModuleHook on {self.module}"

    def _reset_aggregated_state(self) -> None:
        r"""
        Resets the aggregated state of the clipper to be zero for
        the batch size and zero tensors for the per-layer thresholds
        """
        self._aggr_batch_size = 0
        self._aggr_thresh = torch.zeros(1)

    def _get_aggregated_state(self) -> Tuple[torch.Tensor, int]:
        r"""
        Returns an aggregated state of the clipper consisting of the
        list of layer thresholds (for those providing gradient norms)
        as well as the aggregate batch size

        Returns:
            Aggregated state (layer thresholds and batch size)
        """
        return self._aggr_thresh, self._aggr_batch_size

    def pre_step(self) -> Tuple[torch.Tensor, int]:
        r"""
        Prepares the ``.grad`` field of the parameters and provides statistics on the
        maximum gradient norm which should be used to scale noise in the privacy engine
        (:class:``~opacus.privacy_engine.PrivacyEngine``). This function is called before
        the optimizer ``step()``.

        Returns:
            The maximum gradient norm per batch (repeated in batch dimension
            as a tensor) and the batch size
        """

        # check if we've already accumulated clipped gradients for this batch
        if self._aggr_batch_size == 0:
            raise ValueError("You need to call clip_and_accumulate first")

        threshs, batch_size = self._get_aggregated_state()
        # now that we know the full batch size, we can average the gradients
        n = 0
        for _, p in self._named_params():
            p.grad = self._scale_summed_grad(p.summed_grad, batch_size)
            n += 1
            del p.summed_grad

        # NOTE: For Renyi-based epsilon calculation, we will calculate a flat
        # max norm equal to the norm of all clip values per layer.
        max_norm = threshs.new_full((n,), threshs.norm(2))
        self._reset_aggregated_state()
        return max_norm, batch_size

    def clip_and_accumulate(self) -> None:
        r"""
        Clips and sums up per-sample gradients into an accumulator. When this function is called
        ``N >= 1`` times on mini-batches of size ``B`` (could be smaller on final batch), a call to
        :meth:`~opacus.per_sample_gradient_clip.PerSampleGradientClipper.pre_step`
        will populate the ``.grad`` field with the average gradient over the entire batch of size
        ``(N-1)* B + b`` with ``b <= B``.
        """
        # step 0 : calculate the layer norms
        all_norms = calc_sample_norms(
            named_params=self._named_grad_samples(),
            flat=not self.norm_clipper.is_per_layer,
        )

        # step 1: calculate the clipping factors based on the noise
        clipping_factor = self.norm_clipper.calc_clipping_factors(all_norms)

        # step 2: update the aggreagated thresholds and batch size
        self._aggr_thresh = torch.max(
            self._aggr_thresh, self.norm_clipper.thresholds
        )  # retain the largest clipping thresholds accross the entire batch
        batch_size = next(p.shape[0] for (_, p) in self._named_grad_samples())
        # The size for every param.grad_sample is the batch size
        self._aggr_batch_size += batch_size

        for i, (clip_factor, named_param) in enumerate(
            zip(clipping_factor, self._named_params())
        ):
            # Do the clipping
            name, p = named_param
            summed_grad = self._weighted_sum(clip_factor, p.grad_sample)
            clipping_thresh = self.norm_clipper.thresholds[
                i if len(self.norm_clipper.thresholds) > 1 else 0
            ]
            per_sample_norm = all_norms[i if len(all_norms) > 1 else 0]
            # accumulate the summed gradient for this mini-batch
            if hasattr(p, "summed_grad"):
                p.summed_grad += summed_grad
            else:
                p.summed_grad = summed_grad

            self._on_batch_clip(
                name,
                clip_factor,
                clipping_thresh,
                per_sample_norm,
                p.grad_sample,
                grad_before_clip=p.grad,
                grad_after_clip=self._scale_summed_grad(summed_grad, batch_size),
            )

            # remove the per-sample gradients
            del p.grad_sample
        self._on_batch_clip()  # inform analysis of the whole module

    def zero_grad(self):
        """
        Deletes the added attributes, ``grad_sample`` and ``summed_grad``.

        The two mentioned attributes are
        automatically deleted when ``pre_step`` or
        ``clip_and_accumulate`` are properly called. This is a safety measure
        to avoid further issues if regular use has not been followed.
        """
        for _, param in self._named_params():
            if hasattr(param, "grad_sample"):
                del param.grad_sample
            if hasattr(param, "summed_grad"):
                del param.summed_grad

    def _named_params(self) -> Iterator[Tuple[str, nn.Parameter]]:
        r"""
        Helper function to get parameter with their names that require grad

        Returns:
            Iterator over parameters with their names
        """
        return ((n, p) for n, p in self.module.named_parameters() if p.requires_grad)

    def _named_grad_samples(self) -> Iterator[Tuple[str, torch.Tensor]]:
        r"""
        Helper function to get names and per-sample gradients for parameters
        that required grad.

        Returns:
            Iterator of parameter names and per-sample gradients
        """

        no_grad_samples = [
            n
            for n, p in self.module.named_parameters()
            if p.requires_grad and not hasattr(p, "grad_sample")
        ]
        if len(no_grad_samples) >= 1:
            raise AttributeError(
                f"The following layers do not have gradients: {no_grad_samples}. Are you sure they were included in the backward pass?"
            )

        return (
            (n, p.grad_sample)
            for n, p in self.module.named_parameters()
            if p.requires_grad
        )

    def _scale_summed_grad(
        self, summed_grad: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        r"""
        Depending on the loss type, this function averages the summed gradient over batch
        if attribute ``loss_reduction`` is set to "mean", else it returns the input summed
        gradient tensor.

        Args:
            summed_grad: Summed gradient tensor which might be averaged depending on loss_reduction

            batch_size: Batch size of gradient tensor

        Returns:
            Summed gradient tensor if loss_reduction is set to sum else averaged over batch.

        Raises:
            ValueError
                If the loss reduction is not defined to be either 'sum' or 'mean'
        """
        if self.loss_reduction == "mean":
            return summed_grad / batch_size
        elif self.loss_reduction == "sum":
            return summed_grad.detach()
        else:
            raise ValueError(
                f"Loss reduction must be either sum or mean. Got {self.loss_reduction}"
            )

    def _weighted_sum(
        self, batch_weight: torch.Tensor, param: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Helper function to calculate a weighted sum of tensor ``param``
        along the batch dimension weighted by tensor ``batch_weight``.

        Args:
            batch_weight: Tensor of shape ``B`` (where ``B`` is the batch size) corresponding
                to weights along the batch dimension. Each sample in the batch has its own weight.
            param: Tensor to be weighted, is of shape ``[B,...]`` where ``B`` represents the
                batch size.

        Returns:
            Weighted sum tensor for ``param`` along the batch dimension weighted by batch_weight.
        """
        return torch.einsum("i,i...", batch_weight, param)

    def _on_batch_clip(
        self,
        param_name: Optional[str] = None,
        clipping_factor: Optional[torch.Tensor] = None,
        clipping_threshold: Optional[torch.Tensor] = None,
        per_sample_norm: Optional[torch.Tensor] = None,
        per_sample_grad: Optional[torch.Tensor] = None,
        grad_before_clip: Optional[torch.Tensor] = None,
        grad_after_clip: Optional[torch.Tensor] = None,
    ):
        r"""
        Calls a pre-specified function (for example, for clipping stats computation) and
        grants access to that function about current parameter state during the back propagation
        of each batch.

        Args:
            param_name: Name of the parameter, the parameter could be accessed by
                ``self.module.state_dict()[param_name]``. A value of ``None``
                indicates that all parameters have been processed.
            clipping_factor: Scaling factor used in gradient clipping.
            clipping_threshold: Threshold used in gradient clipping.
            per_sample_norm: Per-sample gradient norms for clipping
            per_sample_grad: Raw per sample gradients for parameter
            grad_before_clip: Aggregated gradient before clipping (``= per_sample_grad.mean()``)
            grad_after_clip: Aggregated gradients after clipping
        """
        if self.on_batch_clip_func:
            self.on_batch_clip_func(
                param_name=param_name,
                clipping_factor=clipping_factor,
                clipping_threshold=clipping_threshold,
                per_sample_norm=per_sample_norm,
                per_sample_grad=per_sample_grad,
                grad_before_clip=grad_before_clip,
                grad_after_clip=grad_after_clip,
            )
