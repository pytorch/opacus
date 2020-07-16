#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
In order to apply the Gaussian mechanism to the gradient computation, it is
necessary to bound its sensitivity.
This can be achieved via **per-sample gradient clipping** (in short,
*grad_sample clip*).
Normally if you have a matrix of parameters of size [m, n], the size of the
gradients will match it. This means that they get aggregated over the batch.
Here, we will keep them per-example ie we will have a tensor of size [b_sz, m, n].

grad_sample clip has to be achieved under the following constraints:

1. The norm of the grad_sample of the loss wrt all model parameters has
to be clipped so that if they were to be put in a single vector together, the
total norm will be at most C.
Or in code, let `T = torch.cat([p.grad_sample.flatten() for p in model.parameters()])`.
T will have size `[B, N_TOTAL_PARAMS]`. The total L2 norm of each row of T
cannot be greater than C.
2. This clipping should not backpropagate. This means that clipping layer i+1
should not affect computing the gradient of layer i. To make sure this doesn't
happen, we will first compute the grad_sample of all layers
**without clipping**. In a second pass, we will go back to the per-sample
gradients, clip them, and accumulate them in `.grad`
(thus replacing the "real" gradients). Note: there is only a single .backward()
call as the second pass just works on top of the store grad_sample.
"""

from typing import Callable, List, Tuple

import torch
from torch import nn
from .utils.tensor_utils import calc_sample_norms

from . import autograd_grad_sample
from .utils.clipping import NormClipper


class PerSampleGradientClipper:
    def __init__(
        self,
        module: nn.Module,
        norm_clipper: NormClipper,
        batch_first: bool = True,
        loss_reduction: str = "mean",
    ):
        """
        Attaches to a module, and clips all grad_sample in the backward
        pass. It then puts them in each parameter's .grad.

        Attributes:
            module: reference to the model that is being trained
            norm_clipper: the class that provides clipping factor for
                the given gradients, look at `NormClipper`
            batch_first: if `True` [default] the first dimension represents
                the batch, if `False` the second dimension is the batch.
        """
        self.module = module
        autograd_grad_sample.add_hooks(
            self.module, batch_first=batch_first, loss_reduction=loss_reduction
        )
        self.hooks_attached = True
        self.norm_clipper = norm_clipper
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction
        self._reset_aggregated_state()

        self.on_batch_clip_func = None

    def set_on_batch_clip_func(self, on_batch_clip_func: Callable[..., None]):
        self.on_batch_clip_func = on_batch_clip_func

    def __del__(self):
        self.close()

    def close(self):
        if self.hooks_attached:  # do not close twice
            autograd_grad_sample.remove_hooks(self.module)
        self.hooks_attached = False

    def __repr__(self):
        return f"PerSampleGradientClipModuleHook on {self.module}"

    def _reset_aggregated_state(self) -> None:
        self._aggr_batch_size = 0
        self._aggr_thresh = torch.zeros_like(self.norm_clipper.thresholds)

    def _get_aggregated_state(self) -> Tuple[List[float], int]:
        return self._aggr_thresh, self._aggr_batch_size

    def pre_step(self) -> Tuple[List[float], int]:
        """
        Needs to be called before any optimizer step, this function
        prepares the `.grad` field of the parameters and provides
        statistics on the `max_nrom` which should be used to scale
        noise in the privacy engine.
        """

        # check if we've already accumulated clipped gradients for this batch
        if self._aggr_batch_size == 0:
            raise ValueError("You need to call `clip_and_accumulate` first")

        threshs, batch_size = self._get_aggregated_state()
        # now that we know the full batch size, we can average the gradients
        n = 0
        for _, p in self._named_params():
            p.grad = self._scale_summed_grad(p.summed_grad, batch_size)
            n += 1
            del p.summed_grad

        # NOTE: For Renyi-basedy of epsilon calculation, we will caclulate a flat
        # max norm equal to the norm of all clip values per layer.
        max_norm = threshs.new_full((n,), threshs.norm(2))
        self._reset_aggregated_state()
        return max_norm, batch_size

    def clip_and_accumulate(self) -> None:
        """
        Clips and sums up per-sample gradients into an accumulator.
        After calling `clip_and_aggregate` `N >= 1` times on mini-batches of
        size B (could be smaller on final batch), a call to
        `prepare_for_optimizer` will populate the `.grad` field with the
        average gradient over the entire batch of size `(N-1)* B + b`
        with `b <= B`.
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
        # The  size for every param.grad_sample is the batch size
        self._aggr_batch_size += batch_size

        for i, (clip_factor, named_param) in enumerate(
            zip(clipping_factor, self._named_params())
        ):
            # Do the clipping
            name, p = named_param
            summed_grad = self._weighted_sum(clip_factor, p.grad_sample)
            clipping_thresh = self._get_ith(self.norm_clipper.thresholds, i)
            per_sample_norm = self._get_ith(all_norms, i)
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

    def _named_params(self):
        """helper function to get named_params that required grad"""
        return ((n, p) for n, p in self.module.named_parameters() if p.requires_grad)

    def _named_grad_samples(self):
        """helper function to get named_params that required grad"""
        return (
            (n, p.grad_sample)
            for n, p in self.module.named_parameters()
            if p.requires_grad
        )

    def _scale_summed_grad(self, summed_grad, batch_size):
        """ Depending on the loss type, summed grad might need to be averaged over batch
        """
        if self.loss_reduction == "mean":
            return summed_grad / batch_size
        elif self.loss_reduction == "sum":
            return summed_grad.detach()
        else:
            raise ValueError(
                f"Loss reduction must be either sum or mean. Got {self.loss_reduction}"
            )

    def _weighted_sum(self, batch_weight, param):
        """
        helper function to calculate a weighted sum of `param`
        along the batch weighted by batch `batch_weight`.
        Arguments:
            param: is of shape BXYZ or XBYZ where B represents the batch.
            batch_weight: of shape B defines the factor for each batch.
        """
        return torch.einsum("i,i...", batch_weight, param)

    def _get_ith(self, all_data, layer_index):
        """helper to get ith value for lists of layer_data"""
        return all_data[layer_index if len(all_data) > 1 else 0]

    def _on_batch_clip(
        self,
        param_name=None,
        clipping_factor=None,
        clipping_threshold=None,
        per_sample_norm=None,
        per_sample_grad=None,
        grad_before_clip=None,
        grad_after_clip=None,
    ):
        """
        Grants access to current parameter state during the back propagation
        of each bach.

        Arguments:
            param_name: name of the parameter, the parameter could be accessed by
                `self.module.state_dict()[param_name]`. A `param_name` value `None`
                indicates that all parameters have been processed.
            per_sample_grad: raw per_sample_gradients
            grad_before_clip: aggregated gradient before clipping
                (`= per_sample_grad.mean()`)
            grad_after_clip: aggregated gradients after clipping
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
