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

from typing import List, Tuple, Union

import torch

from . import autograd_grad_sample, stats
from .utils import ClippingMethod, calculate_thresh_value


class GradientClipper:
    """
    Clips gradients of a model based on the batch-statistics.
    """

    def __init__(
        self,
        named_params: List[Tuple[str, torch.Tensor]],
        current_max: Union[List[float], float],
        clip_per_layer: bool = False,
        enable_stat: bool = False,
        **param_kwargs,
    ):
        self.named_params = named_params
        self.clip_per_layer = clip_per_layer
        if not isinstance(current_max, list):
            current_max = [current_max] * (
                len(named_params) if self.clip_per_layer else 1
            )
        self.current_max = current_max
        self.clipping_method = param_kwargs.get(
            "clipping_method", ClippingMethod.STATIC
        )
        if self.clipping_method != ClippingMethod.STATIC:
            print(
                "Warning! Current implementations of dynamic clipping "
                "are not privacy safe; Caclulated privacy loss is not "
                "indicative of a proper bound."
            )
        self.ratio = param_kwargs.get("ratio", 0.0)
        self.stat = {} if enable_stat else None
        self.accumulation_state = {}

    def _get_per_layer_norms(self, named_param) -> torch.Tensor:
        name, p = named_param
        aggregation_dims = list(range(1, len(p.shape)))  # All dims except the first
        p_squared = p * p  # elementwise
        batch_norms = torch.sqrt(p_squared.sum(dim=aggregation_dims))
        if self.stat is not None:
            normalized_per_coordinate_value = (
                p.abs().sum(dim=aggregation_dims) / p[0].numel()
            )
            self.stat[f"{name}:max"] = normalized_per_coordinate_value.max()
            self.stat[f"{name}:mean"] = normalized_per_coordinate_value.mean()
            self.stat[f"{name}:median"] = normalized_per_coordinate_value.median()
        return batch_norms

    def get_all_layer_norms(self) -> torch.Tensor:
        all_layers_norms = [
            self._get_per_layer_norms((name, p.grad_sample))
            for name, p in self.named_params
        ]
        # aggregate if layers are not independently clipped
        aggregated_norms = torch.stack(all_layers_norms, dim=-1)
        aggregated_norms = aggregated_norms.norm(2, dim=1)
        if not self.clip_per_layer:
            all_layers_norms = [aggregated_norms]

        # update the stats first
        if self.stat is not None:
            stats.update(
                stats.StatType.CLIPPING,
                "AllLayers",
                max=aggregated_norms.max(),
                mean=aggregated_norms.mean(),
                median=aggregated_norms.median(),
            )
            stats.update(stats.StatType.CLIPPING, "PerLayer", **self.stat)
            self.stat = {}

        return all_layers_norms

    def calc_thresh_value(self, layer_norms):
        thresh_norm = []
        if len(layer_norms) != len(self.current_max):
            raise ValueError(
                f"Provided grad norm max's size {len(self.current_max)}"
                f" does not match the number of layers {len(layer_norms)}"
            )
        for norm, current_max in zip(layer_norms, self.current_max):
            thresh_norm.append(
                (
                    calculate_thresh_value(
                        norm, current_max, self.clipping_method, self.ratio
                    ),
                    norm,
                )
            )
        return (
            thresh_norm if self.clip_per_layer else thresh_norm * len(self.named_params)
        )

    def clip(self, accumulate: bool = False) -> List[float]:
        r"""
        Clips the grad_sample stored in .grad_sample by computing a per-sample
        norm clip factor, using it to rescale each sample's gradient in
        .grad_sample to norm clip, then averaging them back into .grad.
        If 'accumulate' is set, the clipped gradients are summed up into a 
        temporary accumulator instead.

        The gradients of the model's parameters are modified in-place.

        We assume the batch size is the first dimension.

        Arguments:
            accumulate (bool): if set, the clipped gradients in this mini-batch 
            are summed up into a accummulator for a larger batch

        Returns:
            A list of clipping thresholds
        """

        # check if we've already accumulated all the clipped gradients for this batch
        if self.accumulation_state and not accumulate:
            return self.finalize_batch()

        # step 0 : calculate the layer norms and thresholds
        all_norms = self.get_all_layer_norms()
        thresh_norms = self.calc_thresh_value(all_norms)
        batch_size = len(thresh_norms[0][1])
        threshs = []
        for thresh_norm, named_param in zip(thresh_norms, self.named_params):
            # step 1 : Find the clipping factor per layer (per parameter set)
            thresh, norm = thresh_norm
            per_sample_clip_factor = thresh / (norm + 1e-6)
            # We are *clipping* the gradient, so if the factor is ever >1 we set it to 1
            per_sample_clip_factor = per_sample_clip_factor.clamp(max=1.0)
            # step 2: Do the clipping
            name, p = named_param
            pre_clip_pos = p.grad_sample.mean(0) > 0
            summed_grad = torch.einsum("i,i...", per_sample_clip_factor, p.grad_sample)

            if accumulate:
                # accumulate the summed gradient for this mini-batch
                if p in self.accumulation_state:
                    self.accumulation_state[p] += summed_grad
                else:
                    self.accumulation_state[p] = summed_grad
            else:
                p.grad = summed_grad / batch_size

            post_clip_pos = p.grad > 0
            sign_switched = (pre_clip_pos ^ post_clip_pos).sum()
            total_num = post_clip_pos.numel()
            if self.stat is not None:
                self.stat[f"{name}:clip"] = thresh
                self.stat[f"{name}:percent"] = (
                    (norm > thresh).to(dtype=torch.float64).mean()
                )
                self.stat[f"{name}:switch"] = float(sign_switched) / total_num
            threshs.append(thresh)
        if self.stat is not None:
            stats.update(stats.StatType.CLIPPING, "ClippingStats", **self.stat)
            self.stat = {}

        if accumulate:
            # check if we're adding to an existing accumulator
            if "clip_threshs" in self.accumulation_state:
                # retain the largest clipping thresholds accross the entire batch
                curr_thresh = self.accumulation_state["clip_threshs"]
                self.accumulation_state["clip_threshs"] = [
                    max(n1, n2) for (n1, n2) in zip(threshs, curr_thresh)
                ]

                # keep track of the total batch size
                self.accumulation_state["batch_size"] += batch_size

            else:
                # start a new accumulator
                self.accumulation_state["clip_threshs"] = threshs
                self.accumulation_state["batch_size"] = batch_size

        return threshs, batch_size

    def finalize_batch(self):
        """
        Averages the clipped gradients aggregated over multiple mini-batches and
        stores them in the .grad field
        """

        batch_size = self.accumulation_state["batch_size"]
        # now that we know the full batch size, we can average the gradients
        for _, p in self.named_params:
            acc_grad = self.accumulation_state[p]
            p.grad = acc_grad / batch_size

        threshs = self.accumulation_state["clip_threshs"].copy()
        self.erase_accumulated_grads()
        return threshs, batch_size

    def erase_accumulated_grads(self):
        """Deletes any accumulateds gradient state"""
        self.accumulation_state = {}


class PerSampleGradientClipper:
    def __init__(self, module, max_norm, batch_dim=0, **kwargs):
        """
        Attaches to a module, and clips all grad_sample in the backward
        pass. It then puts them in each parameter's .grad.
        """
        self.module = module
        autograd_grad_sample.add_hooks(self.module, batch_dim=batch_dim)
        self.max_norm = max_norm
        self.hooks_attached = True
        self.batch_dim = batch_dim
        self.gradient_clipper = GradientClipper(
            [(n, p) for n, p in module.named_parameters() if p.requires_grad],
            self.max_norm,
            **kwargs,
        )

    def __del__(self):
        self.close()

    def close(self):
        if (
            hasattr(self, "hooks_attached") and self.hooks_attached
        ):  # do not close twice
            autograd_grad_sample.remove_hooks(self.module)
        self.hooks_attached = False

    def __repr__(self):
        return f"PerSampleGradientClipModuleHook on {self.module}"

    def step(self):
        clip_threshs, batch_size = self.gradient_clipper.clip()
        return clip_threshs, batch_size

    def accumulate_grads(self):
        """
        Clips and sums up per-sample gradients into an accumulator.
        After calling self.accumulate_grads() N times on mini-batches of 
        B per-sample gradients, a call to self.step() will 
        populate the .grad field with the average gradient over
        the entire batch of size N*B.
        """
        self.gradient_clipper.clip(accumulate=True)
        # reset the accumulation of per-sample gradients
        autograd_grad_sample.clear_grad_sample(self.module)

    def zero_grad(self):
        """
        Erase any aggregated gradients when we zero-out the gradient.
        This is useful when gradients were accumulated but not processed in the
        last mini-batches of a training epoch.
        """
        self.gradient_clipper.erase_accumulated_grads()
