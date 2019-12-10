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

import torch

from . import autograd_grad_sample


def clip_per_sample_grad_norm_(model, max_norm):
    r"""Clips the grad_sample stored in .grad_sample by computing a per-sample
    norm clip factor, using it to rescale each sample's gradient in
    .grad_sample to norm clip, then averaging them back into .grad.

    The gradients of the model's parameters are modified in-place.

    We assume the batch size is the first dimension.

    Arguments:
        tensor (Tensor): a single Tensor whose norm will be normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        New total norm of the tensor.
    """
    max_norm = float(max_norm)
    per_sample_norm = get_total_per_sample_grad_norm(model)

    # Each sample gets clipped independently. This is a tensor of size B
    per_sample_clip_factor = max_norm / (per_sample_norm + 1e-6)

    # We are *clipping* the gradient, so if the factor is ever >1 we set it to 1
    per_sample_clip_factor = per_sample_clip_factor.clamp(max=1.0)
    b_sz = len(per_sample_clip_factor)

    # We recompute .grad from .grad_sample by simply averaging it over the B dim
    for p in model.parameters():
        p.grad = torch.einsum("i,i...", per_sample_clip_factor, p.grad_sample) / b_sz
    return


def get_per_sample_norm(t):
    aggregation_dims = [i for i in range(1, len(t.shape))]  # All dims except the first
    t_squared = t * t  # elementwise
    return torch.sqrt(t_squared.sum(dim=aggregation_dims))


def get_total_per_sample_grad_norm(model):
    all_layers_norms = torch.stack(
        [get_per_sample_norm(p.grad_sample) for p in model.parameters()], dim=-1
    )
    return all_layers_norms.norm(2, dim=1)


class PerSampleGradientClipper:
    def __init__(self, module, max_norm):
        """
        Attaches to a module, and clips all grad_sample in the backward
        pass. It then puts them in each parameter's .grad.
        """
        self.module = module
        autograd_grad_sample.add_hooks(self.module)
        self.max_norm = max_norm

    def close(self):
        autograd_grad_sample.remove_hooks(self.module)

    def __repr__(self):
        return f"PerSampleGradientClipModuleHook on {self.module}"

    def step(self):
        autograd_grad_sample.compute_grad_sample(self.module)

        # The first dim of param.grad_sample is b_sz for every param.
        # To look up what that value is, we just pick one
        self.batch_size = next(p.grad_sample.shape[0] for p in self.module.parameters())

        clip_per_sample_grad_norm_(self.module, self.max_norm)
        autograd_grad_sample.clear_backprops(self.module)
