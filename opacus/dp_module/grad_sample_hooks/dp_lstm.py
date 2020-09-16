#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from .utils import create_or_extend_grad_sample


def compute_grad_sample(module, A, B, batch_dim=0):
    """
    Computes per sample gradients for ``DPLSTM`` module
    Parameters
    ----------
    module : opacus.modules.dp_lstm.DPLSTM
        module
    A : torch.Tensor
        Activations
    B : torch.Tensor
        Backpropagations
    batch_dim : int, optional
        Batch dimension position
    """
    lstm_params = [
        module.weight_ih_l0,
        module.weight_hh_l0,
        module.bias_ih_l0,
        module.bias_hh_l0,
    ]
    lstm_out_dim = module.hidden_size

    x = torch.unbind(A, dim=1)
    hooks_delta = torch.unbind(B, dim=1)

    SEQ_LENGTH = len(x)
    BATCH_SIZE = B.shape[0]

    h_init = torch.zeros(1, BATCH_SIZE, lstm_out_dim, device=A.device)
    c_init = torch.zeros(1, BATCH_SIZE, lstm_out_dim, device=A.device)

    delta_h = {}
    delta_h[SEQ_LENGTH - 1] = 0
    f_last = 0
    dc_last = 0

    for t in range(SEQ_LENGTH - 1, -1, -1):
        f_next = f_last if t == SEQ_LENGTH - 1 else module.cells[t + 1].f_t
        dc_next = dc_last if t == SEQ_LENGTH - 1 else module.cells[t + 1].dc_t
        c_prev = c_init if t == 0 else module.cells[t - 1].c_t
        delta_h[t - 1] = module.cells[t].backward(
            x[t], delta_h[t], hooks_delta[t], f_next, dc_next, c_prev
        )

    grad_sample = {param: 0 for param in lstm_params}

    for t in range(0, SEQ_LENGTH):
        h_prev = h_init[0, :] if t == 0 else module.cells[t - 1].h_t[0, :]
        grad_sample[module.weight_ih_l0] += torch.einsum(
            "ij,ik->ijk", module.cells[t].dgates_t, x[t]
        )
        grad_sample[module.weight_hh_l0] += torch.einsum(
            "ij,ik->ijk", module.cells[t].dgates_t, h_prev
        )
        grad_sample[module.bias_ih_l0] += module.cells[t].dgates_t
        grad_sample[module.bias_hh_l0] += module.cells[t].dgates_t

    for param, grad_value in grad_sample.items():
        # pyre-ignore[6]
        create_or_extend_grad_sample(param, grad_value, batch_dim)
