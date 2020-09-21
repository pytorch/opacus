#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class DPLSTMCell(nn.Module):
    def __init__(self, input_dim, lstm_out_dim):
        super(DPLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.lstm_out_dim = lstm_out_dim
        self.sigmoid_fn = nn.Sigmoid()
        self.tanh_fn = nn.Tanh()

    def initialize_weights(self, weight_params):
        [
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
        ] = weight_params

    def forward(self, x_t, h_prev, c_prev):
        self.split_input = (
            F.linear(x_t, self.weight_ih_l0, self.bias_ih_l0)
            + F.linear(h_prev, self.weight_hh_l0, self.bias_hh_l0)
        )[0, :]
        self.i_t_input, self.f_t_input, self.g_t_input, self.o_t_input = torch.split(
            self.split_input, self.lstm_out_dim, 1
        )
        self.i_t = self.sigmoid_fn(self.i_t_input)
        self.f_t = self.sigmoid_fn(self.f_t_input)
        self.g_t = self.tanh_fn(self.g_t_input)
        self.o_t = self.sigmoid_fn(self.o_t_input)
        self.c_t = self.f_t * c_prev + self.i_t * self.g_t
        self.h_t = self.o_t * self.tanh_fn(self.c_t)
        return self.h_t[0, :]

    def backward(self, x_t, delta_h_t, delta_t, f_next, dc_next, c_prev):
        self.dh_t = delta_t + delta_h_t
        self.dc_t = (
            self.dh_t * self.o_t * (1 - torch.pow(self.tanh_fn(self.c_t), 2))
            + dc_next * f_next
        )
        self.dg_t = self.dc_t * self.i_t * (1 - torch.pow(self.g_t, 2))
        self.di_t = self.dc_t * self.g_t * self.i_t * (1 - self.i_t)
        self.df_t = self.dc_t * c_prev * self.f_t * (1 - self.f_t)
        self.do_t = self.dh_t * self.tanh_fn(self.c_t) * self.o_t * (1 - self.o_t)
        self.dgates_t = torch.cat([self.di_t, self.df_t, self.dg_t, self.do_t], 2)[0, :]
        self.delta_h_prev = torch.matmul(self.dgates_t, self.weight_hh_l0)
        return self.delta_h_prev


class DPLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
    ):
        super(DPLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.cells_initialized = False
        self.cells = []

        self.validate_parameters()

        self.weight_ih_l0 = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def validate_parameters(self):
        if self.num_layers > 1 or not self.bias or self.dropout or self.bidirectional:
            raise ValueError(
                "DPLSTM Layer initialized with unsupported non-default flag. "
                "Only supported flags are bias=True, bidirectional=False "
                "num_layers=1, dropout=False, and initial state set to zero tensors"
            )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def _rearrange_batch_dim(self, x):
        if self.batch_first:  # batch is by default in second dimension
            x = x.transpose(0, 1)
        return x

    def initialize_weights(self, weight_params):
        [
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
        ] = weight_params

    def forward(self, x, state_init=None):
        x = self._rearrange_batch_dim(x)
        seq_length, batch_sz, _ = x.shape
        if not self.cells_initialized:
            for t in range(0, seq_length):
                self.cells.append(DPLSTMCell(self.input_size, self.hidden_size))
                self.cells[t].initialize_weights(
                    [
                        self.weight_ih_l0,
                        self.weight_hh_l0,
                        self.bias_ih_l0,
                        self.bias_hh_l0,
                    ]
                )
            self.cells_initialized = True

        x = torch.unbind(x, dim=0)
        h = [None] * seq_length

        if state_init:
            h_init, c_init = state_init
        else:
            h_init = torch.zeros(self.num_layers, batch_sz, self.hidden_size)
            c_init = torch.zeros(self.num_layers, batch_sz, self.hidden_size)

        h[0] = self.cells[0](x[0].unsqueeze(0), h_init, c_init)
        for t in range(1, seq_length):
            h[t] = self.cells[t](
                x[t].unsqueeze(0), self.cells[t - 1].h_t, self.cells[t - 1].c_t
            )
        return (
            self._rearrange_batch_dim(torch.stack(h)),
            (self.cells[-1].h_t, self.cells[-1].c_t),
        )
