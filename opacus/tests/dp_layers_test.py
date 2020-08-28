#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from opacus.layers import DPLSTM, DPMultiheadAttention
from torch import nn
from torch.nn import LSTM
from torch.nn.modules.activation import MultiheadAttention
from torch.testing import assert_allclose


class DPLayersTest(unittest.TestCase):
    def setUp(self):
        self.EMBED_SIZE = 32

    def _reset_seeds(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)

    def _run_multihead(self, q, k, v, **kwargs):
        original_layer = MultiheadAttention(self.EMBED_SIZE, **kwargs)
        dp_layer = DPMultiheadAttention(self.EMBED_SIZE, **kwargs)
        dp_layer.load_state_dict(original_layer.state_dict())

        self._reset_seeds()
        original_y, original_attn_weights = original_layer(q, k, v)

        self._reset_seeds()
        dp_y, dp_attn_weights = dp_layer(q, k, v)

        self.assertTrue(torch.allclose(original_y, dp_y, atol=10e-4, rtol=10e-2))
        self.assertTrue(
            torch.allclose(
                original_attn_weights, dp_attn_weights, atol=10e-4, rtol=10e-2
            )
        )

    def _run_multihead_x(self, **kwargs):
        x = torch.randn(16, 24, self.EMBED_SIZE)
        self._run_multihead(x, x, x, **kwargs)

    def _run_multihead_qkv(self, **kwargs):
        q = torch.randn(16, 24, self.EMBED_SIZE)
        k = torch.randn(20, 24, kwargs["kdim"] if "kdim" in kwargs else self.EMBED_SIZE)
        v = torch.randn(20, 24, kwargs["vdim"] if "vdim" in kwargs else self.EMBED_SIZE)
        self._run_multihead(q, k, v, **kwargs)

    def test_multihead_attention(self):
        for num_heads in (1, 2, 16):
            self._run_multihead_x(num_heads=num_heads)
            self._run_multihead_qkv(num_heads=num_heads)

            self._run_multihead_x(num_heads=num_heads, dropout=0.05)
            self._run_multihead_x(num_heads=num_heads, bias=False)
            self._run_multihead_x(num_heads=num_heads, add_bias_kv=True)
            self._run_multihead_x(num_heads=num_heads, bias=False, add_bias_kv=True)
            self._run_multihead_x(num_heads=num_heads, add_zero_attn=True)

            self._run_multihead_qkv(num_heads=num_heads, kdim=24, vdim=24)


class DPLSTMTest(unittest.TestCase):
    def setUp(self):
        self.SEQ_LENGTH = 20
        self.INPUT_DIM = 25
        self.MINIBATCH_SIZE = 30
        self.LSTM_OUT_DIM = 12

        self.h_init = torch.randn(1, self.MINIBATCH_SIZE, self.LSTM_OUT_DIM)
        self.c_init = torch.randn(1, self.MINIBATCH_SIZE, self.LSTM_OUT_DIM)
        hidden = (self.h_init, self.c_init)

        self.x = torch.randn(self.MINIBATCH_SIZE, self.SEQ_LENGTH, self.INPUT_DIM)

        self.original_lstm = LSTM(self.INPUT_DIM, self.LSTM_OUT_DIM, batch_first=True)
        self.dp_lstm = DPLSTM(self.INPUT_DIM, self.LSTM_OUT_DIM, batch_first=True)

        self.dp_lstm.initialize_weights(
            [
                self.original_lstm.weight_ih_l0,
                self.original_lstm.weight_hh_l0,
                self.original_lstm.bias_ih_l0,
                self.original_lstm.bias_hh_l0,
            ]
        )

        self.lstm_out, self.lstm_state = self.original_lstm(self.x, hidden)
        self.dplstm_out, self.dplstm_state = self.dp_lstm(self.x, hidden)

    def _reset_seeds(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)

    def test_lstm_forward(self):
        params_to_test = [
            (self.lstm_out, self.dplstm_out, "LSTM and DPLSTM output"),
            (self.lstm_state[0], self.dplstm_state[0], "LSTM and DPLSTM state `h`"),
            (self.lstm_state[1], self.dplstm_state[1], "LSTM and DPLSTM state `c`"),
        ]
        for param, dp_param, message in params_to_test:
            assert_allclose(
                actual=param,
                expected=dp_param,
                atol=10e-5,
                rtol=10e-3,
                msg=f"Tensor value mismatch between {message}",
            )

    def test_lstm_backward(self):
        y = torch.randn(self.MINIBATCH_SIZE, self.SEQ_LENGTH, self.LSTM_OUT_DIM)
        criterion = nn.MSELoss()

        loss = criterion(y, self.lstm_out)
        loss.backward()

        dp_loss = criterion(y, self.dplstm_out)
        dp_loss.backward()

        params_to_test = [
            (
                self.original_lstm.weight_ih_l0.grad,
                self.dp_lstm.weight_ih_l0.grad,
                "LSTM and DPLSTM `weight_ih_l0` gradients",
            ),
            (
                self.original_lstm.bias_ih_l0.grad,
                self.dp_lstm.bias_ih_l0.grad,
                "LSTM and DPLSTM `bias_ih_l0` gradients",
            ),
            (
                self.original_lstm.weight_hh_l0.grad,
                self.dp_lstm.weight_hh_l0.grad,
                "LSTM and DPLSTM `weight_hh_l0` gradients",
            ),
            (
                self.original_lstm.bias_hh_l0.grad,
                self.dp_lstm.bias_hh_l0.grad,
                "LSTM and DPLSTM `bias_hh_l0` gradients",
            ),
        ]

        for param, dp_param, message in params_to_test:
            assert_allclose(
                actual=param,
                expected=dp_param,
                atol=10e-5,
                rtol=10e-3,
                msg=f"Tensor value mismatch between {message}",
            )
