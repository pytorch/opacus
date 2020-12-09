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


class SimpleDPLSTMTest(unittest.TestCase):
    def setUp(self):
        self.SEQ_LENGTH = 20
        self.INPUT_DIM = 25
        self.MINIBATCH_SIZE = 30
        self.LSTM_OUT_DIM = 12
        self.NUM_LAYERS = 1
        self.bidirectional = False
        self.batch_first = False

        self.num_directions = 2 if self.bidirectional else 1
        self.h_init = torch.randn(
            self.NUM_LAYERS * self.num_directions,
            self.MINIBATCH_SIZE,
            self.LSTM_OUT_DIM,
        )
        self.c_init = torch.randn(
            self.NUM_LAYERS * self.num_directions,
            self.MINIBATCH_SIZE,
            self.LSTM_OUT_DIM,
        )

        self.original_lstm = LSTM(
            self.INPUT_DIM,
            self.LSTM_OUT_DIM,
            batch_first=self.batch_first,
            num_layers=self.NUM_LAYERS,
            bidirectional=self.bidirectional,
        )
        self.dp_lstm = DPLSTM(
            self.INPUT_DIM,
            self.LSTM_OUT_DIM,
            batch_first=self.batch_first,
            num_layers=self.NUM_LAYERS,
            bidirectional=self.bidirectional,
        )

        self.dp_lstm.load_state_dict(self.original_lstm.state_dict())

    def _reset_seeds(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)

    def test_lstm_forward(self):
        x = (
            torch.randn(self.MINIBATCH_SIZE, self.SEQ_LENGTH, self.INPUT_DIM)
            if self.batch_first
            else torch.randn(self.SEQ_LENGTH, self.MINIBATCH_SIZE, self.INPUT_DIM)
        )
        hidden = (self.h_init, self.c_init)

        out, (hn, cn) = self.original_lstm(x, hidden)
        dp_out, (dp_hn, dp_cn) = self.dp_lstm(x, hidden)

        outputs_to_test = [
            (out, dp_out, "LSTM and DPLSTM output"),
            (hn, dp_hn, "LSTM and DPLSTM state `h`"),
            (cn, dp_cn, "LSTM and DPLSTM state `c`"),
        ]

        for output, dp_output, message in outputs_to_test:
            assert_allclose(
                actual=dp_output.expand_as(output),
                expected=output,
                atol=10e-6,
                rtol=10e-5,
                msg=f"Tensor value mismatch between {message}",
            )

    def test_lstm_backward(self):
        x = (
            torch.randn(self.MINIBATCH_SIZE, self.SEQ_LENGTH, self.INPUT_DIM)
            if self.batch_first
            else torch.randn(self.SEQ_LENGTH, self.MINIBATCH_SIZE, self.INPUT_DIM)
        )
        criterion = nn.MSELoss()

        hidden = (self.h_init, self.c_init)

        out, (hn, cn) = self.original_lstm(x, hidden)
        y = torch.zeros_like(out)
        loss = criterion(out, y)
        loss.backward()

        dp_out, (dp_hn, dp_cn) = self.dp_lstm(x, hidden)
        dp_loss = criterion(dp_out, y)
        dp_loss.backward()

        dp_lstm_params = dict(self.dp_lstm.named_parameters())
        for param_name, param in self.original_lstm.named_parameters():
            dp_param = dp_lstm_params[param_name]
            assert_allclose(
                actual=dp_param,
                expected=param,
                atol=10e-5,
                rtol=10e-3,
                msg=f"Tensor value mismatch in the parameter '{param_name}'",
            )
            assert_allclose(
                actual=dp_param.grad,
                expected=param.grad,
                atol=10e-6,
                rtol=10e-5,
                msg=f"Tensor value mismatch in the gradient of parameter '{param_name}'",
            )

    def test_lstm_param_update(self):
        x = (
            torch.randn(self.MINIBATCH_SIZE, self.SEQ_LENGTH, self.INPUT_DIM)
            if self.batch_first
            else torch.randn(self.SEQ_LENGTH, self.MINIBATCH_SIZE, self.INPUT_DIM)
        )
        criterion = nn.MSELoss()

        optimizer = torch.optim.SGD(self.original_lstm.parameters(), lr=0.5)
        dp_optimizer = torch.optim.SGD(self.dp_lstm.parameters(), lr=0.5)

        # Train original LSTM for one step
        logits, (h_n, c_n) = self.original_lstm(x)
        y = torch.zeros_like(logits)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Train DP LSTM for one step
        dp_logits, (dp_h_n, dp_c_n) = self.dp_lstm(x)
        dp_loss = criterion(dp_logits, y)
        dp_loss.backward()
        dp_optimizer.step()

        dp_lstm_params = dict(self.dp_lstm.named_parameters())
        for param_name, param in self.original_lstm.named_parameters():
            dp_param = dp_lstm_params[param_name]
            assert_allclose(
                actual=dp_param,
                expected=param,
                atol=10e-6,
                rtol=10e-5,
                msg=f"Tensor value mismatch in the parameter '{param_name}'",
            )
            assert_allclose(
                actual=dp_param.grad,
                expected=param.grad,
                atol=10e-6,
                rtol=10e-5,
                msg=f"Tensor value mismatch in the gradient of parameter '{param_name}'",
            )


class ComplexDPLSTMTest(unittest.TestCase):
    def setUp(self):
        self.SEQ_LENGTH = 20
        self.INPUT_DIM = 25
        self.MINIBATCH_SIZE = 30
        self.LSTM_OUT_DIM = 12
        self.NUM_LAYERS = 3
        self.bidirectional = True
        self.batch_first = False

        self.num_directions = 2 if self.bidirectional else 1

        self.h_init = torch.randn(
            self.NUM_LAYERS * self.num_directions,
            self.MINIBATCH_SIZE,
            self.LSTM_OUT_DIM,
        )
        self.c_init = torch.randn(
            self.NUM_LAYERS * self.num_directions,
            self.MINIBATCH_SIZE,
            self.LSTM_OUT_DIM,
        )

        self.original_lstm = LSTM(
            self.INPUT_DIM,
            self.LSTM_OUT_DIM,
            batch_first=self.batch_first,
            num_layers=self.NUM_LAYERS,
            bidirectional=self.bidirectional,
        )
        self.dp_lstm = DPLSTM(
            self.INPUT_DIM,
            self.LSTM_OUT_DIM,
            batch_first=self.batch_first,
            num_layers=self.NUM_LAYERS,
            bidirectional=self.bidirectional,
        )

        self.dp_lstm.load_state_dict(self.original_lstm.state_dict())

    def _reset_seeds(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)

    def test_lstm_forward(self):
        x = (
            torch.randn(self.MINIBATCH_SIZE, self.SEQ_LENGTH, self.INPUT_DIM)
            if self.batch_first
            else torch.randn(self.SEQ_LENGTH, self.MINIBATCH_SIZE, self.INPUT_DIM)
        )
        hidden = (self.h_init, self.c_init)

        out, (hn, cn) = self.original_lstm(x, hidden)
        dp_out, (dp_hn, dp_cn) = self.dp_lstm(x, hidden)

        outputs_to_test = [
            (out, dp_out, "LSTM and DPLSTM output"),
            (hn, dp_hn, "LSTM and DPLSTM state `h`"),
            (cn, dp_cn, "LSTM and DPLSTM state `c`"),
        ]

        for output, dp_output, message in outputs_to_test:
            assert_allclose(
                actual=dp_output.expand_as(output),
                expected=output,
                atol=10e-6,
                rtol=10e-5,
                msg=f"Tensor value mismatch between {message}",
            )

    def test_lstm_backward(self):
        x = (
            torch.randn(self.MINIBATCH_SIZE, self.SEQ_LENGTH, self.INPUT_DIM)
            if self.batch_first
            else torch.randn(self.SEQ_LENGTH, self.MINIBATCH_SIZE, self.INPUT_DIM)
        )
        criterion = nn.MSELoss()

        hidden = (self.h_init, self.c_init)

        out, (hn, cn) = self.original_lstm(x, hidden)
        y = torch.zeros_like(out)
        loss = criterion(out, y)
        loss.backward()

        dp_out, (dp_hn, dp_cn) = self.dp_lstm(x, hidden)
        dp_loss = criterion(dp_out, y)
        dp_loss.backward()

        dp_lstm_params = dict(self.dp_lstm.named_parameters())
        for param_name, param in self.original_lstm.named_parameters():
            dp_param = dp_lstm_params[param_name]
            assert_allclose(
                actual=dp_param,
                expected=param,
                atol=10e-5,
                rtol=10e-3,
                msg=f"Tensor value mismatch in the parameter '{param_name}'",
            )
            assert_allclose(
                actual=dp_param.grad,
                expected=param.grad,
                atol=10e-6,
                rtol=10e-5,
                msg=f"Tensor value mismatch in the gradient of parameter '{param_name}'",
            )

    def test_lstm_param_update(self):
        x = (
            torch.randn(self.MINIBATCH_SIZE, self.SEQ_LENGTH, self.INPUT_DIM)
            if self.batch_first
            else torch.randn(self.SEQ_LENGTH, self.MINIBATCH_SIZE, self.INPUT_DIM)
        )
        criterion = nn.MSELoss()

        optimizer = torch.optim.SGD(self.original_lstm.parameters(), lr=0.5)
        dp_optimizer = torch.optim.SGD(self.dp_lstm.parameters(), lr=0.5)

        # Train original LSTM for one step
        logits, (h_n, c_n) = self.original_lstm(x)
        y = torch.zeros_like(logits)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Train DP LSTM for one step
        dp_logits, (dp_h_n, dp_c_n) = self.dp_lstm(x)
        dp_loss = criterion(dp_logits, y)
        dp_loss.backward()
        dp_optimizer.step()

        dp_lstm_params = dict(self.dp_lstm.named_parameters())
        for param_name, param in self.original_lstm.named_parameters():
            dp_param = dp_lstm_params[param_name]
            assert_allclose(
                actual=dp_param,
                expected=param,
                atol=10e-6,
                rtol=10e-5,
                msg=f"Tensor value mismatch in the parameter '{param_name}'",
            )
            assert_allclose(
                actual=dp_param.grad,
                expected=param.grad,
                atol=10e-6,
                rtol=10e-5,
                msg=f"Tensor value mismatch in the gradient of parameter '{param_name}'",
            )
