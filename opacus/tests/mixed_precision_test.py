#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.layers import DPGRU, DPLSTM, DPRNN, DPMultiheadAttention
from torch.utils.data import DataLoader, TensorDataset


def create_random_data(
    model_class: Type[nn.Module],
    batch_size: int = 2,
    input_dim: int = 4,
    output_dim: int = 4,
    num_batches: int = 2,
    seq_len: int = 4,
    device: torch.device = torch.device("cuda"),
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
    """Create random data for different model types"""

    # Common y tensor creation for all model types
    y = torch.randint(
        0,
        output_dim,
        (batch_size * num_batches,),
        device=device,
    )

    # Dictionary mapping model classes to their input tensor creation
    model_data_map = {
        SimpleLinearModel: lambda: torch.randn(
            batch_size * num_batches, input_dim, device=device
        ),
        SimpleLinearReluModel: lambda: torch.randn(
            batch_size * num_batches, input_dim, device=device
        ),
        Conv1DModel: lambda: torch.randn(
            batch_size * num_batches, 3, 16, device=device
        ),
        Conv2DModel: lambda: torch.randn(
            batch_size * num_batches, 3, 16, 16, device=device
        ),
        Conv3DModel: lambda: torch.randn(
            batch_size * num_batches, 3, 8, 8, 8, device=device
        ),
        RNNModel: lambda: torch.randn(
            batch_size * num_batches, seq_len, input_dim, device=device
        ),
        AttentionModel: lambda: torch.randn(
            batch_size * num_batches, seq_len, input_dim, device=device
        ),
        ComplexModel: lambda: torch.randn(
            batch_size * num_batches, seq_len, input_dim, device=device
        ),
        EmbeddingModel: lambda: torch.randint(
            0, 100, (batch_size * num_batches, 10), device=device
        ),
        EmbeddingBagModel: lambda: torch.randint(
            0, 100, (batch_size * num_batches, 10), device=device
        ),
    }

    # Get the appropriate input tensor creation function
    if model_class not in model_data_map:
        raise ValueError(f"Unknown model class: {model_class.__name__}")

    x = model_data_map[model_class]()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader, dataset


class SimpleLinearReluModel(nn.Module):
    """Simple model with just two linear layers and a ReLU activation"""

    def __init__(self, input_dim=16, hidden_dim=32, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SimpleLinearModel(nn.Module):
    """Simple model with linear layers and normalization layers"""

    def __init__(self, input_dim=16, hidden_dim=32, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gn = nn.GroupNorm(num_groups=4, num_channels=hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.rms_norm = nn.RMSNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x_reshaped = x.unsqueeze(-1)
        x = self.gn(x_reshaped)
        x = x.squeeze(-1)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.rms_norm(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


class Conv1DModel(nn.Module):
    """Model with 1D convolutional layer and instance normalization"""

    def __init__(self, input_channels=3, hidden_dim=32, output_dim=10):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.in1d = nn.InstanceNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.in1d(x)
        x = F.relu(x)
        x = F.avg_pool1d(x, x.size(-1))
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class Conv2DModel(nn.Module):
    """Model with 2D convolutional layer and instance normalization"""

    def __init__(self, input_channels=3, hidden_dim=32, output_dim=10):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.in2d = nn.InstanceNorm2d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.in2d(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


class Conv3DModel(nn.Module):
    """Model with 3D convolutional layer and instance normalization"""

    def __init__(self, input_channels=3, hidden_dim=32, output_dim=10):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.in3d = nn.InstanceNorm3d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.in3d(x)
        x = F.relu(x)
        x = F.avg_pool3d(x, (x.size(-3), x.size(-2), x.size(-1)))
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


class RNNModel(nn.Module):
    """Model with RNN layers (LSTM, GRU, and RNN)"""

    def __init__(self, input_dim=16, hidden_dim=32, output_dim=10):
        super().__init__()
        self.lstm = DPLSTM(input_dim, hidden_dim, batch_first=True)
        self.gru = DPGRU(hidden_dim, hidden_dim, batch_first=True)
        self.rnn = DPRNN(hidden_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.gru(x)
        x, _ = self.rnn(x)
        x = x[:, -1]
        x = self.fc(x)
        return x


class AttentionModel(nn.Module):
    """Model with multihead attention layers"""

    def __init__(self, input_dim=16, hidden_dim=32, output_dim=10, num_heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attention = DPMultiheadAttention(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        attn_output, _ = self.attention(x, x, x)
        x = torch.mean(attn_output, dim=1)

        x = self.fc(x)
        return x


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size=100, embedding_dim=16, output_dim=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=1)
        x = F.relu(x)
        x = self.fc(x)
        return x


class EmbeddingBagModel(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=16, output_dim=10):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x_flat = x.reshape(-1)
        offsets = torch.arange(0, batch_size * seq_len, seq_len, device=x.device)
        x = self.embedding_bag(x_flat, offsets)

        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.fc(x)
        return x


class ComplexModel(nn.Module):
    """Model combining multiple layer types"""

    def __init__(self, input_dim=16, hidden_dim=32, output_dim=10, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = DPLSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = DPMultiheadAttention(hidden_dim, 4)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc_in(x)
        x_conv = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        x_conv = self.conv1d(x_conv)  # [batch_size, hidden_dim, seq_len]
        x_conv = x_conv.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]
        x = x + x_conv
        x_rnn, _ = self.lstm(x)
        x = x + x_rnn
        x_attn, _ = self.attention(x, x, x)
        x = x + x_attn
        x = torch.mean(x, dim=1)
        x = self.fc_out(x)
        return x


class MixedPrecisionTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def setUp(self):

        # Since this test only runs when CUDA is available, always use CUDA
        self.device = torch.device("cuda")
        self.input_dim = 4
        self.hidden_dim = 16
        self.output_dim = 4
        self.seq_len = 4
        self.batch_size = 2
        self.num_batches = 2

        # Check if bfloat16 is supported
        self.bf16_supported = hasattr(torch, "bfloat16")

    def _get_training_components(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        grad_sample_mode: str,
        dtype: torch.dtype,
    ):

        model = model.to(self.device)
        model = model.to(dtype)
        model = model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        privacy_engine = PrivacyEngine()

        # Make the model private with the specified precision
        if grad_sample_mode in ["hooks", "functorch", "ew"]:
            model, optimizer, dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=1,
                max_grad_norm=1,
                grad_sample_mode=grad_sample_mode,
                poisson_sampling=False,
            )
        elif grad_sample_mode == "ghost":
            model, optimizer, criterion, dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                criterion=criterion,
                max_grad_norm=1,
                noise_multiplier=1,
                grad_sample_mode="ghost",
                poisson_sampling=False,
            )

        return model, optimizer, criterion, dataloader, privacy_engine

    def _train_mixed_precision(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        dtype: torch.dtype,
        grad_sample_mode: str = "hooks",
    ):

        model, optimizer, criterion, dataloader, privacy_engine = (
            self._get_training_components(
                model, dataloader, grad_sample_mode, dtype=torch.float32
            )
        )
        # model weights should be in high precision (fp32)
        for p in model.parameters():
            self.assertTrue(p.dtype == torch.float32)

        for batch in dataloader:
            x, y = batch
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=dtype):
                outputs = model(x)
                self.assertTrue(outputs.dtype == dtype)

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()

            # the gradients should have been cast up to high precision (fp32)
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(p.grad.dtype == torch.float32)

            optimizer.step()

        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        optimizer.zero_grad()
        self._test_after_trainig(epsilon, model, x)

    def _train_low_precision(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        dtype: torch.dtype,
        grad_sample_mode: str = "hooks",
    ):

        model, optimizer, criterion, dataloader, privacy_engine = (
            self._get_training_components(
                model, dataloader, grad_sample_mode, dtype=dtype
            )
        )

        for p in model.parameters():
            self.assertTrue(p.dtype == dtype)

        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            if x.is_floating_point():  # for embedding layers, keep input as int
                x = x.to(dtype)
            outputs = model(x)
            self.assertTrue(outputs.dtype == dtype)

            loss = criterion(outputs, y)
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(p.grad.dtype == dtype)

            if grad_sample_mode == "ghost":
                per_sample_norms = model.get_norm_sample()
                self.assertTrue(per_sample_norms.dtype == dtype)

            optimizer.step()

        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        optimizer.zero_grad()
        self._test_after_trainig(epsilon, model, x)

    def _test_after_trainig(self, epsilon, model, x):
        self.assertTrue(epsilon > 0)
        model.eval()
        output = model(x)
        self.assertEqual(output.shape[1], self.output_dim)
        model.train()

    def _test_precision_training(
        self,
        model_class: Type[nn.Module],
        model_kwargs: Optional[Dict] = None,
    ):
        """Test training a model with different precision types"""
        if model_kwargs is None:
            model_kwargs = {}

        # Create random data
        dataloader, _ = create_random_data(
            model_class,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_batches=self.num_batches,
            seq_len=self.seq_len,
        )

        low_precision_type = [torch.float32, torch.float16]
        if self.bf16_supported:
            low_precision_type.append(torch.bfloat16)

        # Test with low precision
        for grad_sample_mode in ["ew", "functorch", "hooks", "ghost"]:
            for dtype in low_precision_type:
                # skip test for models with layers not supported by ew
                if grad_sample_mode == "ew" and model_class in [
                    SimpleLinearModel,
                    EmbeddingModel,
                    EmbeddingBagModel,
                    ComplexModel,
                ]:
                    continue
                # functorch does not support EmbeddingBagModel
                if grad_sample_mode == "functorch" and model_class == EmbeddingBagModel:
                    continue
                print(
                    f"Testing {model_class.__name__} model with low {dtype} precision and grad sample mode {grad_sample_mode}"
                )
                self._train_low_precision(
                    model=model_class(**model_kwargs),  # Create a fresh model
                    dataloader=dataloader,
                    dtype=dtype,
                    grad_sample_mode=grad_sample_mode,
                )

        # Test mixed FP32 + BF16/FP16
        for grad_sample_mode in ["functorch", "hooks", "ghost"]:
            for dtype in low_precision_type:
                if grad_sample_mode == "functorch" and model_class == EmbeddingBagModel:
                    continue
                print(
                    f"Testing {model_class.__name__} with mixed FP32 + {dtype} precision and grad sample mode {grad_sample_mode}"
                )
                self._train_mixed_precision(
                    model=model_class(**model_kwargs),  # Create a fresh model
                    dataloader=dataloader,
                    dtype=dtype,
                    grad_sample_mode=grad_sample_mode,
                )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_conv2d_model_precision(self):
        """Test mixed precision training with 2D convolutional layer"""
        self._test_precision_training(
            model_class=Conv2DModel,
            model_kwargs={
                "input_channels": 3,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_conv3d_model_precision(self):
        """Test mixed precision training with 3D convolutional layer"""
        self._test_precision_training(
            model_class=Conv3DModel,
            model_kwargs={
                "input_channels": 3,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_rnn_model_precision(self):
        """Test mixed precision training with RNN layers"""
        self._test_precision_training(
            model_class=RNNModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_attention_model_precision(self):
        """Test mixed precision training with attention layers"""
        self._test_precision_training(
            model_class=AttentionModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_heads": 4,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_complex_model_precision(self):
        """Test mixed precision training with a complex model combining multiple layer types"""
        self._test_precision_training(
            model_class=ComplexModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "seq_len": self.seq_len,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_conv1d_model_precision(self):
        """Test mixed precision training with 1D convolutional layer"""
        self._test_precision_training(
            model_class=Conv1DModel,
            model_kwargs={
                "input_channels": 3,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_linear_model_precision(self):
        """Test mixed precision training with a simple linear model"""
        self._test_precision_training(
            model_class=SimpleLinearModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_embedding_model_precision(self):
        """Test mixed precision training with embedding layer"""
        self._test_precision_training(
            model_class=EmbeddingModel,
            model_kwargs={
                "vocab_size": 100,
                "embedding_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_embedding_bag_model_precision(self):
        """Test mixed precision training with embedding bag layer"""
        self._test_precision_training(
            model_class=EmbeddingBagModel,
            model_kwargs={
                "vocab_size": 100,
                "embedding_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_linear_relu_model_precision(self):
        """Test mixed precision training with a simple linear-relu model"""
        self._test_precision_training(
            model_class=SimpleLinearReluModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )


if __name__ == "__main__":
    unittest.main()
