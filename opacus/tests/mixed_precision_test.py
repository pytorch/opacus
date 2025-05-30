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
        x_reshaped = x.unsqueeze(-1)  # [batch_size, hidden_dim, 1]
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
        x = F.avg_pool1d(
            x, x.size(-1)
        )  # Global average pooling over sequence dimension
        x = x.squeeze(-1)  # Remove the sequence dimension
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
        x = F.avg_pool2d(
            x, (x.size(-2), x.size(-1))
        )  # Global average pooling over spatial dimensions
        x = x.squeeze(-1).squeeze(-1)  # Remove the spatial dimensions
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
        x = F.avg_pool3d(
            x, (x.size(-3), x.size(-2), x.size(-1))
        )  # Global average pooling over volumetric dimensions
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove the volumetric dimensions
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
        # Take the last output
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
        x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
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
        # For EmbeddingBag, we need to flatten the input and provide offsets
        batch_size = x.size(0)
        seq_len = x.size(1)
        x_flat = x.reshape(-1)  # Flatten the input
        offsets = torch.arange(0, batch_size * seq_len, seq_len, device=x.device)
        x = self.embedding_bag(x_flat, offsets)  # [batch_size, embedding_dim]

        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.fc(x)
        return x


class ComplexModel(nn.Module):
    """Model combining multiple layer types"""

    def __init__(self, input_dim=16, hidden_dim=32, output_dim=10, seq_len=10):
        super().__init__()
        self.seq_len = seq_len

        # Linear layers
        self.fc_in = nn.Linear(input_dim, hidden_dim)

        # Convolutional layers
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # RNN layers
        self.lstm = DPLSTM(hidden_dim, hidden_dim, batch_first=True)

        # Attention layers
        self.attention = DPMultiheadAttention(hidden_dim, 4)

        # Output layers
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # Linear transformation
        x = self.fc_in(x)  # [batch_size, seq_len, hidden_dim]

        # Convolutional processing
        x_conv = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        x_conv = self.conv1d(x_conv)  # [batch_size, hidden_dim, seq_len]
        x_conv = x_conv.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]

        # Add residual connection
        x = x + x_conv

        # RNN processing
        x_rnn, _ = self.lstm(x)

        # Add residual connection
        x = x + x_rnn

        # Self-attention
        x_attn, _ = self.attention(x, x, x)

        # Add residual connection
        x = x + x_attn

        # Take the mean over the sequence dimension
        x = torch.mean(x, dim=1)

        # Output layer
        x = self.fc_out(x)
        return x


class MixedPrecisionTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def setUp(self):
        self.batch_size = 2
        self.input_dim = 4
        self.hidden_dim = 8
        self.output_dim = 4
        self.seq_len = 4
        self.num_batches = 2

        # Since this test only runs when CUDA is available, always use CUDA
        self.device = torch.device("cuda")

        # Check if bfloat16 is supported
        self.bf16_supported = hasattr(torch, "bfloat16")

    def _create_random_data(
        self, model_type: str
    ) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
        """Create random data for different model types"""
        if model_type in ["linear", "simple_linear"]:
            # Data for SimpleLinearModel
            x = torch.randn(
                self.batch_size * self.num_batches, self.input_dim, device=self.device
            )
            y = torch.randint(
                0,
                self.output_dim,
                (self.batch_size * self.num_batches,),
                device=self.device,
            )
            dataset = TensorDataset(x, y)

        elif model_type == "conv1d":
            # Data for Conv1DModel
            x = torch.randn(
                self.batch_size * self.num_batches, 3, 16, device=self.device
            )
            y = torch.randint(
                0,
                self.output_dim,
                (self.batch_size * self.num_batches,),
                device=self.device,
            )
            dataset = TensorDataset(x, y)

        elif model_type == "conv2d":
            # Data for Conv2DModel
            x = torch.randn(
                self.batch_size * self.num_batches, 3, 16, 16, device=self.device
            )
            y = torch.randint(
                0,
                self.output_dim,
                (self.batch_size * self.num_batches,),
                device=self.device,
            )
            dataset = TensorDataset(x, y)

        elif model_type == "conv3d":
            # Data for Conv3DModel
            x = torch.randn(
                self.batch_size * self.num_batches, 3, 8, 8, 8, device=self.device
            )
            y = torch.randint(
                0,
                self.output_dim,
                (self.batch_size * self.num_batches,),
                device=self.device,
            )
            dataset = TensorDataset(x, y)

        elif model_type == "rnn":
            # Data for RNNModel
            x = torch.randn(
                self.batch_size * self.num_batches,
                self.seq_len,
                self.input_dim,
                device=self.device,
            )
            y = torch.randint(
                0,
                self.output_dim,
                (self.batch_size * self.num_batches,),
                device=self.device,
            )
            dataset = TensorDataset(x, y)

        elif model_type == "attention":
            # Data for AttentionModel
            x = torch.randn(
                self.batch_size * self.num_batches,
                self.seq_len,
                self.input_dim,
                device=self.device,
            )
            y = torch.randint(
                0,
                self.output_dim,
                (self.batch_size * self.num_batches,),
                device=self.device,
            )
            dataset = TensorDataset(x, y)

        elif model_type == "complex":
            # Data for ComplexModel
            x = torch.randn(
                self.batch_size * self.num_batches,
                self.seq_len,
                self.input_dim,
                device=self.device,
            )
            y = torch.randint(
                0,
                self.output_dim,
                (self.batch_size * self.num_batches,),
                device=self.device,
            )
            dataset = TensorDataset(x, y)

        elif model_type == "embedding":
            # Data for EmbeddingModel
            vocab_size = 100
            seq_len = 10
            x = torch.randint(
                0,
                vocab_size,
                (self.batch_size * self.num_batches, seq_len),
                device=self.device,
            )
            y = torch.randint(
                0,
                self.output_dim,
                (self.batch_size * self.num_batches,),
                device=self.device,
            )
            dataset = TensorDataset(x, y)

        # We've incorporated instance norm and rms norm into existing models

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        # Get a sample batch for testing
        sample_batch = next(iter(dataloader))

        return dataloader, *sample_batch

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
                if p.grad != None:
                    self.assertTrue(p.grad.dtype == torch.float32)

            optimizer.step()

        # Return privacy budget spent
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
                if p.grad != None:
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
        model_type: str,
        model_kwargs: Optional[Dict] = None,
    ):
        """Test training a model with different precision types"""
        if model_kwargs is None:
            model_kwargs = {}

        # Create random data
        dataloader, _, _ = self._create_random_data(model_type)

        low_precision_type = [torch.float16]
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
                    f"Testing {model_type} model with low {dtype} precision and grad sample mode {grad_sample_mode}"
                )
                self._train_low_precision(
                    model=model_class(**model_kwargs),  # Create a fresh model
                    dataloader=dataloader,
                    dtype=dtype,
                    grad_sample_mode=grad_sample_mode,
                )

        # Test mixed FP32 + BF16/FP16
        for grad_sample_mode in ["hooks", "ghost"]:
            for dtype in low_precision_type:
                print(
                    f"Testing {model_class} with mixed FP32 + {dtype} precision and grad sample mode {grad_sample_mode}"
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
            model_type="conv2d",
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
            model_type="conv3d",
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
            model_type="rnn",
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
            model_type="attention",
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
            model_type="complex",
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
            model_type="conv1d",
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
            model_type="linear",
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
            model_type="embedding",
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
            model_type="embedding",  # Reuse the same data generation as embedding
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
            model_type="simple_linear",
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    # TODO: test distributed training too with mixed precision


if __name__ == "__main__":
    unittest.main()
