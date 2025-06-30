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

from typing import Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Model with RNN layers (LSTM, GRU, and RNN) and LayerNorm between them"""

    def __init__(self, input_dim=16, hidden_dim=32, output_dim=10):
        super().__init__()
        self.lstm = DPLSTM(input_dim, hidden_dim, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gru = DPGRU(hidden_dim, hidden_dim, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.rnn = DPRNN(hidden_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.ln1(x)
        x, _ = self.gru(x)
        x = self.ln2(x)
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
