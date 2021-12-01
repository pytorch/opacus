#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Optional

import pytorch_lightning as pl
import torch
from opacus.data_loader import DPDataLoader


class DPLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.generator = generator

    def prepare_data(self) -> None:
        self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.datamodule.setup(stage)

    def train_dataloader(self):
        dataloader = self.datamodule.train_dataloader()
        return DPDataLoader.from_data_loader(dataloader, distributed=False)

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def predict_dataloader(self):
        return self.datamodule.predict_dataloader()

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        return self.datamodule.transfer_batch_to_device(batch, device, dataloader_idx)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.datamodule.on_before_batch_transfer(batch, dataloader_idx)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.datamodule.on_after_batch_transfer(batch, dataloader_idx)
