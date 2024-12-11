import pytorch_lightning as pl
from torch import nn
import torch

class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method must be implemented by subclass.")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("training_step must be implemented by subclass.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("validation_step must be implemented by subclass.")

    def configure_optimizers(self):
        raise NotImplementedError("configure_optimizers must be implemented by subclass.")
