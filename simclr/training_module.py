# training_module.py
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
import pytorch_lightning as pl
import torch.nn as nn
import torch
from inception import Inception


class SimCLRModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = Inception(
            in_channels=60, n_filters=32
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1536, 128)
        hidden_dim = 128
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):

        h = self.backbone(x)
        h = self.flatten(h)
        h = self.linear(h)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1) = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 2)
        return [optim], [scheduler]

    def train_dataloader(self):
        pass

    def test_step(self, batch, batch_idx):
        x0, x1 = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        acc = (z0.argmax(dim=1) == z1.argmax(dim=1)).float().mean()
        self.log("test_acc",acc)
        return loss
