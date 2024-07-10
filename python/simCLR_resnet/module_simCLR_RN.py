import pytorch_lightning as pl
import torch
import torch.nn as nn
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from tsai.models.ResNet import ResNet as tsResNet


class SimCLRModuleRNAll(pl.LightningModule):
    def __init__(
        self, input_channels=12, hidden_dim=1024, output_dim=128, learning_rate=0.02
    ):
        super().__init__()
        self.save_hyperparameters()

        resnet = tsResNet(input_channels, 1)
        embedding_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection_head = SimCLRProjectionHead(
            embedding_dim, embedding_dim, output_dim
        )
        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.size(0), -1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1) = batch
        z0 = self(x0)
        z1 = self(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x0, x1) = batch
        z0 = self(x0)
        z1 = self(x1)
        loss = self.criterion(z0, z1)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]

    def get_h(self, x):
        x = x.transpose(1, 2)
        h = self.backbone(x)
        return h.view(h.size(0), -1)


class SimCLRModuleRN(pl.LightningModule):
    def __init__(
        self, input_channels=12, hidden_dim=1024, output_dim=128, learning_rate=0.02
    ):
        super().__init__()
        self.save_hyperparameters()

        resnet = tsResNet(input_channels, 1)
        embedding_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection_head = SimCLRProjectionHead(
            embedding_dim, embedding_dim, output_dim
        )
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.backbone(x)
        h = h.view(h.size(0), -1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1) = batch
        z0 = self(x0)
        z1 = self(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x0, x1) = batch
        z0 = self(x0)
        z1 = self(x1)
        loss = self.criterion(z0, z1)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]

    def get_h(self, x):
        x = x.unsqueeze(1).transpose(1, 3)
        h = self.backbone(x)
        return h.view(h.size(0), -1)
