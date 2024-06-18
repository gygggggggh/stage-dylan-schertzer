# training_module.py
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
import pytorch_lightning as pl
import torch.nn as nn
import torch
from inception import Inception, InceptionBlock
import numpy as np
import torchvision


class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = np.transpose(out_shape)

    def forward(self, x):
        return x


InceptionTime = nn.Sequential(
    Reshape(out_shape=(60, 60)),  # bizarre mais ca marche je crois
    InceptionBlock(
        in_channels=12,
        n_filters=32,
        kernel_sizes=[5, 11, 23],
        bottleneck_channels=32,
        use_residual=True,
        activation=nn.ReLU(),
    ),
    InceptionBlock(
        in_channels=32 * 4,
        n_filters=32,
        kernel_sizes=[5, 11, 23],
        bottleneck_channels=32,
        use_residual=True,
        activation=nn.ReLU(),
    ),
    nn.AdaptiveAvgPool1d(output_size=1),
    Flatten(out_features=32 * 4 * 1),
    
)


class SimCLRModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = InceptionTime
        hidden_dim = 128
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.backbone(x)
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

    def get_h(self, x):
        x = x.transpose(1, 2)
        h = self.backbone(x)
        return h


class SimCLResnet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

        # Set requires_grad=True for all parameters in the backbone and projection_head
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.projection_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = x.view(-1, 1, 1, 1).float()
        x = x.repeat(1, 3, 1, 1)  # repeat the single channel to get 3 channels
        h = self.backbone(x)
        h = h.view(h.size(0), -1)
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

    def get_h(self, x):
        x = x.transpose(1, 2)
        h = self.backbone(x)
        return h