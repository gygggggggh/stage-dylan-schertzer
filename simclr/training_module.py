from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
import pytorch_lightning as pl
import torchvision
import torch.nn as nn
import torch


class SimCLRModule(pl.LightningModule):
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
        dataset = torchvision.datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=128, num_workers=4, drop_last=True, shuffle=True
        )
        return dataloader
