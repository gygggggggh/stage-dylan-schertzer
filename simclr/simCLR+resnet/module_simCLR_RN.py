# training_module.py
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np
import torchvision




class SimCLRModuleRN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        resnet.conv1 = nn.Conv2d(12,64, kernel_size=7, stride=2, padding=3, bias=False)  
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = x.transpose(1, 3)
        h = self.backbone(x)
        h = h.view(h.size(0), -1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch: tuple) -> torch.Tensor:
        (x0, x1) = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 2)
        return [optim], [scheduler]

    def train_dataloader(self):
        pass

    def get_h(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()
        if x.dim() == 4:
            if x.size(1) != 12:
                x = x.permute(0, 3, 1, 2)  # Change the shape to [batch, channels, height, width]
        elif x.dim() == 3:
            x = x.unsqueeze(1) 
            x = x.permute(0, 3, 1, 2)  
        
        h = self.backbone(x)
        return h
