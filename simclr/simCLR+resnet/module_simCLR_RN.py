import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead


class SimCLRModuleRN(pl.LightningModule):
    def __init__(self, input_channels=12, hidden_dim=512, output_dim=1024, learning_rate=0.002):
        super().__init__()
        self.save_hyperparameters()

        # Create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, output_dim)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = x.unsqueeze(1).transpose(1, 3)
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

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        return [optimizer], [scheduler]

    def get_h(self, x):
        x = x.unsqueeze(1).transpose(1, 3)
        h = self.backbone(x)
        return h.view(h.size(0), -1)
