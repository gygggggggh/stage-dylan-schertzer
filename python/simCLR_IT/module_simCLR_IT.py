import pytorch_lightning as pl
import torch
import torch.nn as nn
from python.simCLR_IT.inception import InceptionBlock
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead


class Flatten(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class SimCLRModuleIT(pl.LightningModule):
    def __init__(self, learning_rate=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = nn.Sequential(
            InceptionBlock(
                in_channels=12,
                n_filters=256,
                kernel_sizes=[9, 19, 39],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU(),
            ),
            InceptionBlock(
                in_channels=256 * 4,
                n_filters=256,
                kernel_sizes=[9, 19, 39],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU(),
            ),
            InceptionBlock(
                in_channels=256 * 4,
                n_filters=256,
                kernel_sizes=[9, 19, 39],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU(),
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=256 * 4),
        )
        self.projection_head = SimCLRProjectionHead(256 * 4, 256 * 4, 128)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.backbone(x)
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
        x = x.to(self.device)
        x = x.transpose(1, 2)
        return self.backbone(x)
