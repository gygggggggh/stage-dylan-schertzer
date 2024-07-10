import logging
from typing import Any, Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from python.dataset import NPYDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def load_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data."""
    try:
        x_train = np.load(config["x_train_path"])
        y_train = np.load(config["y_train_path"])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading data: {e}")

    return x_train, y_train


def create_data_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation."""
    train_dataset = NPYDataset(x_train, y_train)
    val_dataset = NPYDataset(x_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=True,
    )

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, config):
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints-{config['model_type']}",
        filename=f"simclr-{config['model_type']}-{{epoch:02d}}",
        save_top_k=config["max_epochs"],
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        devices=1,
    )

    trainer.fit(model, train_loader, val_loader)
    return model


def main(config: Dict[str, Any], model_class) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pl.seed_everything(42)

    x_train, y_train = load_data(config)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=config["val_split"], random_state=42
    )
    train_loader, val_loader = create_data_loaders(
        x_train, y_train, x_val, y_val, config
    )

    model = model_class(learning_rate=config["learning_rate"]).to(device)

    train_model(model, train_loader, val_loader, config)
