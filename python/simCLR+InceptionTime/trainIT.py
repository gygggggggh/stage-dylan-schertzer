import logging
from typing import Any, Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from dataset import NPYDataset

from module_simCLR_IT import SimCLRModuleIT
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('high')

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG = {
    "x_train_path": "weights/x_train_40k.npy",
    "y_train_path": "weights/y_train_40k.npy",
    "x_test_path": "weights/x_test.npy",
    "y_test_path": "weights/y_test.npy",
    "model_save_path": "python/simCLR+InceptionTime/simCLR+IT.pth",
    "batch_size": 1024,
    "num_workers": 8,
    "max_epochs": 200,
    "learning_rate": 0.02,
    "val_split": 0.1
}


def load_data(
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training and testing data."""
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
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    train_dataset = NPYDataset(x_train, y_train)
    val_dataset = NPYDataset(x_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=True
    )

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, config):
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="simclr-it-{epoch:02d}",
        save_top_k=200,
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


def main(config: Dict[str, Any]) -> None:
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

    model = SimCLRModuleIT(learning_rate=config["learning_rate"]).to(device)

    train_model(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main(CONFIG)
    logger.info("Training completed.")
