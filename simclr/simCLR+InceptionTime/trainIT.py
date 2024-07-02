import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple
import logging
from pathlib import Path

# Import your custom modules
from module_simCLR_IT import SimCLRModuleIT
from dataset import NPYDataset, NPYDatasetAll  # Ensure these are implemented correctly


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG = {
    "x_train_path": "simclr/x_train_40k.npy",
    "y_train_path": "simclr/y_train_40k.npy",
    "x_test_path": "stage_dylan/visulisation/npy/x_test.npy",
    "y_test_path": "stage_dylan/visulisation/npy/y_test.npy",
    "model_save_path": "simclr/simCLR+InceptionTime/simCLR+IT.pth",
    "batch_size": 64,
    "num_workers": 8,
    "max_epochs": 100,
    "learning_rate": 0.002,
    "val_split": 0.2,
    "patience": 20,
}



def load_data(
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training and testing data."""
    try:
        x_train = np.load(config["x_train_path"])
        y_train = np.load(config["y_train_path"])
        x_test = np.load(config["x_test_path"])
        y_test = np.load(config["y_test_path"])
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise

    logger.info(
        f"Data shapes: x_train {x_train.shape}, y_train {y_train.shape}, x_test {x_test.shape}, y_test {y_test.shape}"
    )
    return x_train, y_train, x_test, y_test


def create_data_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    train_dataset = NPYDataset(x_train, y_train)
    val_dataset = NPYDataset(x_val, y_val)
    test_dataset = NPYDatasetAll(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, config):
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="simclr-it-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=config["patience"], verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, val_loader)
    return model

def main(config: Dict[str, Any]) -> None:
    """Main training function."""
    # Set random seeds for reproducibility
    pl.seed_everything(42)

    # Load the data
    x_train, y_train, x_test, y_test = load_data(config)

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=config["val_split"], random_state=42
    )

    # Create the data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        x_train, y_train, x_val, y_val, x_test, y_test, config
    )

    # Create the SimCLR model
    model = SimCLRModuleIT(learning_rate=config["learning_rate"])

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, config)

    # Save the model
    torch.save(trained_model.state_dict(), config["model_save_path"])
    logger.info(f"Model saved to {config['model_save_path']}")

    # Optionally, you can add an evaluation step here
    # trained_model.eval()
    # evaluate_model(trained_model, test_loader)


if __name__ == "__main__":
    main(CONFIG)
    logger.info("Training completed.")
