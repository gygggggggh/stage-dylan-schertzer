import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from dataset import NPYDataset, NPYDatasetAll
from module_simCLR_RN import SimCLRModuleRN


CONFIG = {
    "x_train_path": "weights/x_train_40k.npy",
    "y_train_path": "weights/y_train_40k.npy",
    "x_test_path": "weights/x_test.npy",
    "y_test_path": "weights/y_test.npy",
    "model_save_path": "python/simCLR+resnet/simCLR+RN.pth",
    "batch_size": 32,
    "num_workers": 4,
    "max_epochs": 50,
    "val_split": 0.2,
}


def load_data(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training and testing data.
    """
    try:
        x_train = np.load(config["x_train_path"])
        y_train = np.load(config["y_train_path"])
        x_test = np.load(config["x_test_path"])
        y_test = np.load(config["y_test_path"])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading data: {e}")

    print(f"x_train shape before transpose: {x_train.shape}")
    print(f"x_test shape before transpose: {x_test.shape}")

    # Assuming the data is in the format (batch_size, height, width, channels)
    # and you need it to be in the format (batch_size, channels, height, width)

    print(f"x_train shape after transpose: {x_train.shape}")
    print(f"x_test shape after transpose: {x_test.shape}")

    return x_train, y_train, x_test, y_test


def create_data_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    """
    train_dataset = NPYDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    val_dataset = NPYDataset(x_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )

    test_dataset = NPYDatasetAll(x_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def main(config: Dict) -> None:
    """
    Main training function.
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    model = SimCLRModuleRN().to(device)

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Train the model
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        log_every_n_steps=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    trainer.fit(model, train_loader, val_loader)

    # Save the model
    model.eval()
    torch.save(model.state_dict(), config["model_save_path"])
    print(f"Model saved to {config['model_save_path']}")


if __name__ == "__main__":
    try:
        main(CONFIG)
    except Exception as e:
        print(f"An error occurred: {e}")
