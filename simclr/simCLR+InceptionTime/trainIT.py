import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import NPYDataset,NPYDatasetAll
from module_simCLR_IT import SimCLRModuleIT
import torch
from typing import Dict, Any

CONFIG: Dict[str, Any] = {
    "x_train_path": "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/x_train.npy",
    "y_train_path": "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/y_train.npy",
    "x_test_path": "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/x_test.npy",
    "y_test_path": "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/y_test.npy",
    "model_save_path": "simclr/simCLR+InceptionTime/simCLR+IT.pth",
    "batch_size": 256,
    "num_workers": 19,
    "max_epochs": 100,
}

def main(config: Dict[str, Any]) -> None:
    """
    Main training function.
    """
    # Load the data
    x_train = np.load(config["x_train_path"])
    y_train = np.load(config["y_train_path"])
    x_test = np.load(config["x_test_path"])
    y_test = np.load(config["y_test_path"])

    # Create the SimCLR model
    model = SimCLRModuleIT()
    torch.cuda.empty_cache()

    # Create the data loaders
    train_dataset = NPYDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_dataset = NPYDatasetAll(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], drop_last=False)

    # Train the model
    trainer = pl.Trainer(max_epochs=config["max_epochs"], log_every_n_steps=1)
    trainer.fit(model, train_loader, test_loader)

    # Save the model
    model.eval()
    torch.save(model.state_dict(), config["model_save_path"])

if __name__ == "__main__":
    main(CONFIG)