import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from dataset import NPYDataset
from training_module import SimCLRModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os

def _20_random_seeds():
    list = np.random.randint(low=0, high=1e9, size=20)
    logging.info(f"20 random seeds: {list}")
    return list

def select_samples_per_class(x, y, n_samples):
    unique_classes = np.unique(y)
    new_x, new_y = [], []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        selected_indices = (
            cls_indices
            if len(cls_indices) <= n_samples
            else np.random.choice(cls_indices, n_samples, replace=False)
        )
        new_x.append(x[selected_indices])
        new_y.append(y[selected_indices])

    return np.concatenate(new_x), np.concatenate(new_y)



def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load your numpy arrays
    x_train = np.load("/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/x_train.npy")
    y_train = np.load("/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/y_train.npy")
    x_test = np.load("/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/x_test.npy")
    y_test = np.load("/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/y_test.npy")

    # Print shapes of the data arrays
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Create the datasets
    train_dataset = NPYDataset(x_train, y_train)
    test_dataset = NPYDataset(x_test, y_test)

    # Create the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=19,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False, num_workers=19, drop_last=False
    )

    # Create the SimCLR model
    model = SimCLRModule()
    torch.cuda.empty_cache()
    # Train the model
    for i in range(20):
        print(f"Training model with seed {i}")
        torch.manual_seed(i)
    trainer = Trainer(max_epochs=100, enable_model_summary=True)
    trainer.fit(model, train_loader, test_loader)

    # Save the model
    torch.save(model.state_dict(), "model.pth")


    # Load the model
    model = SimCLRModule()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Test the model
    with torch.no_grad():
        test_accuracy = trainer.test(model, test_loader)
    print(f"Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()

