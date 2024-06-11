import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from dataset import NPYDataset
from training_module import SimCLRModule
import torch


def main():
    torch.cuda.empty_cache()
    # Load your numpy arrays
    x_train = np.load(
        "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/x_train.npy"
    )
    y_train = np.load(
        "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/y_train.npy"
    )
    x_test = np.load(
        "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/x_test.npy"
    )
    y_test = np.load(
        "/home/sacha/Desktop/stage-dylan-schertzer/stage_dylan/visulisation/npy/y_test.npy"
    )

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
    trainer = Trainer(
        max_epochs=10,
    )
    trainer.fit(model, train_loader, test_loader)

    # Save the model
    torch.save(model.state_dict(), "model.pth")

    # Load the model
    model = SimCLRModule()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Test the model
    test_accuracy = trainer.test(model, test_loader)
    print(f"Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
