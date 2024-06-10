import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from dataset import NPYDataset
from training_module import SimCLRModule
import torch
def main():
    torch.cuda.empty_cache()
    # Load your numpy arrays
    x_train = np.load("stage_dylan/visulisation/npy/x_train.npy")
    y_train = np.load("stage_dylan/visulisation/npy/y_train.npy")
    x_test = np.load("stage_dylan/visulisation/npy/x_test.npy")
    y_test = np.load("stage_dylan/visulisation/npy/y_test.npy")

    # Print shapes of the data arrays
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Create the datasets
    train_dataset = NPYDataset(x_train, y_train)
    test_dataset = NPYDataset(x_test, y_test)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, drop_last=False)

    # Create the SimCLR model
    model = SimCLRModule()
    torch.cuda.empty_cache()
    # Train the model
    trainer = Trainer(max_epochs=5)
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