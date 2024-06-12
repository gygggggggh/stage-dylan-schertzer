import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import NPYDataset
from training_module import SimCLRModule
import torch
import logging


logging.basicConfig(filename="output.log", level=logging.INFO)

def get20randomSeeds():
    seeds = []
    for i in range(20):
        seeds.append(np.random.randint(0, 1e9))
    return seeds

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

def main(seed):
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

    n_values = [5,10,20,50,100]

    # Create the SimCLR model
    model = SimCLRModule()
    torch.cuda.empty_cache()
    acuracy = []
    # Train the model
    for n in n_values:
        x_train_selected, y_train_selected = select_samples_per_class(x_train, y_train, n)
        train_dataset = NPYDataset(x_train_selected, y_train_selected)
        train_loader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=True,
            num_workers=19,
        )
        for i in range(20):
            torch.manual_seed(seed[i])
            pl.seed_everything(seed[i])
            trainer = pl.Trainer(max_epochs=100)
            trainer.fit(model, train_loader, test_loader)
            test_results = trainer.test(model, test_loader)
            test_accuracy = test_results[0]['test_acc']
            acuracy.append(test_accuracy)
        logging.info(f"the accuracy for n = {n} is {sum(acuracy)/len(acuracy)}")
        acuracy = []




if __name__ == "__main__":
    main(get20randomSeeds())

