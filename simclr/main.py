import numpy as np
from torch.utils.data import DataLoader    
import pytorch_lightning as pl
from dataset import NPYDataset
from training_module import SimCLRModule
import torch
import logging
from sklearn.metrics import accuracy_score
from cuml.linear_model import LogisticRegression

logging.basicConfig(filename="output.log", level=logging.INFO)


def get20randomSeeds():
    return [np.random.randint(0, 1e9) for _ in range(20)]


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


def main(seeds):
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

    n_values = [5,10,50,100]

    # Create the SimCLR model
    model = SimCLRModule()
    torch.cuda.empty_cache()

    for n in n_values:
        x_train_selected, y_train_selected = select_samples_per_class(
            x_train, y_train, n
        )
        logging.info(
            f"Selected data for n = {n} is of size {x_train_selected.shape} and {y_train_selected.shape}"
        )

        train_dataset = NPYDataset(x_train_selected, y_train_selected)
        train_loader = DataLoader(
            train_dataset, batch_size=512, shuffle=True, num_workers=19
        )
        test_dataset = NPYDataset.getall(x_test, y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=512, shuffle=False, num_workers=19, drop_last=False
        )

        accuracies = []
        for seed in seeds:
            torch.manual_seed(seed)
            pl.seed_everything(seed)
            model.inference = False
            trainer = pl.Trainer(max_epochs=100,log_every_n_steps=10)
            trainer.fit(model, train_loader, test_loader)

            # Switch the model to evaluation mode
            model.eval()
            model.inference = True

            # Extract H representations from the model
            H_train = []
            H_test = []
            with torch.no_grad():
                for x, y in train_loader:
                    H_train.append(model.get_h(x).cpu().numpy())
                for x, y in test_loader:
                    H_test.append(model.get_h(x).cpu().numpy())
            H_train = np.concatenate(H_train)
            H_test = np.concatenate(H_test)

            # Train a logistic regression model on top of the representations
            clf = LogisticRegression()
            clf.fit(H_train, y_train_selected)
            y_pred = clf.predict(H_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
        logging.info(f"Accuracy for n = {n}  is {sum(accuracies)/len(accuracies)}")


if __name__ == "__main__":
    seeds = get20randomSeeds()
    main(seeds)

