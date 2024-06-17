# import the simCLR model
from training_module import SimCLRModule
import torch
import logging
from sklearn.metrics import accuracy_score
from cuml.linear_model import LogisticRegression
import numpy as np
from dataset import NPYDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

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
    model.load_state_dict(torch.load("simCLR.pth"))
    model.inference = True
    accuracies = []
    acuracies_bef = []
    for n in n_values:
        for seed in seeds:
            torch.manual_seed(seed)
            pl.seed_everything(seed)
            x_train_selected, y_train_selected = select_samples_per_class(
                x_train, y_train, n
            )

            train_dataset = NPYDataset(x_train_selected, y_train_selected)
            train_loader = DataLoader(
                train_dataset, batch_size=512, shuffle=False, num_workers=19
            )
            test_dataset = NPYDataset(x_test, y_test)
            test_loader = DataLoader(
                test_dataset, batch_size=512, shuffle=False, num_workers=19, drop_last=False
            )

            H_train = []
            H_test = []
            hbef_train = []
            hbef_test = []
            with torch.no_grad():
                for x, y in train_loader:
                    #print(f"shape of x_train: {x.shape}")
                    H_train.append(model.get_h(x).cpu().numpy())
                    hbef_train.append(x.cpu().numpy())
                for x, y in test_loader:
                    #print(f"shape of x_test: {x.shape}")
                    H_test.append(model.get_h(x).cpu().numpy())
                    hbef_test.append(x.cpu().numpy())
            H_train = np.concatenate(H_train)
            H_test = np.concatenate(H_test)
            hbef_train = np.concatenate(hbef_train)
            hbef_test = np.concatenate(hbef_test)
            print(H_train.shape, H_test.shape, hbef_train.shape, hbef_test.shape)
            # Train a logistic regression model on top of the representations
            clf = LogisticRegression(max_iter=1000)
            clf.fit(H_train, y_train_selected)
            y_pred = clf.predict(H_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            # Train a logistic regression model on top of the representations befoore the model
            clf = LogisticRegression(max_iter=1000)
            clf.fit(hbef_train.reshape(hbef_train.shape[0], -1), y_train_selected)
            y_pred = clf.predict(hbef_test.reshape(hbef_test.shape[0], -1))
            acc = accuracy_score(y_test, y_pred)
            acuracies_bef.append(acc)
        logging.info(f" Before Accuracy for n = {n}  is {sum(acuracies_bef)/len(acuracies_bef):.4f}")
        logging.info(f"Accuracy for n = {n}  is {sum(accuracies)/len(accuracies):.4f}")
        accuracies = []
        acuracies_bef = []



if __name__ == "__main__":
    seeds = get20randomSeeds()
    main(seeds)