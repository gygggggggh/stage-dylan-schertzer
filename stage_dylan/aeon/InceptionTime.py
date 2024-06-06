import time
import numpy as np
import os
from aeon.classification.deep_learning import InceptionTimeClassifier
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
import sys


# Environment setup
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load data
x_train = np.load("stage_dylan/visulisation/npy/x_train.npy")
y_train = np.load("stage_dylan/visulisation/npy/y_train.npy")
x_test = np.load("stage_dylan/visulisation/npy/x_test.npy")
y_test = np.load("stage_dylan/visulisation/npy/y_test.npy")


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


def reshape_data(x, y, repeat_factor=100):
    x = x.reshape(-1, 60, 12).transpose(0, 2, 1)
    y = y.repeat(repeat_factor)
    return x, y


def select_and_reshape(x, y, n_values):
    reshaped_results = {}
    for n in n_values:
        selected_x, selected_y = select_samples_per_class(x, y, n)
        reshaped_results[n] = reshape_data(selected_x, selected_y)
    return reshaped_results


def fit_and_score(x_train, y_train, x_test, y_test, N):
    model = InceptionTimeClassifier(n_epochs=100, batch_size=512, verbose=0)
    start_time = time.time()
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    model.fit(x_train, y_train)
    sys.stdout = original_stdout
    training_time = time.time() - start_time

    print(f"Training time: {training_time:.2f} seconds")
    dl_list = []
    dl = DataLoader(
        torch.tensor(x_train), batch_size=256, shuffle=False, drop_last=False
    )

    start_time = time.time()

    for x in dl:
        dl_list.append(model.predict(x.numpy()))
    y_pred = np.concatenate(dl_list)
    y_pred = np.concatenate(dl_list)
    y_pred = y_pred.reshape(-1, 100)
    y_pred = y_pred.repeat(len(y_test) // len(y_pred) // 100)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def main():
    with open("output.txt", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        n_values = [5, 10]
        results = select_and_reshape(x_train, y_train, n_values)

        for n, (x_train_reshaped, y_train_reshaped) in results.items():
            print(f"\nFor N = {n}:")
            print(
                f"x_train.shape: {x_train_reshaped.shape}, y_train.shape: {y_train_reshaped.shape}"
            )

            time = fit_and_score(
                x_train_reshaped,
                y_train_reshaped,
                x_test.reshape(-1, 60, 12).transpose(0, 2, 1),
                y_test.repeat(100),
                n,
            )
        print(f"Accuracy on test data: {time:.4f}")
        print("Done!")

        sys.stdout = original_stdout  # Reset the standard output to its original value


if __name__ == "__main__":
    main()
