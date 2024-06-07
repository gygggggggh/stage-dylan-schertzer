# %%
import time
import numpy as np
import os
from aeon.classification.deep_learning import InceptionTimeClassifier
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
import logging


# Set up logging
logging.basicConfig(filename="output.log", level=logging.INFO)

# Environment setup
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load data
x_train = np.load("stage_dylan/visulisation/npy/x_train.npy")
y_train = np.load("stage_dylan/visulisation/npy/y_train.npy")
x_test = np.load("stage_dylan/visulisation/npy/x_test.npy")
y_test = np.load("stage_dylan/visulisation/npy/y_test.npy")


# %%
def _20_random_seeds():
    list = np.random.randint(low=0, high=1e9, size=20)
    logging.info(f"20 random seeds: {list}")
    return list


# %%


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
    model = InceptionTimeClassifier(n_epochs=50, batch_size=512, verbose=1)
    start_time = time.time()
    model.fit(x_train, y_train)
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


def fit_and_score_majority_vote(x_train, y_train, x_test, y_test, N):
    model = InceptionTimeClassifier(n_epochs=50, batch_size=512, verbose=1)
    start_time = time.time()
    model.fit(x_train, y_train)
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
    samples_per_batch = x_train.shape[0] // len(dl)
    y_pred = y_pred.reshape(-1, samples_per_batch, 100)
    y_pred_majority_vote = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred
    )
    y_pred_majority_vote = y_pred_majority_vote.repeat(
        len(y_test) // len(y_pred_majority_vote // 100)
    )

    accuracy = accuracy_score(y_test, y_pred_majority_vote)
    return accuracy


def fit_and_score_20times(x_train, y_train, x_test, y_test, N, seed):
    accuracy_list = []
    for SEED in seed:
        model = InceptionTimeClassifier(
            n_epochs=50, batch_size=512, verbose=1, random_state=SEED
        )
        start_time = time.time()
        model.fit(x_train, y_train)
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
        y_pred = y_pred.reshape(-1, 100)
        y_pred = y_pred.repeat(len(y_test) // len(y_pred) // 100)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)
    average_accuracy = sum(accuracy_list) / len(accuracy_list)

    return average_accuracy


def fit_and_score_majority_vote_20times(x_train, y_train, x_test, y_test, N, seed):
    accuracy_list = []
    for SEED in seed:
        model = InceptionTimeClassifier(
            n_epochs=50, batch_size=512, verbose=1, random_state=SEED
        )
        start_time = time.time()
        model.fit(x_train, y_train)
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
        y_pred = y_pred.reshape(-1, 100)
        samples_per_batch = dl.batch_size
        y_pred = y_pred.reshape(-1, 100)
        y_pred_majority_vote = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred
        )
        y_pred_majority_vote = y_pred_majority_vote.repeat(
            len(y_test) // len(y_pred_majority_vote)
        )
        accuracy = accuracy_score(y_test, y_pred_majority_vote)
        accuracy_list.append(accuracy)
    average_accuracy = sum(accuracy_list) / len(accuracy_list)

    return average_accuracy


def main():
    total_time = time.time()
    SEED = _20_random_seeds()
    n_values = [5]
    results = select_and_reshape(x_train, y_train, n_values)

    for n, (x_train_reshaped, y_train_reshaped) in results.items():
        logging.info(f"\nFor N = {n}:")
        logging.info(
            f"x_train.shape: {x_train_reshaped.shape}, y_train.shape: {y_train_reshaped.shape}"
        )

        accuracy = fit_and_score_20times(
            x_train_reshaped, y_train_reshaped, x_test, y_test, n, SEED
        )
        logging.info(f"Accuracy on test data: {accuracy:.4f}")

    accuracy_majority_vote = fit_and_score_majority_vote_20times(
        x_train_reshaped, y_train_reshaped, x_test, y_test, n, SEED
    )
    logging.info(f"Accuracy on test data with majority vote: {accuracy_majority_vote}")
    logging.info(f"Total time: {time.time() - total_time:.2f} seconds")


if __name__ == "__main__":
    main()
