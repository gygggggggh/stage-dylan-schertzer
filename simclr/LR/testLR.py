# test the model LR

import numpy as np
from sklearn.metrics import accuracy_score
from cuml.linear_model import LogisticRegression
import logging
import pickle

logging.basicConfig(filename="testLR.log", level=logging.INFO)
logging.info("Logistic Regression")

def get_random_seeds(num_seeds: int = 20, seed_range: int = 1e9) -> list:
    """Generate a list of random seeds."""
    return [np.random.randint(0, seed_range) for _ in range(num_seeds)]


def select_samples_per_class(x: np.ndarray, y: np.ndarray, n_samples: int) -> tuple:
    """Select a fixed number of samples per class."""
    unique_classes = np.unique(y)
    selected_x, selected_y = [], []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        selected_indices = (
            cls_indices
            if len(cls_indices) <= n_samples
            else np.random.choice(cls_indices, n_samples, replace=False)
        )
        selected_x.append(x[selected_indices])
        selected_y.append(y[selected_indices])

    return np.concatenate(selected_x), np.concatenate(selected_y)


def load_data() -> tuple:
    """Load training and testing data."""
    x_train = np.load("stage_dylan/visulisation/npy/x_train.npy")
    y_train = np.load("stage_dylan/visulisation/npy/y_train.npy")
    x_test = np.load("stage_dylan/visulisation/npy/x_test.npy")
    y_test = np.load("stage_dylan/visulisation/npy/y_test.npy")

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test



def evaluate_model(x_train, y_train, x_test, y_test, n_values, seeds):
    accuracies = []
    accuracies_majority = []    

    for n in n_values:
        for seed in seeds:
            x_train_selected, y_train_selected = select_samples_per_class(x_train, y_train, n)
            model = pickle.load(open('simclr/LR/LR_model.pkl', 'rb'))
            model.fit(x_train.reshape(x_train.shape[0], -1), y_train)
            y_pred = model.predict(x_test.reshape(x_test.shape[0], -1))
            accuracy1 = accuracy_score(y_test, y_pred)

            x_train_reshaped = x_train_selected.reshape(-1, 60, 12)
            x_train_repeated = np.repeat(x_train_reshaped, 100, axis=0) 
            x_test_reshaped = x_test.reshape(-1, 60, 12)
            y_train_repeated = np.repeat(y_train, 500)

            model = pickle.load(open('simclr/LR/LR_model.pkl', 'rb'))

            model.fit(x_train_repeated.reshape(
                x_train_repeated.shape[0], -1), y_train_repeated)

            y_pred = model.predict(x_test_reshaped.reshape(x_test_reshaped.shape[0], -1))
            y_pred = y_pred.reshape(-1, 100)
            y_pred_majority_vote = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred)
            accuracies_majority.append(accuracy_score(y_test, y_pred_majority_vote))
        logging.info(f"The Accuracy for n={n}: {np.mean(accuracies):.4f}")
        logging.info(f"Majority Vote Accuracy for n={n}: {np.mean(accuracies_majority):.4f}")

        return accuracies


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    seeds = get_random_seeds()
    n_values = [5, 10, 50, 100]
    evaluate_model(x_train, y_train, x_test, y_test, n_values, seeds)
    print("Done")

