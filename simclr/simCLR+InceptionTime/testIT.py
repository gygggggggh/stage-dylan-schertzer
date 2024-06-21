import logging
import numpy as np
import torch
import pytorch_lightning as pl
from cuml.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from module_simCLR_IT import SimCLRModuleIT
from dataset import NPYDataset

logging.basicConfig(filename="testIT.log", level=logging.INFO)
logging.info("simCLR+InceptionTime")


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


def evaluate_model(
    model: SimCLRModuleIT,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_values: list,
    seeds: list,
) -> None:
    """Evaluate the model with different number of samples per class."""
    accuracies = []
    accuracies_majority = []

    for n in n_values:
        for seed in seeds:
            torch.manual_seed(seed)
            pl.seed_everything(seed)

            x_train_selected, y_train_selected = select_samples_per_class(
                x_train, y_train, n
            )

            train_dataset = NPYDataset(x_train_selected, y_train_selected)
            train_loader = DataLoader(
                train_dataset, batch_size=256, shuffle=False, num_workers=19
            )
            test_dataset = NPYDataset(x_test, y_test)
            test_loader = DataLoader(
                test_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=19,
                drop_last=False,
            )

            H_train, H_test = extract_features(
                model, train_loader, test_loader
            )

            accuracies.append(
                train_and_evaluate_logistic_regression(
                    H_train, y_train_selected, H_test, y_test
                )
            )

            accuracies_majority.append(
                train_and_evaluate_logistic_regression_with_majority_vote(
                    H_train, y_train_selected, H_test, y_test
                )
            )

        log_results(n, accuracies, accuracies_majority)
        accuracies.clear()
        accuracies_majority.clear()


def extract_features(
    model: SimCLRModuleIT, train_loader: DataLoader, test_loader: DataLoader
) -> tuple:
    """Extract features using the SimCLR model."""
    H_train, H_test = [], []

    with torch.no_grad():
        for x, _ in train_loader:
            H_train.append(model.get_h(x).cpu().numpy())
        for x, _ in test_loader:
            H_test.append(model.get_h(x).cpu().numpy())

    return (
        np.concatenate(H_train),
        np.concatenate(H_test),
    )


def train_and_evaluate_logistic_regression(
    H_train: np.ndarray,
    y_train: np.ndarray,
    H_test: np.ndarray,
    y_test: np.ndarray,
    reshape: bool = False,
) -> float:
    """Train and evaluate a logistic regression model."""
    clf = LogisticRegression(max_iter=1000)
    if reshape:
        clf.fit(H_train.reshape(H_train.shape[0], -1), y_train)
        y_pred = clf.predict(H_test.reshape(H_test.shape[0], -1))
    else:
        clf.fit(H_train, y_train)
        y_pred = clf.predict(H_test)
    return accuracy_score(y_test, y_pred)


def train_and_evaluate_logistic_regression_with_majority_vote(
    H_train: np.ndarray, y_train: np.ndarray, H_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Train and evaluate logistic regression with majority vote."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(H_train, y_train)
    y_pred = clf.predict(H_test).reshape(-1, 1)
    y_pred_majority_vote = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred
    )
    return accuracy_score(y_test, y_pred_majority_vote)


def log_results(
    n: int, accuracies: list, accuracies_majority: list
) -> None:
    """Log the evaluation results."""
    logging.info(f"The Accuracy for n={n}: {np.mean(accuracies):.4f}")
    logging.info(
        f"Majority Vote Accuracy for n={n}: {np.mean(accuracies_majority):.4f}"
    )


def main() -> None:
    seeds = get_random_seeds()
    x_train, y_train, x_test, y_test = load_data()

    # Create and load the SimCLR model
    model = SimCLRModuleIT()
    model.load_state_dict(torch.load("simclr/simCLR+InceptionTime/simCLR+IT.pth"))
    model.inference = True

    n_values = [5, 10, 50, 100]
    evaluate_model(model, x_train, y_train, x_test, y_test, n_values, seeds)


if __name__ == "__main__":
    main()
    print("Done")
    print("Check the output.log file for the results.")
