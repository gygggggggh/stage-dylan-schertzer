import logging
from typing import List, Tuple

import cupy as cp
import numpy as np
from cuml.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

logging.basicConfig(
    filename="testLR.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Configuration
CONFIG = {
    "x_train_path": "data/x_train_40k.npy",
    "y_train_path": "data/y_train_40k.npy",
    "x_test_path": "data/x_test.npy",
    "y_test_path": "data/y_test.npy",
    "model_path": "python/LR/LR_model.pkl",
    "n_values": [5, 10, 50, 100],
    "num_seeds": 20,
}


def get_random_seeds(num_seeds: int = 20, seed_range: int = int(1e9)) -> List[int]:
    return [np.random.randint(0, seed_range) for _ in range(num_seeds)]


def select_samples_per_class(
    x: np.ndarray, y: np.ndarray, n_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
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

    selected_x = np.concatenate(selected_x)
    selected_y = np.concatenate(selected_y)

    return selected_x, selected_y


def load_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        x_train = np.load(config["x_train_path"])
        y_train = np.load(config["y_train_path"])
        x_test = np.load(config["x_test_path"])
        y_test = np.load(config["y_test_path"])
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise

    return x_train, y_train, x_test, y_test


def fit_and_evaluate_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    majority: bool = False,
) -> float:
    # Reshape the data
    x_train_gpu = x_train.reshape(x_train.shape[0] * 100, -1)
    x_test_gpu = x_test.reshape(x_test.shape[0] * 100, -1)
    y_train_gpu = y_train.repeat(100)
    y_test_gpu = y_test.repeat(100)

    # Train the model
    model = LogisticRegression(max_iter=5000)
    model.fit(x_train_gpu, y_train_gpu)

    # Predict
    y_pred = model.predict(x_test_gpu)

    # Compute accuracy
    if majority:
        y_pred_reshaped = y_pred.reshape(-1, 100)
        y_pred_majority_vote = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred_reshaped
        )
        accuracy = accuracy_score(
            cp.asnumpy(y_test_gpu[::100]), cp.asnumpy(y_pred_majority_vote)
        )
    else:
        accuracy = accuracy_score(y_test_gpu, y_pred)

    return accuracy


def evaluate_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_values: List[int],
    seeds: List[int],
    config: dict,
):
    for n in tqdm(n_values, desc="Processing n_values"):
        accuracies, accuracies_majority = [], []
        for seed in tqdm(seeds, desc=f"Processing seeds for n={n}", leave=False):
            np.random.seed(seed)
            x_train_selected, y_train_selected = select_samples_per_class(
                x_train, y_train, n
            )

            accuracy = fit_and_evaluate_model(
                x_train_selected,
                y_train_selected,
                x_test,
                y_test,
            )
            accuracies.append(accuracy)

            accuracy_majority = fit_and_evaluate_model(
                x_train_selected,
                y_train_selected,
                x_test,
                y_test,
                majority=True,
            )
            accuracies_majority.append(accuracy_majority)

        logger.info(f"The Accuracy for n={n}: {np.mean(accuracies):.4f}")
        logger.info(
            f"Majority Vote Accuracy for n={n}: {np.mean(accuracies_majority):.4f}"
        )


def main(config: dict):
    logger.info("Starting Logistic Regression evaluation")

    x_train, y_train, x_test, y_test = load_data(config)
    seeds = get_random_seeds(config["num_seeds"])
    evaluate_model(x_train, y_train, x_test, y_test, config["n_values"], seeds, config)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main(CONFIG)
