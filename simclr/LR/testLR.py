import numpy as np
import joblib
from cuml.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import logging
from typing import List, Tuple
from pathlib import Path


# Configure logging
logging.basicConfig(
    filename="testLR.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "x_train_path": "simclr/x_train_40k.npy",
    "y_train_path": "simclr/y_train_40k.npy",
    "x_test_path": "stage_dylan/visulisation/npy/x_test.npy",
    "y_test_path": "stage_dylan/visulisation/npy/y_test.npy",
    "model_path": "simclr/LR/LR_model.pkl",
    "n_values": [5, 10, 50, 100],
    "num_seeds": 20,
    "validation_split": 0.2,
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

    return np.concatenate(selected_x), np.concatenate(selected_y)


def load_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        x_train = np.load(config["x_train_path"])
        y_train = np.load(config["y_train_path"])
        x_test = np.load(config["x_test_path"]).astype(np.float32)
        y_test = np.load(config["y_test_path"]).astype(np.float32)
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise

    logger.info(
        f"Data shapes: x_train {x_train.shape}, y_train {y_train.shape}, x_test {x_test.shape}, y_test {y_test.shape}"
    )
    return x_train, y_train, x_test, y_test


def fit_and_evaluate_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_path: str,
    majority: bool = False,
    validation_split: float = 0.2,
) -> Tuple[float, float]:
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        x_train, y_train, test_size=validation_split, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_split, y_train_split)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    y_val_pred = model.predict(x_val_split)
    val_accuracy = accuracy_score(y_val_split, y_val_pred)
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    if majority:
        x_test_majority = x_test.reshape(x_test.shape[0], -1)
        y_pred = model.predict(x_test_majority)
        y_pred_majority_vote = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred.reshape(-1, 1)
        )
        accuracy = accuracy_score(y_test, y_pred_majority_vote)
    else:
        y_pred = model.predict(x_test.reshape(x_test.shape[0], -1))
        accuracy = accuracy_score(y_test, y_pred)

    return accuracy, val_accuracy


def evaluate_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_values: List[int],
    seeds: List[int],
    config: dict,
):
    for n in n_values:
        accuracies, accuracies_majority = [], []
        for seed in seeds:
            np.random.seed(seed)
            x_train_selected, y_train_selected = select_samples_per_class(
                x_train, y_train, n
            )
            x_train_reshaped = x_train_selected.reshape(x_train_selected.shape[0], -1)
            x_test_reshaped = x_test.reshape(x_test.shape[0], -1)

            accuracy, _ = fit_and_evaluate_model(
                x_train_reshaped,
                y_train_selected,
                x_test_reshaped,
                y_test,
                config["model_path"],
            )
            accuracies.append(accuracy)

            accuracy_majority, _ = fit_and_evaluate_model(
                x_train_reshaped,
                y_train_selected,
                x_test_reshaped,
                y_test,
                config["model_path"],
                majority=True,
            )
            accuracies_majority.append(accuracy_majority)

        logger.info(f"Accuracy for n={n}: {np.mean(accuracies):.4f}")
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
