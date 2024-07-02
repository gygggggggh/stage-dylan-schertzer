import logging
from typing import List, Tuple
import numpy as np
import torch
import pytorch_lightning as pl
from cuml.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from module_simCLR_RN import SimCLRModuleRN
from dataset import NPYDataset
from tqdm import tqdm


# Configuration
TRAIN_DATA_PATH = "simclr/x_train_40k.npy"
TRAIN_LABELS_PATH = "simclr/y_train_40k.npy"
TEST_DATA_PATH = "stage_dylan/visulisation/npy/x_test.npy"
TEST_LABELS_PATH = "stage_dylan/visulisation/npy/y_test.npy"
MODEL_PATH = "simclr/simCLR+resnet/simCLR+RN.pth"
BATCH_SIZE = 512
NUM_WORKERS = 10
N_VALUES = [5, 10, 50, 100]
NUM_SEEDS = 20

# Setup logging
logging.basicConfig(
    filename="testRN.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_random_seeds(
    num_seeds: int = NUM_SEEDS, seed_range: int = int(1e9)
) -> List[int]:
    """Generate a list of random seeds."""
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
    selected_x = selected_x.reshape(-1, 60, 12)
    selected_y = selected_y.repeat(100)
    
    return selected_x, selected_y


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training and testing data."""
    try:
        x_train = np.load(TRAIN_DATA_PATH)
        y_train = np.load(TRAIN_LABELS_PATH)
        x_test = np.load(TEST_DATA_PATH)
        y_test = np.load(TEST_LABELS_PATH)
        x_test = x_test.reshape(-1, 60, 12)
        y_test = y_test.repeat(100)
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise

    logger.info(
        f"Data shapes: x_train {x_train.shape}, y_train {y_train.shape}, x_test {x_test.shape}, y_test {y_test.shape}"
    )
    return x_train, y_train, x_test, y_test


def extract_features(model: SimCLRModuleRN, loader: DataLoader, device: torch.device) -> np.ndarray:
    features = []
    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Extracting features", leave=False):
            x = x.to(device)
            # Reshape the input: [batch_size, 12] -> [batch_size, 12, 7, 7]
            x = x.unsqueeze(2).unsqueeze(3).repeat(1, 1, 7, 7)
            features.append(model.get_h(x).cpu().numpy())
    return np.concatenate(features)

def train_and_evaluate_logistic_regression(
    H_train: np.ndarray, y_train: np.ndarray, H_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Train and evaluate a logistic regression model."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(H_train, y_train)
    y_pred = clf.predict(H_test)
    return accuracy_score(y_test, y_pred)


def train_and_evaluate_logistic_regression_with_majority_vote(
    H_train: np.ndarray, y_train: np.ndarray, H_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Train and evaluate logistic regression with majority vote."""
    H_test
    print(f"H_test: {H_test.shape}")
    print(f"y_test: {y_test.shape}")
    print(f"H_test: {H_test.shape}")
    print(f"y_test: {y_test.shape}")
    clf = LogisticRegression(max_iter=1000)
    print(f"HH_train: {H_train.shape}")
    print(f"yy_train: {y_train.shape}")
    clf.fit(H_train, y_train)
    y_pred = clf.predict(H_test)
    print(y_pred.shape)
    print(y_pred.shape)
    print(y_pred.shape)
    print(y_pred.shape)
    y_pred = y_pred.reshape(-1, 1)
    print(y_pred.shape)
    y_pred_majority_vote = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred
    )
    print(y_pred_majority_vote.shape)
    print(y_pred_majority_vote.shape)
    print(y_pred_majority_vote.shape)
    print(y_pred_majority_vote.shape)
    print(y_pred_majority_vote.shape)
    print(y_pred_majority_vote.shape)
    print(y_pred_majority_vote.shape)
    return accuracy_score(y_test, y_pred_majority_vote)


def evaluate_model(
    model: SimCLRModuleRN,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_values: List[int],
    seeds: List[int],
    device: torch.device,
) -> None:
    """Evaluate the model with different number of samples per class."""
    for n in tqdm(n_values, desc="Evaluating different n values"):
        accuracies, accuracies_majority = [], []
        for seed in tqdm(seeds, desc=f"Processing seeds for n={n}", leave=False):
            torch.manual_seed(seed)
            pl.seed_everything(seed)

            x_train_selected, y_train_selected = select_samples_per_class(
                x_train, y_train, n
            )

            train_dataset = NPYDataset(x_train_selected, y_train_selected)
            test_dataset = NPYDataset(x_test, y_test)
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                drop_last=False,
                pin_memory=True,
            )

            H_train = extract_features(model, train_loader, device)
            H_test = extract_features(model, test_loader, device)

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

        logger.info(f"The Accuracy for n={n}: {np.mean(accuracies):.4f}")
        logger.info(
            f"Majority Vote Accuracy for n={n}: {np.mean(accuracies_majority):.4f}"
        )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    seeds = get_random_seeds()
    x_train, y_train, x_test, y_test = load_data()

    try:
        model = SimCLRModuleRN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.inference = True
    except FileNotFoundError:
        logger.error(f"Model file not found: {MODEL_PATH}")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    evaluate_model(model, x_train, y_train, x_test, y_test, N_VALUES, seeds, device)


if __name__ == "__main__":
    main()
    logger.info("Evaluation complete. Check the testRN.log file for results.")
