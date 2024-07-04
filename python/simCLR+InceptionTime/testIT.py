import logging
import numpy as np
import torch
import pytorch_lightning as pl
from cuml.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from typing import List, Tuple
from tqdm import tqdm
import gc

from module_simCLR_IT import SimCLRModuleIT
from dataset import NPYDatasetAll




# Constants
LOG_FILE = "testIT.log"
MODEL_PATH = "python/simCLR+InceptionTime/simCLR+IT.pth"
TRAIN_DATA_PATH = {"x": "weights/x_train_40k.npy", "y": "weights/y_train_40k.npy"}
TEST_DATA_PATH = {
    "x": "weights/x_test.npy",
    "y": "weights/y_test.npy",
}

# Configuration
CONFIG = {
    "num_seeds": 1,
    "n_values": [5, 10, 50, 100],
    "batch_size": 512,
    "num_workers": 10,
}

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("simCLR+InceptionTime")


def get_random_seeds(
    num_seeds: int = CONFIG["num_seeds"], seed_range: int = int(1e9)
) -> List[int]:
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
    try:
        x_train = np.load(TRAIN_DATA_PATH["x"])
        y_train = np.load(TRAIN_DATA_PATH["y"])
        
        x_test = np.load(TEST_DATA_PATH["x"])
        y_test = np.load(TEST_DATA_PATH["y"])
        x_test = x_test.reshape(-1, 60, 12)
        y_test = y_test.repeat(100)
        logging.info(
            f"Data shapes: x_train {x_train.shape}, y_train {y_train.shape}, x_test {x_test.shape}, y_test {y_test.shape}"
        )
        return x_train, y_train, x_test, y_test
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        raise


def extract_features(
    model: SimCLRModuleIT, loader: DataLoader, device: torch.device
) -> np.ndarray:
    features = []
    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Extracting features", leave=False):
            x = x.to(device)
            batch_features = model.get_h(x).cpu().numpy()
            features.append(batch_features)
            del x, batch_features
            torch.cuda.empty_cache()
    return np.concatenate(features)


def train_and_evaluate_logistic_regression(
    H_train: np.ndarray, y_train: np.ndarray, H_test: np.ndarray, y_test: np.ndarray
) -> float:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(H_train, y_train)
    y_pred = clf.predict(H_test)
    return accuracy_score(y_test, y_pred)


def train_and_evaluate_logistic_regression_with_majority_vote(
    H_train: np.ndarray, y_train: np.ndarray, H_test: np.ndarray, y_test: np.ndarray
) -> float:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(H_train, y_train)
    
    # Process test data in batches
    batch_size = 512
    y_pred = []
    for i in range(0, len(H_test), batch_size):
        batch = H_test[i:i+batch_size]
        y_pred.extend(clf.predict(batch))
    
    y_pred = np.array(y_pred).reshape(-1, 100)
    y_pred_majority_vote = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred
    )
    return accuracy_score(y_test[::100], y_pred_majority_vote)

def evaluate_for_seed(
    model: SimCLRModuleIT,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n: int,
    seed: int,
    device: torch.device = torch.device("cuda"),
) -> Tuple[float, float]:
    torch.manual_seed(seed)
    pl.seed_everything(seed)

    x_train_selected, y_train_selected = select_samples_per_class(x_train, y_train, n)

    train_dataset = NPYDatasetAll(x_train_selected, y_train_selected)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    test_dataset = NPYDatasetAll(x_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        drop_last=False,
    )

    H_train = extract_features(model, train_loader, device)
    H_test = extract_features(model, test_loader, device)

    accuracy = train_and_evaluate_logistic_regression(
        H_train, y_train_selected, H_test, y_test
    )
    accuracy_majority = train_and_evaluate_logistic_regression_with_majority_vote(
        H_train, y_train_selected, H_test, y_test
    )

    del H_train, H_test
    gc.collect()

    return accuracy, accuracy_majority

def evaluate_model(
    model: SimCLRModuleIT,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_values: List[int],
    seeds: List[int],
    device: torch.device = torch.device("cuda"),
) -> None:
    for n in n_values:
        accuracies = []
        accuracies_majority = []
        for seed in tqdm(seeds, desc=f"Evaluating n={n}"):
            accuracy, accuracy_majority = evaluate_for_seed(
                model, x_train, y_train, x_test, y_test, n, seed
            )
            accuracies.append(accuracy)
            accuracies_majority.append(accuracy_majority)

        logging.info(f"The Accuracy for n={n}: {np.mean(accuracies):.4f}")
        logging.info(
            f"Majority Vote Accuracy for n={n}: {np.mean(accuracies_majority):.4f}"
        )

        gc.collect()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = get_random_seeds()
    x_train, y_train, x_test, y_test = load_data()

    try:
        model = SimCLRModuleIT()
        model.load_state_dict(torch.load(MODEL_PATH))
        model = model.to(device)  # Move model to device
        model.eval()  # Set model to evaluation mode
        model.inference = True
    except FileNotFoundError:
        logging.error(f"Model file not found: {MODEL_PATH}")
        return
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    evaluate_model(model, x_train, y_train, x_test, y_test, CONFIG["n_values"], seeds, device)
    

if __name__ == "__main__":
    main()
    print("Done")
    print(f"Check the {LOG_FILE} file for the results.")