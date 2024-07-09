# %%
import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
from cuml.linear_model import LogisticRegression
from dataset import NPYDatasetAll
from module_simCLR_IT import SimCLRModuleIT
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Constants
LOG_FILE = "testIT.log"
MODEL_PATH = "checkpoints/simclr-it-epoch=145.ckpt"
MODEL_PATHS = "checkpoints"

TRAIN_DATA_PATH = {"x": "weights/x_train_40k.npy", "y": "weights/y_train_40k.npy"}
TEST_DATA_PATH = {
    "x": "weights/x_test.npy",
    "y": "weights/y_test.npy",
}

CONFIG_SINGLE = {
    "num_seeds": 20,
    "n_values": [5, 10, 50, 100],
    "batch_size": 1024,
    "num_workers": 8,
}

CONFIG_MULTIPLE = {
    "num_seeds": 4,
    "n_values": [10],
    "batch_size": 1024,
    "num_workers": 8,
}

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("simCLR+InceptionTime")

# %%
def list_model_files(model_paths, suffix):
    try:
        files = os.listdir(model_paths)
        model_files = [file for file in files if os.path.isfile(os.path.join(model_paths, file)) and file.endswith(suffix)]
        return sorted(model_files)
    except FileNotFoundError as e:
        logging.error(f"Error listing files in {model_paths}: {e}")
        return []

def get_random_seeds(num_seeds=20, seed_range=int(1e9)):
    return [np.random.randint(0, seed_range) for _ in range(num_seeds)]

def select_samples_per_class(x, y, n_samples):
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

def load_data():
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

def extract_features(model, loader, device):
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

def train_and_evaluate_logistic_regression_with_majority_vote(H_train, y_train, H_test, y_test):
    clf = LogisticRegression(max_iter=5000)
    clf.fit(H_train, y_train)
    
    y_pred = clf.predict(H_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    y_pred = np.array(y_pred).reshape(-1, 100)
    y_pred_majority_vote = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred
    )
    accuracy_majority = accuracy_score(y_test[::100], y_pred_majority_vote)
    
    return acc, accuracy_majority

def evaluate_for_seed(model, x_train, y_train, x_test, y_test, n, seed, config, device=torch.device("cuda")):
    torch.manual_seed(seed)
    pl.seed_everything(seed)

    x_train_selected, y_train_selected = select_samples_per_class(x_train, y_train, n)
    train_dataset = NPYDatasetAll(x_train_selected, y_train_selected)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )

    test_dataset = NPYDatasetAll(x_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )

    H_train = extract_features(model, train_loader, device)
    H_test = extract_features(model, test_loader, device)

    accuracy, accuracy_majority = train_and_evaluate_logistic_regression_with_majority_vote(
        H_train, y_train_selected, H_test, y_test
    )

    return accuracy, accuracy_majority

def evaluate_model(model, x_train, y_train, x_test, y_test, config, seeds, device=torch.device("cuda")):
    for n in config['n_values']:
        accuracies = []
        accuracies_majority = []
        for seed in tqdm(seeds, desc=f"Evaluating n={n}"):
            accuracy, accuracy_majority = evaluate_for_seed(
                model, x_train, y_train, x_test, y_test, n, seed, config, device
            )
            accuracies.append(accuracy)
            accuracies_majority.append(accuracy_majority)

        logging.info(f"The Accuracy for n={n}: {np.mean(accuracies):.4f}")
        logging.info(
            f"Majority Vote Accuracy for n={n}: {np.mean(accuracies_majority):.4f}"
        )

def main(evaluate_all_checkpoints=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = get_random_seeds()
    x_train, y_train, x_test, y_test = load_data()

    config = CONFIG_MULTIPLE if evaluate_all_checkpoints else CONFIG_SINGLE

    def load_and_evaluate_model(model_path):
        try:
            model = SimCLRModuleIT.load_from_checkpoint(model_path)
            model = model.to(device)  # Move model to device
            model.eval()  # Set model to evaluation mode
            model.inference = True
            logging.info(f"Evaluating with {model_path} : ")
            evaluate_model(model, x_train, y_train, x_test, y_test, config, seeds, device)
        except FileNotFoundError:
            logging.error(f"Model file not found: {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    if evaluate_all_checkpoints:
        model_files = list_model_files(MODEL_PATHS, ".ckpt")
        for model_path in model_files:
            if model_path.endswith('.ckpt'):
                load_and_evaluate_model(os.path.join(MODEL_PATHS, model_path))
    else:
        load_and_evaluate_model(MODEL_PATH)

if __name__ == "__main__":
    main()
    print("Done.")
