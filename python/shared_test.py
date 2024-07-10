import logging
import os
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from cuml.linear_model import LogisticRegression
from python.dataset import NPYDatasetAll
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def list_model_files(model_paths: str, suffix: str = "") -> List[str]:
    try:
        files = os.listdir(model_paths)
        model_files = [file for file in files if os.path.isfile(os.path.join(model_paths, file)) and file.endswith(suffix)]
        return sorted(model_files)
    except FileNotFoundError as e:
        logging.error(f"Error listing files in {model_paths}: {e}")
        return []

def get_random_seeds(num_seeds: int = 20, seed_range: int = int(1e9)) -> List[int]:
    return [np.random.randint(0, seed_range) for _ in range(num_seeds)]

def select_samples_per_class(x: np.ndarray, y: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
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

def load_data(train_data_path: dict, test_data_path: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        x_train = np.load(train_data_path["x"])
        y_train = np.load(train_data_path["y"])
        
        x_test = np.load(test_data_path["x"])
        y_test = np.load(test_data_path["y"])
        x_test = x_test.reshape(-1, 60, 12)
        y_test = y_test.repeat(100)
        logging.info(
            f"Data shapes: x_train {x_train.shape}, y_train {y_train.shape}, x_test {x_test.shape}, y_test {y_test.shape}"
        )
        return x_train, y_train, x_test, y_test
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        raise

def extract_features(model, loader: DataLoader, device: torch.device) -> np.ndarray:
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

def train_and_evaluate_logistic_regression_with_majority_vote(
    H_train: np.ndarray, y_train: np.ndarray, H_test: np.ndarray, y_test: np.ndarray
) -> Tuple[float, float]:
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

def evaluate_for_seed(model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, 
                      n: int, seed: int, config: dict, device: torch.device) -> Tuple[float, float]:
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

def evaluate_model(model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, 
                   config: dict, seeds: List[int], device: torch.device) -> None:
    for n in config['n_values']:
        accuracies = []
        accuracies_majority = []
        for seed in tqdm(seeds, desc=f"Evaluating n={n}"):
            accuracy, accuracy_majority = evaluate_for_seed(
                model, x_train, y_train, x_test, y_test, n, seed, config, device
            )
            accuracies.append(accuracy)
            accuracies_majority.append(accuracy_majority)
            
            print(f'Accuracy for n={n}: {accuracy:.4f}')
            print(f'Majority Vote Accuracy for n={n}: {accuracy_majority:.4f}')

        logging.info(f"The Accuracy for n={n}: {np.mean(accuracies):.4f}")
        logging.info(
            f"Majority Vote Accuracy for n={n}: {np.mean(accuracies_majority):.4f}"
        )

def main(model_class, model_paths: str, train_data_path: dict, test_data_path: dict, config: dict, 
         evaluate_all_checkpoints: bool = False, model_path: str = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = get_random_seeds(config['num_seeds'])
    x_train, y_train, x_test, y_test = load_data(train_data_path, test_data_path)

    def load_and_evaluate_model(model_path: str) -> None:
        try:
            model = model_class.load_from_checkpoint(model_path)
            model = model.to(device)
            model.eval()
            model.inference = True
            logging.info(f"Evaluating with {model_path} : ")
            evaluate_model(model, x_train, y_train, x_test, y_test, config, seeds, device)
        except FileNotFoundError:
            logging.error(f"Model file not found: {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    if evaluate_all_checkpoints:
        model_files = list_model_files(model_paths, ".ckpt")
        for model_file in model_files[::config['checkpoints_skips']]:
            load_and_evaluate_model(os.path.join(model_paths, model_file))
    else:
        load_and_evaluate_model(model_path)