import numpy as np
import joblib
from cuml.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(filename="testLR.log", level=logging.INFO)
logging.info("Logistic Regression")

def get_random_seeds(num_seeds: int = 20, seed_range: int = 1e9) -> list:
    return [np.random.randint(0, seed_range) for _ in range(num_seeds)]

def select_samples_per_class(x: np.ndarray, y: np.ndarray, n_samples: int) -> tuple:
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
    x_train = np.load("stage_dylan/visulisation/npy/x_train.npy")
    y_train = np.load("stage_dylan/visulisation/npy/y_train.npy")
    x_test = np.load("stage_dylan/visulisation/npy/x_test.npy")
    y_test = np.load("stage_dylan/visulisation/npy/y_test.npy")

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test

def fit_and_evaluate_model(x_train, y_train, x_test, y_test, model_path, majority=False):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    joblib.dump(model, model_path)

    if majority:
        x_test_majority = x_test[::5].reshape(-1, x_test.shape[-1])
        y_test_majority = y_test[::5]

        y_pred = model.predict(x_test_majority)

        # Perform majority voting
        y_pred_majority_vote = np.array([np.bincount(y_pred[i:i+5]).argmax() for i in range(0, len(y_pred), 5)])

        # Adjust y_test_majority to match the shape of y_pred_majority_vote
        y_test_majority_adjusted = y_test_majority[:len(y_pred_majority_vote)]

        accuracy = accuracy_score(y_test_majority_adjusted, y_pred_majority_vote)
    else:
        y_pred = model.predict(x_test.reshape(x_test.shape[0], -1))
        accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def evaluate_model(x_train, y_train, x_test, y_test, n_values, seeds):
    for n in n_values:
        accuracies = []
        accuracies_majority = []
        for seed in seeds:
            x_train_selected, y_train_selected = select_samples_per_class(x_train, y_train, n)
            x_train_reshaped = x_train_selected.reshape(x_train_selected.shape[0], -1)
            x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
            y_test_majority = y_test[::5]
            y_train_repeated = np.repeat(y_train_selected, 1)
            y_test_repeated = np.repeat(y_test, 1)

            accuracy = fit_and_evaluate_model(x_train_reshaped, y_train_selected, x_test_reshaped, y_test, 'simclr/LR/LR_model.pkl')
            accuracies.append(accuracy)

            accuracy_majority = fit_and_evaluate_model(x_train_reshaped, y_train_repeated, x_test_reshaped, y_test_repeated, 'simclr/LR/LR_model.pkl', majority=True)
            accuracies_majority.append(accuracy_majority)

        logging.info(f"The Accuracy for n={n}: {np.mean(accuracies):.4f}")
        logging.info(f"Majority Vote Accuracy for n={n}: {np.mean(accuracies_majority):.4f}")

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    seeds = get_random_seeds()
    n_values = [5, 10, 50, 100]
    evaluate_model(x_train, y_train, x_test, y_test, n_values, seeds)
    logging.info("Done")
