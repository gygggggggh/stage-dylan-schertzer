import time
import numpy as np
import os
from aeon.classification.deep_learning import InceptionTimeClassifier

# Environment setup
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

# Load data
x_train = np.load("stage_dylan/visulisation/npy/x_train.npy")
y_train = np.load("stage_dylan/visulisation/npy/y_train.npy")
x_test = np.load("stage_dylan/visulisation/npy/x_test.npy")
y_test = np.load("stage_dylan/visulisation/npy/y_test.npy")


def select_samples_per_class(x, y, n_samples):
    unique_classes = np.unique(y)
    new_x, new_y = [], []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        selected_indices = cls_indices if len(cls_indices) <= n_samples else np.random.choice(
            cls_indices, n_samples, replace=False)
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


def fit_and_score(x_train, y_train, x_test, y_test):
    model = InceptionTimeClassifier(n_epochs=1, batch_size=1, verbose=1)
    start_time = time.time()
    model.fit(x_train, y_train)
    training_time = time.time() - start_time

    accuracy = model.score(x_test, y_test)
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Accuracy on test data: {accuracy:.4f}")

    return accuracy


def main():
    n_values = [5, 10, 50, 100]
    results = select_and_reshape(x_train, y_train, n_values)

    for n, (x_train_reshaped, y_train_reshaped) in results.items():

        print(f"\nFor N = {n}:")
        print(
            f"x_train.shape: {x_train_reshaped.shape}, y_train.shape: {y_train_reshaped.shape}")
        accuracy = fit_and_score(x_train_reshaped, y_train_reshaped,
                                 x_test.reshape(-1, 60, 12).transpose(0, 2, 1), y_test.repeat(100))

    print(f"\nNumber of classes in train: {len(np.unique(y_train))}")
    print(f"Number of classes in test: {len(np.unique(y_test))}")
    print("Done!")


if __name__ == "__main__":
    main()
