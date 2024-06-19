import time
import numpy as np
import geopandas as gpd
from sklearn.metrics import accuracy_score


def load_data():
    x_train = np.load("npy/x_train.npy")
    y_train = np.load("npy/y_train.npy")
    x_test = np.load("npy/x_test.npy")
    y_test = np.load("npy/y_test.npy")

    return x_train, y_train, x_test, y_test


def print_data_info(x_train, y_train, x_test, y_test):
    print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
    print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}")
    print(f"n classes in train: {len(np.unique(y_train))}")
    print(f"n classes in test: {len(np.unique(y_test))}")


def get_model(gpu):
    if gpu:
        from cuml.linear_model import LogisticRegression

        print("Using cuml.linear_model.LogisticRegression")
    else:
        from sklearn.linear_model import LogisticRegression

        print("Using sklearn.linear_model.LogisticRegression")

    return LogisticRegression(max_iter=1000)


def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    start_time = time.time()
    model.fit(x_train.reshape(x_train.shape[0], -1), y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(x_test.reshape(x_test.shape[0], -1))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test data: {accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")


def train_and_evaluate_with_majority_vote(model, x_train, y_train, x_test, y_test):
    x_train_reshaped = x_train.reshape(-1, 60, 12)
    x_test_reshaped = x_test.reshape(-1, 60, 12)
    y_train_repeated = np.repeat(y_train, 100)

    start_time = time.time()
    model.fit(x_train_reshaped.reshape(x_train_reshaped.shape[0], -1), y_train_repeated)
    training_time = time.time() - start_time

    y_pred = model.predict(x_test_reshaped.reshape(x_test_reshaped.shape[0], -1))

    y_pred = y_pred.reshape(-1, 100)
    y_pred_majority_vote = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred
    )

    accuracy = accuracy_score(y_test, y_pred_majority_vote)

    print(f"Accuracy on test data with majority vote: {accuracy}")
    print(f"Training time: {training_time:.2f} seconds")


def main():
    x_train, y_train, x_test, y_test = load_data()
    print_data_info(x_train, y_train, x_test, y_test)
    model = get_model(gpu=True)
    train_and_evaluate(model, x_train, y_train, x_test, y_test)
    train_and_evaluate_with_majority_vote(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
