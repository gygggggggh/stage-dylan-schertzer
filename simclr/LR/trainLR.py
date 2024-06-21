from cuml.linear_model import LogisticRegression
import numpy as np
import pickle

def train_logistic_regression(x_train, y_train, x_test, y_test):
    model = LogisticRegression(max_iter=3000)
    model.fit(x_train.reshape(x_train.shape[0], -1), y_train)
    return model


def main():
    x_train = np.load("stage_dylan/visulisation/npy/x_train.npy")
    y_train = np.load("stage_dylan/visulisation/npy/y_train.npy")
    x_test = np.load("stage_dylan/visulisation/npy/x_test.npy")
    y_test = np.load("stage_dylan/visulisation/npy/y_test.npy")

    model = train_logistic_regression(x_train, y_train, x_test, y_test)
    with open('simclr/LR/LR_model.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    main()