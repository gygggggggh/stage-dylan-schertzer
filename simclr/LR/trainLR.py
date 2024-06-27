from cuml.linear_model import LogisticRegression
import numpy as np
import pickle


def train_logistic_regression(x_train, y_train, x_test, y_test):
    # Convert data to float32 to reduce memory usage
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train.reshape(x_train.shape[0], -1), y_train)
    return model

def main():
    x_train = np.load("simclr/x_train_40k.npy").astype(np.float32)
    y_train = np.load("simclr/y_train_40k.npy").astype(np.float32)
    x_test = np.load("stage_dylan/visulisation/npy/x_test.npy").astype(np.float32)
    y_test = np.load("stage_dylan/visulisation/npy/y_test.npy").astype(np.float32)

    model = train_logistic_regression(x_train, y_train, x_test, y_test)
    with open('simclr/LR/LR_model.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    main()