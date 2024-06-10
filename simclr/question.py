import numpy as np

# Load your numpy arrays
x_train = np.load("stage_dylan/visulisation/npy/x_train.npy")
y_train = np.load("stage_dylan/visulisation/npy/y_train.npy")
x_test = np.load("stage_dylan/visulisation/npy/x_test.npy")
y_test = np.load("stage_dylan/visulisation/npy/y_test.npy")

# Print shapes of the data arrays
print(f"x_train shape: {x_train[0].shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")