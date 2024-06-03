# %%
import time
import numpy as np
import geopandas as gpd
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

x_train = np.load('npy/x_train.npy')
y_train = np.load('npy/y_train.npy')
meta_train = gpd.read_parquet('npy/meta_train.parquet')

x_test = np.load('npy/x_test.npy')
y_test = np.load('npy/y_test.npy')
meta_test = gpd.read_parquet('npy/meta_test.parquet')

print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
print(f'x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}')
print(f'n classes in train: {len(np.unique(y_train))}, n classes in test: {len(np.unique(y_test))}')

gpu = True
if gpu:
    from cuml.linear_model import LogisticRegression
    print('Using cuml.linear_model.LogisticRegression')
else:
    from sklearn.linear_model import LogisticRegression
    print('Using sklearn.linear_model.LogisticRegression')
# %% Training on 100 pixel time series at once
model = LogisticRegression(max_iter=1000)

start_time = time.time()
model.fit(x_train.reshape(x_train.shape[0], -1), y_train)
training_time = time.time() - start_time

y_pred = model.predict(x_test.reshape(x_test.shape[0], -1))

accuracy1 = accuracy_score(y_test, y_pred)
print(f'Accuracy on test data: {accuracy1:.4f}')
print(f'Training time: {training_time:.2f} seconds')

# %% Training on each pixel time series individually and doing a majority vote
x_train_reshaped = x_train.reshape(-1, 60, 12)
x_test_reshaped = x_test.reshape(-1, 60, 12)
y_train_repeated = np.repeat(y_train, 100)

model = LogisticRegression(max_iter=1000)

start_time = time.time()
model.fit(x_train_reshaped.reshape(x_train_reshaped.shape[0], -1), y_train_repeated)
training_time = time.time() - start_time

y_pred = model.predict(x_test_reshaped.reshape(x_test_reshaped.shape[0], -1))

y_pred = y_pred.reshape(-1, 100)
y_pred_majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred)

# %%
accuracy2 = accuracy_score(y_test, y_pred_majority_vote)

print(f'Accuracy on test data with majority vote: {accuracy2}')
print(f'Training time: {training_time:.2f} seconds')

# %% graphique 
sample_index = 21
sample = x_train[sample_index]

plt.figure()

plt.plot(sample[:, 21])
plt.title(f'Crop type: {meta_train.iloc[sample_index].CODE_CULTU}')

plt.xlabel('time_step')
plt.ylabel('pixel_value')

plt.show()

# %%
