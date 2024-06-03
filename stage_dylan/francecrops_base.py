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

import csv

def create_crop_dict(csv_file_path):
    crop_dict = {}
    with open(csv_file_path, newline='',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            crop_dict[row[0]] = row[1]
    return crop_dict


crop_dict = create_crop_dict('REF_CULTURES_GROUPES_CULTURES_2021.csv')
print(crop_dict)

def plot_sample(x_train, meta_train, sample_index):
    sample = x_train[sample_index]

    plt.figure()

    for i in range(sample.shape[1]):  
        plt.plot(sample[:, i])

    crop_type = crop_dict.get(meta_train.iloc[sample_index]["CODE_CULTU"])
    plt.title(crop_type)
    plt.xlabel('time_step')
    plt.ylabel('pixel_value')

    plt.legend([f'Line {i}' for i in range(sample.shape[1])])  

    plt.show()

plot_sample(x_train_reshaped, meta_train, 0)

# %%
