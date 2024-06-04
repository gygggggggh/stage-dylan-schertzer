# %%
import time
import numpy as np
import geopandas as gpd
from sklearn.metrics import accuracy_score
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

x_train = np.load('npy/x_train.npy')
y_train = np.load('npy/y_train.npy')
meta_train = gpd.read_parquet('npy/meta_train.parquet')

x_test = np.load('npy/x_test.npy')
y_test = np.load('npy/y_test.npy')
meta_test = gpd.read_parquet('npy/meta_test.parquet')

print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}')
print(f'x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}')
print(f'n classes in train: {len(np.unique(y_train))}')
print(f'n classes in test: {len(np.unique(y_test))}')
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
model.fit(x_train_reshaped.reshape(
    x_train_reshaped.shape[0], -1), y_train_repeated)
training_time = time.time() - start_time

y_pred = model.predict(x_test_reshaped.reshape(x_test_reshaped.shape[0], -1))

y_pred = y_pred.reshape(-1, 100)
y_pred_majority_vote = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred)

# %%
accuracy2 = accuracy_score(y_test, y_pred_majority_vote)

print(f'Accuracy on test data with majority vote: {accuracy2}')
print(f'Training time: {training_time:.2f} seconds')

# %% graphique


def create_crop_dict(csv_file_path):
    crop_dict = {}
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            crop_dict[row[0]] = row[1]
    return crop_dict


crop_dict = create_crop_dict('REF_CULTURES_GROUPES_CULTURES_2021.csv')


def plot_sample(x_train, meta_train, sample_index):
    sample = x_train[sample_index]

    plt.figure()

    for i in range(sample.shape[1]):
        plt.plot(sample[:, i])

    crop_type = crop_dict.get(meta_train.iloc[sample_index]["CODE_CULTU"])
    plt.title(crop_type)
    plt.xlabel('time_step')
    plt.ylabel('pixel_value')

    legendList = ["Aerosols", "Blue", "Green", "Red",
                  "Red Edge 1", "Red Edge 2",
                  "Red Edge 3", "NIR", "Red Edge 4",
                  "Water vapor", "SWIR 1", "SWIR 2"]

    plt.legend([f'{legendList[i]}' for i in range(sample.shape[1])])

    plt.show()


plot_sample(x_train_reshaped, meta_train, 34)

# %% show the france map usin shapefile


def plot_france(meta, crop_dict, geojson_path):
    meta = meta.to_crs(epsg=3857)
    meta['geometry'] = meta.centroid

    france = gpd.read_file(geojson_path)
    france = france.to_crs(epsg=3857)

    fig, ax = plt.subplots()
    france.plot(ax=ax, color='white', edgecolor='black')

    # Create a color map with a unique color for each CODE_CULTU
    cmap = plt.cm.get_cmap('nipy_spectral', len(meta['CODE_CULTU'].unique()))

    # Assign a color to each CODE_CULTU
    for i, crop_type in enumerate(meta['CODE_CULTU'].unique()):
        color = cmap(i)
        meta[meta['CODE_CULTU'] == crop_type].plot(
            ax=ax, color=color, markersize=5, label=crop_dict.get(crop_type, "Unknown"))

    # Add a legend to the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()


plot_france(meta_train, crop_dict, 'france.geojson')
# %% un seule


def plot_point_on_map(meta, crop_dict, geojson_path, point_index):
    meta = meta.to_crs(epsg=3857)
    meta['geometry'] = meta.centroid

    france = gpd.read_file(geojson_path)
    france = france.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    france.plot(ax=ax, color='white', edgecolor='black')
    point = meta.iloc[point_index]

    ax.scatter(point.geometry.x, point.geometry.y, color='red', s=100)

    # Create a custom legend
    crop_name = crop_dict.get(point['CODE_CULTU'], "Unknown")
    red_patch = mpatches.Patch(color='red', label=crop_name)
    ax.legend(handles=[red_patch])

    plt.show()


plot_point_on_map(meta_train, crop_dict, 'france.geojson', 34)

# %% random


def user_crop(meta, crop_dict, geojson_path, x_train_reshaped, crop_type):
    samples = meta[meta['CODE_CULTU'] == crop_type]

    if not samples.empty:
        # Get indices that are in both samples and x_train_reshaped
        common_indices = set(samples.index) & set(x_train_reshaped.index)

        if common_indices:
            # Pick a random sample from the common indices
            sample_index = random.choice(list(common_indices))
            print(f"Sample index: {sample_index}")

            # Plot the sample on the map
            plot_point_on_map(meta, crop_dict, geojson_path, sample_index)

            # Plot the sample
            plot_sample(x_train_reshaped, meta, sample_index)
        else:
            print(f"No common samples found for crop type {crop_type} in x_train_reshaped")
    else:
        print(f"No samples found for crop type {crop_type}")

        
user_crop(meta_train, crop_dict, 'france.geojson', x_train_reshaped, 'MIS')

# %%
