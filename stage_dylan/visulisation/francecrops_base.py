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


plot_sample(x_train_reshaped, meta_train, 934)

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

    crop_name = crop_dict.get(point['CODE_CULTU'], "Unknown")
    red_patch = mpatches.Patch(color='red', label=crop_name)
    ax.legend(handles=[red_patch])

    plt.show()


plot_point_on_map(meta_train, crop_dict, 'france.geojson', 934)

# %% random


def plot_random_sample_of_crop(meta, crop_dict, geojson_path, x_train_reshaped):
    # Filter the meta DataFrame to only include rows with the specified crop code
    crop_code = random.choice(meta['CODE_CULTU'].unique())
    crop_meta = meta[meta['CODE_CULTU'] == crop_code]
    # If there are no rows with the specified crop code, print an error message and return
    if len(crop_meta) == 0:
        print(f"No samples found for crop code {crop_code}")
        return

    sample = crop_meta.sample(1)

    sample_index = sample.index[0] % 2000
    print(f"Selected sample index: {sample_index}")
    if sample_index >= len(x_train_reshaped):
        print(f"Index {sample_index} is out of range for x_train_reshaped")
        return


    plot_sample(x_train_reshaped, meta, sample_index)

    plot_point_on_map(meta, crop_dict, geojson_path, sample_index)


plot_random_sample_of_crop(meta_train, crop_dict, 'france.geojson', x_train_reshaped,)

# %%
def plot_sample_by_crop(x_train, meta_train, crop_code):
    indices = np.where(meta_train['CODE_CULTU'] == crop_code)[0]
    if len(indices) == 0:
        print(f"No samples found for crop code {crop_code}")
        return

    sample_index = np.random.choice(indices)

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

plot_sample_by_crop(x_train_reshaped, meta_train, 'MIS')
# %%
def plot_point_on_map_by_crop(meta, crop_dict, geojson_path, crop_code):
    indices = np.where(meta['CODE_CULTU'] == crop_code)[0]
    if len(indices) == 0:
        print(f"No samples found for crop code {crop_code}")
        return

    point_index = np.random.choice(indices)

    meta = meta.to_crs(epsg=3857)
    meta['geometry'] = meta.centroid

    france = gpd.read_file(geojson_path)
    france = france.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    france.plot(ax=ax, color='white', edgecolor='black')
    point = meta.iloc[point_index]

    ax.scatter(point.geometry.x, point.geometry.y, color='red', s=100)

    crop_name = crop_dict.get(point['CODE_CULTU'], "Unknown")
    red_patch = mpatches.Patch(color='red', label=crop_name)
    ax.legend(handles=[red_patch])

    plt.show()

# Example usage:
plot_point_on_map_by_crop(meta_train, crop_dict, 'france.geojson', 'MIS')

# %%
def plot_crop_sample_and_map(x_train, meta, crop_dict, geojson_path):
    crop_code = input("Enter a crop code: ")

    plot_sample_by_crop(x_train, meta, crop_code)

    plot_point_on_map_by_crop(meta, crop_dict, geojson_path, crop_code)

plot_crop_sample_and_map(x_train_reshaped, meta_train, crop_dict, 'france.geojson')
