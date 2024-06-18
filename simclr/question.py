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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Prepare the data
data = {
    'Method': ['Logistic Regression', 'MiniRocket', 'InceptionTime', 'TimeMAE', 'SimCLR + GaPP + AvgP'],
    '5': [42, 47, 56, 53, 62],
    '10': [49, 58, 65, 61, 70],
    '50': [68, 75, 77, 75, 80],
    '100': [74, 79, 80, 79, 82]
}

df = pd.DataFrame(data)
df.set_index('Method', inplace=True)

# Step 2: Create the table using seaborn and matplotlib
fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size

# Hide axes
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 rowLabels=df.index,
                 cellLoc='center',
                 loc='center',
                 colColours=sns.color_palette("coolwarm", len(df.columns)))

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

# Set a title
plt.title('Performance of Different Methods', fontsize=16)

# Save the table as an image (optional)
plt.savefig('table.png', bbox_inches='tight')

# Show the table
plt.show()
