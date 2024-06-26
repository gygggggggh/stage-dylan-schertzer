import pandas as pd
import matplotlib.pyplot as plt

def parse_log(filename):
    data = []
    majority_data = []
    with open(filename, 'r') as file:
        for line in file:
            if "The Accuracy for n=" in line:
                n = int(line.split('=')[1].split(':')[0])
                acc = round(float(line.split('=')[1].split(':')[1].strip())*100, 0)
                data.append((filename, n, acc))
            if "Majority Vote Accuracy for n=" in line:
                n_majority = int(line.split('=')[1].split(':')[0])
                acc_majority = round(float(line.split('=')[1].split(':')[1].strip())*100, 0)
                majority_data.append((filename, n_majority, acc_majority))
    return data, majority_data


# Parse the log files
data_LR, majority_data_LR = parse_log('testLR.log')
data_RN, majority_data_RN = parse_log('testRN.log')
data_IT, majority_data_IT = parse_log('testIT.log')

# Combine the data
data = data_LR + data_RN + data_IT
majority_data = majority_data_LR + majority_data_RN + majority_data_IT

# Create DataFrames
df = pd.DataFrame(data, columns=['Model', 'n', 'Accuracy'])
df_majority = pd.DataFrame(majority_data, columns=['Model', 'n', 'Accuracy'])

# Create pivot tables
table = pd.pivot_table(df, values='Accuracy', index='n', columns='Model')
table_majority = pd.pivot_table(df_majority, values='Accuracy', index='n', columns='Model')

# Plot the tables
fig, axs = plt.subplots(1, 2, figsize=(15, 8))

# Function to plot a table
def plot_table(ax, table, title):
    ax.axis('tight')
    ax.axis('off')
    table_plot = ax.table(cellText=table.values, colLabels=table.columns, rowLabels=table.index, cellLoc='center', loc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(12)
    table_plot.scale(1.2, 1.2)
    ax.set_title(title, fontsize=14)

# Plot the accuracy table
plot_table(axs[0], table, 'Accuracy')

# Plot the majority vote accuracy table
plot_table(axs[1], table_majority, 'Majority Vote Accuracy')

plt.tight_layout()
plt.show()
