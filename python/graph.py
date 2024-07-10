import pandas as pd
import matplotlib.pyplot as plt

def parse_log(filename):
    data = []
    majority_data = []
    with open(filename, 'r') as file:
        for line in file:
            if "The Accuracy for n=" in line:
                n, acc = extract_data(line, "The Accuracy for n=")
                data.append((filename, n, acc))
            elif "Majority Vote Accuracy for n=" in line:
                n, acc = extract_data(line, "Majority Vote Accuracy for n=")
                majority_data.append((filename, n, acc))
    return data, majority_data

def extract_data(line, prefix):
    parts = line.split('=')[1].split(':')
    n = int(parts[0])
    acc = round(float(parts[1].strip()) * 100, 0)
    return n, acc

def parse_all_logs(filenames):
    all_data = []
    all_majority_data = []
    for filename in filenames:
        data, majority_data = parse_log(filename)
        all_data.extend(data)
        all_majority_data.extend(majority_data)
    return all_data, all_majority_data

def create_dataframes(data, majority_data):
    df = pd.DataFrame(data, columns=['Model', 'n', 'Accuracy'])
    df_majority = pd.DataFrame(majority_data, columns=['Model', 'n', 'Accuracy'])
    return df, df_majority

def create_pivot_tables(df, df_majority):
    table = pd.pivot_table(df, values='Accuracy', index='n', columns='Model')
    table_majority = pd.pivot_table(df_majority, values='Accuracy', index='n', columns='Model')
    return table, table_majority

def plot_table(ax, table, title):
    ax.axis('tight')
    ax.axis('off')
    table_plot = ax.table(cellText=table.values, colLabels=table.columns, 
                          rowLabels=table.index, cellLoc='center', loc='center')
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(12)
    table_plot.scale(1.2, 1.2)
    ax.set_title(title, fontsize=14)

def plot_results(table, table_majority):
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    plot_table(axs[0], table, 'Accuracy')
    plot_table(axs[1], table_majority, 'Majority Vote Accuracy')
    plt.tight_layout()
    plt.show()

def main():
    filenames = ['testLR.log', 'testRN.log', 'testIT.log']
    data, majority_data = parse_all_logs(filenames)
    df, df_majority = create_dataframes(data, majority_data)
    table, table_majority = create_pivot_tables(df, df_majority)
    plot_results(table, table_majority)

if __name__ == "__main__":
    main()