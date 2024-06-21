import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_log(filename):
    data = []
    majority_data = []
    with open(filename, 'r') as file:
        for line in file:
            if "Accuracy for n=" in line:
                n = int(line.split('=')[1].split(':')[0])
                acc = float(line.split('=')[1].split(':')[1].strip())
                data.append((filename, n, acc))
            if "Majority Vote Accuracy for n=" in line:
                n_majority = int(line.split('=')[1].split(':')[0])
                acc_majority = float(line.split('=')[1].split(':')[1].strip())
                majority_data.append((filename, n_majority, acc_majority))
    return data, majority_data

# Parse the log files
data_LR, majority_data_LR = parse_log('testLR.log')
data_RN, majority_data_RN = parse_log('testRN.log')
data_IT, majority_data_IT = parse_log('testIT.log')



# Combine the data
data = data_LR + data_RN + data_IT
majority_data = majority_data_LR + majority_data_RN + majority_data_IT

# Create a DataFrame
df = pd.DataFrame(data, columns=['Model', 'n', 'Accuracy'])
df_majority = pd.DataFrame(majority_data, columns=['Model', 'n', 'Accuracy'])

#create a table
table = pd.pivot_table(df, values='Accuracy', index='n', columns='Model')
table_majority = pd.pivot_table(df_majority, values='Accuracy', index='n', columns='Model')

# Print the table

print(table)
print(table_majority)