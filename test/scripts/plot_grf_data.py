import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

config_filename = "plot_config.yaml"

with open(config_filename, "r") as file:
    config_options = yaml.load(file, Loader=yaml.FullLoader)

csv_filename = config_options["data_filename"]
force_column_indxs = config_options["force_column_indxs"]
df = pd.read_csv(csv_filename)

# Compute the low bias for each force column
low_bias_values = df.iloc[:, force_column_indxs].min()

# Subtract the low bias from all measurements in each force column
df.iloc[:, force_column_indxs] -= low_bias_values.values

# Get column names using .iloc
column_names = df.columns.tolist()

# Print the biases for each force column
print("Bias values:")
for idx, bias in zip(force_column_indxs, low_bias_values):
    print(f"Force {column_names[idx]} --> {bias}")

# Plot data
for i, idx in enumerate(force_column_indxs):
    plt.figure(figsize=(10, 6))
    plt.plot(df.iloc[:, 0].values, df.iloc[:, idx].values)  # Use column index from config
    plt.xlabel(df.columns.values[0])  # Set xlabel dynamically from the first column
    plt.ylabel(df.columns.values[idx])  # Set ylabel dynamically
    plt.title(f"{df.columns[idx]} vs. {df.columns[0]}")
    plt.ylim(0, df.iloc[:, idx].max())  # Set y-axis limit from 0 to max value
    plt.show()
