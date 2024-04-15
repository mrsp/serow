'''
Plots the data stored in the file specified at plot_config.
Prints the bias in the forces for each leg
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

config_filename = "plot_config.yaml"

with open(config_filename, "r") as file:
  config_options = yaml.load(file, Loader=yaml.FullLoader)

csv_filename = config_options["data_filename"]

df = pd.read_csv(csv_filename)

# Extract y-labels from column names
y_labels = [label.split(",")[0] for label in df.columns[1:]]


# Plot data
for i, column in enumerate(df.columns[1:]):
    plt.figure(figsize=(10, 6))
    plt.plot(df.iloc[:, 0], df[column])
    plt.xlabel(df.columns[0])  # Set xlabel dynamically from the first column
    plt.ylabel(column)
    plt.title(f"{column} vs. {df.columns[0]}")
    plt.show()