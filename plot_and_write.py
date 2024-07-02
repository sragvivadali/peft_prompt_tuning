import matplotlib.pyplot as plt
import numpy as np 
import os
import csv
import math

def write_to_output(var_names, values, filename):
    # Create the folder if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs('output')
    
    # Path to the file inside the folder
    file_path = os.path.join("output", filename)
    
    # Create the file if it doesn't exist
    if not os.path.exists(filename):
        with open(filename, "w") as file:
            file.write("")

    # Write values to the file
    with open(filename, "a") as file:
        for var_name, value in zip(var_names, values):
            file.write(f"{var_name}: {value}\n")

        file.write("----------------------\n")

    
def plot_predicted_vs_ground_truth(pred_values, gt_values, title="Predicted vs. Ground Truth", output = "PredictedvGT.png", query = "imputation"):
    # Read the input CSV file
    with open('./benchmark/test.csv', 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)

    # Set up the number of rows and columns for subplots
    num_plots = len(pred_values)
    cols = 2
    rows = (num_plots // cols) + (num_plots % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15 * cols, 15 * rows))  # Set figure size for 15x15 subplots
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i in range(len(pred_values)):
        if i + 1 < len(pred_values):
            true_result = []
            pred_result = []
            corrupt_data = lines[i + 1][0].split(', ')
            j = 0

            if query == "imputation":
                for x in corrupt_data:
                    if x != 'nan':
                        true_result.append(float(x))
                        pred_result.append(float(x))
                    else:
                        if j < len(gt_values):
                            true_result.append(float(gt_values[i][j]))
                        if j < len(pred_values):
                            pred_result.append(float(pred_values[i][j]))
            elif query == "extrapolation":  # Extrapolation
                valid_data = [float(x) for x in corrupt_data if not math.isnan(float(x))]
                true_result = valid_data + list(map(float, gt_values[i]))
                pred_result = valid_data + list(map(float, pred_values[i][:len(gt_values[i])]))

            # Plot the data for this line on a subplot
            ax = axes[i - 1]
            ax.plot(true_result, label='Combined Data with Ground Truth', color='blue')
            ax.plot(pred_result, label='Predicted String', color='red')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.set_title(f'{title} - Line {i}')
            ax.legend()
            ax.grid(True)

    plt.savefig(output)
    plt.show()