import matplotlib.pyplot as plt
import numpy as np 
import os
import csv

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

    
def plot_predicted_vs_ground_truth(input_file, title="Predicted vs. Ground Truth", output = "PredictedvGT.png", query = "imputation"):
    # Read the input CSV file
    with open('./benchmark/test.csv', 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)

    with open(input_file, "r") as file:
        content = file.read()

    blocks = content.split("----------------------")

    # Set up the number of rows and columns for subplots
    num_plots = len(blocks)
    cols = 2
    rows = (num_plots // cols) + (num_plots % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15 * cols, 15 * rows))  # Set figure size for 15x15 subplots
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, block in enumerate(blocks):
        content_lines = block.strip().split("\n")

        pred_str = None
        for line in content_lines:
            if line.startswith("pred_str:"):
                pred_str = line.split(":")[1].strip()
            elif line.startswith("target_str:"):
                target_str = line.split(":")[1].strip()

        if pred_str and target_str:
            pred_values = []

            for x in pred_str.split(', '):
                x = x.strip()
                try:
                    pred_values.append(float(x))
                except ValueError:
                    print(f"Skipping invalid value in pred_str: {x}")


        if i + 1 < len(blocks):
            true_result = []
            pred_result = []
            corrupt_data = lines[i + 1][0].split(', ')
            gt = lines[i + 1][1].split(', ')
            j = 0
            k = 0
            
            if query == "imputation":
                for x in corrupt_data:
                    if x != 'nan':
                        value = float(x)
                    else:
                        if j < len(gt):
                            value = float(gt[j])
                            j += 1
                    true_result.append(value)

                for y in corrupt_data:
                    if y != 'nan':
                        value = float(y)
                    else:
                        if k < len(pred_values):
                            value = float(pred_values[k])
                            k += 1
                    pred_result.append(value)
            else:
                true_result = corrupt_data + gt
                pred_result = corrupt_data + pred_values

            # Plot the data for this line on a subplot
            ax = axes[i - 1]
            ax.plot(true_result, label='Combined Data with Ground Truth', color='blue')
            ax.plot(pred_result, label='Predicted String', color='red')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.set_title(f'{title} - Line {i}')
            ax.legend()
            ax.grid(True)


    # Adjust layout
    plt.tight_layout()
    plt.savefig(output)
    plt.show()
