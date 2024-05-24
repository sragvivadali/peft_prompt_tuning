import os
import csv
import numpy as np

def initialize_files(folder, text, label, train_length = 100, test_length = 50, eval_length = 20, pred_len = 50):
    """
    Initialize output CSV files and populate them with data from .npy files.

    This function creates three CSV files for evaluation, testing, and training data.
    Each CSV file contains rows with 'corrupted ecg' and 'gt values' columns.
    The 'corrupted ecg' column contains 1-dimensional arrays extracted from .npy files,
    and the 'gt values' column contains corresponding ground truth values.
    """
    output_csv_file = ["./benchmark/train.csv", "./benchmark/test.csv", "./benchmark/eval.csv"]
    start = [1, train_length + 1, train_length + test_length + 1]
    end = [train_length, train_length + test_length, train_length + test_length + eval_length]
    
    npy_folder = folder
    # List all .npy files in the specified directory
    npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]

    if (train_length + test_length + eval_length) > 2000:
        print(f"Invalid lengths")
        return -1

    # Open the CSV file for writing
    for i in range(len(output_csv_file)):
        with open(output_csv_file[i], mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([text, label])
            # Process each .npy file
            for npy_file in npy_files:
                if 'gt' in npy_file:
                    continue
                
                idx = int(npy_file.split('_')[0])
                midx = int(npy_file.split('_')[-1].split('.npy')[0])

                if idx >= start[i] and idx <= end[i] and midx==pred_len:
                    # Load the 1-dimensional array from the .npy file
                    array = np.load(os.path.join(npy_folder, npy_file))
                    
                    # Ensure the array is 1-dimensional
                    if array.ndim != 1:
                        print(f"Warning: {npy_file} contains a non-1-dimensional array. Skipping.")
                        continue

                    max_val = max(abs(array))
                    array = [(x * 100.00) / max_val for x in array]
                    
                    array_str =[]
                    for a in array:
                        if np.isnan(a):
                            array_str.append('nan')
                        else:
                            array_str.append(str(a.astype(np.int64)))
                    array_str = ', '.join(array_str)

                    # also put down gt for reference
                    org_file = npy_file.split('.npy')[0]+'_gt.npy'
                    org_array = np.load(os.path.join(npy_folder, org_file))

                    max_val = max(abs(org_array))
                    org_array = [(x * 100.00) / max_val for x in org_array]


                    org_array_value = np.array(org_array, dtype=np.int64)
                    org_array_value_str = ', '.join([str(a) for a in org_array_value])
                    # Write the array as a new row in the CSV file
                    writer.writerow([array_str, org_array_value_str])
                    # break 

    print(f"All 1-dimensional arrays have been saved to {output_csv_file}.")


initialize_files("./benchmark/ppg-imputation","corrupted", "gt")