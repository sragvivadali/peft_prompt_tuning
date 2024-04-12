import numpy as np
import csv
import os
import pdb 

# Specify the directory containing the .npy files and the output CSV file name
npy_directory = './benchmark/ecg_data-imputation/'
# output_csv_file = './benchmark/ecg_imp_train2.csv'
# start, end = 101, 110

# output_csv_file = './benchmark/ecg_imp_eval2.csv'
# start, end = 91, 100

output_csv_file = './benchmark/ecg_imp_test2.csv'
start, end = 1, 90

# List all .npy files in the specified directory
npy_files = [f for f in os.listdir(npy_directory) if f.endswith('.npy')]

# Open the CSV file for writing
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['corrupted ecg', 'gt values'])
    # Process each .npy file
    for npy_file in npy_files:

        if 'gt' in npy_file:
            continue
        
        idx = int(npy_file.split('_')[0])
        midx = int(npy_file.split('_')[-1].split('.npy')[0])

        if idx >= start and idx <= end and midx==50:
            # Load the 1-dimensional array from the .npy file
            array = np.load(os.path.join(npy_directory, npy_file))
            
            # Ensure the array is 1-dimensional
            if array.ndim != 1:
                print(f"Warning: {npy_file} contains a non-1-dimensional array. Skipping.")
                continue
            
            array = 1000 * array
            array_str =[]
            for a in array:
                if np.isnan(a):
                    array_str.append('nan')
                else:
                    array_str.append(str(a.astype(np.int64)))
            array_str = ', '.join(array_str)

            # also put down gt for reference
            org_file = npy_file.split('.npy')[0]+'_gt.npy'
            org_array = np.load(os.path.join(npy_directory, org_file))

            org_array_value = (1000*org_array).astype(np.int64)
            org_array_value_str = ', '.join([str(a) for a in org_array_value])
            
            # Write the array as a new row in the CSV file
            writer.writerow([array_str, org_array_value_str])
            # break 

print(f"All 1-dimensional arrays have been saved to {output_csv_file}.")
