#!/bin/bash

# Define the values for x and y
x_values=(10 90 150)
y_values=(5 10 25 50)

# Create the output directory if it doesn't exist
mkdir -p output

# Loop over each combination of x and y values
for i in "${!x_values[@]}"; do
    for j in "${!y_values[@]}"; do
        x=${x_values[i]}
        y=${y_values[j]}
        
        # Execute the command with the current x and y values asynchronously
        output_file="output/train_${x}_pred_${y}.txt"
        accelerate launch finetune_llm.py -f ./benchmark/ecg_data-imputation --train "$x" --pred "$y" -p "Predict the unknown values of the ECG data" > "$output_file"

        echo "Output saved to: $output_file"
    done
done


mkdir -p "$1"
mv output "$1"

for i in "${!x_values[@]}"; do
    for j in "${!y_values[@]}"; do
        x=${x_values[i]}
        y=${y_values[j]}

        mv "output_for_train_${x}_pred_${y}.txt" "$1"
        mv "output_for_train_${x}_pred_${y}.png" "$1"
    done
done