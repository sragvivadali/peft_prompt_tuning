#!/bin/bash

# Define the values for x and y
x_values=(10)
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
        accelerate launch finetune_llm.py -f ./benchmark/ppg-extrapolation --query "extrapolation" --train "$x" --pred "$y" -p "Predict the next values given the PPG data" > "$output_file"

        echo "Output saved to: $output_file"
    done
done
