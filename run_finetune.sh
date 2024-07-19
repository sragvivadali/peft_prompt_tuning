#!/bin/bash

# Define the values for x and y
x_values=(10 90 150)
y_values=(5 10 25 50)

if [ -z "$1" ]; then
    echo "Parameter is null or empty"
    exit 1
fi


# Create the output directory inside the first argument folder
mkdir -p "$1"

# Loop over each combination of x and y values
for i in "${!x_values[@]}"; do
    for j in "${!y_values[@]}"; do
        x=${x_values[i]}
        y=${y_values[j]} 

        # Run the command and save the output using tee
        accelerate launch finetune_llm.py -f ./benchmark/"$1" --train "$x" --pred "$y" -p "$2" --query "$3"

        # Define expected filenames
        expected_txt="output_for_${x}_and_${y}.txt"
        expected_png="output_for_${x}_and_${y}.png"

        # Move files if they exist
        if [ -f "$expected_txt" ]; then
            mv "$expected_txt" "$1"
        fi
        if [ -f "$expected_png" ]; then
            mv "$expected_png" "$1"
        fi
    done
done
