import matplotlib.pyplot as plt
import numpy as np

# Example corrupted data
corrupt_data = [1, 2, 4, 4, 5, float('nan'), float('nan'), float('nan'), float('nan'), 8, 9, 10, 11]

# Initialize lists to store segments
segments_x = []
segments_y = []

# Iterate through corrupt_data to collect segments
current_segment_x = []
current_segment_y = []
for i, value in enumerate(corrupt_data):
    if not np.isnan(value):  # Check if value is not NaN
        current_segment_x.append(i)
        current_segment_y.append(value)
    else:
        if current_segment_x:  # If current segment has values, add to segments
            segments_x.append(current_segment_x)
            segments_y.append(current_segment_y)
            current_segment_x = []
            current_segment_y = []

# Add last segment if exists
if current_segment_x:
    segments_x.append(current_segment_x)
    segments_y.append(current_segment_y)

# Plotting
plt.figure(figsize=(8, 6))  # Adjust figure size as needed

# Plot each segment separately
for seg_x, seg_y in zip(segments_x, segments_y):
    plt.plot(seg_x, seg_y, marker='o', color='blue')

# Set labels, title, legend, and grid for the plot
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Corrupted Data with Disjointed Lines')
plt.grid(True)

# Save the plot
plt.savefig('corrupted_data_disjointed_lines.png')

# Show the plot
plt.show()
