import os
import re
import sys

import pandas as pd

sys.path.append(os.getcwd())

# Initialize lists to store extracted data
video_names = []
bin_labels = []
ego_involves = []

# Read and parse the txt file
with open('critical_classification/critical_dataset/Crash-1500.txt', 'r') as file:
    for line in file:
        # Split the line by commas
        parts = line.strip().split(',')

        # Extract relevant information
        video_name = f'{parts[0]}.mp4'
        # Use regex to find the part within square brackets
        bin_label = re.search(r'\[.*?\]', line).group()[1: -4]

        ego_involve = parts[-1]  # Last element in the list

        # Append to lists
        video_names.append(video_name)
        bin_labels.append(bin_label)
        ego_involves.append(ego_involve)

# Create a DataFrame
df = pd.DataFrame({
    'Video Name': video_names,
    'Binary Labels': bin_labels,
    'Ego Involve': ego_involves
})

# Write the DataFrame to an Excel file
df.to_excel('critical_classification/critical_dataset/car_crash_metadata.xlsx', index=False)

print("Data has been successfully written")
